
from pathlib import Path
import re
from typing import Tuple

import pandas as pd
import networkx as nx

try:
	import yaml  # used by some callers, safe to keep
except Exception:
	yaml = None


def safe_filename(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def build_hfacs_category_columns(df: pd.DataFrame, hfacs_categories: dict) -> pd.DataFrame:
	"""Create top-level HFACS category columns by OR-ing mapped sub-columns.

	This function tries to be forgiving about minor differences in column
	naming (commas, spaces, case). If a requested sub-column name isn't
	found exactly, it attempts to match by a normalized form. If a unique

	match is found, it's used; ambiguous or missing mappings still raise.
	"""
	df = df.copy()

	def _norm(s: str) -> str:
		return re.sub(r"[^A-Za-z0-9]+", "", str(s)).lower()

	# build a map of normalized -> actual column names (may be multiple)
	col_map = {}
	for c in df.columns:
		col_map.setdefault(_norm(c), []).append(c)

	for category, subcols in hfacs_categories.items():
		if not isinstance(subcols, list) or len(subcols) == 0:
			raise ValueError(f"hfacs_categories[{category}] must be a non-empty list")

		# allow config entries that contain comma-separated column names
		flat_subcols = []
		for sc in subcols:
			if isinstance(sc, str) and "," in sc:
				parts = [p.strip() for p in sc.split(",") if p.strip()]
				# if later parts look like fragments (no underscore), try
				# prefixing them with the first part's leading token
				if parts:
					first = parts[0]
					prefix = ""
					if isinstance(first, str) and "_" in first:
						prefix = first.split("_", 1)[0]
					new_parts = []
					for i, p in enumerate(parts):
						if i == 0:
							new_parts.append(p)
						else:
							if p in df.columns:
								new_parts.append(p)
							elif ("_" not in p) and prefix:
								candidate = f"{prefix}_{p}".strip()
								if candidate in df.columns:
									new_parts.append(candidate)
								else:
									new_parts.append(p)
							else:
								new_parts.append(p)
					parts = new_parts
				flat_subcols.extend(parts)
			else:
				flat_subcols.append(sc)

		mapped = []
		missing = []
		ambiguous = []
		for sc in flat_subcols:
			# exact match first
			if sc in df.columns:
				mapped.append(sc)
				continue

			# try normalized match
			n = _norm(sc)
			matches = col_map.get(n, [])
			if len(matches) == 1:
				mapped.append(matches[0])
			elif len(matches) > 1:
				ambiguous.append((sc, matches))
			else:
				missing.append(sc)

		if ambiguous:
			raise ValueError(
				f"Ambiguous sub-column mappings for category '{category}': {ambiguous}\n"
				f"Please disambiguate these names in your config or dataframe."
			)

		if missing:
			raise ValueError(
				f"Missing sub-columns for category '{category}': {missing}\n"
				f"Available columns: {list(df.columns)}\n"
				f"Your processed CSV must contain these columns (0/1)."
			)

		# Convert to numeric 0/1 safely and OR them (row-wise max)
		block = df[mapped].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
		df[category] = (block.max(axis=1) > 0).astype(int)

	return df


def default_hfacs_hierarchy() -> dict:
	return {
		"Organizational_Climate": [
			"Inadequate_Supervision",
			"Failed_to_Correct_Problem",
			"Planned_Inappropriate_Operations",
			"Supervisory_Violation",
		],
		"Resource_Management/Organizational_Process": [
			"Inadequate_Supervision",
			"Failed_to_Correct_Problem",
			"Planned_Inappropriate_Operations",
			"Supervisory_Violation",
		],
		"Inadequate_Supervision": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
		"Failed_to_Correct_Problem": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
		"Planned_Inappropriate_Operations": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
		"Supervisory_Violation": ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"],
		"Condition_of_Operators": ["Error", "Violation"],
		"Personnel_Factors": ["Error", "Violation"],
		"Situational_Factors": ["Error", "Violation"],
	}


def get_hfacs_hierarchy(cfg: dict) -> dict:
	if "hfacs_hierarchy" in cfg and cfg["hfacs_hierarchy"]:
		return cfg["hfacs_hierarchy"]
	return default_hfacs_hierarchy()


def compute_edge_weights(df: pd.DataFrame, hierarchy: dict) -> pd.DataFrame:
	rows = []
	for parent, children in hierarchy.items():
		if parent not in df.columns:
			raise ValueError(f"Missing HFACS category column in df: {parent}")

		df_p = df[df[parent] == 1]
		n_parent = int(len(df_p))

		for child in children:
			if child not in df.columns:
				raise ValueError(f"Missing HFACS category column in df: {child}")

			if n_parent == 0:
				n_joint = 0
				p = 0.0
			else:
				n_joint = int((df_p[child] == 1).sum())
				p = round(n_joint / n_parent, 6)

			rows.append(
				{
					"parent": parent,
					"child": child,
					"N_parent": n_parent,
					"N_joint": n_joint,
					"P_child_given_parent": p,
				}
			)

	return pd.DataFrame(rows)


def build_dag_from_edges(edge_df: pd.DataFrame) -> nx.DiGraph:
	G = nx.DiGraph()
	for _, r in edge_df.iterrows():
		G.add_edge(r["parent"], r["child"], weight=float(r["P_child_given_parent"]))
	return G


def export_outputs(G: nx.DiGraph, edge_df: pd.DataFrame, output_dir: str) -> None:
	outdir = Path(output_dir)
	outdir.mkdir(parents=True, exist_ok=True)

	edge_df.to_csv(outdir / "hfacs_dag_edges.csv", index=False)

	nodes = list(G.nodes())
	A = nx.to_pandas_adjacency(G, nodelist=nodes, weight="weight")
	A.to_csv(outdir / "hfacs_dag_adjacency_matrix.csv")

	try:
		from networkx.drawing.nx_pydot import write_dot

		write_dot(G, str(outdir / "hfacs_dag.dot"))
	except Exception:
		# pydot optional
		pass

	with open(outdir / "hfacs_dag_summary.txt", "w", encoding="utf-8") as f:
		f.write(f"Nodes: {G.number_of_nodes()}\n")
		f.write(f"Edges: {G.number_of_edges()}\n")
		f.write(f"Is DAG: {nx.is_directed_acyclic_graph(G)}\n")
		if nx.is_directed_acyclic_graph(G):
			order = list(nx.topological_sort(G))
			f.write("One topological order:\n")
			f.write(" -> ".join(order) + "\n")


def run_hfacs_dag(df: pd.DataFrame, cfg: dict, output_dir: str = "./data/processed") -> Tuple[nx.DiGraph, pd.DataFrame]:
	hfacs_categories = cfg.get("hfacs_categories")
	if hfacs_categories is None:
		raise ValueError("Config must contain 'hfacs_categories'")

	df = build_hfacs_category_columns(df, hfacs_categories)
	hierarchy = get_hfacs_hierarchy(cfg)
	edge_df = compute_edge_weights(df, hierarchy)
	G = build_dag_from_edges(edge_df)

	if not nx.is_directed_acyclic_graph(G):
		cycles = list(nx.simple_cycles(G))
		raise ValueError(f"Hierarchy is not acyclic. Found cycles: {cycles[:5]}")

	export_outputs(G, edge_df, output_dir)
	return G, edge_df


def plot_hfacs_layered(G: nx.DiGraph, save_path: str = None) -> None:
	try:
		import matplotlib.pyplot as plt
	except Exception as e:
		raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib") from e

	# assign levels using longest-path-from-roots dynamic programming on topological order
	topo = list(nx.topological_sort(G))
	level = {n: 0 for n in topo}
	for n in topo:
		preds = list(G.predecessors(n))
		if preds:
			level[n] = max(level[p] + 1 for p in preds)

	# group by level and create positions
	levels = {}
	for n, lv in level.items():
		levels.setdefault(lv, []).append(n)

	pos = {}
	for lv, nodes in levels.items():
		width = max(1, len(nodes))
		for i, n in enumerate(sorted(nodes)):
			pos[n] = (i - (width - 1) / 2.0, -lv)

	plt.figure(figsize=(10, 6))
	weights = [G[u][v].get("weight", 0.0) for u, v in G.edges()]
	edge_widths = [max(0.5, w * 6) for w in weights]
	nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="#ffd966", edgecolors="black")
	# Draw directed edges with visible arrowheads. Use a small curvature to separate parallel edges.
	nx.draw_networkx_edges(
		G,
		pos,
		width=edge_widths,
		arrows=True,
		arrowstyle='-|>',
		arrowsize=20,
		connectionstyle='arc3,rad=0.08',
		edge_color='gray',
	)
	nx.draw_networkx_labels(G, pos, font_size=9)
	plt.axis("off")
	if save_path:
		Path(save_path).parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


