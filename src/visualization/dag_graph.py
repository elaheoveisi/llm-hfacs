
# Add required imports at the top
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def prettify_label(label):
	"""
	Prettify label for node and legend display.
	"""
	custom_mapping = {
		"Error": "Error (Unsafe Act)",
		"Violation": "Violation (Unsafe Act)",
		"Situational_Factors": "Situational Factors",
		"Personnel_Factors": "Personnel Factors",
		"Condition_of_Operators": "Condition of Operators",
		"Inadequate_Supervision": "Inadequate Supervision",
		"Failed_to_Correct_Problem": "Failed to Correct Problem",
		"Planned_Inappropriate_Operations": "Planned Inappropriate Operations",
		"Supervisory_Violation": "Supervisory Violation",
		"Organizational_Climate": "Organizational Climate",
		"Resource_Management/Organizational_Process": "Resource Management / Organizational Process",
	}
	if label in custom_mapping:
		return custom_mapping[label]
	# Fallback: Title Case, replace _ and /
	return label.replace("_", " ").replace("/", " / ").title()

def plot_dag(
	G,
	save_path: str | None = None,
	layout_seed: int = 42,
	title: str = "Learned HFACS DAG (GES optimized)",
) -> None:
	def get_node_colors(labels):
		palette = [
			"#ffd966", "#a4c2f4", "#b6d7a8", "#f4cccc", "#d9d2e9", "#ffe599",
			"#76a5af", "#e06666", "#6aa84f", "#674ea7", "#c27ba0", "#f6b26b",
			"#8e7cc3", "#cfe2f3", "#ea9999", "#b4a7d6", "#fff2cc", "#b6d7a8",
			"#a2c4c9", "#e69138", "#38761d", "#134f5c", "#990000", "#0b5394"
		]
		colors = []
		for i, label in enumerate(labels):
			label_lower = label.lower()
			if label_lower == "error":
				colors.append("#003366")
			elif label_lower == "violation":
				colors.append("black")
			else:
				colors.append(palette[i % len(palette)])
		return colors


	pos = nx.circular_layout(G)
	plt.figure(figsize=(12, 8))
	node_labels = [prettify_label(n) for n in G.nodes]
	node_colors = get_node_colors(node_labels)


	# Try to load conditional probabilities for edge labels
	import os
	condprob_path = os.path.join(os.path.dirname(save_path) if save_path else '.', "learned_dag_conditional_probabilities.csv")
	condprobs = None
	if os.path.exists(condprob_path):
		import pandas as pd
		condprobs = pd.read_csv(condprob_path)
		condprob_dict = {(row['parent'], row['child']): row['P(child=1|parent=1)'] for _, row in condprobs.iterrows()}
	else:
		condprob_dict = {}

	# Draw all edges with the same thickness
	nx.draw_networkx_edges(
		G,
		pos,
		ax=plt.gca(),
		width=2.0,  # constant thickness for all edges
		edge_color="dimgray",
		arrows=True,
		arrowstyle="-|>",
		arrowsize=60,  # make arrowheads even larger and more obvious
		min_source_margin=15,
		min_target_margin=15,
	)

	# Draw edge labels for conditional probabilities if available
	if condprobs is not None:
		edge_labels = {}
		for u, v in G.edges():
			p = condprob_dict.get((u, v), None)
			if p is not None and not pd.isna(p):
				edge_labels[(u, v)] = f"{p:.2f}"
		nx.draw_networkx_edge_labels(
			G,
			pos,
			edge_labels=edge_labels,
			font_color='blue',
			font_size=14,
			label_pos=0.7  # Move labels further from nodes
		)

	# Draw only rectangles with category names (no networkx node shapes)
	for node, (x, y) in pos.items():
		label = prettify_label(node)
		idx = list(G.nodes).index(node)
		plt.text(x, y, label, fontsize=16, ha='center', va='center',
				 bbox=dict(boxstyle='round,pad=0.4', fc=node_colors[idx], ec='black', lw=2))

	plt.title(title)
	plt.axis('off')
	plt.tight_layout()

	if save_path:
		out_path = Path(save_path)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		# Save as PDF
		plt.savefig(str(out_path.with_suffix('.pdf')), bbox_inches="tight")
		# Save as PNG
		plt.savefig(str(out_path.with_suffix('.png')), bbox_inches="tight")
		plt.close()
	else:
		plt.show()

