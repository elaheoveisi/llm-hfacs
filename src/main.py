
import yaml
import os
from pathlib import Path
import csv
import pandas as pd
from features.hfacs_order_probability import (
    HFACS_ORDER,
    compute_all_full_hfacs_chains,
    compute_combined_hfacs_matrix,
    compute_hfacs_ordered_probabilities,
)
from features.hfacs_dag import run_hfacs_dag, plot_hfacs_layered
from utils import skip_run




#


def run_bayesian_dag(processed_dir):
    from features import bayesian as bayesian_module
    edges_csv = Path(processed_dir) / "hfacs_dag_edges.csv"
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    df_bayes = bayesian_module.load_edges_csv(str(edges_csv))
    # Calculate w_bayes for each edge
    df_bayes["w_bayes"] = df_bayes.apply(
        lambda r: bayesian_module.bayes_edge_mean(int(r["N_joint"]), int(r["N_parent"]), 1.0, 1.0), axis=1
    )
    # Save the full Bayesian edge table
    df_bayes.to_csv(out_dir / "hfacs_bayesian_dag_edges.csv", index=False)
    bayesian_module.draw_dag_pdf(
        df_bayes,
        str(out_dir / "bayesian_dag.pdf"),
        "Bayesian HFACS DAG (thickness=weight)",
    )

    # Prune using the function from bayesian.py
    pruned = bayesian_module.threshold_prune_edges(
        df_bayes,
        alpha=1.0,
        beta=1.0,
        min_keep_score=-3.0,
    )
    pruned.to_csv(out_dir / "hfacs_bayesian_dag_edges_threshold_pruned.csv", index=False)
    bayesian_module.draw_dag_pdf(
        pruned,
        str(out_dir / "bayesian_dag_threshold_pruned.pdf"),
        "Bayesian HFACS DAG (after threshold pruning)",
    )


# HFACS chain function
# from models.prediction import compute_full_chain


def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] pandas read_csv failed for {csv_path}: {e}")
        print("[INFO] Retrying with Python csv parser fallback...")
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return pd.DataFrame(rows)


with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]
processed_dir = os.path.dirname(paths["processed_csv"]) or "./data/processed"
processed_csv_path = Path(paths["processed_csv"])

if not processed_csv_path.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {processed_csv_path}. "
        "This pipeline is configured to use processed_output.csv as the source dataset."
    )

print(f"[INFO] Using existing processed dataset: {processed_csv_path}")
df = read_csv_robust(processed_csv_path)


with skip_run("run", "load_raw_dataset") as check:
    if check():
        print("[INFO] Skipping raw dataset load (using processed_output.csv).")


with skip_run("run", "create_hfacs_category_data") as check:
    if check():
        print("[INFO] Skipping HFACS regeneration/writes (using processed_output.csv).")


with skip_run("run", "hfacs_ordered_probabilities") as check:
    if check():
        print("[INFO] Computing HFACS-ordered conditional probabilities...")
        compute_hfacs_ordered_probabilities(
            df,
            hfacs_order=HFACS_ORDER,
            output_dir="./data/processed",
        )
        print("[INFO] HFACS ordered probability tables saved.")


with skip_run("run", "hfacs_full_chains") as check:
    if check():
        print("[INFO] Computing ALL HFACS full chains (Error + Violation)...")

        all_chains_df = compute_all_full_hfacs_chains(
            hfacs_order=HFACS_ORDER,
            processed_dir="./data/processed",
        )
        all_chains_df.to_csv(
            "./data/processed/HFACS_all_L4_to_L1_chains.csv",
            index=False,
        )

        print("[INFO] HFACS full-chain computation complete.")
        print(all_chains_df.head())


with skip_run("run", "hfacs_combined_matrix") as check:
    if check():
        print("[INFO] Computing combined HFACS matrix (L4 → L1)...")

        combined_df = compute_combined_hfacs_matrix(
            hfacs_order=HFACS_ORDER,
            processed_dir="./data/processed",
            filename="HFACS_L4_to_L1_combined.csv",
        )

        print("[INFO] Combined HFACS matrix saved.")
        print(combined_df)


with skip_run("run", "hfacs_dag") as check:
    if check():
        print("[INFO] Building HFACS DAG and exports...")
        G, edge_df = run_hfacs_dag(df, config, output_dir=processed_dir)
        # try to produce a layered PNG visualization (optional dependency: matplotlib)
        try:
            plot_hfacs_layered(G, save_path=str(Path(processed_dir) / "hfacs_dag.pdf"))
        except Exception:
            pass
        print("[INFO] HFACS DAG exported to:")
        print(f" - {processed_dir}/hfacs_dag.dot (if pydot installed)")
        print(f" - {processed_dir}/hfacs_dag_edges.csv")
        print(f" - {processed_dir}/hfacs_dag_adjacency_matrix.csv")


with skip_run("skip", "bayesian") as check:
    if check():
        print("[INFO] Running Bayesian HFACS processing...")
        run_bayesian_dag(processed_dir)




with skip_run("skip", "svm") as check:
    if check():
        print("[INFO] Running SVM analysis...")
        from features.svm import run_svm_analysis

        svm_cfg = config.get("svm", {})
        svm_data_path = Path(processed_dir) / svm_cfg.get("data_file", "step3_hfacs_categories.csv")
        svm_df = read_csv_robust(svm_data_path)

        xcols = svm_cfg.get("feature_columns", ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"])
        ycols = svm_cfg.get("target_columns", ["Error", "Violation"])

        run_svm_analysis(
            df=svm_df,
            xcols=xcols,
            ycols=ycols,
            n_splits=svm_cfg.get("n_splits", 5),
            seed=svm_cfg.get("seed", 7),
        )
        print("[INFO] SVM analysis complete.")


# Also print HFACS category totals across the dataset (if available)
#