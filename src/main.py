import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
import csv
from data.dataset import build_processed_dataset
from features.hfacs_order_probability import (
    HFACS_ORDER,
    compute_all_full_hfacs_chains,
    compute_combined_hfacs_matrix,
    compute_hfacs_ordered_probabilities,
)
from utils import skip_run
from features.bayesian import run_bayesian_workflow
from project_config import chdir_project_root, load_config, get_optional_path, get_processed_dir, CONFIG_PATH


chdir_project_root()
config = load_config()

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]
processed_csv_path = Path(paths["processed_csv"])
processed_dir = get_processed_dir(config)
processed_excel_path = get_optional_path(
    config,
    "processed_excel",
    default=str(processed_csv_path.with_suffix(".xlsx")),
)
hfacs_full_chains_path = get_optional_path(
    config,
    "hfacs_full_chains_csv",
    default=str(processed_dir / "HFACS_all_L4_to_L1_chains.csv"),
)
hfacs_combined_matrix_path = get_optional_path(
    config,
    "hfacs_combined_matrix_csv",
    default=str(processed_dir / "HFACS_L4_to_L1_combined.csv"),
)
all_conditional_probabilities_path = get_optional_path(
    config,
    "all_conditional_probabilities_csv",
    default=str(processed_dir / "all_conditional_probabilities.csv"),
)
svm_output_dir = get_optional_path(config, "svm_output_dir", default="./data/processed/svm")
learned_dag_probability_matrix_path = get_optional_path(
    config,
    "learned_dag_probability_matrix_csv",
    default=str(processed_dir / "learned_dag_probability_matrix.csv"),
)



def save_svm_results(y_test, y_pred, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=["Pred_Neither", "Pred_Error", "Pred_Violation"],
                        index=["True_Neither", "True_Error", "True_Violation"])
    cm_df.to_csv(out_dir / "svm_confusion_matrix.csv")

    # Save classification report
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "svm_classification_report.csv")



def balance_joint_categories(df, error_col="Error", violation_col="Violation", random_state=42):
    """
    Downsample all (Error, Violation) joint categories to the size of the smallest group.
    Returns a balanced DataFrame.
    """
    joint_label = df[error_col].astype(int) * 2 + df[violation_col].astype(int)
    min_count = joint_label.value_counts().min()
    balanced_df = (
        df.assign(_joint=joint_label)
        .groupby("_joint", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=random_state))
        .drop(columns=["_joint"])
        .reset_index(drop=True)
    )
    return balanced_df


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

if __name__ == "__main__":
    with skip_run("run", "load_raw_dataset") as check:
        with check():
            if processed_csv_path.exists():
                print(f"[INFO] Using existing processed dataset: {processed_csv_path}")
            else:
                print("[INFO] Building processed dataset from raw input...")
                build_processed_dataset(config)
                print(f"[INFO] Saved processed dataset: {processed_csv_path}")

    if not processed_csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_csv_path}. "
            "The dataset build step did not create the expected processed_output.csv file."
        )

    df = read_csv_robust(processed_csv_path)

    # Count and print the number in each of the four joint categories (Error, Violation)
    if "Error" in df.columns and "Violation" in df.columns:
        joint_label = df["Error"].astype(int) * 2 + df["Violation"].astype(int)
        print("[INFO] Counts for each (Error, Violation) joint category:")
        for idx, count in joint_label.value_counts().sort_index().items():
            print(f"  Category {idx} (Error={idx//2}, Violation={idx%2}): {count}")

    with skip_run("skip", "create_hfacs_category_data") as check:
        with check():
            print("[INFO] Skipping HFACS regeneration/writes (using processed_output.csv).")

    with skip_run("skip", "hfacs_ordered_probabilities") as check:
        with check():
            print("[INFO] Computing HFACS-ordered conditional probabilities...")
            compute_hfacs_ordered_probabilities(
                df,
                hfacs_order=HFACS_ORDER,
                output_dir=str(processed_dir),
            )
            print("[INFO] HFACS ordered probability tables saved.")

    with skip_run("skip", "hfacs_full_chains") as check:
        with check():
            print("[INFO] Computing ALL HFACS full chains (Error + Violation)...")
            all_chains_df = compute_all_full_hfacs_chains(
                hfacs_order=HFACS_ORDER,
                processed_dir=str(processed_dir),
            )
            hfacs_full_chains_path.parent.mkdir(parents=True, exist_ok=True)
            all_chains_df.to_csv(hfacs_full_chains_path, index=False)
            print("[INFO] HFACS full-chain computation complete.")
            print(all_chains_df.head())

    with skip_run("skip", "hfacs_combined_matrix") as check:
        with check():
            print("[INFO] Computing combined HFACS matrix (L4 → L1)...")
            combined_df = compute_combined_hfacs_matrix(
                hfacs_order=HFACS_ORDER,
                processed_dir=str(hfacs_combined_matrix_path.parent),
                filename=hfacs_combined_matrix_path.name,
            )
            print("[INFO] Combined HFACS matrix saved.")
            print(combined_df)

    with skip_run("skip", "hfacs_conditional_probabilities") as check:
        with check():
            print("[INFO] Computing conditional probabilities for all HFACS categories...")
            from features.conditional_prob import compute_all_hfacs_probabilities, save_all_conditional_probabilities_to_csv
            results = compute_all_hfacs_probabilities(
                df,
                hfacs_map,
                output_dir=str(processed_dir),
            )
            save_all_conditional_probabilities_to_csv(results, output_path=str(all_conditional_probabilities_path))
            print(f"[INFO] Conditional probability tables saved to {processed_dir} and {all_conditional_probabilities_path.name}")

    with skip_run("skip", "bayesian") as check:
        with check():
            print("[INFO] Running Bayesian Network prediction...")
            run_bayesian_workflow(
                config_path=str(CONFIG_PATH),
                data_file=str(processed_csv_path),
            )

    with skip_run("skip", "llm") as check:
        with check():
            try:
                from llm.data_create.generate_llm_dataset import generate_llm_dataset
                llm_cfg = config.get("llm", {})
                print("[INFO] Running LLM prediction...")
                generate_llm_dataset(
                    input_path=str(processed_csv_path),
                    output_path=paths.get("llm_predictions_csv", "./data/processed/llm_predictions.csv"),
                    style=llm_cfg.get("style", "cot"),
                    limit=llm_cfg.get("limit"),
                )
            except Exception as e:
                print(f"[ERROR] LLM prediction failed: {e}")

    with skip_run("skip", "svm") as check:
        with check():
            from features import svm
            print("[INFO] Running SVM prediction...")
            svm.main()
            # Try to save SVM results if y_test and y_pred are available from svm module
            if hasattr(svm, "y_test") and hasattr(svm, "y_pred"):
                save_svm_results(svm.y_test, svm.y_pred, svm_output_dir)
            else:
                print("[WARN] y_test and y_pred not found in svm module; results not saved.")

    with skip_run("skip", "save_dag_probability_matrix") as check:
        with check():
            try:
                from features.DAG import save_dag_probability_matrix
                import networkx as nx
                print("[INFO] Saving learned DAG probability matrix...")
                dag_edges_path = processed_dir / "learned_dag_edges.csv"
                dag_nodes = []
                dag_edges = []
                if dag_edges_path.exists():
                    df_edges = pd.read_csv(dag_edges_path)
                    dag_edges = list(df_edges.itertuples(index=False, name=None))
                    dag_nodes = list(set([e[0] for e in dag_edges] + [e[1] for e in dag_edges]))
                    G = nx.DiGraph()
                    G.add_nodes_from(dag_nodes)
                    G.add_edges_from(dag_edges)
                    save_dag_probability_matrix(G, df, output_path=str(learned_dag_probability_matrix_path))
                    print("[INFO] Probability matrix saved as learned_dag_probability_matrix.csv")
                else:
                    print("[WARN] learned_dag_edges.csv not found, skipping probability matrix save.")
            except Exception as e:
                print(f"[ERROR] Could not save learned DAG probability matrix: {e}")
