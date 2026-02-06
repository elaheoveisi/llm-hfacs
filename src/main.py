import yaml
import os
from pathlib import Path

import yaml
import os
from pathlib import Path

from data.dataset import (
    create_hfacs_categories,
    extract_factor_columns,
    load_raw_dataset,
    save_outputs,
)
from features.hfacs_order_probability import (
    HFACS_ORDER,
    compute_all_full_hfacs_chains,
    compute_combined_hfacs_matrix,
    compute_hfacs_ordered_probabilities,
)
from features.hfacs_dag import run_hfacs_dag, plot_hfacs_layered
from utils import skip_run
from utils import skip_run

# HFACS chain function
# from models.prediction import compute_full_chain


with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]
processed_dir = os.path.dirname(paths["processed_csv"]) or "./data/processed"


with skip_run("run", "load_raw_dataset") as check:
    if check():
        print("[INFO] Loading raw dataset...")
        df = load_raw_dataset(
            paths["raw_data"],
            save_cleaned=True,
        )
        print("[INFO] Raw shape:", df.shape)


with skip_run("run", "create_hfacs_category_data") as check:
    if check():
        print("[INFO] Extracting factor columns...")
        df = extract_factor_columns(
            df,
            source_columns,
            save_step=True,
        )
        print("[INFO] Creating HFACS categories...")
        df = create_hfacs_categories(
            df,
            hfacs_map,
            save_step=True,
        )
        # ensure output directories exist before saving
        csv_dir = os.path.dirname(paths["processed_csv"]) or "."
        excel_dir = os.path.dirname(paths["processed_excel"]) or "."
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(excel_dir, exist_ok=True)

        csv_path = paths["processed_csv"]
        excel_path = paths["processed_excel"]

        try:
            save_outputs(
                df,
                csv_path,
                excel_path,
            )
            print("[INFO] Saved CSV & Excel")
        except PermissionError as e:
            print(f"[ERROR] Permission denied writing {csv_path}: {e}")
            # diagnostics and best-effort fix: try to make file writable and retry
            try:
                if os.path.exists(csv_path):
                    os.chmod(csv_path, 0o666)
                    print(f"[INFO] Changed permissions for {csv_path}, retrying...")
                    save_outputs(df, csv_path, excel_path)
                    print("[INFO] Saved CSV & Excel after chmod")
                else:
                    print("[ERROR] Target file does not exist. Checking directory permissions...")
                    print(" - dir exists:", os.path.exists(csv_dir))
                    print(" - dir writable:", os.access(csv_dir, os.W_OK))
                    raise
            except Exception as e2:
                print(f"[ERROR] Retry failed: {e2}")
                print("Diagnostic:")
                try:
                    print(" - file exists:", os.path.exists(csv_path))
                    print(" - file writable:", os.access(csv_path, os.W_OK))
                    print(" - dir exists:", os.path.exists(csv_dir))
                    print(" - dir writable:", os.access(csv_dir, os.W_OK))
                except Exception:
                    pass
                raise


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


# Also print HFACS category totals across the dataset (if available)
try:
    from scripts.count_hfacs import count_hfacs
except Exception as imp_err:
    # try a robust fallback: load the module directly from the scripts file
    try:
        import importlib.util

        scripts_file = Path.cwd() / "scripts" / "count_hfacs.py"
        if scripts_file.exists():
            spec = importlib.util.spec_from_file_location("count_hfacs", str(scripts_file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            count_hfacs = getattr(mod, "count_hfacs")
        else:
            raise FileNotFoundError(f"{scripts_file} not found")
    except Exception as load_err:
        print(f"[WARN] Cannot import count_hfacs: {imp_err}; fallback failed: {load_err}")
        count_hfacs = None

if callable(globals().get("count_hfacs", None)):
    try:
        csv_path = Path(processed_dir) / "step3_hfacs_categories.csv"
        counts, total_rows = count_hfacs(csv_path)
        print("[INFO] HFACS category totals across dataset:")
        print(f"Total rows: {total_rows}")
        for k, v in counts.items():
            print(f" - {k}: {v}")

        # save counts to CSV
        out_csv = Path(processed_dir) / "hfacs_category_counts.csv"
        try:
            import csv as _csv

            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="", encoding="utf-8") as cf:
                writer = _csv.writer(cf)
                writer.writerow(["category", "count"])
                for k, v in counts.items():
                    writer.writerow([k, v])
                writer.writerow(["__total_rows", total_rows])
            print(f"[INFO] HFACS counts saved to: {out_csv}")
        except Exception as werr:
            print(f"[WARN] Failed to write counts CSV: {werr}")
    except Exception as e:
        print(f"[WARN] Could not compute HFACS counts: {e}")
else:
    print("[INFO] HFACS counting function not available; skipping counts.")