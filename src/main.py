import yaml
import os

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
from utils import skip_run

# HFACS chain function
# from models.prediction import compute_full_chain


with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]


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