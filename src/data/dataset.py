import pandas as pd
import os
import tempfile
from datetime import datetime


def load_raw_dataset(path, save_cleaned=True):
    # Load CSV with two-row header
    df = pd.read_csv(path, header=[0, 1])

    # Flatten MultiIndex columns → "Events_Anomaly"
    df.columns = df.columns.map(lambda x: f"{x[0]}_{x[1]}".strip())

    # Save cleaned header file
    if save_cleaned:
        df.to_csv("./data/processed/step1_cleaned_header.csv", index=False)

    return df


def extract_factor_columns(df, source_columns, save_step=True):
    result = df.copy()

    # Collect unique factors for each key
    factor_sets = {key: set() for key in source_columns}

    def split_factors(x):
        if not isinstance(x, str):
            return []
        return [p.strip() for p in x.split(";") if p.strip()]

    # Build factor sets
    for key, col in source_columns.items():
        for val in df[col]:
            factor_sets[key].update(split_factors(val))

    # Create binary 1/0 factor columns
    for key, col in source_columns.items():
        for factor in sorted(factor_sets[key]):  # alphabetical consistency
            new_col = f"{key}_{factor}"
            result[new_col] = df[col].apply(
                lambda x: int(factor in split_factors(x))
            )

    # Save step result
    if save_step:
        result.to_csv("./data/processed/step2_factor_expanded.csv", index=False)

    return result


def create_hfacs_categories(df, category_map, save_step=True):
    result = df.copy()

    for cat, cols in category_map.items():

        # Only include existing columns
        valid = [c for c in cols if c in result.columns]
        if not valid:
            continue

        # HFACS category is 1 if ANY contributing factor is present
        result[cat] = result[valid].sum(axis=1).apply(lambda v: 1 if v > 0 else 0)

    # Save HFACS step
    if save_step:
        result.to_csv("./data/processed/step3_hfacs_categories.csv", index=False)

    return result


def save_outputs(df, csv_path, excel_path):
    # Ensure output directories exist
    csv_dir = os.path.dirname(csv_path) or "."
    excel_dir = os.path.dirname(excel_path) or "."
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)

    # Write CSV to a temp file in the same dir, then atomically replace
    tmp_csv_fd, tmp_csv_path = tempfile.mkstemp(suffix=".csv", dir=csv_dir)
    os.close(tmp_csv_fd)
    try:
        df.to_csv(tmp_csv_path, index=False)
        try:
            os.replace(tmp_csv_path, csv_path)
        except PermissionError:
            # target may be locked; fall back to timestamped file
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = os.path.splitext(csv_path)[0] + f"_{ts}.csv"
            os.replace(tmp_csv_path, fallback)
            print(f"[WARN] Could not replace {csv_path}; wrote fallback {fallback}")
    except Exception:
        if os.path.exists(tmp_csv_path):
            try:
                os.remove(tmp_csv_path)
            except Exception:
                pass
        raise

    # Write Excel similarly
    tmp_xl_fd, tmp_xl_path = tempfile.mkstemp(suffix=".xlsx", dir=excel_dir)
    os.close(tmp_xl_fd)
    try:
        df.to_excel(tmp_xl_path, index=False)
        try:
            os.replace(tmp_xl_path, excel_path)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_xl = os.path.splitext(excel_path)[0] + f"_{ts}.xlsx"
            os.replace(tmp_xl_path, fallback_xl)
            print(f"[WARN] Could not replace {excel_path}; wrote fallback {fallback_xl}")
    except Exception:
        if os.path.exists(tmp_xl_path):
            try:
                os.remove(tmp_xl_path)
            except Exception:
                pass
        raise