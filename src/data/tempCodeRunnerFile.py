from pathlib import Path

import pandas as pd


def load_raw_dataset(path):
    df = pd.read_csv(path, header=[0, 1])
    df.columns = df.columns.map(lambda x: f"{x[0]}_{x[1]}".strip())
    return df


def extract_factor_columns(df, source_columns):
    result = df.copy()

    for key, col in source_columns.items():
        split_series = df[col].fillna("").astype(str).apply(
            lambda x: [p.strip() for p in x.split(";") if p.strip()]
        )
        unique_factors = set(f for factors in split_series for f in factors)
        for factor in sorted(unique_factors):
            result[f"{key}_{factor}"] = split_series.apply(lambda x: int(factor in x))

    return result


def create_hfacs_categories(df, category_map):
    result = df.copy()

    for cat, cols in category_map.items():
        valid = [c for c in cols if c in result.columns]
        if not valid:
            continue
        result[cat] = (result[valid].sum(axis=1) > 0).astype(int)

    return result


def build_processed_dataset(config):
    paths = config["paths"]
    csv_path = paths["processed_csv"]
    excel_path = paths.get("processed_excel", str(Path(csv_path).with_suffix(".xlsx")))

    df = load_raw_dataset(paths["raw_data"])
    df = extract_factor_columns(df, config["source_columns"])
    df = create_hfacs_categories(df, config["hfacs_categories"])

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    Path(excel_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)
    return df
