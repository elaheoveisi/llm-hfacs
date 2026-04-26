from pathlib import Path
import polars as pl


def load_raw_dataset(raw_dir):
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for f in csv_files:
        # Read the two header rows to build flat column names
        headers = pl.read_csv(f, n_rows=2, has_header=False, infer_schema_length=0)
        row0 = [v or "" for v in headers.row(0)]
        row1 = [v or "" for v in headers.row(1)]
        col_names = [f"{a}_{b}".strip() for a, b in zip(row0, row1)]

        df = pl.read_csv(f, skip_rows=3, has_header=False, new_columns=col_names, infer_schema_length=0)
        frames.append(df)

    combined = pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]
    empty_cols = [c for c in combined.columns if combined[c].null_count() == combined.height]
    return combined.drop(empty_cols)


def extract_factor_columns(df, source_columns):
    for key, col in source_columns.items():
        split_col = (
            df[col]
            .fill_null("")
            .str.split(";")
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        )

        unique_factors = set()
        for row in split_col.to_list():
            if row:
                unique_factors.update(row)

        df = df.with_columns([
            split_col.list.contains(factor).cast(pl.Int32).alias(f"{key}_{factor}")
            for factor in sorted(unique_factors)
        ])

    return df


def create_hfacs_categories(df, category_map):
    for cat, cols in category_map.items():
        valid = [c for c in cols if c in df.columns]
        if not valid:
            continue
        df = df.with_columns(
            (pl.sum_horizontal([pl.col(c) for c in valid]) > 0).cast(pl.Int32).alias(cat)
        )
    return df


def build_processed_dataset(config):
    paths = config["paths"]
    csv_path = paths["processed_csv"]
    excel_path = paths.get("processed_excel", str(Path(csv_path).with_suffix(".xlsx")))

    df = load_raw_dataset(paths["raw_data_dir"])
    df = extract_factor_columns(df, config["source_columns"])

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(csv_path)
    Path(excel_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(excel_path)

    return df.to_pandas()
