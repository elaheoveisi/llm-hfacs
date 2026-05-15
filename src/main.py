from pathlib import Path

import pandas as pd
import yaml

from data.dataset import (
    build_processed_dataset,
    extract_factor_columns,
    load_raw_dataset,
    undersample_ae100,
)
from eda import (
    build_eda,
    default_input_from_config,
    print_report,
    read_table,
    save_tables,
)
from features.utils import load_dataset
from llm.classify import run as run_classify
from models.ghfacs import ghfacs_bayes, ghfacs_svm, ghfacs_svm_nonbalance
from models.ghfacs_rf import ghfacs_rf as run_ghfacs_rf
from models.svm import svm as run_svm
from models.bayes_p import bayes_p as run_bayes_p
from models.svm_p import svm_p as run_svm_p
from models.rf_p import rf_p as run_rf_p
from utils import skip_run

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


with skip_run("skip", "load_raw_dataset") as check, check():
    processed_csv_path = Path(config["paths"]["processed_csv"])
    build_processed_dataset(config)
    print(f"[INFO] Saved processed dataset: {processed_csv_path}")

with skip_run("skip", "undersampled_dataset") as check, check():
    undersample_ae100(config)

with skip_run("skip", "eda") as check, check():
    eda_output_dir = Path(config["paths"]["eda_output_dir"])
    df_eda = read_table(default_input_from_config(config))
    tables = build_eda(df_eda)
    print_report(tables, len(df_eda))
    save_tables(tables, eda_output_dir)
    print(f"\n[INFO] Saved EDA tables to: {eda_output_dir}")

with skip_run("skip", "svm") as check, check():
    df_asrs = extract_factor_columns(
        load_raw_dataset(config["paths"]["raw_data_dir"]), config["source_columns"]
    ).to_pandas()
    run_svm(config, df_asrs)

with skip_run("skip", "svm_p") as check, check():
    run_svm_p(config)

with skip_run("skip", "bayes_p") as check, check():
    run_bayes_p(config)

with skip_run("skip", "rf_p") as check, check():
    run_rf_p(config)



with skip_run("run", "ghfacs_svm") as check, check():
    df = pd.read_excel(config["paths"]["ghfacs_raw_data"])
    ghfacs_svm(config, df)

with skip_run("run", "ghfacs_rf") as check, check():
    df = pd.read_excel(config["paths"]["ghfacs_raw_data"])
    run_ghfacs_rf(config, df)

with skip_run("skip", "ghfacs_svm_nonbalance") as check, check():
    df = pd.read_excel(config["paths"]["ghfacs_raw_data"])
    ghfacs_svm_nonbalance(config, df)



with skip_run("skip", "ghfacs_bayes") as check, check():
    df = pd.read_excel(config["paths"]["ghfacs_raw_data"])
    ghfacs_bayes(config, df)

with skip_run("skip", "classify") as check, check():
    run_classify(config)
