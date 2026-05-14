from pathlib import Path

import yaml

from data.dataset import build_processed_dataset, undersample_ae100
from features.ghfacs_bayes import ghfacs_bayes as run_ghfacs_bayes
from features.ghfacs_svm import ghfacs_svm as run_ghfacs_svm
from features.ghfacs_svm_nonbalance import (
    ghfacs_svm_nonbalance as run_ghfacs_svm_nonbalance,
)
from features.svm import svm as run_svm
from features.utils import load_dataset
from llm.classify import run as run_classify
from utils import skip_run

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_csv_path = Path(config["paths"]["processed_csv"])

with skip_run("skip", "load_raw_dataset") as check, check():
    build_processed_dataset(config)
    print(f"[INFO] Saved processed dataset: {processed_csv_path}")

with skip_run("skip", "undersampled_dataset") as check, check():
    df = undersample_ae100(config)

with skip_run("skip", "svm") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    run_svm(config, df)

with skip_run("skip", "ghfacs_svm") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    run_ghfacs_svm(config, df)

with skip_run("skip", "ghfacs_svm_nonbalance") as check, check():
    run_ghfacs_svm_nonbalance(config)

with skip_run("run", "ghfacs_bayes") as check, check():
    run_ghfacs_bayes(config)

with skip_run("skip", "classify") as check, check():
    run_classify(config)
