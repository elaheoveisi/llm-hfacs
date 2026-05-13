import os
import sys
import yaml
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import build_processed_dataset
from utils import skip_run
from features.svm import svm as run_svm
from features.ghfacs_svm import ghfacs_svm as run_ghfacs_svm
from features.ghfacs_svm_nonbalance import ghfacs_svm_nonbalance as run_ghfacs_svm_nonbalance
from features.ghfacs_rf import ghfacs_rf as run_ghfacs_rf
from features.ghfacs_bayes import ghfacs_bayes as run_ghfacs_bayes
from llm.prompts.GHFACS.classify import run as run_classify

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_csv_path = Path(config["paths"]["processed_csv"])

with skip_run("skip", "load_raw_dataset") as check:
    with check():
            build_processed_dataset(config)
            print(f"[INFO] Saved processed dataset: {processed_csv_path}")

with skip_run("skip", "svm") as check:
    with check():
        run_svm()

with skip_run("run", "ghfacs_svm") as check:
    with check():
        run_ghfacs_svm()

with skip_run("skip", "ghfacs_svm_nonbalance") as check:
    with check():
        run_ghfacs_svm_nonbalance()


with skip_run("skip", "ghfacs_bayes") as check:
    with check():
        run_ghfacs_bayes()



with skip_run("skip", "classify") as check:
    with check():
        run_classify()
