import os
import sys
import yaml
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import build_processed_dataset
from utils import skip_run
from features.svm import svm as run_svm

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_csv_path = Path(config["paths"]["processed_csv"])

with skip_run("skip", "load_raw_dataset") as check:
    with check():
            build_processed_dataset(config)
            print(f"[INFO] Saved processed dataset: {processed_csv_path}")

with skip_run("run", "svm") as check:
    with check():
        run_svm()

