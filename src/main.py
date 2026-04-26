import os
import yaml
from pathlib import Path
from data.dataset import build_processed_dataset
from utils import skip_run

os.chdir(Path(__file__).parent.parent)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

processed_csv_path = Path(config["paths"]["processed_csv"])

with skip_run("run", "load_raw_dataset") as check:
    with check():
            build_processed_dataset(config)
            print(f"[INFO] Saved processed dataset: {processed_csv_path}")

