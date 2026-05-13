import os
import pandas as pd
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.utils import load_config, make_four_class_target

TARGET_N = 1400
RANDOM_STATE = 42


def undersample_ae100():
    config = load_config()
    data_dir = config["paths"]["ghfacs_data_dir"]
    input_file = config["llm"]["input"]
    input_path = os.path.join(data_dir, input_file)

    df = pd.read_excel(input_path)
    y4 = make_four_class_target(df)

    print("Class distribution before undersampling:")
    for cls, count in y4.value_counts().sort_index().items():
        print(f"  {cls}: {count}")

    ae100_idx = y4[y4 == "AE100 only"].index
    if len(ae100_idx) <= TARGET_N:
        print(f"\n'AE100 only' already has {len(ae100_idx)} rows (<= {TARGET_N}), no drop needed.")
        return df

    drop_idx = ae100_idx.to_series().sample(n=len(ae100_idx) - TARGET_N, random_state=RANDOM_STATE).index
    df_out = df.drop(index=drop_idx).reset_index(drop=True)

    y_out = make_four_class_target(df_out)
    print("\nClass distribution after undersampling:")
    for cls, count in y_out.value_counts().sort_index().items():
        print(f"  {cls}: {count}")

    out_stem = Path(input_file).stem
    out_path = os.path.join(data_dir, f"{out_stem}_undersampled.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return df_out


if __name__ == "__main__":
    undersample_ae100()
