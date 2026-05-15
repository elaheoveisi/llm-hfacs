# Coding Philosophy

These rules apply to all work in this repository.

---

## 1. Single source of truth for configuration

Every hyperparameter, file path, and runtime setting belongs in `configs/config.yaml`. Python files read from it — they never define magic values themselves.

```python
# WRONG
model = RandomForest(n_estimators=100, random_state=42)
output_path = "data/processed/results.csv"

# RIGHT
cfg = config["models"]["random_forest"]
model = RandomForest(n_estimators=cfg["n_estimators"], random_state=config["random_state"])
output_path = config["paths"]["results_csv"]
```

If a key is missing from config, let it raise a `KeyError` — never add a silent default fallback inside the function. Missing config is a bug, not a recoverable error.

Random seeds always come from config (`config["random_state"]`) — never hardcode `42` or any seed directly in a function.

Experiments are config changes, not code changes. To try a new hyperparameter, add or change a config entry. Never edit a function just to try a different value.

If the same value is needed in multiple places (e.g., `random_state`, `test_size`), it lives under one config key. Never copy-paste a constant across files.

---

## 2. Single point of entry

`src/main.py` is the only file you run. It orchestrates the full pipeline using `skip_run` blocks to toggle steps on/off. Nothing else should be executed directly.

```python
# main.py — sparse, sequential, no inline logic
with skip_run("run", "train_model") as check, check():
    df = load_dataset(config["paths"]["training_csv"])
    train_model(config, df)
```

---

## 3. main.py stays sparse

Each `skip_run` block is at most 8–10 lines: load data, call a function, done. If a step needs more than 3 lines, wrap it in a function inside the appropriate module and call that from `main.py`.

Folder structure carries the logic:

```
models/        — training, evaluation, prediction
features/      — feature engineering, splitting, balancing, targets
data/          — raw loading and preprocessing
llm/           — LLM integration
visualization/ — plots and figures
```

---

## 4. No path discovery

Never construct paths relative to `__file__` or the current working directory:

```python
# WRONG
Path(__file__).parent.parent / "data" / "output.csv"
os.path.dirname(__file__) + "/../../configs/config.yaml"

# RIGHT
Path(config["paths"]["output_csv"])
```

All paths in `config.yaml` are relative to the project root. `main.py` is always run from the project root.

---

## 5. Functions accept config — they don't load it

Pipeline functions receive `config` (and data) as arguments. They never open a config file or discover data paths internally.

```python
# WRONG
def train_model():
    config = yaml.safe_load(open("configs/config.yaml"))
    df = pd.read_csv("data/processed/training.csv")

# RIGHT
def train_model(config: dict, df: pd.DataFrame) -> None:
    cfg = config["models"]["my_model"]
    out_dir = config["paths"]["model_output_dir"]
```

---

## 6. Function design

- **One function, one responsibility.** If a function description needs the word "and", split it.
- **Keep functions short** — aim for ~30 lines. If you need to scroll to read a function, refactor it.
- **Type hints on all public function signatures**: `def train(config: dict, df: pd.DataFrame) -> None`
- **Never mutate the input DataFrame** inside a function — call `.copy()` if you need to modify it.
- **No silent exception swallowing** — never `except: pass` or catch-and-continue without at least a log message.

---

## 7. Save all outputs — no ephemeral results

Every run should save its results (metrics, confusion matrices, predictions, plots) to the output directory specified in config. Never rely on reading terminal output after the fact.

---

## 8. Code hygiene

- **No commented-out code.** Delete dead code — git has the history.
- **No `print()` debugging** left in committed code. Remove debug prints or replace with proper logging.
- **No `TODO` comments** in code files. Track open work in issues or a task list, not in-file.
- **Imports at the top** of every file. The only exception is lazy-loading a heavy dependency inside a function to avoid a slow import at startup (e.g., `causallearn`, `pymc`).

---

## 9. Data validation at load time

Check shape, required columns, and nulls right after loading — not scattered throughout the pipeline:

```python
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["col_a", "col_b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df
```

---

## Pre-submission checklist

- [ ] No hardcoded numbers or strings in `.py` files — all in `configs/config.yaml`
- [ ] No path construction using `__file__`, `os.getcwd()`, or relative `..` navigation
- [ ] `main.py` blocks are ≤ 3 lines of logic
- [ ] All pipeline functions accept `(config, ...)` — no internal config or data loading
- [ ] New config values added to `configs/config.yaml` under a sensible section
- [ ] No commented-out code, no debug `print()` statements, no `TODO` comments
- [ ] All outputs saved to paths from config — no ephemeral results
- [ ] Input DataFrames not mutated inside functions (use `.copy()`)
- [ ] No silent exception handling (`except: pass`)
- [ ] Type hints on all public function signatures
