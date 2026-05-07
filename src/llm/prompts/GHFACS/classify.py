import os, json, time, tempfile, shutil, yaml, pandas as pd, openai
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR, CFG_PATH = ROOT / "data" / "GHFACS", ROOT / "configs" / "config.yaml"
DEFAULT_PROMPT = "initial_prompting"
FINAL = {"Final_Answer", "Final_Class", "Final_HFACS_Code", "Codes_Selected", "Final_Justification", "Confidence"}
JSON_MODELS, NARR = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}, "narr_accf"
CLASSES = ["AE100 only", "AE200 only", "Both", "anyofthem"]

def dget(d, k, v=None): return d.get(k, v) if isinstance(d, dict) else v
def o2n(v): return None if v in (None, "", "none", "null") else int(v)
def flatten(x): return {f"{k}__{a}": b for k, v in x.items() if isinstance(v, dict) for a, b in v.items()} | {k: ", ".join(map(str, v)) if isinstance(v, list) else v for k, v in x.items() if not isinstance(v, dict)}
def finals(x): return {k: v for k, v in x.items() if any(k == f or k.endswith(f"__{f}") for f in FINAL) or k in {"original_index", NARR, "skip_reason", "error"}}

def _reader(path):
    return pd.read_csv if Path(path).suffix.lower() == ".csv" else pd.read_excel

def llm_paths(llm):
    inp = dget(llm, "input")
    if not inp: raise ValueError("Missing input. Set llm.input in config.")
    compact = bool(dget(llm, "compact", False))
    prompt_name = dget(llm, "prompt", DEFAULT_PROMPT)
    inp_path = DATA_DIR / inp
    out = DATA_DIR / f"{inp_path.stem}_LLM_Output_{prompt_name}{'_compact' if compact else ''}.csv"
    return inp_path, out, compact, prompt_name

def four_class(ae100, ae200):
    if ae100 and ae200: return "Both"
    if ae100: return "AE100 only"
    if ae200: return "AE200 only"
    return "anyofthem"

def normalize_class(value):
    if pd.isna(value): return None
    value = str(value).strip().lower()
    aliases = {
        "ae100": "AE100 only", "ae100 only": "AE100 only",
        "ae200": "AE200 only", "ae200 only": "AE200 only",
        "both": "Both", "multiple": "Both",
        "none": "anyofthem", "anyofthem": "anyofthem",
    }
    return aliases.get(value)

def call(client, msgs, llm, temp, retries, wait):
    model = dget(llm, "model", "gpt-4o-mini")
    for i in range(1, retries + 1):
        try:
            kw = {"model": model, "messages": msgs, "temperature": temp}
            if model in JSON_MODELS: kw["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kw).choices[0].message.content
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e) or "billing" in str(e).lower():
                raise RuntimeError(f"Quota exhausted — add credits at platform.openai.com/settings/organization/billing\n{e}") from e
            if i == retries: raise
            time.sleep(wait * i)
        except (openai.APIConnectionError, openai.APITimeoutError):
            if i == retries: raise
            time.sleep(wait * i)

def process_row(args):
    i, n, client, sys_p, usr_t, step2_t, llm, temp, retries, wait, compact = args
    if not n or n.lower() in {"nan", "none", ""}: return i, {"original_index": i, "skip_reason": "empty_narrative"}
    def msgs(content): return ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": content}]
    try:
        step1_out = call(client, msgs(usr_t.replace("{narrative}", n).replace("{NARRATIVE_TEXT}", n)), llm, temp, retries, wait)
        if step2_t:
            x = flatten(json.loads(call(client, msgs(step2_t.replace("{preconditions}", step1_out)), llm, temp, retries, wait)))
        else:
            x = flatten(json.loads(step1_out))
    except RuntimeError: raise
    except Exception as e: x = {"error": str(e)}
    x |= {"original_index": i, NARR: n}
    return i, finals(x) if compact else x

def read_input_table(path, tries=3, wait=2):
    path = Path(path)
    reader = _reader(path)
    for i in range(1, tries + 1):
        try:
            return reader(path)
        except PermissionError as e:
            if i == tries:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir) / path.name
                        shutil.copy2(path, tmp_path)
                        return reader(tmp_path)
                except PermissionError:
                    raise RuntimeError(
                        f"Cannot open input file (locked by Excel/OneDrive): {path}\n"
                        "Close the workbook if open, pause or finish OneDrive sync, then retry.\n"
                        "If it still fails, save/export the workbook as CSV and set llm.input to that file in config.yaml."
                    ) from e
            time.sleep(wait)

def evaluate_labels(llm_out, gt_path):
    """Evaluate four-class Final_Class predictions against ground truth."""
    llm = pd.read_csv(llm_out, keep_default_na=False, na_values=[""])
    gt = _reader(gt_path)(gt_path)
    gt = gt[["AE100", "AE200"]].reset_index().rename(columns={"index": "original_index"})
    merged = llm.merge(gt, on="original_index", how="inner")
    if merged.empty:
        raise ValueError("No rows matched between LLM output and ground truth — check original_index alignment.")
    class_col = next((c for c in ("Final_Class", "Final_HFACS_Code") if c in merged.columns), None)
    if class_col is None:
        raise ValueError(f"Missing Final_Class or Final_HFACS_Code column in LLM output. Columns: {list(llm.columns)}")

    y_true = [four_class(r["AE100"] == "AE100", r["AE200"] == "AE200") for _, r in merged.iterrows()]
    y_pred = [normalize_class(r[class_col]) for _, r in merged.iterrows()]

    valid = [p in CLASSES for p in y_pred]
    invalid = merged[[not v for v in valid]][["original_index", class_col]]
    y_true_v = [t for t, v in zip(y_true, valid) if v]
    y_pred_v = [p for p, v in zip(y_pred, valid) if v]

    cm = confusion_matrix(y_true_v, y_pred_v, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in CLASSES], columns=[f"Pred: {c}" for c in CLASSES])

    print("\n=== Four-Class Evaluation (Final_Class vs. Ground Truth) ===")
    print(f"Total rows: {len(y_true)}")
    if not invalid.empty:
        print(f"Excluded (blank/invalid Final_Class): {len(invalid)}")
        print(invalid.to_string(index=False))
    print(f"Evaluated rows: {len(y_true_v)}")
    print(f"Accuracy: {accuracy_score(y_true_v, y_pred_v):.4f}")
    print("\n--- Confusion Matrix ---")
    print(cm_df.to_string())
    print("\n--- Classification Report ---")
    print(classification_report(y_true_v, y_pred_v, labels=CLASSES, zero_division=0))
    return cm_df

if __name__ == "__main__":
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    llm = dget(cfg if isinstance(cfg, dict) else {}, "llm", {})
    inp_path, out, compact, prompt_name = llm_paths(llm)

    if dget(llm, "mode") == "evaluate":
        evaluate_labels(out, inp_path)
    else:
        key = (dget(llm, "api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key: raise RuntimeError("Missing API key. Set llm.api_key or OPENAI_API_KEY.")
        prompt_path = Path(__file__).with_name(f"{prompt_name}.yaml")
        prm = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        sys_p = dget(prm, "system_prompt", dget(prm, "system", "")) if isinstance(prm, dict) else ""
        step2_t = dget(prm, "step2_prompt") if isinstance(prm, dict) else None
        usr_t = (dget(prm, "step1_prompt") or dget(prm, "user_prompt_template", dget(prm, "prompt_template", ""))) if isinstance(prm, dict) else prm
        temp = float(dget(prm, "temperature", 0.0) if isinstance(prm, dict) else 0.0)
        limit = o2n(dget(llm, "limit"))
        retries, wait = int(dget(llm, "max_retries", 3)), int(dget(llm, "retry_wait", 60))
        workers = int(dget(llm, "workers", 20))
        df = read_input_table(inp_path)
        if NARR not in df.columns: raise ValueError(f"Missing {NARR} column.")
        client = openai.OpenAI(api_key=key)
        subset = df.head(limit) if limit else df
        tasks = [(i, str(r[NARR]).strip(), client, sys_p, usr_t, step2_t, llm, temp, retries, wait, compact) for i, r in subset.iterrows()]
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_row, t): t[0] for t in tasks}
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"[{prompt_name}]"):
                i, row = f.result()
                results[i] = row
        rows = [results[i] for i in sorted(results)]
        DATA_DIR.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig"); print(f"Saved {len(rows)} rows to: {out}")
        if bool(dget(llm, "evaluate", False)):
            evaluate_labels(out, inp_path)
