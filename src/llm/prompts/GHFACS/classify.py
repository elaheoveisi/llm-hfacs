import os, json, time, tempfile, shutil, yaml, pandas as pd, openai
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR, CFG_PATH = ROOT / "data" / "GHFACS", ROOT / "configs" / "config.yaml"
PROMPT_PATH = Path(__file__).with_name("initial_prompting.yaml")
FINAL = {"Final_Answer", "Final_HFACS_Code", "Codes_Selected", "Final_Justification", "Confidence"}
JSON_MODELS, NARR = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}, "narr_accf"
LABEL_PRED_COLS = {
    "AE100": "Q2_AE100_Performance_Skill_Error__Final_Answer",
    "AE200": "Q3_AE200_Judgment_Decision_Error__Final_Answer",
    "AD000": "Q4_AD000_Unknown_Deviation__Final_Answer",
}

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
    inp_path = DATA_DIR / inp
    out = DATA_DIR / f"{inp_path.stem}_LLM_Output_initial_prompt{'_compact' if compact else ''}.csv"
    return inp_path, out, compact

def binary_metrics(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
    return {
        "N": len(y_true), "Support": int(y_true.sum()),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

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
    i, n, client, sys_p, usr_t, llm, temp, retries, wait, compact = args
    if not n or n.lower() in {"nan", "none", ""}: return i, {"original_index": i, "skip_reason": "empty_narrative"}
    try: x = flatten(json.loads(call(client, ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": usr_t.replace("{narrative}", n).replace("{NARRATIVE_TEXT}", n)}], llm, temp, retries, wait)))
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
    """Compare LLM predictions for AE100, AE200, AD000 against ground-truth dataset labels."""
    llm = pd.read_csv(llm_out)
    gt = _reader(gt_path)(gt_path)
    gt = gt[list(LABEL_PRED_COLS)].reset_index().rename(columns={"index": "original_index"})
    merged = llm.merge(gt, on="original_index", how="inner")
    if merged.empty:
        raise ValueError("No rows matched between LLM output and ground truth — check original_index alignment.")
    rows = []
    for label, pred_col in LABEL_PRED_COLS.items():
        y_true = (merged[label] == label).astype(int)
        y_pred = (merged[pred_col].str.strip().str.lower() == "yes").astype(int)
        metrics = binary_metrics(y_true, y_pred)
        rows.append({"Label": label, **metrics})
        tp, tn, fp, fn = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]
        print(f"\n--- Confusion Matrix: {label} ---")
        print(f"{'':20s} {'Pred: YES':>10} {'Pred: NO':>10}")
        print(f"{'True: YES (positive)':20s} {tp:>10} {fn:>10}")
        print(f"{'True: NO  (negative)':20s} {fp:>10} {tn:>10}")
    result = pd.DataFrame(rows).set_index("Label")
    print("\n=== Label Accuracy vs. Ground Truth ===")
    print(result[["N", "Support", "Accuracy", "Precision", "Recall", "F1"]].to_string())
    return result

if __name__ == "__main__":
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    llm = dget(cfg if isinstance(cfg, dict) else {}, "llm", {})
    inp_path, out, compact = llm_paths(llm)

    if dget(llm, "mode") == "evaluate":
        evaluate_labels(out, inp_path)
    else:
        key = (dget(llm, "api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key: raise RuntimeError("Missing API key. Set llm.api_key or OPENAI_API_KEY.")
        prm = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
        sys_p = dget(prm, "system_prompt", dget(prm, "system", "")) if isinstance(prm, dict) else ""
        usr_t = dget(prm, "user_prompt_template", dget(prm, "prompt_template", "")) if isinstance(prm, dict) else prm
        temp = float(dget(prm, "temperature", 0.0) if isinstance(prm, dict) else 0.0)
        limit = o2n(dget(llm, "limit"))
        retries, wait = int(dget(llm, "max_retries", 3)), int(dget(llm, "retry_wait", 60))
        workers = int(dget(llm, "workers", 20))
        df = read_input_table(inp_path)
        if NARR not in df.columns: raise ValueError(f"Missing {NARR} column.")
        client = openai.OpenAI(api_key=key)
        subset = df.head(limit) if limit else df
        tasks = [(i, str(r[NARR]).strip(), client, sys_p, usr_t, llm, temp, retries, wait, compact) for i, r in subset.iterrows()]
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_row, t): t[0] for t in tasks}
            for f in tqdm(as_completed(futures), total=len(futures), desc="[initial_prompt]"):
                i, row = f.result()
                results[i] = row
        rows = [results[i] for i in sorted(results)]
        DATA_DIR.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig"); print(f"Saved {len(rows)} rows to: {out}")
        if bool(dget(llm, "evaluate", False)):
            evaluate_labels(out, inp_path)
