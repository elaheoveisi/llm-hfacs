import os, json, time, argparse, tempfile, shutil, yaml, pandas as pd, openai
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR, CFG_PATH = ROOT / "data" / "GHFACS", ROOT / "configs" / "config.yaml"
PROMPTS = {"tot": Path(__file__).parent / "initial_prompting.yaml"}
FINAL = {"Final_Answer", "Final_HFACS_Code", "Codes_Selected", "Final_Justification", "Confidence"}
JSON_MODELS, NARR = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}, "narr_accf"

def dget(d, k, v=None): return d.get(k, v) if isinstance(d, dict) else v
def o2n(v): return None if v in (None, "", "none", "null") else int(v)
def flatten(x): return {f"{k}__{a}": b for k, v in x.items() if isinstance(v, dict) for a, b in v.items()} | {k: ", ".join(map(str, v)) if isinstance(v, list) else v for k, v in x.items() if not isinstance(v, dict)}
def finals(x): return {k: v for k, v in x.items() if any(k == f or k.endswith(f"__{f}") for f in FINAL) or k in {"original_index", NARR, "skip_reason", "error"}}

def call(client, msgs, model, temp, retries, wait):
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
    i, n, client, sys_p, usr_t, model, temp, retries, wait, compact = args
    if not n or n.lower() in {"nan", "none", ""}: return i, {"original_index": i, "skip_reason": "empty_narrative"}
    try: x = flatten(json.loads(call(client, ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": usr_t.replace("{narrative}", n).replace("{NARRATIVE_TEXT}", n)}], model, temp, retries, wait)))
    except RuntimeError: raise
    except Exception as e: x = {"error": str(e)}
    x |= {"original_index": i, NARR: n}
    return i, finals(x) if compact else x

def read_input_table(path, tries=3, wait=2):
    path = Path(path)
    reader = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
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
                        "If it still fails, save/export the workbook as CSV and run with --input that_file.csv."
                    ) from e
            time.sleep(wait)

def read_input_excel(path, tries=3, wait=2):
    return read_input_table(path, tries, wait)

def evaluate_labels(llm_out, gt_path):
    """Compare LLM predictions for AE100, AE200, AD000 against ground-truth dataset labels."""
    llm = pd.read_csv(llm_out)
    suffix = Path(gt_path).suffix.lower()
    gt = (pd.read_excel if suffix in {".xlsx", ".xls"} else pd.read_csv)(gt_path)
    gt = gt[["AE100", "AE200", "AD000"]].reset_index().rename(columns={"index": "original_index"})
    merged = llm.merge(gt, on="original_index", how="inner")
    if merged.empty:
        raise ValueError("No rows matched between LLM output and ground truth — check original_index alignment.")
    label_map = {
        "AE100": "Q2_AE100_Performance_Skill_Error__Final_Answer",
        "AE200": "Q3_AE200_Judgment_Decision_Error__Final_Answer",
        "AD000": "Q4_AD000_Unknown_Deviation__Final_Answer",
    }
    rows = []
    for label, pred_col in label_map.items():
        y_true = (merged[label] == label).astype(int)
        y_pred = (merged[pred_col].str.strip().str.lower() == "yes").astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        n = len(y_true)
        acc  = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"Label": label, "N": n, "Support": int(y_true.sum()),
                     "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                     "Recall": round(rec, 4), "F1": round(f1, 4),
                     "TP": tp, "TN": tn, "FP": fp, "FN": fn})
        print(f"\n--- Confusion Matrix: {label} ---")
        print(f"{'':20s} {'Pred: YES':>10} {'Pred: NO':>10}")
        print(f"{'True: YES (positive)':20s} {tp:>10} {fn:>10}")
        print(f"{'True: NO  (negative)':20s} {fp:>10} {tn:>10}")
    result = pd.DataFrame(rows).set_index("Label")
    print("\n=== Label Accuracy vs. Ground Truth ===")
    print(result[["N", "Support", "Accuracy", "Precision", "Recall", "F1"]].to_string())
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classify GHFACS with prompt styles from config.yaml")
    p.add_argument("--config", default=str(CFG_PATH)); p.add_argument("--input"); p.add_argument("--style", choices=["tot"])
    p.add_argument("--model"); p.add_argument("--api-key"); p.add_argument("--limit", type=int); p.add_argument("--compact", action="store_true")
    p.add_argument("--max-retries", type=int); p.add_argument("--retry-wait", type=int); p.add_argument("--workers", type=int)
    p.add_argument("--evaluate", action="store_true", help="Compare LLM output against ground-truth labels and print per-label metrics")
    p.add_argument("--llm-output", help="Path to LLM output CSV for --evaluate (defaults to the standard output path)")
    a = p.parse_args(); cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")); llm = dget(cfg, "llm", {})
    style, model = a.style or dget(llm, "style", "tot"), a.model or dget(llm, "model", "gpt-4o-mini")
    inp, key = a.input or dget(llm, "input"), (a.api_key or dget(llm, "api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    if not inp: raise ValueError("Missing input. Set --input or llm.input in config.")
    compact = a.compact or bool(dget(llm, "compact", False))
    out = DATA_DIR / f"{Path(inp).stem}_LLM_Output_{style}{'_compact' if compact else ''}.csv"
    run_eval = a.evaluate or bool(dget(llm, "evaluate", False))
    if a.evaluate and not (a.config or a.input):
        # --evaluate only: skip classification, just print metrics for existing output
        evaluate_labels(Path(a.llm_output) if a.llm_output else out, DATA_DIR / inp)
    else:
        if not key: raise RuntimeError("Missing API key. Set --api-key, llm.api_key, or OPENAI_API_KEY.")
        prm = yaml.safe_load(PROMPTS[style].read_text(encoding="utf-8"))
        sys_p = dget(prm, "system_prompt", dget(prm, "system", "")) if isinstance(prm, dict) else ""
        usr_t = dget(prm, "user_prompt_template", dget(prm, "prompt_template", "")) if isinstance(prm, dict) else prm
        temp = float(dget(prm, "temperature", 0.0) if isinstance(prm, dict) else 0.0)
        limit = a.limit if a.limit is not None else o2n(dget(llm, "limit"))
        retries, wait = a.max_retries or int(dget(llm, "max_retries", 3)), a.retry_wait or int(dget(llm, "retry_wait", 60))
        workers = a.workers or int(dget(llm, "workers", 20))
        df = read_input_table(DATA_DIR / inp)
        if NARR not in df.columns: raise ValueError(f"Missing {NARR} column.")
        client = openai.OpenAI(api_key=key)
        subset = df.head(limit) if limit else df
        tasks = [(i, str(r[NARR]).strip(), client, sys_p, usr_t, model, temp, retries, wait, compact) for i, r in subset.iterrows()]
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_row, t): t[0] for t in tasks}
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"[{style}]"):
                i, row = f.result()
                results[i] = row
        rows = [results[i] for i in sorted(results)]
        DATA_DIR.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig"); print(f"Saved {len(rows)} rows to: {out}")
        if run_eval:
            evaluate_labels(out, DATA_DIR / inp)
