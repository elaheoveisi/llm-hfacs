import os, json, time, argparse, tempfile, shutil, yaml, pandas as pd, openai
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR, CFG_PATH = ROOT / "data" / "GHFACS", ROOT / "configs" / "config.yaml"
PROMPTS = {"tot": Path(__file__).parent / "tot.yaml"}
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

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classify GHFACS with prompt styles from config.yaml")
    p.add_argument("--config", default=str(CFG_PATH)); p.add_argument("--input"); p.add_argument("--style", choices=["tot"])
    p.add_argument("--model"); p.add_argument("--api-key"); p.add_argument("--limit", type=int); p.add_argument("--compact", action="store_true")
    p.add_argument("--max-retries", type=int); p.add_argument("--retry-wait", type=int); p.add_argument("--workers", type=int)
    a = p.parse_args(); llm = dget(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")), "llm", {})
    style, model = a.style or dget(llm, "style", "tot"), a.model or dget(llm, "model", "gpt-4o-mini")
    inp, key = a.input or dget(llm, "input"), (a.api_key or dget(llm, "api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    if not inp: raise ValueError("Missing input. Set --input or llm.input in config.")
    if not key: raise RuntimeError("Missing API key. Set --api-key, llm.api_key, or OPENAI_API_KEY.")
    prm = yaml.safe_load(PROMPTS[style].read_text(encoding="utf-8"))
    sys_p = dget(prm, "system_prompt", dget(prm, "system", "")) if isinstance(prm, dict) else ""
    usr_t = dget(prm, "user_prompt_template", dget(prm, "prompt_template", "")) if isinstance(prm, dict) else prm
    temp = float(dget(prm, "temperature", 0.0) if isinstance(prm, dict) else 0.0)
    limit, compact = a.limit if a.limit is not None else o2n(dget(llm, "limit")), (a.compact or bool(dget(llm, "compact", False)))
    retries, wait = a.max_retries or int(dget(llm, "max_retries", 3)), a.retry_wait or int(dget(llm, "retry_wait", 60))
    workers = a.workers or int(dget(llm, "workers", 20))
    df = read_input_table(DATA_DIR / inp)
    out = DATA_DIR / f"{Path(inp).stem}_LLM_Output_{style}{'_compact' if compact else ''}.csv"
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
