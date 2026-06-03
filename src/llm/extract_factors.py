from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
import yaml
from tqdm import tqdm

from llm.utils import resolve_openai_api_key


def _retry_delay(error: Exception, attempt: int, llm_cfg: dict) -> float:
    """Prefer OpenAI's suggested wait, then add a small stagger for parallel calls."""
    retry_wait = float(llm_cfg["retry_wait"])
    match = re.search(r"try again in ([0-9.]+)s", str(error), flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + retry_wait + random.uniform(0.25, 1.25)
    return retry_wait * attempt + random.uniform(0.25, 1.25)


def _build_prompt_parts(cols: list[str], factor_defs: dict[str, str]) -> tuple[str, str]:
    """Return (factors_text, template_json) for the prompt placeholders."""
    factor_lines = [f"  - {col}: {factor_defs.get(col, col)}" for col in cols]
    factors_text = "\n".join(factor_lines)
    template = json.dumps({col: 0 for col in cols}, indent=2)
    return factors_text, template


def _call_llm(client, msgs: list, llm_cfg: dict) -> str:
    model = llm_cfg["model"]
    retries = int(llm_cfg["max_retries"])
    json_models = set(llm_cfg["json_models"])
    no_temp_models = set(llm_cfg["no_temperature_models"])
    for attempt in range(1, retries + 1):
        try:
            kw = {"model": model, "messages": msgs}
            if model not in no_temp_models:
                kw["temperature"] = 0.0
            if model in json_models:
                kw["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kw).choices[0].message.content
        except openai.AuthenticationError as e:
            raise RuntimeError(
                "OpenAI authentication failed. Check the API key available to this "
                "Python process; the current run is not using a valid key."
            ) from e
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            if isinstance(e, openai.RateLimitError) and (
                "insufficient_quota" in str(e) or "billing" in str(e).lower()
            ):
                raise RuntimeError(f"Quota exhausted — add credits at platform.openai.com\n{e}") from e
            if attempt == retries:
                raise
            time.sleep(_retry_delay(e, attempt, llm_cfg))


def run(config: dict) -> None:
    """Extract new HFACS factor columns from Report 1_Narrative via LLM.

    Reads processed_output.csv (untouched), sends each narrative to the LLM,
    and saves a new file (new_features_asrs.csv) containing only the row ID
    and the extracted 0/1 factor columns.
    """
    llm_cfg = config["llm"]
    processed_path = Path(config["paths"]["processed_csv"])
    out_path = Path(config["paths"]["new_features_csv"])

    df = pd.read_csv(processed_path, low_memory=False)
    narr_col = "Report 1_Narrative"
    if narr_col not in df.columns:
        raise ValueError(f"Missing column '{narr_col}' in {processed_path}")

    # Load prompt — the factors dict in the YAML is the authoritative list of what to extract
    prompts_dir = Path(config["paths"]["prompts_dir"]).parent / "ASRS"
    prm = yaml.safe_load(
        (prompts_dir / "extract_hfacs_factors.yaml").read_text(encoding="utf-8")
    )
    sys_p = prm["system_prompt"].strip()
    usr_t = prm["user_prompt_template"]
    factor_defs: dict[str, str] = prm.get("factors", {})
    # Only extract columns that are not already in processed_output
    new_cols = [c for c in factor_defs if c not in df.columns]

    print(f"[INFO] Extracting {len(new_cols)} HFACS factor columns via LLM:")
    for c in new_cols:
        print(f"  - {c}")
    factors_text, template_json = _build_prompt_parts(new_cols, factor_defs)

    key = resolve_openai_api_key(llm_cfg)
    client = openai.OpenAI(api_key=key)

    def process(i: int, narrative: str) -> tuple[int, dict]:
        if not narrative or narrative.lower() in {"nan", "none", ""}:
            return i, {col: 0 for col in new_cols}
        prompt = (
            usr_t
            .replace("{narrative}", narrative)
            .replace("{factors}", factors_text)
            .replace("{template}", template_json)
        )
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}]
        try:
            raw = _call_llm(client, msgs, llm_cfg)
            data = json.loads(raw)
            return i, {col: int(bool(data.get(col, 0))) for col in new_cols}
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError):
            raise
        except Exception as e:
            print(f"\n[WARN] Row {i} failed ({type(e).__name__}): {e}")
            return i, {col: 0 for col in new_cols}

    lim = llm_cfg.get("limit")
    subset = df if lim in (None, "", "none", "null") else df.head(int(lim))
    tasks = [(i, str(row[narr_col]).strip()) for i, row in subset.iterrows()]

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=int(llm_cfg["workers"])) as executor:
        futures = {executor.submit(process, *t): t[0] for t in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc="[extract_factors]"):
            i, row_data = f.result()
            results[i] = row_data

    # Add new columns to the subset and save — original processed_output.csv is untouched
    out_df = subset.copy()
    for col in new_cols:
        out_df[col] = out_df.index.map(lambda idx, c=col: results.get(idx, {}).get(c, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Saved {len(out_df)} rows × {len(out_df.columns)} columns to: {out_path}")
