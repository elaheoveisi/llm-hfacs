import os
import json
import time
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # llm-hfacs/

DATA_DIR = WORKSPACE_ROOT / "data" / "GHFACS"

NARRATIVE_COLUMN = "narr_accf"
DEFAULT_MODEL = "gpt-4o-mini"
RETRY_WAIT = 5
MAX_RETRIES = 3

PROMPT_FILES = {
    "tot": SCRIPT_DIR / "tot.yaml",
    "cot": SCRIPT_DIR / "cot.yaml",
    "zeroshot": SCRIPT_DIR / "zeroshot.yaml",
}

# Models known to support response_format=json_object
VALID_JSON_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}


def load_prompt(style: str) -> dict:
    """Return {"system": str, "user": str, "model": str, "temperature": float}."""
    prompt_path = PROMPT_FILES[style]
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        yml = yaml.safe_load(f)

    # Support structured YAML (system_prompt + user_prompt_template)
    # and simple YAML (single prompt_template string)
    if isinstance(yml, str):
        return {"system": "", "user": yml, "model": DEFAULT_MODEL, "temperature": 0.0}

    system = yml.get("system_prompt", yml.get("system", "")).strip()
    user = yml.get("user_prompt_template", yml.get("prompt_template", "")).strip()
    raw_model = yml.get("model", DEFAULT_MODEL)
    model = raw_model if any(raw_model.startswith(m) for m in ("gpt-4", "gpt-3")) else DEFAULT_MODEL
    temperature = float(yml.get("temperature", 0.0))
    return {"system": system, "user": user, "model": model, "temperature": temperature}


def fill_prompt(template: str, narrative: str) -> str:
    # Support both placeholder conventions
    return template.replace("{narrative}", narrative).replace("{NARRATIVE_TEXT}", narrative)


def call_llm(system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    use_json_format = any(model.startswith(m) for m in VALID_JSON_MODELS)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(model=model, messages=messages, temperature=temperature)
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except openai.RateLimitError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
            else:
                raise
    return ""


def parse_output(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": raw}


def flatten_result(parsed: dict) -> dict:
    flat = {}
    for key, val in parsed.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[f"{key}__{sub_key}"] = sub_val
        elif isinstance(val, list):
            flat[key] = ", ".join(str(v) for v in val)
        else:
            flat[key] = val
    return flat


def run(style: str, input_file: Path, limit: int = None):
    prompt = load_prompt(style)

    print(f"Reading: {input_file}")
    df = pd.read_excel(input_file)

    if NARRATIVE_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{NARRATIVE_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    rows_to_process = df.head(limit) if limit else df
    results = []

    for idx, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc=f"[{style}]"):
        narrative = str(row[NARRATIVE_COLUMN]).strip()
        if not narrative or narrative.lower() in ("nan", "none", ""):
            results.append({"original_index": idx, "skip_reason": "empty_narrative"})
            continue

        user_msg = fill_prompt(prompt["user"], narrative)
        try:
            raw = call_llm(prompt["system"], user_msg, prompt["model"], prompt["temperature"])
            parsed = parse_output(raw)
        except Exception as e:
            parsed = {"error": str(e)}

        flat = flatten_result(parsed)
        flat["original_index"] = idx
        flat[NARRATIVE_COLUMN] = narrative
        results.append(flat)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    output_file = DATA_DIR / f"{stem}_LLM_Output_{style}.csv"
    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(results)} rows to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify GHFACS narratives using a specified LLM prompt style."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Excel filename inside data/GHFACS/ (e.g. GAHFACS_Version3.xlsx).",
    )
    parser.add_argument(
        "--style",
        choices=["tot", "cot", "zeroshot"],
        required=True,
        help="Prompt style to use: tot, cot, or zeroshot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (for testing).",
    )
    args = parser.parse_args()
    input_file = DATA_DIR / args.input
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    run(style=args.style, input_file=input_file, limit=args.limit)
