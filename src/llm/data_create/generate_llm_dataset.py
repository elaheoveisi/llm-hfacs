import os
import sys
import yaml
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import openai

# Use environment variable for API key (do NOT hardcode keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

PROMPT_DIR = Path(__file__).parent


PROMPT_FILES = {
    "cot": PROMPT_DIR / "cot.yaml",
    "tot": PROMPT_DIR / "tot.yaml",
    "zeroshot": PROMPT_DIR / "zeroshot.yaml",
}

# Columns to use from ASRS
target_columns = [
    "Report 1_Narrative",
    "Report 2_Narrative",
    "Report 1_Callback",
    "Report 2_Callback",
    "Report 1_Synopsis",
]

# Choose your LLM backend here
LLM_BACKEND = "openai"  # or "hf" for HuggingFace
LLM_MODEL = "gpt-3.5-turbo"  # or e.g. "mistralai/Mistral-7B-Instruct-v0.2"

HFACS_KEYS = [
    "Error",
    "Violation",
    "Situational_Factors",
    "Personnel_Factors",
    "Condition_of_Operators",
    "Inadequate_Supervision",
    "Failed_to_Correct_Problem",
    "Planned_Inappropriate_Operations",
    "Supervisory_Violation",
    "Organizational_Climate",
    "Resource_Management/Organizational_Process",
]


def load_prompt(style):
    with open(PROMPT_FILES[style], "r", encoding="utf-8") as f:
        yml = yaml.safe_load(f)
    return yml["prompt_template"]


def fill_prompt(template, row):
    # Replace placeholders with actual row values
    return template.replace("[Report 1_Narrative]", str(row["Report 1_Narrative"])) \
        .replace("[Report 2_Narrative]", str(row["Report 2_Narrative"])) \
        .replace("[Report 1_Callback]", str(row["Report 1_Callback"])) \
        .replace("[Report 2_Callback]", str(row["Report 2_Callback"])) \
        .replace("[Report 1_Synopsis]", str(row["Report 1_Synopsis"]))


def query_llm(prompt, backend=LLM_BACKEND, model=LLM_MODEL):
    if backend == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content
    elif backend == "hf":
        raise NotImplementedError("HF Transformers call not implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def parse_llm_output(output):
    """Parse the model response into a flat dict for CSV export.

    The prompts now request a JSON object with HFACS keys only. We parse that
    JSON when possible and fall back to an error marker plus raw output.
    """
    raw_output = (output or "").strip()
    result = {key: pd.NA for key in HFACS_KEYS}
    result["parse_ok"] = 0
    result["raw_output"] = raw_output

    if not raw_output:
        result["parse_error"] = "empty_output"
        return result

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        result["parse_error"] = f"json_decode_error: {exc.msg}"
        return result

    if not isinstance(parsed, dict):
        result["parse_error"] = "json_not_object"
        return result

    for key in HFACS_KEYS:
        value = parsed.get(key, pd.NA)
        if value is pd.NA:
            result[key] = pd.NA
            continue
        try:
            result[key] = int(value)
        except (TypeError, ValueError):
            result[key] = pd.NA

    result["parse_ok"] = 1
    result["parse_error"] = ""
    return result



def generate_llm_dataset(
    input_path,
    output_path,
    style="cot",
    limit=None,
    backend=LLM_BACKEND,
    model=LLM_MODEL
):
    prompt_template = load_prompt(style)
    df = pd.read_csv(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if limit and idx >= limit:
            break
        prompt = fill_prompt(prompt_template, row)
        try:
            output = query_llm(prompt, backend=backend, model=model)
        except Exception as e:
            print(f"[ERROR] LLM call failed for row {idx}: {e}")
            output = ""
        parsed = parse_llm_output(output)
        results.append(
            {
                "index": idx,
                "prompt_style": style,
                **parsed,
            }
        )
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved LLM predictions to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Input CSV path")
    parser.add_argument("--output", type=str, required=False, help="Output CSV path")
    parser.add_argument("--style", choices=["cot", "tot", "zeroshot"], default="cot", help="Prompt style to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows (for testing)")
    args = parser.parse_args()
    # Default to old paths if not provided
    WORKSPACE_ROOT = PROMPT_DIR.parent.parent
    input_path = Path(args.input) if args.input else (WORKSPACE_ROOT / "data" / "processed" / "processed_output.csv")
    output_path = Path(args.output) if args.output else (WORKSPACE_ROOT / "data" / "processed" / "llm_predictions.csv")
    generate_llm_dataset(
        input_path=input_path,
        output_path=output_path,
        style=args.style,
        limit=args.limit
    )
