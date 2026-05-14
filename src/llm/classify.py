import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


def dget(d, k, v=None):
    return d.get(k, v) if isinstance(d, dict) else v


CFG_PATH = Path(
    os.getenv("CONFIG_PATH")
    or next(
        p / "configs" / "config.yaml"
        for p in Path(__file__).resolve().parents
        if (p / "configs" / "config.yaml").exists()
    )
)
_cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

CLASSES = dget(
    dget(_cfg, "llm", {}), "classes", ["AE100 only", "AE200 only", "Both", "None"]
)
AE100_ONLY, AE200_ONLY, BOTH, ANY_OF_THEM = CLASSES


def flatten(x):
    return {
        f"{k}__{a}": b
        for k, v in x.items()
        if isinstance(v, dict)
        for a, b in v.items()
    } | {
        k: ", ".join(map(str, v)) if isinstance(v, list) else v
        for k, v in x.items()
        if not isinstance(v, dict)
    }


def finals(x, narr, final_cols):
    return {
        k: v
        for k, v in x.items()
        if any(k == f or k.endswith(f"__{f}") for f in final_cols)
        or k in {"original_index", narr, "skip_reason", "error"}
    }


def _reader(path):
    return pd.read_csv if Path(path).suffix.lower() == ".csv" else pd.read_excel


def four_class(ae100, ae200):
    if ae100 and ae200:
        return BOTH
    if ae100:
        return AE100_ONLY
    if ae200:
        return AE200_ONLY
    return ANY_OF_THEM


def call(client, msgs, llm, prm):
    model = dget(llm, "model", "gpt-4o-mini")
    retries = int(dget(llm, "max_retries", 3))
    json_models = set(dget(llm, "json_models", ["gpt-4o", "gpt-4o-mini"]))
    no_temp_models = set(dget(llm, "no_temperature_models", []))
    for i in range(1, retries + 1):
        try:
            kw = {"model": model, "messages": msgs}
            if model not in no_temp_models:
                kw["temperature"] = float(dget(prm, "temperature", 0.0))
            if model in json_models:
                kw["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kw).choices[0].message.content
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as e:
            if isinstance(e, openai.RateLimitError) and (
                "insufficient_quota" in str(e) or "billing" in str(e).lower()
            ):
                raise RuntimeError(
                    f"Quota exhausted — add credits at platform.openai.com/settings/organization/billing\n{e}"
                ) from e
            if i == retries:
                raise
            time.sleep(int(dget(llm, "retry_wait", 60)) * i)


def evaluate_labels(llm_out, gt_path, llm):
    llm_df = pd.read_csv(llm_out, keep_default_na=False, na_values=[""])
    gt_cols = dget(llm, "ground_truth_columns", ["AE100", "AE200"])
    gt = (
        _reader(gt_path)(gt_path)[gt_cols]
        .reset_index()
        .rename(columns={"index": "original_index"})
    )
    merged = llm_df.merge(gt, on="original_index", how="inner")
    if merged.empty:
        raise ValueError(
            "No rows matched between LLM output and ground truth — check original_index alignment."
        )
    class_col = next(
        (c for c in ("Final_Class", "Final_HFACS_Code") if c in merged.columns), None
    )
    if class_col is None:
        raise ValueError(
            f"Missing Final_Class or Final_HFACS_Code column in LLM output. Columns: {list(llm_df.columns)}"
        )

    _lookup = {
        alias: cls for cls in CLASSES for alias in (cls.lower(), cls.lower().split()[0])
    }
    _lookup |= {"multiple": _lookup.get("both"), "none": _lookup.get("None")}
    merged["y_true"] = merged.apply(
        lambda r: four_class(*[r[c] == c for c in gt_cols]), axis=1
    )
    merged["y_pred"] = merged[class_col].apply(
        lambda v: _lookup.get(str(v).strip().lower()) if not pd.isna(v) else None
    )
    valid = merged["y_pred"].isin(CLASSES)
    invalid = merged.loc[~valid, ["original_index", class_col]]
    y_true_v, y_pred_v = (
        merged.loc[valid, "y_true"].tolist(),
        merged.loc[valid, "y_pred"].tolist(),
    )

    cm = confusion_matrix(y_true_v, y_pred_v, labels=CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {c}" for c in CLASSES],
        columns=[f"Pred: {c}" for c in CLASSES],
    )

    print("\n=== Four-Class Evaluation (Final_Class vs. Ground Truth) ===")
    print(f"Total rows: {len(merged)}")
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


def run():
    llm = dget(_cfg, "llm", {})
    data_dir = CFG_PATH.parents[1] / dget(
        dget(_cfg, "paths", {}), "ghfacs_data_dir", "data/GHFACS"
    )
    if not (inp := dget(llm, "input")):
        raise ValueError("Missing input. Set llm.input in config.")
    prompt_name = dget(llm, "prompt", dget(llm, "default_prompt", "initial_prompting"))
    inp_path = data_dir / inp
    out = (
        data_dir
        / f"{inp_path.stem}_LLM_Output_{prompt_name}{'_compact' if dget(llm, 'compact', False) else ''}.csv"
    )

    if dget(llm, "mode") == "evaluate":
        evaluate_labels(out, inp_path, llm)
    else:
        key = (dget(llm, "api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("Missing API key. Set llm.api_key or OPENAI_API_KEY.")
        prm = yaml.safe_load(
            Path(__file__).with_name(f"{prompt_name}.yaml").read_text(encoding="utf-8")
        )
        narr = dget(llm, "narrative_column", "narr_accf")
        df = _reader(inp_path)(inp_path)
        if narr not in df.columns:
            raise ValueError(f"Missing {narr} column.")
        client = openai.OpenAI(api_key=key)

        def process(i, n):
            compact = dget(llm, "compact", False)
            final_cols = set(
                dget(
                    llm,
                    "output_columns",
                    [
                        "Final_Answer",
                        "Final_Class",
                        "Final_HFACS_Code",
                        "Codes_Selected",
                        "Final_Justification",
                        "Confidence",
                    ],
                )
            )
            sys_p = dget(prm, "system_prompt", dget(prm, "system", ""))
            step2_t = dget(prm, "step2_prompt")
            usr_t = dget(
                prm,
                "step1_prompt",
                dget(prm, "user_prompt_template", dget(prm, "prompt_template", "")),
            )
            if not n or n.lower() in {"nan", "none", ""}:
                return i, {"original_index": i, "skip_reason": "empty_narrative"}

            def msgs(content):
                return ([{"role": "system", "content": sys_p}] if sys_p else []) + [
                    {"role": "user", "content": content}
                ]

            try:
                step1_out = call(
                    client, msgs(usr_t.replace("{narrative}", n)), llm, prm
                )
                raw = (
                    call(
                        client,
                        msgs(step2_t.replace("{preconditions}", step1_out)),
                        llm,
                        prm,
                    )
                    if step2_t
                    else step1_out
                )
                x = flatten(json.loads(raw))
            except RuntimeError:
                raise
            except Exception as e:
                x = {"error": str(e)}
            x |= {"original_index": i, narr: n}
            return i, finals(x, narr, final_cols) if compact else x

        _lim = dget(llm, "limit")
        subset = df.head(int(_lim)) if _lim not in (None, "", "none", "null") else df
        tasks = [(i, str(r[narr]).strip()) for i, r in subset.iterrows()]
        results = {}
        with ThreadPoolExecutor(max_workers=int(dget(llm, "workers", 20))) as executor:
            futures = {executor.submit(process, *t): t[0] for t in tasks}
            for f in tqdm(
                as_completed(futures), total=len(futures), desc=f"[{prompt_name}]"
            ):
                i, row = f.result()
                results[i] = row
        rows = [v for _, v in sorted(results.items())]
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved {len(rows)} rows to: {out}")
        if dget(llm, "evaluate", False):
            evaluate_labels(out, inp_path, llm)
