#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------
# HTTP / OpenAI-compat API
# -----------------------
def post_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> Tuple[str, Dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:2000]}")
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            return str(text), data
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_s * (2 ** attempt))
            else:
                break

    raise RuntimeError(f"Request failed after retries. Last error: {last_err!r}")


# -----------------------
# File helpers
# -----------------------
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------
# Prompt building
# -----------------------
def build_system_prompt_holistic() -> str:
    return (
        "You are an expert evaluator of multi-turn conversations between assistants and users.\n"
        "You are evaluating the ASSISTANT's responses, not the user's inputs.\n"
        "Follow the rubric exactly on each metric's guideline and anchor and pay close attention to score definitions.\n\n"
        "EVALUATION FOCUS:\n"
        "- You do not need to closely scrutinize the technical details of the solution steps.\n"
        "- Please focus on the assistant's emotional tone, responsiveness to the user's state, and whether the response supports progress, rather than on whether every technical step is correct.\n"
        "- Base all ratings only on the assistant's responses, not on the user's messages.\n"
        "CRITICAL SCORING GUIDELINES:\n"
        "- Score 1: Severe failure/mismatch according to rubric anchors\n"
        "- Score 2: Noticeable issues/poor performance according to rubric anchors\n"
        "- Score 3: Adequate/appropriate performance with minor issues\n"
        "- Score 4: Good performance with strong alignment to expectations\n"
        "- Score 5: Excellent performance with precise calibration/perfection according to rubric anchors\n"
        "EVALUATION APPROACH:\n"
        "- Critically analyze the entire conversation for potential flaws, not just strengths\n"
        "- Look for specific evidence of imperfections even in generally good conversations\n"
        "- Remember that most real-world conversations will have some imperfections and should receive scores of 2, 3, or 4\n"
        "- Reserve score 5 for conversations that demonstrate truly exceptional performance with no discernible flaws\n"
        "OUTPUT RULES:\n"
        "1) Output MUST be valid JSON only.\n"
        "2) Use ONLY metric_id values provided (exact match).\n"
        "3) Include number_id for each metric and it must match the rubric's number_id for that metric_id.\n"
        "4) Scores must be integers 1-5.\n"
        "5) Provide a short rationale for each metric.\n"
    )


def rubric_for_single_metric(rubric: Dict[str, Any], scope: str, metric_id: str) -> Dict[str, Any]:
    """Extract a single metric from the rubric."""
    metrics = [m for m in rubric.get("metrics", []) if m.get("scope") == scope and m.get("id") == metric_id]
    if not metrics:
        return {}

    m = metrics[0]
    return {
        "global_instructions": rubric.get("global_instructions", []),
        "metrics": [
            {
                "metric_id": m.get("id"),
                "number_id": m.get("number_id"),
                "scope": m.get("scope"),
                "definition": m.get("definition"),
                "guidance": m.get("guidance"),
                "anchors": m.get("anchors"),
            }
        ],
    }


def conversation_text(turns: List[Dict[str, Any]]) -> str:
    """
    Build full conversation context for holistic evaluation.
    Includes all turns in the conversation.
    """
    pieces: List[str] = []
    for t in turns:
        n = t.get("turn")
        u = (t.get("user_content") or "").strip()
        a = (t.get("assistant_content") or "").strip()

        # Truncate long user/assistant content to save tokens
        max_content_length = 500  # Limit each message to 500 characters
        if len(u) > max_content_length:
            u = u[:max_content_length] + "..."
        if len(a) > max_content_length:
            a = a[:max_content_length] + "..."

        pieces.append(f"T{n} USER:\n{u}\n")
        pieces.append(f"T{n} ASSISTANT:\n{a}\n")
        pieces.append("-" * 30)
    return "\n".join(pieces)


def build_user_prompt_holistic(
    scenario_id: str,
    conversation_text: str,
    rubric_holistic: Dict[str, Any],
    metric_id: str
) -> str:
    required_output = {
        "scenario_id": scenario_id,
        "holistic_evaluation": {
            "metrics": [
                {
                    "metric_id": metric_id,
                    "number_id": rubric_holistic["metrics"][0]["number_id"] if rubric_holistic.get("metrics") else 0,
                    "score": 3,
                    "rationale": "...",
                    "evidence": "..."
                }
            ],
        },
    }

    return (
        f"SCENARIO_ID: {scenario_id}\n\n"
        "FULL_CONVERSATION:\n"
        f"{conversation_text}\n\n"
        "RUBRIC (HOLISTIC METRICS ONLY):\n"
        f"{json.dumps(rubric_holistic, ensure_ascii=False)}\n\n"
        "SCORING REMINDER: Assign scores based on how well the entire conversation matches the specific criteria for each score level. "
        "Do not default to high scores - use the full range from 1-5 based on actual performance. "
        "Score 5 means exceptional performance across the entire conversation, while score 1 means severe failure. "
        "Return JSON only in this shape:\n"
        f"{json.dumps(required_output, ensure_ascii=False)}"
    )


# -----------------------
# Parsing / validation (holistic-only)
# -----------------------
def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    # First try to parse as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If that fails, try to clean up common issues
    # Remove trailing commas before closing brackets/braces
    cleaned = re.sub(r',(\s*[}\]])', r'\1', text)

    # Remove extra trailing braces/brackets that might have been added
    # Find the position of the last valid JSON structure
    for i in range(len(cleaned)):
        try:
            chunk = cleaned[:len(cleaned)-i]
            parsed = json.loads(chunk)
            # If it parses, check if it has the expected structure
            if isinstance(parsed, dict) and "scenario_id" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Try another approach - find the longest valid JSON prefix
    # This is more aggressive cleaning
    bracket_count = 0
    brace_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0

    for i, char in enumerate(cleaned):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

        # If we have balanced brackets and braces, this might be a valid endpoint
        if not in_string and bracket_count == 0 and brace_count == 0 and i > 0:
            try:
                potential = cleaned[:i+1]
                parsed = json.loads(potential)
                if isinstance(parsed, dict) and "scenario_id" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


def build_rubric_map_single(rubric_holistic: Dict[str, Any]) -> Tuple[Dict[str, int], List[str]]:
    id_to_number: Dict[str, int] = {}
    ids: List[str] = []
    for m in rubric_holistic.get("metrics", []):
        mid = m.get("metric_id")
        num = m.get("number_id")
        if mid and isinstance(num, int):
            id_to_number[mid] = num
            ids.append(mid)
    return id_to_number, ids


def validate_holistic_obj_single(
    obj: Dict[str, Any],
    scenario_id: str,
    rubric_holistic: Dict[str, Any],
) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Judge output is not a JSON object."
    if obj.get("scenario_id") != scenario_id:
        return False, "scenario_id mismatch."

    he = obj.get("holistic_evaluation")
    if not isinstance(he, dict):
        return False, "holistic_evaluation must be an object."

    metrics = he.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 1:
        return False, "metrics must be a list with exactly one item."

    id_to_number, allowed_ids = build_rubric_map_single(rubric_holistic)

    seen = set()
    for m in metrics:
        if not isinstance(m, dict):
            return False, "metric entry must be an object."
        mid = m.get("metric_id")
        num = m.get("number_id")
        score = m.get("score")
        if mid not in allowed_ids:
            return False, f"Unknown metric_id: {mid}"
        if not isinstance(num, int) or id_to_number.get(mid) != num:
            return False, f"number_id mismatch for {mid}"
        if not isinstance(score, int) or not (1 <= score <= 5):
            return False, f"Invalid score for {mid}: {score}"
        if mid in seen:
            return False, f"Duplicate metric_id: {mid}"
        seen.add(mid)

    if set(seen) != set(allowed_ids):
        return False, "Missing or extra holistic metrics."

    return True, ""


# -----------------------
# Output writers
# -----------------------
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["scenario_id", "judge_model", "turn", "metric_id", "number_id", "score"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--metric-id", required=True, help="Specific metric ID to evaluate")

    ap.add_argument("--input-dir", default="../pilot_final_responses")
    ap.add_argument("--rubric", default="rubric.json")

    ap.add_argument("--out-dir", default="../pilot_llm_evaluation_holistic")

    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff-s", type=float, default=1.0)

    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    rubric_path = (here / args.rubric).resolve()
    in_dir = (here / args.input_dir).resolve()

    out_root = (here / args.out_dir).resolve()
    out_json_dir = out_root / "json"
    out_csv_dir = out_root / "csv"
    ensure_dir(out_json_dir)
    ensure_dir(out_csv_dir)

    rubric = load_json(rubric_path)

    # Get the specific metric to evaluate
    rubric_single_metric = rubric_for_single_metric(rubric, "holistic", args.metric_id)
    if not rubric_single_metric.get("metrics"):
        print(f"Metric '{args.metric_id}' not found in holistic metrics.")
        return

    lnum_pat = re.compile(r"^L\d+.*\.json$", re.IGNORECASE)
    scenario_files = sorted(f for f in in_dir.glob("L*.json") if lnum_pat.match(f.name))

    if not scenario_files:
        print(f"No scenario files found in {in_dir} matching L<digits>*.json")
        return

    # Collect all rows for the combined CSV
    all_combined_rows: List[Dict[str, Any]] = []

    for fp in scenario_files:
        scenario = load_json(fp)
        scenario_id = scenario.get("scenario_id", fp.stem)
        turns = scenario.get("turns", [])
        if not turns:
            print(f"[SKIP] {fp.name}: no turns")
            continue

        # Evaluate the specific holistic metric
        all_rows: List[Dict[str, Any]] = []

        conv_text = conversation_text(turns)

        messages = [
            {"role": "system", "content": build_system_prompt_holistic()},
            {"role": "user", "content": build_user_prompt_holistic(scenario_id, conv_text, rubric_single_metric, args.metric_id)},
        ]

        print(f"Judging HOLISTIC metric '{args.metric_id}': {scenario_id} ...")

        t0 = time.time()
        err: Optional[str] = None
        judge_text = ""
        judge_obj: Optional[Dict[str, Any]] = None

        try:
            judge_text, _raw = post_chat(
                base_url=args.base_url,
                model=args.judge_model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                retries=args.retries,
                backoff_s=args.backoff_s,
            )
            judge_obj = try_parse_json(judge_text)
            if judge_obj is None:
                err = "Judge output was not valid JSON."
            else:
                ok, why = validate_holistic_obj_single(judge_obj, scenario_id, rubric_single_metric)
                if not ok:
                    err = why
                    judge_obj = None
        except Exception as e:
            err = repr(e)

        elapsed_s = round(time.time() - t0, 4)

        # Save rich holistic artifact
        out_json_path = out_json_dir / f"{scenario_id}.{args.metric_id}.holistic.judge.json"
        rich_payload = {
            "scenario_id": scenario_id,
            "judge_model": args.judge_model,
            "source_file": fp.name,
            "metric_id": args.metric_id,
            "turn": 0,  # Always 0 for holistic evaluations
            "elapsed_s": elapsed_s,
            "error": err,
            "judge_text": judge_text,
            "judge_json": judge_obj,
        }
        write_json(out_json_path, rich_payload)

        # Flatten to lean rows
        out_csv_path = out_csv_dir / f"{scenario_id}.{args.metric_id}.holistic.csv"
        if judge_obj is not None:
            metrics = judge_obj["holistic_evaluation"]["metrics"]
            rows = [
                {
                    "scenario_id": scenario_id,
                    "judge_model": args.judge_model,
                    "metric_id": m["metric_id"],
                    "number_id": m["number_id"],
                    "score": m["score"],
                    "turn": 0,  # Always 0 for holistic evaluations
                }
                for m in metrics
            ]
            write_csv(out_csv_path, rows)
            all_rows.extend(rows)
            all_combined_rows.extend(rows)  # Add to combined rows
            print(f"  ✓ wrote JSON: {out_json_path}")
            print(f"  ✓ wrote CSV : {out_csv_path}")
        else:
            error_rows = [
                {
                    "scenario_id": scenario_id,
                    "judge_model": args.judge_model,
                    "metric_id": "__ERROR__",
                    "number_id": "",
                    "score": "",
                    "turn": 0,  # Always 0 for holistic evaluations
                }
            ]
            write_csv(out_csv_path, error_rows)
            all_combined_rows.extend(error_rows)  # Add to combined rows
            print(f"  ✗ judge failed: {err}")
            print(f"  ✓ wrote JSON: {out_json_path}")
            print(f"  ! wrote CSV (error marker): {out_csv_path}")

    # Write combined CSV for all scenarios
    combined_csv_path = out_csv_dir / f"ALL_{args.metric_id}_holistic.csv"
    write_csv(combined_csv_path, all_combined_rows)
    print(f"  ✓ wrote combined CSV: {combined_csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()