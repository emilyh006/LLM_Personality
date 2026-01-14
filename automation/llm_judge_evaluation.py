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
            r.raise_for_status()
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
def build_system_prompt() -> str:
    return (
        "You are an expert evaluator of assistant responses in multi-turn conversations.\n"
        "Follow the rubric exactly.\n\n"
        "OUTPUT RULES:\n"
        "1) Output MUST be valid JSON only. No markdown. No extra text.\n"
        "2) Use ONLY metric_id values provided (exact match).\n"
        "3) Include number_id for each metric and it must match the rubric's number_id for that metric_id.\n"
        "4) Scores must be integers 1-5.\n"
        "5) Provide a short rationale and short evidence for each metric.\n"
    )


def render_transcript(turns: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for t in turns:
        n = t.get("turn")
        u = (t.get("user_content") or "").strip()
        a = (t.get("assistant_content") or "").strip()
        parts.append(f"TURN {n} — USER:\n{u}\n")
        parts.append(f"TURN {n} — ASSISTANT:\n{a}\n")
        parts.append("-" * 40)
    return "\n".join(parts)


def rubric_to_prompt_blob(rubric: Dict[str, Any]) -> Dict[str, Any]:
    metrics = rubric.get("metrics", [])
    return {
        "rubric_name": rubric.get("rubric_name"),
        "version": rubric.get("version"),
        "global_instructions": rubric.get("global_instructions", []),
        "metrics": [
            {
                "metric_id": m.get("id"),
                "number_id": m.get("number_id"),
                "scope": m.get("scope"),
                "name": m.get("name"),
                "definition": m.get("definition"),
                "guidance": m.get("guidance"),
                "anchors": m.get("anchors"),
            }
            for m in metrics
        ],
    }


def build_user_prompt(
    scenario_id: str,
    transcript: str,
    rubric: Dict[str, Any],
) -> str:
    rubric_blob = rubric_to_prompt_blob(rubric)

    required_output_schema = {
        "scenario_id": "string",
        "turn_evaluations": [
            {
                "turn": "integer",
                "metrics": [
                    {
                        "metric_id": "string (exact from rubric.metrics[].metric_id)",
                        "number_id": "integer (must match rubric.metrics[].number_id for that metric_id)",
                        "score": "integer 1-5",
                        "rationale": "string",
                        "evidence": "string"
                    }
                ]
            }
        ],
        "holistic_evaluation": {
            "metrics": [
                {
                    "metric_id": "string (exact from rubric.metrics[].metric_id)",
                    "number_id": "integer (must match rubric.metrics[].number_id for that metric_id)",
                    "score": "integer 1-5",
                    "rationale": "string",
                    "evidence": "string"
                }
            ]
        }
    }

    return (
        f"SCENARIO_ID: {scenario_id}\n\n"
        "TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "RUBRIC:\n"
        f"{json.dumps(rubric_blob, ensure_ascii=False)}\n\n"
        "Return your evaluation in the following JSON schema (JSON only):\n"
        f"{json.dumps(required_output_schema, ensure_ascii=False)}"
    )


# -----------------------
# Parsing / validation
# -----------------------
def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_rubric_maps(rubric: Dict[str, Any]) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Returns:
      id_to_number: metric_id -> number_id
      turn_metric_ids: metric_ids where scope == 'turn'
      hol_metric_ids: metric_ids where scope == 'holistic'
    """
    id_to_number: Dict[str, int] = {}
    turn_ids: List[str] = []
    hol_ids: List[str] = []

    for m in rubric.get("metrics", []):
        mid = m.get("id")
        num = m.get("number_id")
        scope = m.get("scope")
        if not mid or scope not in ("turn", "holistic"):
            continue
        if not isinstance(num, int):
            # If you later change number_id to string, update this check accordingly.
            continue

        id_to_number[mid] = num
        if scope == "turn":
            turn_ids.append(mid)
        else:
            hol_ids.append(mid)

    return id_to_number, turn_ids, hol_ids


def validate_metric_entry(
    entry: Dict[str, Any],
    allowed_ids: List[str],
    id_to_number: Dict[str, int],
    context: str,
) -> Tuple[bool, str]:
    mid = entry.get("metric_id")
    num = entry.get("number_id")
    score = entry.get("score")

    if mid not in allowed_ids:
        return False, f"{context}: Unknown or wrong-scope metric_id: {mid}"

    if not isinstance(num, int):
        return False, f"{context}: number_id must be integer for metric_id {mid}"

    expected_num = id_to_number.get(mid)
    if expected_num is None:
        return False, f"{context}: metric_id {mid} missing from rubric map"
    if num != expected_num:
        return False, f"{context}: number_id mismatch for {mid} (got {num}, expected {expected_num})"

    if not isinstance(score, int) or not (1 <= score <= 5):
        return False, f"{context}: Invalid score for {mid}: {score}"

    return True, ""


def validate_judge_obj(
    obj: Dict[str, Any],
    scenario_id: str,
    expected_turns: List[int],
    rubric: Dict[str, Any],
) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Judge output is not a JSON object."

    if obj.get("scenario_id") != scenario_id:
        return False, "scenario_id mismatch in judge output."

    if "turn_evaluations" not in obj or "holistic_evaluation" not in obj:
        return False, "Missing turn_evaluations or holistic_evaluation."

    id_to_number, turn_metric_ids, hol_metric_ids = build_rubric_maps(rubric)

    te = obj.get("turn_evaluations")
    if not isinstance(te, list):
        return False, "turn_evaluations must be a list."

    got_turns: List[int] = []
    for item in te:
        if not isinstance(item, dict):
            return False, "Each turn_evaluations item must be an object."
        t = item.get("turn")
        if not isinstance(t, int):
            return False, "Each turn must be an integer."
        got_turns.append(t)

        metrics = item.get("metrics")
        if not isinstance(metrics, list):
            return False, "Each turn_evaluations.metrics must be a list."

        seen = set()
        for m in metrics:
            if not isinstance(m, dict):
                return False, f"Turn {t}: metric entry must be an object."
            ok, why = validate_metric_entry(m, turn_metric_ids, id_to_number, context=f"Turn {t}")
            if not ok:
                return False, why
            mid = m["metric_id"]
            if mid in seen:
                return False, f"Turn {t}: Duplicate metric_id {mid}"
            seen.add(mid)

        if set(seen) != set(turn_metric_ids):
            return False, f"Turn {t}: missing or extra turn metrics."

    if sorted(got_turns) != sorted(expected_turns):
        return False, "Turn numbers in judge output do not match transcript turns."

    hol = obj.get("holistic_evaluation", {})
    hol_metrics = hol.get("metrics")
    if not isinstance(hol_metrics, list):
        return False, "holistic_evaluation.metrics must be a list."

    seen_h = set()
    for m in hol_metrics:
        if not isinstance(m, dict):
            return False, "Holistic metric entry must be an object."
        ok, why = validate_metric_entry(m, hol_metric_ids, id_to_number, context="Holistic")
        if not ok:
            return False, why
        mid = m["metric_id"]
        if mid in seen_h:
            return False, f"Holistic: Duplicate metric_id {mid}"
        seen_h.add(mid)

    if set(seen_h) != set(hol_metric_ids):
        return False, "Holistic: missing or extra holistic metrics."

    return True, ""


# -----------------------
# Output writers
# -----------------------
def write_judge_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_lean_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["scenario_id", "judge_model", "turn", "metric_id", "number_id", "score"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten_to_lean_rows(
    scenario_id: str,
    judge_model: str,
    judge_obj: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for te in judge_obj["turn_evaluations"]:
        turn_n = te["turn"]
        for m in te["metrics"]:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "judge_model": judge_model,
                    "turn": turn_n,
                    "metric_id": m["metric_id"],
                    "number_id": m["number_id"],
                    "score": m["score"],
                }
            )

    for m in judge_obj["holistic_evaluation"]["metrics"]:
        rows.append(
            {
                "scenario_id": scenario_id,
                "judge_model": judge_model,
                "turn": "ALL",
                "metric_id": m["metric_id"],
                "number_id": m["number_id"],
                "score": m["score"],
            }
        )

    return rows


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--judge-model", required=True)

    ap.add_argument("--input-dir", default="../pilot_final_responses")
    ap.add_argument("--rubric", default="rubric.json")

    # One parent output dir
    ap.add_argument("--out-dir", default="../pilot_llm_evaluation")

    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff-s", type=float, default=1.0)

    # "refresh" after each conversation
    ap.add_argument("--sleep-between-scenarios", type=float, default=1.0)

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

    # Only include files starting with L + digits
    lnum_pat = re.compile(r"^L\d+.*\.json$", re.IGNORECASE)
    scenario_files = sorted(f for f in in_dir.glob("L*.json") if lnum_pat.match(f.name))

    if not scenario_files:
        print(f"No scenario files found in {in_dir} matching L<digits>*.json")
        return

    for fp in scenario_files:
        scenario = load_json(fp)
        scenario_id = scenario.get("scenario_id", fp.stem)
        turns = scenario.get("turns", [])

        if not turns:
            print(f"[SKIP] {fp.name}: no turns")
            continue

        expected_turns = [t.get("turn") for t in turns if isinstance(t.get("turn"), int)]
        transcript = render_transcript(turns)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(scenario_id, transcript, rubric)},
        ]

        print(f"Judging {scenario_id} from {fp.name} ...")

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
                ok, why = validate_judge_obj(judge_obj, scenario_id, expected_turns, rubric)
                if not ok:
                    err = why
                    judge_obj = None

        except Exception as e:
            err = repr(e)

        elapsed_s = round(time.time() - t0, 4)

        # Always save rich artifact (even on error)
        out_json_path = out_json_dir / f"{scenario_id}.judge.json"
        rich_payload = {
            "scenario_id": scenario_id,
            "judge_model": args.judge_model,
            "source_file": fp.name,
            "elapsed_s": elapsed_s,
            "error": err,
            "judge_text": judge_text,
            "judge_json": judge_obj,
        }
        write_judge_json(out_json_path, rich_payload)

        # Write lean CSV only if valid judge_json exists
        out_csv_path = out_csv_dir / f"{scenario_id}.csv"
        if judge_obj is not None:
            rows = flatten_to_lean_rows(scenario_id, args.judge_model, judge_obj)
            write_lean_csv(out_csv_path, rows)
            print(f"  ✓ wrote JSON: {out_json_path}")
            print(f"  ✓ wrote CSV : {out_csv_path}")
        else:
            # Optional: write a minimal CSV marker row on failure
            write_lean_csv(
                out_csv_path,
                [
                    {
                        "scenario_id": scenario_id,
                        "judge_model": args.judge_model,
                        "turn": "ALL",
                        "metric_id": "__ERROR__",
                        "number_id": "",
                        "score": "",
                    }
                ],
            )
            print(f"  ✗ judge failed: {err}")
            print(f"  ✓ wrote JSON: {out_json_path}")
            print(f"  ! wrote CSV (error marker): {out_csv_path}")

        if args.sleep_between_scenarios > 0:
            time.sleep(args.sleep_between_scenarios)

    print("Done.")


if __name__ == "__main__":
    main()
