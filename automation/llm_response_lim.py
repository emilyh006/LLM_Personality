#!/usr/bin/env python3

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError("Input JSON must be a list.")


def post_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> Tuple[str, Dict[str, Any], int]:
    """
    Send a chat completion request to an OpenAI-compatible endpoint.

    Returns:
      assistant_text, raw_json, api_retry_count_used
    """
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
            # attempt = 0 means no API retry; attempt=1 means retried once, etc.
            return str(text), data, attempt

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


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def count_words(text: str) -> int:
    """Simple whitespace word count (validity-neutral)."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="../pilot_final_responses")

    # word-limit behavior
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--length-retries", type=int, default=1)  # retry if > max-words

    # NOTE: keep this high enough so the model can finish a <=400-word answer.
    ap.add_argument("--max-tokens", type=int, default=700)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=2)        # API/network retries
    ap.add_argument("--backoff-s", type=float, default=1.0)
    ap.add_argument("--sleep-s", type=float, default=0.25)

    # summary outputs
    ap.add_argument("--summary-csv", default="wordcount_summary.csv")
    

    args = ap.parse_args()

    scenarios = load_scenarios(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_csv_path = os.path.join(args.out_dir, args.summary_csv)
   

    # Create summary writers once
    csv_fieldnames = [
        "scenario_id",
        "turn",
        "assistant_word_count",
        "over_max_words",
        "length_retry_count",
        "api_retry_count",
        "elapsed_s",
        "error",
    ]

    # Write CSV header (overwrite each run)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fieldnames)
        writer.writeheader()


    for scn in scenarios:
        scenario_id = scn.get("scenario_id", "unknown")
        messages_in = scn.get("messages", [])
        if not messages_in:
            continue

        print(f"Running {scenario_id} ({len(messages_in)} turns)")

        # Keep your current prompt (no overkill)
        base_system_prompt = (
            "You are a helpful assistant. "
            f"Your entire response must be no more than {args.max_words} words. "
            "Do not exceed this limit."
        )

        convo: List[Dict[str, str]] = [
            {"role": "system", "content": base_system_prompt}
        ]

        turns_out: List[Dict[str, Any]] = []

        for m in messages_in:
            turn = m.get("turn")
            role = m.get("role", "user")
            content = m.get("content", "")

            # add user turn
            convo.append({"role": role, "content": content})

            t0 = time.time()
            err = None

            assistant_text = ""
            assistant_word_count = 0
            over_max_words = False

            api_retry_count = 0
            length_retry_count = 0

            # Try once + retry if over-length (discarding the too-long draft)
            while True:
                try:
                    # For length retries, add a *temporary* reminder message only for that call
                    if length_retry_count == 0:
                        messages_for_call = convo
                    else:
                        messages_for_call = convo + [{
                            "role": "system",
                            "content": (
                                f"Reminder: Your response must be no more than {args.max_words} words. "
                                "Rewrite your response to the most recent user message under this limit."
                            ),
                        }]

                    assistant_text, _raw, api_retry_used = post_chat(
                        base_url=args.base_url,
                        model=args.model,
                        messages=messages_for_call,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout_s=args.timeout_s,
                        retries=args.retries,
                        backoff_s=args.backoff_s,
                    )

                    # record the *max* API retries used across attempts
                    api_retry_count = max(api_retry_count, api_retry_used)

                    assistant_word_count = count_words(assistant_text)
                    over_max_words = assistant_word_count > args.max_words

                    # If it's compliant, or we've exhausted length-retries, accept it
                    if (not over_max_words) or (length_retry_count >= args.length_retries):
                        break

                    # otherwise: retry due to length (and do NOT append the overlong answer)
                    length_retry_count += 1
                    continue

                except Exception as e:
                    assistant_text = ""
                    assistant_word_count = 0
                    over_max_words = False
                    err = repr(e)
                    break

            elapsed = round(time.time() - t0, 4)

            # add assistant reply for context (ONLY final accepted text)
            convo.append({"role": "assistant", "content": assistant_text})

            turn_record = {
                "turn": turn,
                "user_content": content,
                "assistant_content": assistant_text,
                "assistant_word_count": assistant_word_count,
                "over_400_words": over_max_words,  # keep your existing key for continuity
                "max_words": args.max_words,
                "length_retry_count": length_retry_count,
                "api_retry_count": api_retry_count,
                "elapsed_s": elapsed,
                "error": err,
            }
            turns_out.append(turn_record)

            # Append to summary CSV + JSONL
            summary_row = {
                "scenario_id": scenario_id,
                "turn": turn,
                "assistant_word_count": assistant_word_count,
                "over_max_words": int(over_max_words),
                "length_retry_count": length_retry_count,
                "api_retry_count": api_retry_count,
                "elapsed_s": elapsed,
                "error": err,
            }

            with open(summary_csv_path, "a", newline="", encoding="utf-8") as csv_f:
                writer = csv.DictWriter(csv_f, fieldnames=csv_fieldnames)
                writer.writerow(summary_row)

        out_path = os.path.join(args.out_dir, sanitize_filename(scenario_id) + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario_id": scenario_id,
                    "model": args.model,
                    "base_url": args.base_url,
                    "response_constraint": {"max_words": args.max_words},
                    "turns": turns_out,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    print("Done.")
    print(f"Summary CSV:   {summary_csv_path}")


if __name__ == "__main__":
    main()
