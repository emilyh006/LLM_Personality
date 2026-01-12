#!/usr/bin/env python3

import argparse
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
) -> Tuple[str, Dict[str, Any]]:
    """
    Send a chat completion request to an OpenAI-compatible endpoint.
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


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def count_words(text: str) -> int:
    """Simple whitespace word count (validity-neutral)."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def main() -> None:
    """
    Execute multi-turn chat scenarios against an OpenAI-compatible chat completion API.

    For each scenario, the function maintains a running conversation
    history, appending both user messages and assistant responses so
    that each turn is evaluated with full prior context. Outputs are
    saved as one JSON file per scenario, including timing and error
    information.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="../pilot_response_word_lim")

    # NOTE: keep this high enough so the model can finish a <=400-word answer.
    # You're measuring word-constraint adaptation, not server-side truncation.
    ap.add_argument("--max-tokens", type=int, default=600)

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff-s", type=float, default=1.0)
    ap.add_argument("--sleep-s", type=float, default=0.25)
    args = ap.parse_args()

    scenarios = load_scenarios(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    for scn in scenarios:
        scenario_id = scn.get("scenario_id", "unknown")
        messages_in = scn.get("messages", [])
        if not messages_in:
            continue

        print(f"Running {scenario_id} ({len(messages_in)} turns)")

        # Add the response constraint ONCE per scenario (applies to all turns)
        convo: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Your entire response must be no more than 400 words. "
                    "Do not exceed this limit."
                ),
            }
        ]

        turns_out: List[Dict[str, Any]] = []

        for m in messages_in:
            turn = m.get("turn")
            role = m.get("role", "user")
            content = m.get("content", "")

            # add user turn
            convo.append({"role": role, "content": content})

            t0 = time.time()
            try:
                assistant_text, _raw = post_chat(
                    base_url=args.base_url,
                    model=args.model,
                    messages=convo,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                    retries=args.retries,
                    backoff_s=args.backoff_s,
                )
                err = None
            except Exception as e:
                assistant_text = ""
                err = repr(e)

            elapsed = round(time.time() - t0, 4)

            # Count words per turn 
            assistant_word_count = count_words(assistant_text)
            over_400_words = assistant_word_count > 400

            # add assistant reply for context
            convo.append({"role": "assistant", "content": assistant_text})

            turns_out.append(
                {
                    "turn": turn,
                    "user_content": content,
                    "assistant_content": assistant_text,
                    "assistant_word_count": assistant_word_count,
                    "over_400_words": over_400_words,
                    "elapsed_s": elapsed,
                    "error": err,
                }
            )

        out_path = os.path.join(args.out_dir, sanitize_filename(scenario_id) + ".json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario_id": scenario_id,
                    "model": args.model,
                    "base_url": args.base_url,
                    "response_constraint": {"max_words": 400},
                    "turns": turns_out,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    print("Done.")


if __name__ == "__main__":
    main()
