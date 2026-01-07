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
        # parameters 
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

    Calls POST {base_url}/v1/chat/completions with the provided conversation history.
    Applies per-request timeouts and retry logic with exponential backoff to handle
    transient failures.

    Parameters
    ----------
    base_url : str
        Base URL of the inference server.
    model : str
        Model identifier expected by the server.
    messages : List[Dict[str, str]]
        Conversation history in OpenAI chat format.
    max_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature (lower values are more deterministic).
    timeout_s : float
        HTTP request timeout in seconds.
    retries : int
        Number of retry attempts on transient errors.
    backoff_s : float
        Base delay for exponential backoff between retries.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Assistant text and the full raw response JSON.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    # Build the full endpoint URL safely (avoid double slashes
    url = base_url.rstrip("/") + "/v1/chat/completions"
    
    # Payload follows OpenAI-compatible schema
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Store last error so we can report a useful final message
    last_err: Optional[Exception] = None
    
    # Attempt initial request + retries (total = retries + 1)
    for attempt in range(retries + 1):
        try:
             # Send request (timeout prevents hanging forever)
            r = requests.post(url, json=payload, timeout=timeout_s)
             # Raise an exception if status code is not 2xx
            r.raise_for_status()

            # Parse JSON body
            data = r.json()
             # Extract assistant message text (OpenAI-style response)
            text = data["choices"][0]["message"]["content"] or ""
            return str(text), data
        
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_s * (2 ** attempt))
            else:
                break

    raise RuntimeError(f"Request failed after retries. Last error: {last_err!r}")


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


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
    ap.add_argument("--out-dir", default="../pilot_response")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff-s", type=float, default=1.0)
    ap.add_argument("--sleep-s", type=float, default=0.25)
    args = ap.parse_args()


    # Load scenarios and prepare output directory
    scenarios = load_scenarios(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    # Loop over scenarios
    for scn in scenarios:
        scenario_id = scn.get("scenario_id", "unknown")
        messages_in = scn.get("messages", [])
        if not messages_in:
            continue

        print(f"Running {scenario_id} ({len(messages_in)} turns)")

        # Fresh conversation per scenario
        convo: List[Dict[str, str]] = []

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

            # add assistant reply for context
            convo.append({"role": "assistant", "content": assistant_text})

            turns_out.append({
                "turn": turn,
                "user_content": content,
                "assistant_content": assistant_text,
                "elapsed_s": elapsed,
                "error": err,
            })

        out_path = os.path.join(
            args.out_dir, sanitize_filename(scenario_id) + ".json"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario_id": scenario_id,
                    "model": args.model,
                    "base_url": args.base_url,
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
