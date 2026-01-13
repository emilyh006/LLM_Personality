#!/usr/bin/env python3
"""
Convert scenario JSON files into readable transcripts for Google Forms / human rating.
Outputs:
- out_txt/<scenario_id>.txt   (one per scenario)
- out_txt/ALL_SCENARIOS.txt   (everything combined)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

# -------------------------
# Markdown cleanup patterns
# -------------------------
_RE_HEADING = re.compile(r"^\s*(#{1,6})\s*(.+?)\s*$", re.MULTILINE)
_RE_RULE = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
_RE_BULLET = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)
_RE_BQ = re.compile(r"^\s*>\s*", re.MULTILINE)
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_RE_CODE = re.compile(r"`([^`]+)`")


def clean_markdown(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Horizontal rules -> divider
    t = _RE_RULE.sub("\n––––––––––––––––––––\n", t)

    # Headings -> plain text headings
    def heading_repl(m: re.Match) -> str:
        level = len(m.group(1))
        title = m.group(2).strip()
        return f"\n{title.upper()}\n" if level <= 3 else f"\n{title}\n"

    t = _RE_HEADING.sub(heading_repl, t)

    # Blockquotes
    t = _RE_BQ.sub("", t)
    t = re.sub(r"^\s*⚠️\s*", "NOTE: ", t, flags=re.MULTILINE)

    # Inline formatting
    t = _RE_BOLD.sub(r"\1", t)
    t = _RE_ITALIC.sub(r"\1", t)
    t = _RE_CODE.sub(r"\1", t)

    # Bullets -> •
    t = _RE_BULLET.sub(lambda m: f"{m.group(1)}• ", t)

    # Ensure blank line after big headings/dividers
    t = re.sub(r"(\n[A-Z0-9 ,:()\"'✅🔥⚠️🚨📱-]{6,}\n)(?!\n)", r"\1\n", t)

    # Trim trailing whitespace per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))

    # Collapse excessive blank lines
    t = re.sub(r"\n{4,}", "\n\n\n", t).strip()

    return t


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:120] if name else "unknown"


def _fmt_val(v: Any) -> str:
    """Nice printable value (especially for None/booleans)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def format_scenario(scn: Dict[str, Any], include_raw_json: bool = True) -> str:
    scenario_id = scn.get("scenario_id", "unknown_scenario")
    model = scn.get("model", "unknown_model")
    base_url = scn.get("base_url", "")
    response_constraint = scn.get("response_constraint", {})
    turns = scn.get("turns", [])

    max_words = ""
    if isinstance(response_constraint, dict) and "max_words" in response_constraint:
        max_words = response_constraint.get("max_words")

    lines: List[str] = [
        f"Conversation: {scenario_id}",
        f"Model: {model}",
    ]

    if base_url:
        lines.append(f"Base URL: {base_url}")
    if max_words != "":
        lines.append(f"Response Constraint: max_words={_fmt_val(max_words)}")
    elif response_constraint:
        # in case it has other fields
        lines.append(f"Response Constraint: {json.dumps(response_constraint, ensure_ascii=False)}")

    lines += [
        "",
        "––––––––––––––––––––",
        "",
    ]

    for t in turns:
        turn_no = t.get("turn", "")

        user_text = clean_markdown((t.get("user_content") or "").strip())
        asst_text = clean_markdown((t.get("assistant_content") or "").strip())

        lines.append(f"Turn {turn_no} — User")
        lines.append("")
        lines.append(user_text)
        lines.append("")

        lines.append(f"Turn {turn_no} — Assistant")
        lines.append("")
        lines.append(asst_text)
        lines.append("")

        # Turn-level metadata (include everything except the big texts)
        meta_keys = [
            "assistant_word_count",
            "over_400_words",
            "max_words",
            "length_retry_count",
            "api_retry_count",
            "elapsed_s",
            "error",
        ]
        meta_present = [k for k in meta_keys if k in t]
        if meta_present:
            lines.append("Turn Metadata")
            for k in meta_present:
                lines.append(f"- {k}: {_fmt_val(t.get(k))}")
            lines.append("")

        lines.append("––––––––––––––––––––")
        lines.append("")

    if include_raw_json:
        lines.append("RAW JSON (Full Record)")
        lines.append("")
        lines.append(json.dumps(scn, ensure_ascii=False, indent=2))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def coerce_scenarios(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
    - dict = one scenario
    - list = list of scenarios
    """
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        # best effort: keep only dict items
        return [x for x in obj if isinstance(x, dict)]
    raise ValueError("Input JSON must be a scenario dict or a list of scenario dicts.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="", help="Optional: single JSON file to convert.")
    ap.add_argument(
        "--input-glob",
        type=str,
        default="L*_multi.json",
        help='Glob pattern (relative to script dir) when --input is not provided. Default: "L*_multi.json"',
    )
    ap.add_argument("--out-dir", type=str, default="out_txt", help="Output directory. Default: out_txt")
    ap.add_argument(
        "--no-raw-json",
        action="store_true",
        help="Do not append RAW JSON at the end of each transcript.",
    )
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    include_raw = not args.no_raw_json

    # Collect input paths
    paths: List[Path] = []
    if args.input:
        p = (base_dir / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
        if not p.exists():
            raise SystemExit(f"❌ Input file not found: {p}")
        paths = [p]
    else:
        paths = sorted(base_dir.glob(args.input_glob))
        if not paths:
            raise SystemExit(f'❌ No JSON files matched glob: "{args.input_glob}" in {base_dir}')

    all_blocks: List[str] = []
    scenario_count = 0

    for path in paths:
        obj = load_json(path)
        scenarios = coerce_scenarios(obj)

        for scn in scenarios:
            scenario_count += 1
            text = format_scenario(scn, include_raw_json=include_raw)

            scenario_id = scn.get("scenario_id", path.stem)
            fname = safe_filename(str(scenario_id)) + ".txt"
            (out_dir / fname).write_text(text, encoding="utf-8")

            all_blocks.append(text)

    # Combined output
    combined = "\n\n" + ("=" * 64) + "\n\n"
    all_text = combined.join(all_blocks).strip() + "\n"
    (out_dir / "ALL_SCENARIOS.txt").write_text(all_text, encoding="utf-8")

    print(f"✅ Converted {scenario_count} scenario(s) from {len(paths)} file(s).")
    print(f"✅ Output folder: {out_dir.resolve()}")
    print(f"✅ Combined transcript: {out_dir.resolve() / 'ALL_SCENARIOS.txt'}")


if __name__ == "__main__":
    main()
