#!/usr/bin/env python3
"""
Convert scenario JSON files into readable transcripts for Google Forms / human rating.

Outputs (default .md):
- out_txt/<scenario_id>.md
- out_txt/ALL_SCENARIOS.md

This version:
- Removes scenario-level metadata (model/base_url/constraints)
- Removes turn-level metadata (word counts, retry counts, etc.)
- Does NOT append RAW JSON
- Preserves markdown formatting (**bold**, *italic*, `code`, headings, bullets, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    """Preserve markdown styling; only normalize newlines and trim trailing whitespace."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    return t.strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:120] if name else "unknown"


def coerce_scenarios(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
    - dict = one scenario
    - list = list of scenarios
    """
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    raise ValueError("Input JSON must be a scenario dict or a list of scenario dicts.")


def format_scenario(scn: Dict[str, Any]) -> str:
    """Output only transcript content (turn labels + user/assistant text)."""
    scenario_id = scn.get("scenario_id", "unknown_scenario")
    turns = scn.get("turns", [])

    lines: List[str] = [f"# Conversation: {scenario_id}", ""]

    divider = "—" * 28

    for t in turns:
        turn_no = t.get("turn", "")

        user_text = normalize_text((t.get("user_content") or "").strip())
        asst_text = normalize_text((t.get("assistant_content") or "").strip())

        lines.append(f"## Turn {turn_no} — User")
        lines.append("")
        lines.append(user_text)
        lines.append("")
        lines.append(f"## Turn {turn_no} — Assistant")
        lines.append("")
        lines.append(asst_text)
        lines.append("")
        lines.append(divider)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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
        "--ext",
        choices=["md", "txt"],
        default="md",
        help="Output extension/format. Default: md",
    )
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            text = format_scenario(scn)

            scenario_id = scn.get("scenario_id", path.stem)
            fname = safe_filename(str(scenario_id)) + f".{args.ext}"
            (out_dir / fname).write_text(text, encoding="utf-8")

            all_blocks.append(text)

    # Combined output
    combined_sep = "\n\n" + ("=" * 64) + "\n\n"
    all_text = combined_sep.join(all_blocks).strip() + "\n"
    (out_dir / f"ALL_SCENARIOS.{args.ext}").write_text(all_text, encoding="utf-8")

    print(f"✅ Converted {scenario_count} scenario(s) from {len(paths)} file(s).")
    print(f"✅ Output folder: {out_dir.resolve()}")
    print(f"✅ Combined transcript: {(out_dir / f'ALL_SCENARIOS.{args.ext}').resolve()}")


if __name__ == "__main__":
    main()
