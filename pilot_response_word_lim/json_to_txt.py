#!/usr/bin/env python3
"""
Batch convert scenario JSON files into Google-Forms-readable transcripts.
Includes BOTH user and assistant messages.

Run from the folder containing:
  L10_angry_multi.json
  L1_neutral_multi.json
  ...
  json_to_txt.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import re

# Markdown Cleanup
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

    # 1) Convert horizontal rules to dividers (keep spacing)
    t = _RE_RULE.sub("\n––––––––––––––––––––\n", t)

    # 2) Convert headings to plain text headings with spacing
    def heading_repl(m: re.Match) -> str:
        level = len(m.group(1))
        title = m.group(2).strip()
        if level <= 3:
            return f"\n{title.upper()}\n"
        else:
            return f"\n{title}\n"

    t = _RE_HEADING.sub(heading_repl, t)

    # 3) Blockquotes: "> ⚠️ ..." -> "NOTE: ..."
    # Keep them visually separated
    t = _RE_BQ.sub("", t)
    t = re.sub(r"^\s*⚠️\s*", "NOTE: ", t, flags=re.MULTILINE)

    # 4) Inline formatting: keep text, remove markers
    t = _RE_BOLD.sub(r"\1", t)
    t = _RE_ITALIC.sub(r"\1", t)
    t = _RE_CODE.sub(r"\1", t)

    # 5) Bullets (preserve indentation/nesting)
    # "  - item" -> "  • item"
    t = _RE_BULLET.sub(lambda m: f"{m.group(1)}• ", t)


    # 6) Ensure blank line AFTER headings/dividers if not already
    t = re.sub(r"(\n[A-Z0-9 ,:()\"'✅🔥⚠️🚨📱-]{6,}\n)(?!\n)", r"\1\n", t)

    # 7) Trim trailing spaces each line
    t = "\n".join(line.rstrip() for line in t.split("\n"))

    # 8) Collapse *excessive* blank lines but keep paragraph breaks
    t = re.sub(r"\n{4,}", "\n\n\n", t).strip()

    return t



def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_scenario(scn: Dict[str, Any]) -> str:
    scenario_id = scn.get("scenario_id", "unknown_scenario")
    model = scn.get("model", "unknown_model")
    turns = scn.get("turns", [])

    lines: List[str] = [
        f"Conversation: {scenario_id}",
        f"Model: {model}",
        "",
        "––––––––––––––––––––",
        ""
    ]

    for t in turns:
        turn_no = t.get("turn", "")

        user_text = clean_markdown(t.get("user_content", "").strip())
        asst_text = clean_markdown(t.get("assistant_content", "").strip())

        lines.append(f"Turn {turn_no} — User")
        lines.append("")
        lines.append(user_text)
        lines.append("")

        lines.append(f"Turn {turn_no} — Assistant")
        lines.append("")
        lines.append(asst_text)
        lines.append("")
        lines.append("––––––––––––––––––––")
        lines.append("")

    return "\n".join(lines).rstrip()


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:120]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "out_txt"
    out_dir.mkdir(exist_ok=True)

    json_files = sorted(base_dir.glob("L*_multi.json"))
    if not json_files:
        raise SystemExit("❌ No L*_multi.json files found.")

    for jf in json_files:
        scn = load_json(jf)
        text = format_scenario(scn)

        fname = safe_filename(scn.get("scenario_id", jf.stem)) + ".txt"
        (out_dir / fname).write_text(text, encoding="utf-8")

    print(f"Converted {len(json_files)} files.")
    print(f"Output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
