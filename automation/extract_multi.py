import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "scenario" / "pilot_all_sample_messages.json"
OUTPUT_PATH = BASE_DIR.parent / "scenario" / "pilot_multi_messages.json"


def is_multi_turn(messages: list) -> bool:
    if len(messages) <= 1:
        return False
    turns = [m.get("turn", 0) for m in messages]
    return max(turns) >= 1


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    multi_only = [
        scn for scn in data
        if is_multi_turn(scn.get("messages", []))
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(multi_only, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(multi_only)} multi-turn scenarios → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
