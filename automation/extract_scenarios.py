import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "scenario" / "pilot_all_sample_messages.json"

SINGLE_OUT = BASE_DIR.parent / "scenario" / "pilot_single_messages.json"
MULTI_OUT = BASE_DIR.parent / "scenario" / "pilot_multi_messages.json"


def is_multi_turn(messages: list) -> bool:
    if len(messages) <= 1:
        return False
    turns = [m.get("turn", 0) for m in messages]
    return max(turns) >= 1


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    single_turn = []
    multi_turn = []

    for scn in data:
        messages = scn.get("messages", [])
        if is_multi_turn(messages):
            multi_turn.append(scn)
        else:
            single_turn.append(scn)

    with open(SINGLE_OUT, "w", encoding="utf-8") as f:
        json.dump(single_turn, f, ensure_ascii=False, indent=2)

    with open(MULTI_OUT, "w", encoding="utf-8") as f:
        json.dump(multi_turn, f, ensure_ascii=False, indent=2)

    print(f"✅ Single-turn scenarios: {len(single_turn)} → {SINGLE_OUT}")
    print(f"✅ Multi-turn scenarios:  {len(multi_turn)} → {MULTI_OUT}")


if __name__ == "__main__":
    main()
