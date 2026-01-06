import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "scenario" / "pilot_raw_all.json"
OUTPUT_PATH = BASE_DIR.parent / "scenario" / "pilot_all_sample_messages.json"


def to_messages(scn: dict) -> list:
    # If already has messages, use them (but ensure turn exists)
    if isinstance(scn.get("messages"), list) and scn["messages"]:
        msgs = []
        for i, m in enumerate(scn["messages"], start=1):
            msgs.append({
                "turn": m.get("turn", i),
                "role": m.get("role", "user"),
                "content": m.get("content", "")
            })
        return msgs

    fmt = (scn.get("format") or "").lower()

    if "single" in fmt:
        return [{
            "turn": 0,
            "role": "user",
            "content": scn["user_message"]
        }]

    if "multi" in fmt:
        return [{
            "turn": i,
            "role": "user",
            "content": t["user_message"]
        } for i, t in enumerate(scn["turns"], start=1)]

    raise ValueError(f"Unknown scenario format: {scn.get('format')}")


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]

    model_inputs = []
    for scn in scenarios:
        model_inputs.append({
            "scenario_id": scn["scenario_id"],
            "messages": to_messages(scn),
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_inputs, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(model_inputs)} model-ready inputs → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
