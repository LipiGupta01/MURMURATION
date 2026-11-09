import json
from pathlib import Path

path = Path("data/raw/pilot__adaption__gemma-4-31b-it.jsonl")

rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

for condition in "ABCDE":
    group = [
        r for r in rows
        if r["condition"] == condition and not r.get("api_error")
    ]

    if not group:
        continue

    # Inspect the positive adoption case where available.
    trial = next(
        (r for r in group if r["outcome"]["adopted"]),
        group[0]
    )

    print("\n" + "=" * 65)
    print("CONDITION:", condition)
    print("TRIAL:", trial["trial_id"])
    print("WORDING:", trial["wording"])
    print("SENDERS:", trial["senders"])
    print("OUTCOME:", trial["outcome"])

    for event in trial.get("env_snapshot", {}).get("events", []):
        print("\nSTEP:", event["step"])
        print("TOOL:", event["tool"])
        print("ARGS:", event["args"])

        if event["step"] == 1:
            print("FIRST TOOL RESULT / INBOX:")
            print(event["result"][:2500])
