import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "public_eval.jsonl"
OUT = ROOT / "dev" / "split.json"
HOLDOUT = 10


def main():
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category = defaultdict(list)
    for q in questions:
        by_category[q["category"]].append(q["id"])
    rng = random.Random(42)
    queues = []
    for category in sorted(by_category, key=lambda c: (-len(by_category[c]), c)):
        ids = sorted(by_category[category])
        rng.shuffle(ids)
        queues.append(ids)
    holdout = []
    while len(holdout) < HOLDOUT:
        for ids in queues:
            if ids and len(holdout) < HOLDOUT:
                holdout.append(ids.pop())
    holdout = sorted(holdout)
    tuning = sorted(q["id"] for q in questions if q["id"] not in holdout)
    OUT.write_text(json.dumps({"tuning": tuning, "holdout": holdout}, indent=1), encoding="utf-8")
    print(len(tuning), "tuning,", len(holdout), "holdout:", ", ".join(holdout))


if __name__ == "__main__":
    main()
