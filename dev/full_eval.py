import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "public_eval.jsonl"
OUT = ROOT / "eval" / "full_eval.jsonl"


def main():
    paraphrases = json.loads((ROOT / "dev" / "paraphrases.json").read_text(encoding="utf-8"))
    paraphrases.update(json.loads((ROOT / "dev" / "paraphrases_holdout.json").read_text(encoding="utf-8")))
    items = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        q["group"] = q["id"]
        items.append(q)
        for n, text in enumerate(paraphrases.get(q["id"], []), 1):
            items.append({**q, "id": f"{q['id']}-p{n}", "question": text})
    OUT.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in items) + "\n", encoding="utf-8")
    print(len(items), "questions in", OUT.name)


if __name__ == "__main__":
    main()
