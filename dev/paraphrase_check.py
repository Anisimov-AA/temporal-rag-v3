import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution"), str(ROOT / "dev")]
import solution  # noqa: E402
import tournament  # noqa: E402
from run_eval import load_snapshots  # noqa: E402
from scoring import _numbers, score_item  # noqa: E402

PARAPHRASES = json.loads((ROOT / "dev" / "paraphrases.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "paraphrases"


def main(run="1"):
    k = solution.build(solution.CONFIG)
    for snapshot in load_snapshots(ROOT / "snapshots"):
        k.ingest(snapshot)
    k.answer("warm up")
    rows = []
    for q in tournament.questions():
        for n, text in enumerate([q["question"]] + PARAPHRASES[q["id"]]):
            start = time.perf_counter()
            answer = k.answer(text, q.get("as_of"))
            s = score_item(q, answer)
            rows.append({"id": q["id"], "n": n, "question": text, "text": answer.text, "confident": answer.confident,
                         "sources": [x.url for x in answer.sources], "correctness": s["correctness"], "score": s["score"],
                         "seconds": round(time.perf_counter() - start, 2)})
            print(q["id"], n, round(s["score"], 2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"run{run}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    groups = {}
    for r in rows:
        groups.setdefault(r["id"], []).append(r)
    ok = 0
    for qid, members in groups.items():
        all_right = all(m["correctness"] >= 0.999 for m in members)
        same = len({frozenset(_numbers(m["text"])) for m in members}) == 1
        ok += all_right or same
        if not (all_right or same):
            print("differs:", qid, [round(m["score"], 2) for m in members])
    print(f"paraphrase consistency {ok}/{len(groups)} = {ok / len(groups):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
