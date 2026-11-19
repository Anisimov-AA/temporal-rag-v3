import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
import solution  # noqa: E402
from run_eval import load_snapshots  # noqa: E402
from retrieval.reorder_by_date import reorder_by_date as newest_effective_first  # noqa: E402
from retrieval.rerank_chunks import rerank_chunks as rerank  # noqa: E402
from retrieval.retrieve_chunks import fuse  # noqa: E402

QUESTIONS = ROOT / "eval" / "public_eval.jsonl"
SPLIT = ROOT / "dev" / "split.json"
OUT = ROOT / "results" / "search_check.json"
CANDIDATES = 20
TOPS = [3, 5, 10]
MODES = ["vectors", "vectors + rerank", "vectors + bm25", "vectors + bm25 + rerank", "rerank + newest effective"]
WINDOW = 5


def text_of(chunk):
    return chunk.metadata["url"] + "\n" + chunk.page_content


def first_rank(ranked, words):
    for n in range(1, len(ranked) + 1):
        text = "\n".join(text_of(c) for c in ranked[:n])
        if all(w in text for w in words):
            return n
    return None


def stale_first(ranked, good, bad):
    for chunk in ranked:
        text = text_of(chunk)
        if any(w in text for w in good):
            return False
        if any(w in text for w in bad):
            return True
    return False


def ranked_lists(k, question, snapshot_id):
    vectors = k.stores[snapshot_id].similarity_search(question, k=CANDIDATES)
    words = k.keywords[snapshot_id].search(question, CANDIDATES)
    fused = fuse([vectors, words])
    reranked = rerank(question, fused)
    return {"vectors": vectors, "vectors + rerank": rerank(question, vectors), "vectors + bm25": fused, "vectors + bm25 + rerank": reranked,
            "rerank + newest effective": newest_effective_first(reranked[:WINDOW]) + reranked[WINDOW:]}


def main():
    tuning = set(json.loads(SPLIT.read_text(encoding="utf-8"))["tuning"])
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    asked = [q for q in questions if q["id"] in tuning and q["must_include"]]
    k = solution.build({**solution.CONFIG, "llm": "gemini-3.5-flash-lite"})
    for snapshot in load_snapshots(ROOT / "snapshots"):
        k.ingest(snapshot)
    rows = []
    for n, q in enumerate(asked, 1):
        lists = ranked_lists(k, q["question"], k.pick(q.get("as_of")))
        row = {"id": q["id"], "question": q["question"], "category": q["category"], "modes": {}}
        for mode, ranked in lists.items():
            row["modes"][mode] = {
                "rank": first_rank(ranked, q["must_include"]),
                "stale_first": stale_first(ranked, q["must_include"], q["must_not_include"]),
            }
        rows.append(row)
        print(f"{n}/{len(asked)} {q['id']}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"questions": len(asked), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for mode in MODES:
        ranks = [r["modes"][mode]["rank"] for r in rows]
        found = " · ".join(f"top {t}: {sum(1 for x in ranks if x and x <= t)}/{len(rows)}" for t in TOPS)
        mrr = sum(1 / x for x in ranks if x) / len(rows)
        stale = sum(r["modes"][mode]["stale_first"] for r in rows)
        print(f"{mode:26} {found} · MRR {mrr:.2f} · old value first: {stale}")


if __name__ == "__main__":
    main()
