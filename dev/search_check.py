import json
import statistics
import sys
import time
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
OUT = ROOT / "results" / "search_check"
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


def ranked_lists(k, question, snapshot_id, candidates):
    vectors = k.stores[snapshot_id].similarity_search(question, k=candidates)
    words = k.keywords[snapshot_id].search(question, candidates)
    fused = fuse([vectors, words])
    reranked = rerank(question, fused)
    return {"vectors": vectors, "vectors + rerank": rerank(question, vectors), "vectors + bm25": fused, "vectors + bm25 + rerank": reranked,
            "rerank + newest effective": newest_effective_first(reranked[:WINDOW]) + reranked[WINDOW:]}


def check(k, asked, candidates):
    k.config["candidates"] = candidates
    rows = []
    for q in asked:
        snapshot_id = k.pick(q.get("as_of"))
        lists = ranked_lists(k, q["question"], snapshot_id, candidates)
        t0 = time.perf_counter()
        chunks = k.search(q["question"], snapshot_id)
        seconds = time.perf_counter() - t0
        row = {"id": q["id"], "question": q["question"], "category": q["category"], "modes": {},
               "prompt_chars": sum(len(c.page_content) for c in chunks), "search_seconds": round(seconds, 3)}
        for mode, ranked in lists.items():
            row["modes"][mode] = {
                "rank": first_rank(ranked, q["must_include"]),
                "stale_first": stale_first(ranked, q["must_include"], q["must_not_include"]),
            }
        rows.append(row)
    return rows


def summary(rows, mode):
    ranks = [r["modes"][mode]["rank"] for r in rows]
    found = " · ".join(f"top {t}: {sum(1 for x in ranks if x and x <= t)}/{len(rows)}" for t in TOPS)
    mrr = sum(1 / x for x in ranks if x) / len(rows)
    stale = sum(r["modes"][mode]["stale_first"] for r in rows)
    return f"{found} · MRR {mrr:.2f} · old value first: {stale}"


def main(caps="4000", candidates="20"):
    tuning = set(json.loads(SPLIT.read_text(encoding="utf-8"))["tuning"])
    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    asked = [q for q in questions if q["id"] in tuning and q["must_include"]]
    OUT.mkdir(parents=True, exist_ok=True)
    for cap in map(int, caps.split(",")):
        k = solution.build({**solution.CONFIG, "chunk_size": cap})
        for snapshot in load_snapshots(ROOT / "snapshots"):
            k.ingest(snapshot)
        k.search("warm up", k.pick(None))
        for n in map(int, candidates.split(",")):
            rows = check(k, asked, n)
            name = f"cap{cap}_c{n}"
            (OUT / f"{name}.json").write_text(json.dumps({"questions": len(asked), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\n{name}")
            for mode in MODES:
                print(f"  {mode:26} {summary(rows, mode)}")
            print(f"  prompt chars {statistics.mean(r['prompt_chars'] for r in rows):.0f} · search {statistics.mean(r['search_seconds'] for r in rows):.2f} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
