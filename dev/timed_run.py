import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution"), str(ROOT / "dev")]
import solution  # noqa: E402
import tournament  # noqa: E402
from contract import Answer  # noqa: E402
from generation.ask_the_llm import GAPS, ask_the_llm  # noqa: E402
from generation.check_the_answer import check_the_answer  # noqa: E402
from generation.write_the_prompt import FORMATS, write_the_prompt  # noqa: E402
from run_eval import load_snapshots  # noqa: E402
from scoring import score_item  # noqa: E402


def main(llm, prompt, top_k, dates, runs="1"):
    config = {**solution.CONFIG, "llm": llm, "prompt": prompt, "top_k": int(top_k), "dates": dates}
    name = f"{llm.replace(':', '_')}__{prompt}__k{top_k}__{'on' if dates != 'none' else 'off'}"
    out = ROOT / "results" / "timed" / name
    out.mkdir(parents=True, exist_ok=True)
    k = solution.build(config)
    for snapshot in load_snapshots(ROOT / "snapshots"):
        k.ingest(snapshot)
    k.answer("warm up")
    for n in range(1, int(runs) + 1):
        lines, total, weight = [], 0, 0
        for q in tournament.questions():
            t0 = time.perf_counter()
            chunks = k.search(q["question"], k.pick(q.get("as_of")))
            t1 = time.perf_counter()
            reply = ask_the_llm(k.llm, GAPS[llm], write_the_prompt(prompt, q["question"], chunks), FORMATS[prompt])
            reply = check_the_answer(q["question"], reply)
            t2 = time.perf_counter()
            ans = Answer(text=solution.DONT_KNOW, confident=False) if not reply.known else Answer(text=reply.answer)
            s = score_item(q, ans)["score"]
            total += s * q["weight"]; weight += q["weight"]
            lines.append({"id": q["id"], "reply": reply.model_dump(), "search_s": round(t1 - t0, 2), "llm_s": round(t2 - t1, 2), "total_s": round(t2 - t0, 2), "score": s})
            print(f"run {n} {q['id']} search {t1 - t0:.1f} s · llm {t2 - t1:.1f} s", flush=True)
        (out / f"run{n}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
        totals = sorted(x["total_s"] for x in lines)
        p95 = totals[int(0.95 * (len(totals) - 1))]
        print(f"run {n}: accuracy {total / weight:.3f} · search {statistics.mean(x['search_s'] for x in lines):.2f} s · "
              f"llm {statistics.mean(x['llm_s'] for x in lines):.2f} s · total mean {statistics.mean(totals):.2f} s · p95 {p95:.2f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
