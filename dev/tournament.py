import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
import solution  # noqa: E402
from contract import Answer, Source  # noqa: E402
from scoring import score_item  # noqa: E402
from steps.answer import DONT_KNOW, GAPS, LLMS, PROMPTS, VALUE, Reply, excerpts, reply, with_value  # noqa: E402

QUESTIONS = ROOT / "eval" / "public_eval.jsonl"
SPLIT = ROOT / "dev" / "split.json"
OUT = ROOT / "results" / "tournament"
CONTEXTS = OUT / "contexts.json"
CONFIG = {**solution.CONFIG, "prompt": "plain", "top_k": 5}
AGENT_PARTS = 3


def questions():
    tuning = set(json.loads(SPLIT.read_text(encoding="utf-8"))["tuning"])
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in rows if q["id"] in tuning]


def contexts():
    from run_eval import load_snapshots

    k = solution.build({**CONFIG, "llm": "llama3.1:8b"})
    for snapshot in load_snapshots(ROOT / "snapshots"):
        k.ingest(snapshot)
    items = []
    for q in questions():
        snapshot_id = k.pick(q.get("as_of"))
        chunks = k.search(q["question"], snapshot_id)
        today = k.dates[snapshot_id]
        prompt = PROMPTS[CONFIG["prompt"]].format(excerpts=excerpts(chunks), question=q["question"], today=today)
        items.append({"id": q["id"], "snapshot_id": snapshot_id, "prompt": prompt,
                      "chunks": [{"url": c.metadata["url"], "text": c.page_content} for c in chunks]})
    OUT.mkdir(parents=True, exist_ok=True)
    CONTEXTS.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), "prompts saved")


def load_contexts():
    return {c["id"]: c for c in json.loads(CONTEXTS.read_text(encoding="utf-8"))}


def run(model, runs="3", first="1"):
    llm = LLMS[model]()
    folder = OUT / model.replace(":", "_")
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(int(first), int(first) + int(runs)):
        lines = []
        for c in load_contexts().values():
            start = time.time()
            try:
                r = reply(llm, GAPS[model], c["prompt"])
                lines.append({"id": c["id"], "reply": r.model_dump(), "seconds": round(time.time() - start, 1)})
            except Exception as error:
                lines.append({"id": c["id"], "error": str(error)[:200], "seconds": round(time.time() - start, 1)})
            print(f"run {n} {c['id']} {lines[-1]['seconds']} s", flush=True)
        (folder / f"run{n}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")


def agent_prompts():
    items = list(load_contexts().values())
    size = -(-len(items) // AGENT_PARTS)
    folder = OUT / "claude-opus"
    folder.mkdir(parents=True, exist_ok=True)
    for part in range(AGENT_PARTS):
        blocks = [f"===== {c['id']} =====\n{c['prompt']}\n" for c in items[part * size:(part + 1) * size]]
        (folder / f"prompts_{part + 1}.txt").write_text("\n".join(blocks), encoding="utf-8")
    print("prompts for the agents in", folder)


def to_answer(r, context):
    if not r.known:
        return Answer(text=DONT_KNOW, confident=False)
    chunks = context["chunks"]
    used = chunks[r.source - 1] if 1 <= r.source <= len(chunks) else chunks[0]
    pages = [(c["url"], c["url"] + "\n" + c["text"]) for c in chunks]
    urls = list(dict.fromkeys([used["url"]] + with_value(r.answer, pages)))
    return Answer(text=r.answer, sources=[Source(url=u, snapshot_id=context["snapshot_id"]) for u in urls])


def load_run(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["id"]] = d
    return rows


def score():
    qs = {q["id"]: q for q in questions()}
    ctx = load_contexts()
    table = {}
    for folder in sorted(p for p in OUT.iterdir() if p.is_dir()):
        runs = [load_run(p) for p in sorted(folder.glob("run?.jsonl"))]
        if not runs:
            continue
        per_run, per_question, seconds = [], {}, []
        for rows in runs:
            total = weight = 0
            for qid, q in qs.items():
                d = rows.get(qid, {})
                ans = to_answer(Reply(**d["reply"]), ctx[qid]) if "reply" in d else Answer(text="", confident=False)
                s = score_item(q, ans)
                total += s["score"] * q["weight"]
                weight += q["weight"]
                per_question.setdefault(qid, []).append((s["score"], sorted(set(VALUE.findall(ans.text))), ans.confident))
                if d.get("seconds") is not None:
                    seconds.append(d["seconds"])
            per_run.append(total / weight)
        same = sum(1 for v in per_question.values() if len({(x[0], tuple(x[1]), x[2]) for x in v}) == 1)
        table[folder.name] = {"accuracy": sum(per_run) / len(per_run), "runs": per_run, "repeat": same / len(per_question),
                              "seconds": sum(seconds) / len(seconds) if seconds else None,
                              "questions": {qid: [x[0] for x in v] for qid, v in per_question.items()}}
    (OUT / "scores.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for name, t in table.items():
        sec = f"{t['seconds']:.1f} s" if t["seconds"] is not None else "-"
        print(f"{name:28} accuracy {t['accuracy']:.3f} ({' / '.join(f'{x:.3f}' for x in t['runs'])}) · repeat {t['repeat']:.2f} · {sec}")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"contexts": contexts, "run": run, "agent_prompts": agent_prompts, "score": score}[command](*args)
