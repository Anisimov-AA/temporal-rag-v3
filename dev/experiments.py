import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
import solution  # noqa: E402
from contract import Answer, Source  # noqa: E402
from scoring import score_item  # noqa: E402
from steps.answer import DONT_KNOW, GAPS, LLMS, PROMPTS, Reply, reply  # noqa: E402

OUT = ROOT / "results" / "experiments"
QUESTIONS = ROOT / "eval" / "public_eval.jsonl"
SEARCH_KEYS = ("clean", "split", "chunk_size", "embedding", "search", "candidates", "top_k")

BASE = {k: v for k, v in solution.CONFIG.items() if k != "llm"}
CONFIGS = {
    "01_plain": {**BASE, "prompt": "plain", "about": "the 10 chunks of search, no rules, a short analysis"},
    "02_cot": {**BASE, "prompt": "cot", "about": "step-by-step analysis before the answer, as in RAG-Challenge-2"},
}


def questions():
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def search_key(config):
    part = {k: CONFIGS[config][k] for k in SEARCH_KEYS}
    return hashlib.sha1(json.dumps(part, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def contexts(config):
    from run_eval import load_snapshots
    from steps.answer import excerpts

    k = solution.build({**CONFIGS[config], "llm": "llama3.1:8b"})
    for snapshot in load_snapshots(ROOT / "snapshots"):
        k.ingest(snapshot)
    items = []
    for q in questions():
        snapshot_id = k.pick(q.get("as_of"))
        chunks = k.search(q["question"], snapshot_id)
        items.append({"id": q["id"], "question": q["question"], "snapshot_id": snapshot_id,
                      "urls": [c.metadata["url"] for c in chunks], "excerpts": excerpts(chunks)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"contexts_{search_key(config)}.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(len(items), "contexts saved for", config)


def load_contexts(config):
    path = OUT / f"contexts_{search_key(config)}.json"
    if not path.exists():
        contexts(config)
    return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))}


def prompt(config, context):
    return PROMPTS[CONFIGS[config]["prompt"]].format(excerpts=context["excerpts"], question=context["question"])


def to_answer(r, context):
    if not r.known:
        return Answer(text=DONT_KNOW, confident=False)
    n = r.source
    url = context["urls"][n - 1] if 1 <= n <= len(context["urls"]) else context["urls"][0]
    return Answer(text=r.answer, sources=[Source(url=url, snapshot_id=context["snapshot_id"])])


def score(config, model, replies):
    ctx = load_contexts(config)
    rows = []
    for q in questions():
        r = replies.get(q["id"])
        ans = to_answer(r, ctx[q["id"]]) if r else Answer(text="", confident=False)
        s = score_item(q, ans)
        s.update(error=None if r else "no reply", answer={"text": ans.text, "confident": ans.confident,
                                                          "sources": [{"url": x.url} for x in ans.sources]})
        rows.append(s)
    folder = OUT / config
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{model}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(CONFIGS[config], indent=2), encoding="utf-8")
    ok = [r for r in rows if not r["error"]]
    acc = sum(r["score"] * r["weight"] for r in ok) / sum(r["weight"] for r in ok)
    print(f"{config} · {model}: accuracy {acc:.3f} on {len(ok)} of {len(rows)}")


def run(config, model):
    llm = LLMS[model]()
    replies = {}
    for c in load_contexts(config).values():
        try:
            replies[c["id"]] = reply(llm, GAPS[model], prompt(config, c))
        except Exception as e:
            print("failed", c["id"], type(e).__name__, str(e)[:120])
    score(config, model, replies)


def agent_prompts(config, parts=4):
    items = list(load_contexts(config).values())
    size = -(-len(items) // int(parts))
    folder = OUT / config
    folder.mkdir(parents=True, exist_ok=True)
    for part in range(int(parts)):
        blocks = [f"===== {c['id']} =====\n{prompt(config, c)}\n" for c in items[part * size:(part + 1) * size]]
        (folder / f"opus_prompts_{part + 1}.txt").write_text("\n".join(blocks), encoding="utf-8")
    print("prompts for the agents in", folder)


def score_agents(config):
    replies = {}
    for f in sorted((OUT / config).glob("opus_answers_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                replies[d["id"]] = Reply(**d)
    score(config, "claude-opus", replies)


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"contexts": contexts, "run": run, "agent_prompts": agent_prompts, "score_agents": score_agents}[command](*args)
