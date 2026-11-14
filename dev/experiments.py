import hashlib
import time
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
    "02_chain_of_thought": {**BASE, "base": "01_plain", "prompt": "chain_of_thought", "about": "chain of thought: think step by step before the answer, as in RAG-Challenge-2"},
    "03_site_words": {**BASE, "base": "01_plain", "prompt": "site_words", "about": "plain + one rule: numbers, counts and names in the words of the site"},
    "04_closest": {**BASE, "base": "01_plain", "prompt": "closest", "about": "plain + no exact answer → the closest one, with where and when it is from"},
    "04b_closest_strict": {**BASE, "base": "04_closest", "prompt": "closest_strict", "about": "04 + known = false when no excerpt gives the value asked for"},
    "05_looking_for": {**BASE, "base": "04b_closest_strict", "prompt": "looking_for", "about": "04b + first write what the member wants, known only if an excerpt gives it"},
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


def score(config, model, replies, seconds=None):
    ctx = load_contexts(config)
    seconds = seconds or {}
    rows = []
    for q in questions():
        r = replies.get(q["id"])
        ans = to_answer(r, ctx[q["id"]]) if r else Answer(text="", confident=False)
        s = score_item(q, ans)
        s.update(error=None if r else "no reply", latency_s=seconds.get(q["id"]),
                 answer={"text": ans.text, "confident": ans.confident, "sources": [{"url": x.url} for x in ans.sources]})
        rows.append(s)
    folder = OUT / config
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{file_name(model)}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(CONFIGS[config], indent=2), encoding="utf-8")
    ok = [r for r in rows if not r["error"]]
    acc = sum(r["score"] * r["weight"] for r in ok) / sum(r["weight"] for r in ok)
    times = sorted(t for t in seconds.values() if t is not None)
    speed = f" · {sum(times) / len(times):.1f} s a question, p95 {times[int(0.95 * len(times))]:.1f} s" if times else ""
    print(f"{config} · {model}: accuracy {acc:.3f} on {len(ok)} of {len(rows)}{speed}")


def file_name(model):
    return model.replace(":", "_")


def run(config, model, limit=None):
    llm = LLMS[model]()
    folder = OUT / config
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"{file_name(model)}.replies.jsonl"
    replies, seconds = {}, {}
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            replies[d["id"]], seconds[d["id"]] = Reply(**d["reply"]), d["seconds"]
    items = list(load_contexts(config).values())[: int(limit) if limit else None]
    for n, c in enumerate(items, 1):
        if c["id"] in replies:
            continue
        start = time.time()
        try:
            replies[c["id"]] = reply(llm, GAPS[model], prompt(config, c))
            seconds[c["id"]] = round(time.time() - start, 1)
            with raw.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": c["id"], "seconds": seconds[c["id"]], "reply": replies[c["id"]].model_dump()}) + "\n")
            print(f"{n}/{len(items)} {c['id']} {seconds[c['id']]} s", flush=True)
        except Exception as e:
            print(f"{n}/{len(items)} {c['id']} failed {type(e).__name__}: {str(e)[:100]}", flush=True)
    score(config, model, replies, seconds)


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
