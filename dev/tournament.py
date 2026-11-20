import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
import solution  # noqa: E402
from contract import Answer  # noqa: E402
from scoring import score_item  # noqa: E402
from generation.ask_the_llm import GAPS, LLMS, Reply, parse_reply, reply_text  # noqa: E402
from generation.write_the_prompt import FORMATS, write_the_prompt  # noqa: E402
from solution import DONT_KNOW  # noqa: E402

QUESTIONS = ROOT / "eval" / "public_eval.jsonl"
SPLIT = ROOT / "dev" / "split.json"
VARIANTS = {"step0json": {"prompt": "plain"}, "step0known": {"prompt": "plain_known"}, "step0reason": {"prompt": "plain_reason"}, "step0text": {"prompt": "plain_text"}, "step2": {}}
for _p in ("plain", "plain_text", "plain_known", "plain_reason"):
    for _k in (3, 5, 10, 15):
        for _d, _rule in (("on", "newest effective first"), ("off", "none")):
            VARIANTS[f"s2_{_p}_k{_k}_{_d}"] = {"prompt": _p, "top_k": _k, "dates": _rule}
KNOWN_LINE = 'If the excerpts do not answer the question, set "known" to false.\n'
RULES = {"A": 'If the excerpts do not state the exact value asked for, set "known" to false.',
         "B": "If excerpts disagree, use the newest one and do not mention older values.",
         "C": "Include the exact figures from the excerpts: rates, times, amounts.",
         "D": "Answer in one or two clear sentences, with the value first.",
         "E": "Use the value whose product, term and tier match the question exactly.",
         "F": 'Examples:\nQuestion: What is the late fee on a personal loan? -> {"answer": "The late fee is $25.", "known": true}\n'
              'Question: Do you offer pet insurance? (the excerpts never mention it) -> {"answer": "", "known": false}'}
QUESTION = re.compile(r"\nMember question: [^\n]*\n")
ROLE_END = "You answer only from the website excerpts you are given.\n"
TEXT_LINE = "If the excerpts do not answer the question, reply exactly: I don't know.\n"
STEP4 = {"s4_": ("plain_known", "step0known", KNOWN_LINE), "s4b_": ("plain", "step0json", ROLE_END), "s4o_": ("plain_text", "step0text", TEXT_LINE)}
VARIANT = os.getenv("VARIANT", "step2")
for _prefix, (_prompt, _, _) in STEP4.items():
    if VARIANT.startswith(_prefix):
        VARIANTS[VARIANT] = {"prompt": _prompt}
SETTINGS = {"s6_base": {}, "s6_cap150": {"num_predict": 150}, "s6_think": {"reasoning": True},
            "s6b_base": {}, "s6b_cap150": {"max_tokens": 150},
            "s6b_think": {"max_tokens": 4096, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
            "t_k10_on_ctx16": {"num_ctx": 16384}, "t_neighbors": {"num_ctx": 12288}}
THINK = re.compile(r"<think>.*?</think>", re.S)
if VARIANT.startswith("f_"):
    VARIANTS[VARIANT] = {"prompt": VARIANT[2:]}
if VARIANT.startswith("o_json"):
    VARIANTS[VARIANT] = {"prompt": "json"}
if VARIANT.startswith("t_"):
    VARIANTS[VARIANT] = {"prompt": "json_known"}
if VARIANT == "s9_clear":
    VARIANTS[VARIANT] = {"prompt": "json_known_pages"}
if VARIANT == "s8_versions":
    VARIANTS[VARIANT] = {"prompt": "json_known_pages", "old_versions": "drop"}
if VARIANT in SETTINGS or VARIANT == "s7_pages":
    VARIANTS[VARIANT] = {"prompt": "json" if VARIANT.startswith("s6b_") else "json_known"}
OUT = ROOT / "results" / "tournament" / "variants" / VARIANT
CONTEXTS = OUT / "contexts.json"
CONFIG = {**solution.CONFIG, **VARIANTS[VARIANT]}
SKIP = {"speed"}
AGENT_PARTS = 3
VALUE = re.compile(r"\d[\d,]*(?:\.\d+)?%?|\d{1,2}:\d{2}")


def questions():
    tuning = set(json.loads(SPLIT.read_text(encoding="utf-8"))["tuning"])
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in rows if q["id"] in tuning]


def contexts():
    from run_eval import load_snapshots

    k = solution.build(CONFIG)
    for snapshot in load_snapshots(ROOT / "snapshots"):
        k.ingest(snapshot)
    items = []
    for q in questions():
        snapshot_id = k.pick(q.get("as_of"))
        chunks = k.search(q["question"], snapshot_id)
        prompt = write_the_prompt(CONFIG["prompt"], q["question"], chunks)
        items.append({"id": q["id"], "snapshot_id": snapshot_id, "prompt": prompt,
                      "chunks": [{"url": c.metadata["url"], "text": c.page_content} for c in chunks]})
    OUT.mkdir(parents=True, exist_ok=True)
    CONTEXTS.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), "prompts saved")


def rules():
    prefix, combo = VARIANT.split("_", 1)
    _, base, anchor = STEP4[prefix + "_"]
    lines = "".join(RULES[name] + "\n" for name in combo if name in RULES)
    items = json.loads((OUT.parent / base / "contexts.json").read_text(encoding="utf-8"))
    for c in items:
        assert c["prompt"].count(anchor) == 1
        c["prompt"] = c["prompt"].replace(anchor, anchor + lines)
        if "G" in combo:
            question = QUESTION.search(c["prompt"]).group(0)
            c["prompt"] = c["prompt"].replace(question, "\n").replace("\nWebsite excerpts:", question + "\nWebsite excerpts:", 1)
    OUT.mkdir(parents=True, exist_ok=True)
    CONTEXTS.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items), "prompts with rules", combo, "on", base)


def load_contexts():
    return {c["id"]: c for c in json.loads(CONTEXTS.read_text(encoding="utf-8"))}


def wait_for_limit(gap, last):
    left = gap - (time.time() - last)
    if left > 0:
        time.sleep(left)


def model_seconds(message):
    meta = message.response_metadata
    if "total_duration" not in meta:
        return None
    return round((meta["total_duration"] - meta.get("load_duration", 0)) / 1e9, 1)


def warm_up(llm, tries=5):
    for _ in range(tries):
        try:
            llm.invoke("Reply OK")
            return
        except Exception as error:
            print("warm-up failed, retrying:", str(error)[:80], flush=True)
            time.sleep(10)


def run(model, runs="3", first="1"):
    llm = LLMS[model]()
    if VARIANT in SETTINGS:
        llm = llm.model_copy(update=SETTINGS[VARIANT])
    gap = GAPS[model]
    folder = OUT / model.replace(":", "_")
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(int(first), int(first) + int(runs)):
        warm_up(llm)
        last = time.time()
        lines = []
        for c in load_contexts().values():
            wait_for_limit(gap, last)
            start = last = time.time()
            try:
                message = llm.invoke(c["prompt"])
                raw = THINK.sub("", reply_text(message)).strip()
                r = parse_reply(raw, FORMATS[CONFIG["prompt"]])
                lines.append({"id": c["id"], "raw": raw, "reply": r.model_dump(), "seconds": round(time.time() - start, 1),
                              "model_seconds": model_seconds(message)})
            except Exception as error:
                lines.append({"id": c["id"], "error": str(error)[:200], "seconds": round(time.time() - start, 1)})
            print(f"run {n} {c['id']} {lines[-1]['seconds']} s", flush=True)
        (folder / f"run{n}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")


def speed(model, count="10", gpu="L4"):
    llm = LLMS[model]()
    warm_up(llm)
    times = []
    for c in list(load_contexts().values())[: int(count)]:
        start = time.time()
        message = llm.invoke(c["prompt"])
        wall = round(time.time() - start, 1)
        times.append(model_seconds(message) or wall)
        print(f"{c['id']} {times[-1]} s", flush=True)
    folder = OUT / "speed"
    folder.mkdir(parents=True, exist_ok=True)
    result = {"model": model, "gpu": gpu, "questions": len(times), "model_seconds": sum(times) / len(times), "each": times}
    (folder / f"{model.replace(':', '_')}_{gpu}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{model} on {gpu}: {result['model_seconds']:.1f} s per answer")


def agent_prompts():
    items = list(load_contexts().values())
    size = -(-len(items) // AGENT_PARTS)
    folder = OUT / "claude-opus"
    folder.mkdir(parents=True, exist_ok=True)
    for part in range(AGENT_PARTS):
        blocks = [f"===== {c['id']} =====\n{c['prompt']}\n" for c in items[part * size:(part + 1) * size]]
        (folder / f"prompts_{part + 1}.txt").write_text("\n".join(blocks), encoding="utf-8")
    print("prompts for the agents in", folder)


def to_answer(r):
    if not r.known:
        return Answer(text=DONT_KNOW, confident=False)
    return Answer(text=r.answer)


def load_run(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows[d["id"]] = d
    return rows


def score():
    qs = {q["id"]: q for q in questions()}
    table = {}
    for folder in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name not in SKIP):
        runs = [load_run(p) for p in sorted(folder.glob("run?.jsonl"))]
        if not runs:
            continue
        per_run, per_question, seconds, model_time = [], {}, [], []
        for rows in runs:
            total = weight = 0
            for qid, q in qs.items():
                d = rows.get(qid, {})
                ans = to_answer(Reply(**d["reply"])) if "reply" in d else Answer(text="", confident=False)
                s = score_item(q, ans)
                total += s["score"] * q["weight"]
                weight += q["weight"]
                per_question.setdefault(qid, []).append((s["score"], sorted(set(VALUE.findall(ans.text))), ans.confident))
                if d.get("seconds") is not None:
                    seconds.append(d["seconds"])
                if d.get("model_seconds") is not None:
                    model_time.append(d["model_seconds"])
            per_run.append(total / weight)
        same = sum(1 for v in per_question.values() if len({(x[0], tuple(x[1]), x[2]) for x in v}) == 1)
        table[folder.name] = {"accuracy": sum(per_run) / len(per_run), "runs": per_run, "repeat": same / len(per_question) if len(runs) > 1 else None,
                              "seconds": sum(seconds) / len(seconds) if seconds else None,
                              "model_seconds": sum(model_time) / len(model_time) if model_time else None,
                              "questions": {qid: [x[0] for x in v] for qid, v in per_question.items()}}
    (OUT / "scores.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for name, t in table.items():
        sec = f"{t['seconds']:.1f} s" if t["seconds"] is not None else "-"
        if t["model_seconds"] is not None:
            sec += f" (model {t['model_seconds']:.1f} s)"
        print(f"{name:28} accuracy {t['accuracy']:.3f} ({' / '.join(f'{x:.3f}' for x in t['runs'])}) · repeat {'-' if t['repeat'] is None else format(t['repeat'], '.2f')} · {sec}")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"contexts": contexts, "rules": rules, "run": run, "speed": speed, "agent_prompts": agent_prompts, "score": score}[command](*args)
