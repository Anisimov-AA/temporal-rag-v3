import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "dev"), str(ROOT / "harness"), str(ROOT / "solution")]
import tournament  # noqa: E402
from steps.answer import Reply  # noqa: E402

OUT = ROOT / "answers.html"
NAMES = {"claude-opus": "Opus 5.5", "gcp_qwen3_32b": "Qwen3 32B", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
         "gcp_qwen3.6_27b": "Qwen3.6 27B", "qwen3_8b": "Qwen3 8B", "gcp_gemma4_26b": "Gemma 4 26B",
         "gcp_mistral-small3.2_24b": "Mistral 3.2", "gcp_command-r_35b": "Command R", "gcp_qwen3.8_27b": "Qwen3.8 27B",
         "gcp_granite4.2_30b": "Granite 4.2", "gcp_llama3.3_70b": "Llama 3.3 70B", "gcp_qwen3_30b-a3b": "Qwen3 30B-A3B",
         "gcp_gpt-oss_20b": "gpt-oss low", "gcp_gpt-oss_20b-medium": "gpt-oss medium",
         "nvidia_nemotron-3.5-super": "Nemotron Super", "nvidia_nemotron-3.5-lightning": "Nemotron Lightning", "nvidia_nemotron-3-ultra": "Nemotron Ultra",
         "nvidia_gemma-4-31b": "Gemma 4 31B", "nvidia_nemotron-3.5-bolt": "Nemotron Bolt", "nvidia_mistral-large-4": "Mistral Large 4", "nvidia_glm-5.3-flash": "GLM-5.3"}

BROKEN = {"dup-005": "scorer gives 0 to any list", "dup-088": "key wants 2023, the site's latest is 2024",
          "dup-049": "key ignores the HSA and money market tiers"}

STRICT = {"dup-087": "accepts only the word fifth, not five times",
          "dup-104": "asks for noon, which the expected answer never mentions"}

STYLE = """
:root{--bg:#f6f8fa;--ink:#1f2328;--muted:#59636e;--line:#d1d9e0}
*{box-sizing:border-box}
i{font-style:normal}
body{margin:0;padding:32px 36px;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin:0 0 20px}
details{border-bottom:1px solid var(--line)}
.qrow{display:flex;align-items:center;gap:4px;font-size:12px;line-height:20px;white-space:nowrap;cursor:pointer;list-style:none}
.qrow::-webkit-details-marker{display:none}
.qrow:hover{background:#eef1f4}
.head{display:flex;gap:4px;font-size:11px;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:4px}
.qt{width:380px;overflow:hidden;text-overflow:ellipsis;flex:none}
.qt u{text-decoration:none;color:var(--muted);font-family:Consolas,monospace;margin-right:6px}
.qh{width:64px;flex:none;text-align:center}
.qc{display:inline-block;width:64px;flex:none;text-align:center;border-radius:3px;font-size:11px;line-height:16px}
.none{color:var(--muted)}.t1{background:#dafbe1;color:#1a7f37}.tp{background:#fff8c5;color:#9a6700}.t0{background:#ffebe9;color:#cf222e}
.qd{margin:2px 0 8px 0;padding:2px 10px;border-left:2px solid var(--line);font-size:12px;line-height:1.45;max-width:900px}
.qx{color:var(--muted);margin-bottom:6px}
.ans{display:flex;gap:8px;margin:4px 0}
.ans .qc{width:52px}
.ans b{font-weight:600;flex:none;width:72px}
nav{margin-bottom:18px;border-bottom:1px solid var(--line);padding-bottom:6px}
nav .step{display:flex;gap:14px;align-items:baseline;margin-bottom:8px}
nav .step b{flex:none;width:200px;font-size:13px}
nav .row{display:flex;align-items:baseline;margin-bottom:4px}
nav .grp{flex:none;width:130px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}

.tab{display:inline-block;font-size:13px;color:var(--muted);text-decoration:none;margin-right:16px;padding-bottom:2px}
.tab.on{color:var(--ink);border-bottom:2px solid var(--ink)}
.broken .qrow{opacity:.45}.key{margin-right:4px}.why{color:#9a6700;margin-bottom:6px}
@media (max-width:600px){body{padding:20px 16px}.qt{width:200px}}
"""


def tone(score):
    return "t1" if score >= 0.85 else "tp" if score >= 0.5 else "t0"


def chip(scores):
    mean = sum(scores) / len(scores)
    label = f"{round(scores[0], 2):g}" if len(set(scores)) == 1 else " · ".join(f"{round(s, 2):g}" for s in scores)
    return f'<i class="qc {tone(mean)}">{label}</i>'


def answer_text(d, context):
    if "reply" not in d:
        return "(no reply)"
    return tournament.to_answer(Reply(**d["reply"]), context).text


RESULTS = ROOT / "results" / "tournament"
VARIANTS = RESULTS / "variants"
QWEN = "gcp_qwen3_32b"
ORDER = ["gcp_gemma4_26b", QWEN, "nvidia_nemotron-3-ultra", "claude-opus", "gemini-3.5-flash-lite"]


def step4(name):
    return [(VARIANTS / f"{name}10", None), (VARIANTS / f"{name}10nodates", QWEN)]


STEPS = [("Step 2 · Pick the LLM", [("", [("step2", "all models", [(RESULTS, None)])])]),
         ("Step 3 · Pick how many chunks", [("", [(v, label, [(VARIANTS / v, None)]) for v, label in
                                                 [("top3", "3 chunks"), ("top10", "10 chunks"), ("top15", "15 chunks"),
                                                  ("top10nodates", "10, no date rule"), ("norerank10", "10, no rerank")]])]),
         ("Step 4 · Pick the prompt", [
             ("", [("base4", "base", [(VARIANTS / "top10", None), (VARIANTS / "top10nodates", QWEN)])]),
             ("best practices", [(f"{v}10", label, step4(v)) for v, label in
                                 [("closest", "closest"), ("conflictshot", "example"), ("checklist", "checklist"),
                                  ("labels", "labels"), ("short", "short"), ("cot", "chain of thought")]]),
             ("from our errors", [(f"{v}10", v, step4(v)) for v in ["strict", "varies", "latest", "clear"]]),
             ("combined", [(f"{v}10", label, step4(v)) for v, label in
                           [("checklistshot", "checklist + example"), ("latestchecklist", "latest + checklist"), ("variesclosest", "varies + closest"), ("variesclear", "varies + clear"), ("closestclear", "closest + clear")]])])]


def load(sources, skip):
    models = []
    for folder, only in sources:
        scores_file = folder / "scores.json"
        if not scores_file.exists():
            continue
        scores = json.loads(scores_file.read_text(encoding="utf-8"))
        ctx = {c["id"]: c for c in json.loads((folder / "contexts.json").read_text(encoding="utf-8"))}
        for m, t in scores.items():
            if (only is None or m == only) and (only is not None or m != QWEN or len(sources) == 1) and m not in skip and m not in [x[0] for x in models]:
                models.append((m, t, ctx, tournament.load_run(folder / m / "run1.jsonl")))
    return sorted(models, key=lambda x: (ORDER.index(x[0]) if x[0] in ORDER else len(ORDER), -x[1]["accuracy"]))


def section(key, sources, qs, skip=(), keep=False):
    models = load(sources, skip)
    if not models:
        return ""
    if keep:
        have = [m for m, *_ in models]
        models = [next(x for x in models if x[0] == m) if m in have else (m, None, None, None) for m in ORDER if m not in skip] + [x for x in models if x[0] not in ORDER]
    head = '<div class="head"><i class="qt"></i>' + "".join(f'<i class="qh">{NAMES.get(m, m)}</i>' for m, *_ in models) + "</div>"
    empty = '<i class="qc none">—</i>'
    rows = []
    for q in qs:
        chips = "".join(chip(t["questions"][q["id"]]) if t else empty for _, t, _, _ in models)
        date = f' · as of {q["as_of"]}' if q.get("as_of") else ""
        answers = "".join(
            f'<div class="ans">{chip(t["questions"][q["id"]])}<b>{NAMES.get(m, m)}</b>'
            f'<span>{html.escape(answer_text(run.get(q["id"], {}), ctx[q["id"]]))}</span></div>'
            for m, t, ctx, run in models if t)
        must = f' · must include: {html.escape(", ".join(q["must_include"]))}' if q["must_include"] else ""
        broken = BROKEN.get(q["id"])
        cls = ' class="broken"' if broken else ""
        why = f' title="answer-key problem: {html.escape(broken)}"' if broken else ""
        note = f'<div class="why">answer-key problem: {html.escape(broken)}</div>' if broken else ""
        strict = STRICT.get(q["id"])
        lock = f'<span class="key" title="strict key: {html.escape(strict)}">🔑</span>' if strict else ""
        if strict:
            note += f'<div class="why">strict key: {html.escape(strict)}</div>'
        rows.append(f'<details{cls}><summary class="qrow"><i class="qt"{why}><u>{q["id"][4:]}</u>{lock}{html.escape(q["question"])}{date}</i>{chips}</summary>'
                    f'<div class="qd">{note}<div class="qx">expected: {html.escape(q["expected_answer"])}{must}</div>{answers}</div></details>')
    return f'<section id="{key}">{head}{"".join(rows)}</section>'


def main():
    qs = tournament.questions()
    nav, sections, first = [], [], None
    for step, groups in STEPS:
        parts = []
        for group, tabs in groups:
            links = []
            for key, label, sources in tabs:
                built = section(key, sources, qs, ("gemini-3.5-flash-lite",) if step.startswith("Step 4") else (), keep=not step.startswith("Step 2"))
                if built:
                    sections.append(built)
                    first = first or key
                    links.append(f'<a class="tab" href="#{key}">{label}</a>')
            if links:
                parts.append(f'<div class="row"><span class="grp">{group}</span>{"".join(links)}</div>')
        nav.append(f'<div class="step"><b>{step}</b><div>{"".join(parts)}</div></div>')
    tabs = "".join(nav)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Answers</title><style>{STYLE}</style></head>
<body><h1>Answers</h1><p class="sub">30 tuning questions · score of each of the 3 runs (one number = the same in all 3) · click a question for the answers (run 1) · grey = answer-key problem, no model can score · 🔑 = strict key: right answers in other words get less</p>
<nav>{tabs}</nav>{"".join(sections)}
<script>
function show() {{
  const key = location.hash.slice(1) || "{first}";
  document.querySelectorAll("section").forEach(s => s.style.display = s.id === key ? "" : "none");
  document.querySelectorAll(".tab").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + key));
}}
window.addEventListener("hashchange", show);
show();
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print("written", OUT, len(sections), "tabs")


if __name__ == "__main__":
    main()
