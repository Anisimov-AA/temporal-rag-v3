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
         "gcp_gpt-oss_20b": "gpt-oss low", "gcp_gpt-oss_20b-medium": "gpt-oss medium"}

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
.t1{background:#dafbe1;color:#1a7f37}.tp{background:#fff8c5;color:#9a6700}.t0{background:#ffebe9;color:#cf222e}
.qd{margin:2px 0 8px 0;padding:2px 10px;border-left:2px solid var(--line);font-size:12px;line-height:1.45;max-width:900px}
.qx{color:var(--muted);margin-bottom:6px}
.ans{display:flex;gap:8px;margin:4px 0}
.ans .qc{width:52px}
.ans b{font-weight:600;flex:none;width:72px}
nav{margin-bottom:14px}
.tab{display:inline-block;font-size:13px;color:var(--muted);text-decoration:none;margin-right:16px;padding-bottom:2px}
.tab.on{color:var(--ink);border-bottom:2px solid var(--ink)}
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


SECTIONS = [("step2", "Step 2 · Pick the LLM", ROOT / "results" / "tournament"),
            ("top3", "Step 3 · 3 chunks", ROOT / "results" / "tournament" / "variants" / "top3"),
            ("top10", "Step 3 · 10 chunks", ROOT / "results" / "tournament" / "variants" / "top10"),
            ("top15", "Step 3 · 15 chunks", ROOT / "results" / "tournament" / "variants" / "top15"),
            ("top10nodates", "Step 3 · 10 chunks, no date rule", ROOT / "results" / "tournament" / "variants" / "top10nodates"),
            ("norerank10", "Step 3 · 10 chunks, no rerank", ROOT / "results" / "tournament" / "variants" / "norerank10")]


def section(key, folder, qs):
    scores_file = folder / "scores.json"
    if not scores_file.exists():
        return ""
    ctx = {c["id"]: c for c in json.loads((folder / "contexts.json").read_text(encoding="utf-8"))}
    scores = json.loads(scores_file.read_text(encoding="utf-8"))
    models = [(m, NAMES.get(m, m)) for m in sorted(scores, key=lambda m: -scores[m]["accuracy"])]
    runs = {m: tournament.load_run(folder / m / "run1.jsonl") for m, _ in models}
    head = '<div class="head"><i class="qt"></i>' + "".join(f'<i class="qh">{name}</i>' for _, name in models) + "</div>"
    rows = []
    for q in qs:
        chips = "".join(chip(scores[m]["questions"][q["id"]]) for m, _ in models)
        date = f' · as of {q["as_of"]}' if q.get("as_of") else ""
        answers = "".join(
            f'<div class="ans">{chip(scores[m]["questions"][q["id"]])}<b>{name}</b>'
            f'<span>{html.escape(answer_text(runs[m].get(q["id"], {}), ctx[q["id"]]))}</span></div>'
            for m, name in models)
        must = f' · must include: {html.escape(", ".join(q["must_include"]))}' if q["must_include"] else ""
        rows.append(f'<details><summary class="qrow"><i class="qt"><u>{q["id"][4:]}</u>{html.escape(q["question"])}{date}</i>{chips}</summary>'
                    f'<div class="qd"><div class="qx">expected: {html.escape(q["expected_answer"])}{must}</div>{answers}</div></details>')
    return f'<section id="{key}">{head}{"".join(rows)}</section>'


def main():
    qs = tournament.questions()
    built = [(key, title, section(key, folder, qs)) for key, title, folder in SECTIONS]
    built = [b for b in built if b[2]]
    tabs = "".join(f'<a class="tab" href="#{key}">{title}</a>' for key, title, _ in built)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Answers</title><style>{STYLE}</style></head>
<body><h1>Answers</h1><p class="sub">30 tuning questions · score of each of the 3 runs (one number = the same in all 3) · click a question for the answers (run 1)</p>
<nav>{tabs}</nav>{"".join(s for _, _, s in built)}
<script>
function show() {{
  const key = location.hash.slice(1) || "{built[0][0]}";
  document.querySelectorAll("section").forEach(s => s.style.display = s.id === key ? "" : "none");
  document.querySelectorAll(".tab").forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + key));
}}
window.addEventListener("hashchange", show);
show();
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print("written", OUT, "sections:", [k for k, _, _ in built])


if __name__ == "__main__":
    main()
