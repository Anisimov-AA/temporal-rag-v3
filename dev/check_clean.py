import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
from run_eval import load_snapshots  # noqa: E402
from scoring import _norm, _numbers  # noqa: E402
from steps.clean import CLEANERS  # noqa: E402
from steps.load import to_documents  # noqa: E402

STEPS = [
    "images",
    "links",
    "markdown_noise",
    "html_noise",
    "blocks",
    "repeated_lines",
    "tidy",
    "head",
    "empty_pages",
    "copies",
]
EVAL = ROOT / "eval" / "public_eval.jsonl"
SHOW = 60


def answer_values():
    questions = [
        json.loads(line)
        for line in EVAL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted({value for q in questions for value in q["must_include"]})


def is_number(value):
    leftover = re.sub(r"[\s$\d.,%]|\b(apr|apy|percent)\b", "", value.lower())
    return bool(_numbers(value)) and not leftover and len(value) <= 20


def has(text, numbers, value):
    if is_number(value):
        return all(
            n in numbers or (n[0] == "" and any(a[1] == n[1] for a in numbers))
            for n in _numbers(value)
        )
    return _norm(value) in text


def where(docs, values):
    found = {value: set() for value in values}
    for doc in docs:
        text, numbers = _norm(doc.page_content), _numbers(doc.page_content)
        for value in values:
            if has(text, numbers, value):
                found[value].add(doc.metadata["url"])
    return found


def around(before, after, value):
    for match in re.finditer(re.escape(value), before, re.IGNORECASE):
        piece = before[max(0, match.start() - SHOW) : match.end() + SHOW]
        if piece not in after:
            return " ".join(piece.split())
    return ""


def name(url):
    return url.split("dupaco.com")[1] or "/"


def main():
    values = answer_values()
    print(f"{len(values)} answer values, steps: {' → '.join(STEPS)}")
    losses = {step: {} for step in STEPS}
    site_losses = {step: set() for step in STEPS}
    for snapshot in load_snapshots(ROOT / "snapshots"):
        docs = to_documents(snapshot)
        before = where(docs, values)
        for step in STEPS:
            old = {doc.metadata["url"]: doc.page_content for doc in docs}
            docs = CLEANERS[step](docs)
            new = {doc.metadata["url"]: doc.page_content for doc in docs}
            after = where(docs, values)
            for value in values:
                for url in sorted(before[value] - after[value]):
                    loss = losses[step].setdefault(
                        (value, url),
                        {
                            "downloads": [],
                            "was": around(old[url], new.get(url, ""), value),
                        },
                    )
                    loss["downloads"].append(snapshot.taken_on[5:])
                if before[value] and not after[value]:
                    site_losses[step].add((value, snapshot.taken_on[5:]))
            before = after
    for step in STEPS:
        gone = sorted(site_losses[step])
        print(
            f"\n{step}: {len(losses[step])} lost from a page · gone from the whole site: {gone if gone else 'nothing'}"
        )
        for (value, url), loss in losses[step].items():
            print(
                f"  {value!r} gone from {name(url)}   in {', '.join(loss['downloads'])}"
            )
            print(f"    was: ...{loss['was']}...")
    total = sum(len(found) for found in losses.values())
    print(
        f"\n{'OK, nothing lost' if total == 0 else 'read each one: is it real text or only an address or code?'}"
    )


if __name__ == "__main__":
    main()
