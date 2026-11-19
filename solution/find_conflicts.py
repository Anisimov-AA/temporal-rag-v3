import re
from collections import Counter, defaultdict

from contract import Conflict, Source

EFFECTIVE = re.compile(r"effective (?:as of )?(\d{1,2}/\d{1,2}/\d{4})", re.I)
NOTE = re.compile(r"(%|\)|APR|APY)\s?\d+$")


def cells(line):
    return [NOTE.sub(r"\1", c.strip()) for c in line.strip().strip("|").split("|")]


def section_above(lines, start):
    for line in reversed(lines[max(0, start - 12):start]):
        text = line.strip().lstrip("#").strip()
        if text and not text.startswith("|") and not EFFECTIVE.search(text) and len(text) < 60:
            return text
    return ""


def effective_below(lines, end):
    for line in lines[end:end + 25]:
        if line.startswith("|"):
            break
        found = EFFECTIVE.search(line)
        if found:
            return found.group(1)
    return ""


def page_rows(doc):
    lines = doc.page_content.splitlines()
    rows, n = [], 0
    while n < len(lines):
        if lines[n].startswith("|") and n + 1 < len(lines) and set(lines[n + 1].replace("|", "").strip()) <= set("-: "):
            head, start, n = cells(lines[n]), n, n + 2
            body = []
            while n < len(lines) and lines[n].startswith("|"):
                body.append(cells(lines[n]))
                n += 1
            section, effective = section_above(lines, start), effective_below(lines, n)
            for row in body:
                if any("%" in c for c in row):
                    rows.append({"snapshot": doc.metadata["snapshot_id"], "taken_on": doc.metadata["taken_on"],
                                 "url": doc.metadata["url"], "section": section, "effective": effective,
                                 "cells": dict(zip(head, row))})
        else:
            n += 1
    return rows


def rate_table(docs):
    rows = []
    for doc in docs:
        rows += page_rows(doc)
    return rows


SERVICE = re.compile(r"payment|example|points|min\.? to|minimum", re.I)
YEARS = re.compile(r"\b(?:19|20)\d\d\b(?:\s*[-–]\s*\d{2,4})?")
RATE = re.compile(r"\d+(?:\.\d+)?(?=\s*%)")
WORD = re.compile(r"[a-z]{3,}")
GENERIC = {"loans", "loan", "rates", "rate", "account", "accounts", "the", "and", "for", "with", "your"}
DISTINCT = {"business", "jumbo", "ira", "iras", "roth", "traditional", "health", "kids", "youth", "student",
            "special", "regular", "refinance", "purchase", "equity", "rewards", "platinum", "boat"}


def norm(text):
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%.,/&+ -]", " ", text)).strip()


def facts(rows):
    count = Counter()
    for r in rows:
        pairs = [(h, v) for h, v in r["cells"].items() if not SERVICE.search(h)]
        label = " ".join(v for _, v in pairs if "%" not in v)
        name = norm(YEARS.sub("", label))
        rates = {h: v for h, v in pairs if "%" in v}
        if name and rates and r["section"]:
            key = (r["snapshot"], r["url"], norm(r["section"]), name)
            count[key] += 1
            yield {**r, "name": f"{name} #{count[key]}", "label": label, "rates": rates}


def signature(rates):
    return tuple(tuple(RATE.findall(v)[:1]) for _, v in sorted(rates.items()))


def describe(rates):
    return "; ".join(f"{column} {value}" for column, value in rates.items())


def topic(fact):
    return f'{fact["section"]}: {fact["label"]}'


def over_time(rows):
    history = defaultdict(list)
    for f in facts(rows):
        history[(f["url"], norm(f["section"]), f["name"])].append(f)
    newest = max(r["taken_on"] for r in rows)
    out, seen = [], set()
    for versions in history.values():
        changes = []
        for f in sorted(versions, key=lambda f: f["taken_on"]):
            if not changes or signature(changes[-1]["rates"]) != signature(f["rates"]):
                changes.append(f)
        same = (changes[0]["name"].split(" #")[0], signature(changes[-1]["rates"]))
        if len(changes) < 2 or same in seen:
            continue
        seen.add(same)
        last = changes[-1]
        older = ", ".join(f'{describe(f["rates"])} ({f["taken_on"]})' for f in changes[:-1])
        shown = "the newest download" if max(f["taken_on"] for f in versions) == newest else f'the last download with it ({last["taken_on"]})'
        out.append(Conflict(topic=topic(last), values=[describe(f["rates"]) for f in changes],
                            sources=[Source(url=f["url"], snapshot_id=f["snapshot"], quote=describe(f["rates"])) for f in changes],
                            resolution=describe(last["rates"]),
                            reason=f'temporal: {shown} shows {describe(last["rates"])}; older downloads showed {older}'))
    return out


def same_product(a, b, name):
    words_a, words_b = set(WORD.findall(a)) - GENERIC, set(WORD.findall(b)) - GENERIC
    return bool(words_a & words_b) and not ((words_a ^ words_b) & DISTINCT - set(WORD.findall(name)))


def between_pages(rows):
    newest = max(r["snapshot"] for r in rows)
    groups = defaultdict(list)
    for f in facts([r for r in rows if r["snapshot"] == newest]):
        groups[(f["name"], frozenset(norm(c) for c in f["rates"]))].append(f)
    out, seen = [], set()
    for (name, _), found in groups.items():
        for first in found:
            group = [f for f in found if same_product(norm(first["section"]), norm(f["section"]), name)]
            by_value = defaultdict(list)
            for f in group:
                by_value[signature(f["rates"])].append(f)
            if len(by_value) < 2 or len({f["url"] for f in group}) < 2:
                continue
            key = (name.split(" #")[0], frozenset(by_value))
            if key in seen:
                continue
            seen.add(key)
            official = [v for v, fs in by_value.items() if any(f["url"].rstrip("/").endswith("/rates") for f in fs)]
            winner = official[0] if len(official) == 1 else max(by_value, key=lambda v: len(by_value[v]))
            why = "the official rates page shows it" if len(official) == 1 else "most pages of the newest download show it"
            others = "; ".join(f'{describe(fs[0]["rates"])} on {", ".join(sorted(f["url"].split("dupaco.com")[-1] for f in fs))}'
                               for v, fs in by_value.items() if v != winner)
            out.append(Conflict(topic=topic(first), values=[describe(fs[0]["rates"]) for fs in by_value.values()],
                                sources=[Source(url=f["url"], snapshot_id=newest, quote=describe(f["rates"])) for fs in by_value.values() for f in fs],
                                resolution=describe(by_value[winner][0]["rates"]), reason=f"authority: {why}; {others}"))
    return out


def find_conflicts(rows):
    return over_time(rows) + between_pages(rows) if rows else []
