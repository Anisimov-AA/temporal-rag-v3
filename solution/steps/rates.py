import re

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
