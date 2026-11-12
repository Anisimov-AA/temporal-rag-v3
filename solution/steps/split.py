import re

from langchain_core.documents import Document

HEADING = re.compile(r"^(#{1,2}) (.*)")
ANY_HEADING = re.compile(r"^#{1,6} ")
SUBHEADING = re.compile(r"^#{3,6} ")
QUESTION = re.compile(r"^[-*] .+\?$")
CAPTION = re.compile(r"^[^.!?:|#]{1,60}$")
TABLE_ROW = re.compile(r"^\s*\|")
CAPTION_LINES = 2


def has_text(lines):
    return any(line.strip() and not ANY_HEADING.match(line) for line in lines)


def carry_headings(lines):
    k = len(lines)
    while k > 0 and (not lines[k - 1].strip() or ANY_HEADING.match(lines[k - 1])):
        k -= 1
    return lines[:k], lines[k:]


def split_by_headings(doc):
    sections, lines, path = [], [], {}
    for line in doc.page_content.splitlines():
        heading = HEADING.match(line)
        if heading and has_text(lines):
            body, lines = carry_headings(lines)
            sections.append((body, path))
        if heading and heading.group(1) == "#":
            path = {"h1": heading.group(2)}
        elif heading:
            path = {**path, "h2": heading.group(2)}
        lines.append(line)
    sections.append((lines, path))
    return [
        Document(page_content="\n".join(lines).strip(), metadata={**doc.metadata, **path})
        for lines, path in sections
        if "".join(lines).strip()
    ]


def caption_start(lines, table):
    start, seen, k = table, 0, table - 1
    while k >= 0 and seen < CAPTION_LINES:
        line = lines[k].strip()
        if line and not CAPTION.match(line):
            break
        if line:
            start, seen = k, seen + 1
        k -= 1
    return start


def unit_starts(lines):
    starts = set()
    for i, line in enumerate(lines):
        line = line.strip()
        if SUBHEADING.match(line) or QUESTION.match(line):
            starts.add(i)
        elif TABLE_ROW.match(line) and (i == 0 or not TABLE_ROW.match(lines[i - 1])):
            starts.add(caption_start(lines, i))
    return starts


def split_into_units(text):
    lines = text.splitlines()
    starts = unit_starts(lines)
    units, unit = [], []
    for i, line in enumerate(lines):
        if i in starts and has_text(unit):
            body, unit = carry_headings(unit)
            units.append(body)
        unit.append(line)
    units.append(unit)
    return ["\n".join(unit).strip() for unit in units if "\n".join(unit).strip()]


def split_into_chunks(docs, size):
    chunks = []
    for doc in docs:
        for section in split_by_headings(doc):
            if len(section.page_content) <= size:
                chunks.append(section)
                continue
            for unit in split_into_units(section.page_content):
                chunks.append(Document(page_content=unit, metadata=section.metadata))
    return chunks
