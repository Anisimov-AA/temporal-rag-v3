import re

from langchain_core.documents import Document

HEADING = re.compile(r"^(#{1,2}) (.*)")
ANY_HEADING = re.compile(r"^#{1,6} ")


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


def split_into_chunks(docs):
    return [section for doc in docs for section in split_by_headings(doc)]
