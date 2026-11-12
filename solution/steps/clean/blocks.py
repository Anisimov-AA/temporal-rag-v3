import re

from langchain_core.documents import Document

HEADING = re.compile(r"^(#{1,6})\s")
BLOCKS = {
    "## Handpicked for You": None,
    "## Recent News": None,
    "#### Filter": "Load More",
    "✕": "Continue Cancel",
}
TEASER_TITLE = re.compile(r"\(dupaco\.com/[^)]*\)$")
TEASER_CLOSER = re.compile(r"^((get|read|see)( the)? (more|all)\b|keep reading)", re.IGNORECASE)
FIELD = re.compile(r"^(?!\|)[A-Z][^|]{0,90}(\*|\(Required\)\*?)$")
SUBMIT = re.compile(r"^(Previous)?Submit$")
PROMISE = re.compile(r"^We[’']ll only use your information")
INTRO_LINES = 3


def level(line):
    found = HEADING.match(line.strip())
    return len(found.group(1)) if found else None


def cut(lines, spans):
    drop = {i for first, last in spans for i in range(first, last + 1)}
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def each_page(find):
    def step(docs):
        cleaned = []
        for doc in docs:
            lines = doc.page_content.splitlines()
            cleaned.append(
                Document(page_content=cut(lines, find(lines)), metadata=doc.metadata)
            )
        return cleaned

    return step


def teasers_end(lines, first):
    last = first
    for i in range(first + 1, len(lines)):
        line = lines[i].strip()
        if level(line) and not TEASER_TITLE.search(line):
            break
        if TEASER_CLOSER.match(line):
            last = i
    return last


def block_end(lines, first, end):
    if not end:
        return teasers_end(lines, first)
    for i in range(first + 1, len(lines)):
        if lines[i].strip().startswith(end):
            return i
    return first


def find_blocks(lines):
    spans, i = [], 0
    while i < len(lines):
        start = lines[i].strip()
        if start in BLOCKS:
            last = block_end(lines, i, BLOCKS[start])
            spans.append((i, last))
            i = last
        i += 1
    return spans


def form_start(lines, first):
    k, seen = first - 1, 0
    while k >= 0 and seen < INTRO_LINES:
        line = lines[k].strip()
        if level(line):
            return k if level(line) > 1 else first
        if line:
            seen += 1
        k -= 1
    return first


def form_end(lines, first):
    promise = None
    for i in range(first + 1, len(lines)):
        if level(lines[i]):
            break
        if SUBMIT.match(lines[i].strip()):
            return i
        if PROMISE.match(lines[i].strip()):
            promise = i
    return promise


def find_forms(lines):
    spans, i = [], 0
    while i < len(lines):
        if FIELD.match(lines[i].strip()):
            last = form_end(lines, i)
            if last is not None:
                spans.append((form_start(lines, i), last))
                i = last
        i += 1
    return spans


def find_all(lines):
    return find_blocks(lines) + find_forms(lines)


remove_blocks = each_page(find_all)
