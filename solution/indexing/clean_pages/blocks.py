import re

from langchain_core.documents import Document

HEADING = re.compile(r"^(#{1,6})\s")
BLOCKS = {
    "## Handpicked for You": None,
    "## Recent News": None,
    "#### Filter": "Load More",
    "✕": "Continue Cancel",
    "#### Build a Vehicle": "Get a New Quote",
}
MARKERS = {marker.lower(): end for marker, end in BLOCKS.items()}
TEASER_TITLE = re.compile(r"\(dupaco\.com/[^)]*\)$")
TEASER_CLOSER = re.compile(r"^((get|read|see)( the)? (more|all)\b|keep reading)", re.IGNORECASE)
FIELD = re.compile(r"^(?!\|)[A-Z][^|]{0,90}(\*|\(Required\)\*?)$")
SUBMIT = re.compile(r"^(Previous)?Submit$")
PROMISE = re.compile(r"^We[’']ll only use your information")
INTRO_LINES = 4
WIDGET_LINES = 4
WIDGET_TITLE = re.compile(r"^(?![-*#|])(?!.*\.$)[^!:|()]{2,40}$")
TITLE_LIKE = re.compile(r"^(?![-*#|])[^.!:|()]{2,40}$")
LOCATOR = re.compile(r"^With \d+ miles$")
RESULTS = re.compile(r"^\d+ Results$")
ZOOM = "### Zoom in"
MAP_LINE = re.compile(r"^(Closed|Open\b.*|Dupaco Branches|Shared Branches|Surcharge-Free ATMs|\d+)$")
CALC_END = re.compile(
    r"data points: "
    r"|^(Estimates are for educational purposes only"
    r"|Title, and other taxes and fees are not included"
    r"|This material is provided for general and educational purposes)"
)
CALC_TAIL = re.compile(r"^(powered by\b|Disclosure$)")
CALC_INPUT = re.compile(r"^(\$[\d,.]*|\d+ months)$")
LABEL = re.compile(r"^(?!#)[^.!?|]{1,40}$")
DOUBLED = re.compile(r"^[^|#].{0,59}$")


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
        if TEASER_CLOSER.match(line) or line.endswith("..."):
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
        start = lines[i].strip().lower()
        if start in MARKERS:
            last = block_end(lines, i, MARKERS[start])
            spans.append((i, last))
            i = last
        i += 1
    return spans


def form_heading(lines, first):
    k, seen, title = first - 1, 0, None
    while k >= 0 and seen < INTRO_LINES:
        line = lines[k].strip()
        if level(line):
            return k if level(line) > 1 else title
        if line:
            seen += 1
            if TITLE_LIKE.match(line):
                title = k
        k -= 1
    return title


def form_start(lines, first):
    heading = form_heading(lines, first)
    if heading is not None:
        return heading
    start, k = first, first - 1
    while k >= 0 and (not lines[k].strip() or LABEL.match(lines[k].strip())):
        if lines[k].strip():
            start = k
        k -= 1
    return start


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


def next_text(lines, i):
    for k in range(i + 1, len(lines)):
        if lines[k].strip():
            return k
    return None


def find_locators(lines):
    spans = []
    for i, line in enumerate(lines):
        k = next_text(lines, i)
        if line.strip() != "Go" or k is None or not LOCATOR.match(lines[k].strip()):
            continue
        results = [j for j in range(k, len(lines)) if RESULTS.match(lines[j].strip())]
        zoom = next((j for j in range(k, len(lines)) if lines[j].strip() == ZOOM), None)
        if len(results) < 2 or zoom is None:
            continue
        spans.append((i, results[1]))
        spans += [(j, j) for j in range(results[1] + 1, zoom) if MAP_LINE.match(lines[j].strip())]
        spans.append((zoom, next_text(lines, zoom)))
    return spans


def is_input(rows, n):
    doubled = n + 1 < len(rows) and rows[n + 1] == rows[n] and DOUBLED.match(rows[n])
    return bool(doubled or CALC_INPUT.match(rows[n]))


def widget_title(rows, first, start):
    title = None
    for k in range(first - 1, max(start, first - WIDGET_LINES) - 1, -1):
        if level(rows[k]):
            break
        if WIDGET_TITLE.match(rows[k]):
            title = k
    return title


def find_calculators(lines):
    spans = []
    full = [i for i, line in enumerate(lines) if line.strip()]
    rows = [lines[i].strip() for i in full]
    start = n = 0
    while n < len(rows):
        if not CALC_END.search(rows[n]):
            n += 1
            continue
        end = n
        while end + 1 < len(rows) and (CALC_END.search(rows[end + 1]) or CALC_TAIL.match(rows[end + 1])):
            end += 1
        first = next((k for k in range(start, n) if is_input(rows, k)), None)
        if first is not None:
            while first > start and LABEL.match(rows[first - 1]):
                first -= 1
            title = widget_title(rows, first, start)
            first = first if title is None else title
            spans.append((full[first], full[end]))
        start = n = end + 1
    return spans


def find_all(lines):
    return find_blocks(lines) + find_forms(lines) + find_locators(lines) + find_calculators(lines)


remove_blocks = each_page(find_all)
