import re

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

HEADING = re.compile(r"^(#{1,2}) (.*)")
ANY_HEADING = re.compile(r"^#{1,6} ")
SUBHEADING = re.compile(r"^#{3,6} ")
QUESTION = re.compile(r"^[-*] .+\?$")
CAPTION = re.compile(r"^[^.!?:|#]{1,60}$")
TABLE_ROW = re.compile(r"^\s*\|")
TABLE_LINE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
CAPTION_LINES = 2
TINY = 100


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


def tables_and_text(text):
    parts, lines = [], []
    for line in text.splitlines():
        if lines and bool(TABLE_ROW.match(line)) != bool(TABLE_ROW.match(lines[-1])):
            parts.append(lines)
            lines = []
        lines.append(line)
    parts.append(lines)
    return parts


def cut_table(rows, size):
    header = rows[:2] if len(rows) > 1 and TABLE_LINE.match(rows[1]) else rows[:1]
    pieces, piece = [], list(header)
    for row in rows[len(header):]:
        if len(piece) > len(header) and len("\n".join(piece + [row])) > size:
            pieces.append("\n".join(piece))
            piece = list(header)
        piece.append(row)
    pieces.append("\n".join(piece))
    return pieces


def glue(pieces, size):
    glued = []
    for piece in pieces:
        if glued and len(glued[-1]) + 1 + len(piece) <= size:
            glued[-1] += "\n" + piece
        else:
            glued.append(piece)
    return glued


def cut_long_unit(text, size):
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=size, chunk_overlap=0
    )
    pieces = []
    for lines in tables_and_text(text):
        if TABLE_ROW.match(lines[0]):
            pieces += cut_table(lines, size)
        elif "\n".join(lines).strip():
            pieces += splitter.split_text("\n".join(lines))
    return glue(pieces, size)


def same_page(a, b):
    return a.metadata["url"] == b.metadata["url"]


def joined(first, second, metadata):
    text = first.page_content + "\n\n" + second.page_content
    return Document(page_content=text, metadata=metadata)


def glue_tiny(chunks):
    forward = []
    for chunk in chunks:
        last = forward[-1] if forward else None
        if last and same_page(last, chunk) and len(last.page_content) < TINY:
            forward[-1] = joined(last, chunk, chunk.metadata)
        else:
            forward.append(chunk)
    glued = []
    for chunk in forward:
        last = glued[-1] if glued else None
        if last and same_page(last, chunk) and len(chunk.page_content) < TINY:
            glued[-1] = joined(last, chunk, last.metadata)
        else:
            glued.append(chunk)
    return glued


def split_into_chunks(docs, size):
    chunks = []
    for doc in docs:
        for section in split_by_headings(doc):
            if len(section.page_content) <= size:
                chunks.append(section)
                continue
            for unit in split_into_units(section.page_content):
                for piece in cut_long_unit(unit, size) if len(unit) > size else [unit]:
                    chunks.append(Document(page_content=piece, metadata=section.metadata))
    return glue_tiny(chunks)
