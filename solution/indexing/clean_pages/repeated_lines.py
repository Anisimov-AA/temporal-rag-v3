import re
from collections import Counter

from langchain_core.documents import Document

SHARE = 0.03
NUMBER = re.compile(r"\d+")
LETTER = re.compile(r"[A-Za-z]")
KEEP = re.compile(
    r"^(#|\|)|9/9/9|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? 9, 9\b"
)
FAMILIES = [
    re.compile(r"^\d+ min read$"),
    re.compile(r"^([^()]+ \(dupaco\.com/[^)]*/(author|tag)/[^)]*\)\s*)+$"),
    re.compile(r"^#{1,6}\s*\**$"),
    re.compile(r"^\\# # #$"),
]


def shape(line):
    return NUMBER.sub("9", line.strip())


def find_repeated(docs):
    counts = Counter(
        s
        for doc in docs
        for s in {shape(line) for line in doc.page_content.splitlines()}
    )
    least = SHARE * len(docs)
    return {
        s
        for s, pages in counts.items()
        if pages >= least and LETTER.search(s) and not KEEP.search(s)
    }


def is_junk(line, repeated):
    s = line.strip()
    return shape(s) in repeated or any(rx.match(s) for rx in FAMILIES)


def remove_repeated_lines(docs):
    repeated = find_repeated(docs)
    cleaned = []
    for doc in docs:
        lines = [
            line
            for line in doc.page_content.splitlines()
            if not is_junk(line, repeated)
        ]
        cleaned.append(Document(page_content="\n".join(lines), metadata=doc.metadata))
    return cleaned
