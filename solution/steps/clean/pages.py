import re
from datetime import datetime
from urllib.parse import urlparse

from langchain_core.documents import Document

MIN_WORDS = 10
APP_PAGES = ("/subscription-preferences", "/mobile/eNotifiers")
TITLE = re.compile(r"^# ")
DATE = re.compile(r"^[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}$")
BYLINE = re.compile(r"^Written by .*\| ([A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})")
BUTTON = re.compile(r"^[^.!?]{0,60}\([^)]*\)$")
LEFTOVER = re.compile(r"^(title|description): |^[\w.-]+\.com(/\S*)?$")
ADDRESS = re.compile(r"\([^)]*\)")
WORD = re.compile(r"[A-Za-z]{2,}")


def to_iso(date):
    date = date.replace(".", "")
    for form in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(date, form).date().isoformat()
        except ValueError:
            pass
    return None


def title_line(lines):
    for i, line in enumerate(lines):
        if TITLE.match(line):
            return i
    return 0


def read_head(text):
    lines = text.splitlines()
    title = title_line(lines)
    published = None
    kept = []
    for i, line in enumerate(lines):
        byline = BYLINE.match(line)
        if byline:
            published = published or to_iso(byline.group(1))
        elif i < title and DATE.match(line):
            published = published or to_iso(line)
        elif i < title and (BUTTON.match(line) or LEFTOVER.match(line)):
            continue
        else:
            kept.append(line)
    return "\n".join(kept).strip(), published


def move_head(docs):
    moved = []
    for doc in docs:
        text, published = read_head(doc.page_content)
        moved.append(
            Document(
                page_content=text, metadata={**doc.metadata, "published": published}
            )
        )
    return moved


def own_words(text):
    count = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or TITLE.match(s) or DATE.match(s) or BUTTON.match(s):
            continue
        count += len(WORD.findall(ADDRESS.sub("", s)))
    return count


def drop_app_pages(docs):
    return [doc for doc in docs if not urlparse(doc.metadata["url"]).path.startswith(APP_PAGES)]


def drop_empty_pages(docs):
    return [doc for doc in docs if own_words(doc.page_content) >= MIN_WORDS]


def drop_copies(docs):
    kept = {}
    for doc in sorted(docs, key=lambda d: (len(d.metadata["url"]), d.metadata["url"])):
        text = doc.page_content
        if text in kept:
            kept[text].metadata["also_at"].append(doc.metadata["url"])
        else:
            kept[text] = Document(
                page_content=text, metadata={**doc.metadata, "also_at": []}
            )
    return list(kept.values())
