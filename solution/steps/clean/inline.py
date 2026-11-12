import re
from urllib.parse import urlparse

URL = r"(?:[^()\s]|\([^()\s]*\))*"
TITLE = r'(?:\s+"[^"]*")?'
IMAGE = re.compile(rf"!\[[^\]]*\]\({URL}{TITLE}\)")
LINK = re.compile(rf"\[([^\]]*)\]\(({URL}){TITLE}\)")

BOLD = re.compile(
    r"\*\*(?=\S)([^\n]*?\w[^\n]*?)(?<=\S)\*\*|__(?=\S)([^\n]*?[A-Za-z][^\n]*?)(?<=\S)__"
)
ITALIC = re.compile(
    r"(?<![*\w\\])\*(?=\S)([^*\n]*?\w[^*\n]*?)(?<=\S)\*(?![*\w])|(?<![_\w\\])_(?=[^\s_])([^_\n]*?[A-Za-z][^_\n]*?)(?<=[^\s_])_(?![_\w])"
)
STRIKE = re.compile(r"~~[^~\n]+~~")
CODE = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"(?m)^\s*```.*$")
DIVIDER = re.compile(r"(?m)^\s*([-*_])(\s*\1){2,}\s*$")
LINE_END = re.compile(r"(?m)\\+\s*$")
ESCAPED = re.compile(r"\\([*_\[\]()`~!])")


def remove_images(text):
    return IMAGE.sub("", text)


def short_address(url):
    parts = urlparse(url)
    if parts.scheme in ("mailto", "tel"):
        return parts.path
    if not parts.netloc or url.endswith("#"):
        return ""
    return parts.netloc.removeprefix("www.") + parts.path.rstrip("/")


def link_text(match):
    text, address = match.group(1).strip(), short_address(match.group(2))
    if not address:
        return text
    return f"{text} ({address})" if text else address


def links_to_text(text):
    return LINK.sub(link_text, text)


def remove_markdown_noise(text):
    text = BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = ITALIC.sub(lambda m: m.group(1) or m.group(2), text)
    text = STRIKE.sub("", text)
    text = CODE.sub(r"\1", text)
    text = FENCE.sub("", text)
    text = DIVIDER.sub("", text)
    text = LINE_END.sub("", text)
    return ESCAPED.sub(r"\1", text)
