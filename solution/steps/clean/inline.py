import re
from urllib.parse import urlparse

URL = r"(?:[^()\s]|\([^()\s]*\))*"
TITLE = r'(?:\s+"[^"]*")?'
IMAGE = re.compile(rf"!\[[^\]]*\]\({URL}{TITLE}\)")
LINK = re.compile(rf"\[([^\]]*)\]\(({URL}){TITLE}\)")


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
