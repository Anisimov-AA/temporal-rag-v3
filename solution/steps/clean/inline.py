import re

URL = r"(?:[^()\s]|\([^()\s]*\))*"
TITLE = r'(?:\s+"[^"]*")?'
IMAGE = re.compile(rf"!\[[^\]]*\]\({URL}{TITLE}\)")


def remove_images(text):
    return IMAGE.sub("", text)
