import re

ODD_SPACE = re.compile(r"[\xa0\t]")
MANY_SPACES = re.compile(r"(?<=\S) {2,}")
EMPTY_LINES = re.compile(r"\n{3,}")


def tidy_line(line):
    line = ODD_SPACE.sub(" ", line)
    line = MANY_SPACES.sub(" ", line)
    return line.rstrip()


def tidy(text):
    text = "\n".join(tidy_line(line) for line in text.splitlines())
    return EMPTY_LINES.sub("\n\n", text).strip()
