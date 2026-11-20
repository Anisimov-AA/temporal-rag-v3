ROLE = """You are the member assistant of Dupaco Community Credit Union.
You answer only from the website excerpts you are given.
"""

CLEAR = """Start with the current value for the date of the question in one sentence. Do not list older values unless the member asks about the past. If the question asks for several products, terms or rates, give every one from the excerpts as a short list. No headings, no bold, nothing the member did not ask for.
"""

KNOWN_JSON = """If the excerpts do not answer the question, set "known" to false.
"""

KNOWN_TEXT = """If the excerpts do not answer the question, reply exactly: I don't know.
"""

BODY = """
Website excerpts:

{excerpts}

Member question: {question}
"""

JSON = """
Reply as JSON: {{"answer": …, "known": true/false}}"""

REASON = """
Reply as JSON: {{"analysis": …, "answer": …, "known": true/false}}"""

PAGES = """
Reply as JSON: {{"answer": …, "known": true/false, "pages": [numbers of the excerpts you used]}}"""

PROMPTS = {
    "text": ROLE + BODY,
    "text_known": ROLE + KNOWN_TEXT + BODY,
    "json": ROLE + BODY + JSON,
    "json_known": ROLE + KNOWN_JSON + BODY + JSON,
    "json_known_clear": ROLE + CLEAR + KNOWN_JSON + BODY + JSON,
    "text_known_clear": ROLE + CLEAR + KNOWN_TEXT + BODY,
    "reason": ROLE + BODY + REASON,
    "json_known_pages": ROLE + KNOWN_JSON + BODY + PAGES,
}
FORMATS = {name: "text" if name.startswith("text") else "json" for name in PROMPTS}


def excerpts(chunks):
    pieces = [(c.metadata["url"], c.metadata.get("published") or c.metadata["taken_on"], c.page_content) for c in chunks]
    return "\n\n".join(f"[{n}] {url} · {date}\n{text}" for n, (url, date, text) in enumerate(pieces, 1))


def write_the_prompt(prompt_name, question, chunks):
    return PROMPTS[prompt_name].format(excerpts=excerpts(chunks), question=question)
