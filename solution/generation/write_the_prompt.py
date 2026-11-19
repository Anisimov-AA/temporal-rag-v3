PLAIN = """You are the member assistant of Dupaco Community Credit Union.
You answer only from the website excerpts you are given.

Website excerpts:

{excerpts}

Member question: {question}

Reply as JSON: {{"answer": …, "known": true/false}}"""

PROMPTS = {"plain": PLAIN}


def excerpts(chunks):
    pieces = [(c.metadata["url"], c.metadata.get("published") or c.metadata["taken_on"], c.page_content) for c in chunks]
    return "\n\n".join(f"[{n}] {url} · {date}\n{text}" for n, (url, date, text) in enumerate(pieces, 1))


def write_the_prompt(prompt_name, question, chunks):
    return PROMPTS[prompt_name].format(excerpts=excerpts(chunks), question=question)
