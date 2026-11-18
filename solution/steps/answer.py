import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, field_validator

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GCP = os.getenv("GCP_OLLAMA", "http://localhost:11435")

GAPS = {"gemini-3.5-flash-lite": 4.5, "gemma-4-31b-it": 35, "llama3.1:8b": 0, "qwen3:8b": 0,
        "gcp:qwen3:32b": 0, "gcp:qwen3:30b-a3b": 0, "gcp:qwen3:30b-a3b-thinking": 0, "gcp:qwen3.6:35b-a3b": 0, "gcp:gpt-oss:20b": 0, "gcp:gemma4:31b": 0}
LLMS = {
    "gemini-3.5-flash-lite": lambda: ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=0, api_key=KEY),
    "gemma-4-31b-it": lambda: ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0, max_retries=4, api_key=KEY),
    "llama3.1:8b": lambda: ChatOllama(model="llama3.1:8b", temperature=0, seed=42, num_ctx=8192),
    "qwen3:8b": lambda: ChatOllama(model="qwen3:8b", temperature=0, seed=42, num_ctx=8192, reasoning=False),
    "gcp:qwen3:32b": lambda: ChatOllama(model="qwen3:32b", base_url=GCP, temperature=0, seed=42, num_ctx=8192, reasoning=False),
    "gcp:qwen3:30b-a3b": lambda: ChatOllama(model="qwen3:30b-a3b-instruct-2507-q4_K_M", base_url=GCP, temperature=0, seed=42, num_ctx=8192, num_predict=1024),
    "gcp:qwen3:30b-a3b-thinking": lambda: ChatOllama(model="qwen3:30b-a3b-thinking-2507-q4_K_M", base_url=GCP, temperature=0, seed=42, num_ctx=16384,
                                                     num_predict=4096, reasoning=True),
    "gcp:qwen3.6:35b-a3b": lambda: ChatOllama(model="qwen3.6:35b-a3b", base_url=GCP, temperature=0, seed=42, num_ctx=8192, num_predict=1024, reasoning=False),
    "gcp:gpt-oss:20b": lambda: ChatOllama(model="gpt-oss:20b", base_url=GCP, temperature=0, seed=42, num_ctx=8192, reasoning="low"),
    "gcp:gemma4:31b": lambda: ChatOllama(model="gemma4:31b", base_url=GCP, temperature=0, seed=42, num_ctx=8192),
}
DONT_KNOW = "I don't know. The Dupaco website does not say this. Please contact Dupaco and a member services rep can help."

PLAIN = """You are the member assistant of Dupaco Community Credit Union.
You answer only from the website excerpts you are given.

Website excerpts:

{excerpts}

Member question: {question}

Reply as JSON: {{"analysis": …, "answer": …, "source": <excerpt number>, "known": true/false}}"""

CHAIN_OF_THOUGHT = """You are the member assistant of Dupaco Community Credit Union.
You answer only from the website excerpts you are given.

Before the final answer, think out loud, step by step. Pay special attention to the wording of the question.
- The excerpt with the answer may use other words than the question.
- A value can look like the answer but belong to a similar product, another date or another branch: check that it is the one asked for.

Website excerpts:

{excerpts}

Member question: {question}

Reply as JSON: {{"analysis": <step by step, at least 5 steps and 150 words>, "summary": <the reasoning in about 50 words>, "answer": …, "source": <excerpt number>, "known": true/false}}"""

SITE_WORDS = PLAIN.replace("You answer only from the website excerpts you are given.\n",
                           "You answer only from the website excerpts you are given.\n"
                           "Write numbers, counts and names with the same words as the excerpts.\n")

CLOSEST = PLAIN.replace("You answer only from the website excerpts you are given.\n",
                        "You answer only from the website excerpts you are given.\n"
                        "If the excerpts have no exact answer but a close one (an older year, one branch, a news post), "
                        "give it and say where it comes from and its date.\n"
                        "Example: the question asks for this year, an excerpt from 2023 says \"N members\" "
                        "→ answer \"As of 2023, N members\", known = true.\n"
                        "Set known to false only when no excerpt is about the question.\n")

CLOSEST_STRICT = CLOSEST.replace("Set known to false only when no excerpt is about the question.\n",
                                 "Set known to false when no excerpt gives the value asked for, even if the excerpts describe the product.\n")

LOOKING_FOR = (CLOSEST_STRICT
               .replace("Set known to false when no excerpt gives the value asked for, even if the excerpts describe the product.\n",
                        "First write what the member wants, in a few words (looking_for).\n"
                        "Set known to true only if an excerpt gives that value, even an older or partial one. "
                        "Describing the product is not enough.\n")
               .replace('Reply as JSON: {{"analysis"', 'Reply as JSON: {{"looking_for": …, "analysis"'))

QUOTE = (LOOKING_FOR
         .replace("First write what the member wants, in a few words (looking_for).\n",
                  "First write what the member wants, in a few words (looking_for).\n"
                  "Then copy the exact words of the excerpts that give it (quote), or \"\" if none do.\n")
         .replace('{{"looking_for": …, "analysis"', '{{"looking_for": …, "quote": …, "analysis"'))

EXAMPLES = """Example 1
Excerpt [2] a news post · 2022-09-07: "Branch hours: Monday-Friday N a.m.-N p.m."
Question: What are your hours?
Reply: {{"analysis": "Only one branch's hours, from a 2022 news post.", "answer": "Per a 2022 news post, the X branch is open Monday-Friday N a.m.-N p.m.", "source": 2, "known": true}}

Example 2
Excerpt [1] a product page: "The Y loan has a fixed rate for the first year."
Question: What is the rate on the Y loan?
Reply: {{"analysis": "The product is described, but no rate is given.", "answer": "", "source": 1, "known": false}}

"""

FEW_SHOT = CLOSEST.replace("Website excerpts:\n", EXAMPLES + "Website excerpts:\n", 1)

TOP = "You answer only from the website excerpts you are given.\n"


def plain_plus(rule, fields=""):
    prompt = PLAIN.replace(TOP, TOP + rule, 1)
    return prompt.replace('Reply as JSON: {{"analysis"', 'Reply as JSON: {{' + fields + '"analysis"', 1)


ONLY_STRICT = plain_plus("Set known to false when no excerpt gives the value asked for, even if the excerpts describe the product.\n")
ONLY_LOOKING_FOR = plain_plus("First write what the member wants, in a few words (looking_for).\n", '"looking_for": …, ')
ONLY_QUOTE = plain_plus("Copy the exact words of the excerpts that give the answer (quote), or \"\" if none do.\n", '"quote": …, ')
ONLY_FEW_SHOT = PLAIN.replace("Website excerpts:\n", EXAMPLES + "Website excerpts:\n", 1)

NEWEST = CLOSEST.replace(TOP, TOP + "If the excerpts disagree, trust the one with the newest date and say what changed.\n", 1)
COMBINE = CLOSEST.replace(TOP, TOP + "Build the answer from every excerpt that has a part of it, not from one excerpt only.\n", 1)
TODAY = CLOSEST.replace(TOP, TOP + "Today is {today}.\n", 1)

MAX = (QUOTE.replace(TOP, TOP + "Today is {today}.\n"
                         "If the excerpts disagree, trust the one with the newest date and say what changed.\n"
                         "Build the answer from every excerpt that has a part of it, not from one excerpt only.\n", 1)
       .replace("Website excerpts:\n", EXAMPLES + "Website excerpts:\n", 1))

PROMPTS = {"plain": PLAIN, "chain_of_thought": CHAIN_OF_THOUGHT, "site_words": SITE_WORDS, "closest": CLOSEST,
           "closest_strict": CLOSEST_STRICT, "looking_for": LOOKING_FOR, "quote": QUOTE, "few_shot": FEW_SHOT,
           "only_strict": ONLY_STRICT, "only_looking_for": ONLY_LOOKING_FOR, "only_quote": ONLY_QUOTE, "only_few_shot": ONLY_FEW_SHOT,
           "newest": NEWEST, "combine": COMBINE, "today": TODAY, "max": MAX}


class Reply(BaseModel):
    looking_for: str = ""
    quote: str = ""
    analysis: str
    summary: str = ""
    answer: str
    source: int
    known: bool

    @field_validator("looking_for", "quote", "analysis", "summary", "answer", mode="before")
    @classmethod
    def as_text(cls, value):
        if isinstance(value, list):
            return "\n".join(str(x) for x in value)
        return "" if value is None else str(value)

    @field_validator("source", mode="before")
    @classmethod
    def first_number(cls, value):
        found = re.search(r"\d+", str(value))
        return int(found.group()) if found else 0


VALUE = re.compile(r"\d[\d,]*(?:\.\d+)?%?|\d{1,2}:\d{2}")


def with_value(answer, pages):
    values = {v for v in VALUE.findall(answer) if len(v) > 1}
    return [url for url, text in pages if any(v in text for v in values)]


KINDS = (("/newsroom/", "news post"), ("/blog/", "blog post"), (".pdf", "PDF"), ("/rates", "rates page"))


def kind(url):
    return next((name for part, name in KINDS if part in url), "page")


def newest_first(pieces, today):
    ordered = sorted(pieces, key=lambda p: p[1], reverse=True)
    parts = []
    for n, (url, date, text) in enumerate(ordered, 1):
        when = f"page as of {date}" if date == today else f"published {date}"
        parts.append(f"[{n}] {kind(url)} · {when} · {url}\n{text}")
    return "\n\n".join(parts), [url for url, _, _ in ordered]


def excerpts(chunks, order="score", today=""):
    pieces = [(c.metadata["url"], c.metadata.get("published") or c.metadata["taken_on"], c.page_content) for c in chunks]
    if order == "newest":
        return newest_first(pieces, today)[0]
    return "\n\n".join(f"[{n}] {url} · {date}\n{text}" for n, (url, date, text) in enumerate(pieces, 1))


last_call = [0.0]


def reply(llm, gap, prompt):
    wait = gap - (time.time() - last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_call[0] = time.time()
    return Reply(**(llm | JsonOutputParser()).invoke(prompt))


def ask(llm, gap, prompt_name, question, chunks, today="", order="score"):
    prompt = PROMPTS[prompt_name].format(excerpts=excerpts(chunks, order, today), question=question, today=today)
    return reply(llm, gap, prompt)
