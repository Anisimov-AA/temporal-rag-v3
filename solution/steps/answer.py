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
        "gcp:qwen3:32b": 0, "gcp:gpt-oss:20b": 0, "gcp:gemma4:31b": 0}
LLMS = {
    "gemini-3.5-flash-lite": lambda: ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=4, api_key=KEY),
    "gemma-4-31b-it": lambda: ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0, max_retries=4, api_key=KEY),
    "llama3.1:8b": lambda: ChatOllama(model="llama3.1:8b", temperature=0, num_ctx=8192),
    "qwen3:8b": lambda: ChatOllama(model="qwen3:8b", temperature=0, num_ctx=8192, reasoning=False),
    "gcp:qwen3:32b": lambda: ChatOllama(model="qwen3:32b", base_url=GCP, temperature=0, num_ctx=8192, reasoning=False),
    "gcp:gpt-oss:20b": lambda: ChatOllama(model="gpt-oss:20b", base_url=GCP, temperature=0, num_ctx=8192, reasoning="low"),
    "gcp:gemma4:31b": lambda: ChatOllama(model="gemma4:31b", base_url=GCP, temperature=0, num_ctx=8192),
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

PROMPTS = {"plain": PLAIN, "chain_of_thought": CHAIN_OF_THOUGHT, "site_words": SITE_WORDS, "closest": CLOSEST,
           "closest_strict": CLOSEST_STRICT, "looking_for": LOOKING_FOR}


class Reply(BaseModel):
    looking_for: str = ""
    analysis: str
    summary: str = ""
    answer: str
    source: int
    known: bool

    @field_validator("looking_for", "analysis", "summary", "answer", mode="before")
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


def excerpts(chunks):
    parts = []
    for n, chunk in enumerate(chunks, 1):
        m = chunk.metadata
        date = m.get("published") or m["taken_on"]
        parts.append(f"[{n}] {m['url']} · {date}\n{chunk.page_content}")
    return "\n\n".join(parts)


last_call = [0.0]


def reply(llm, gap, prompt):
    wait = gap - (time.time() - last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_call[0] = time.time()
    return Reply(**(llm | JsonOutputParser()).invoke(prompt))


def ask(llm, gap, prompt_name, question, chunks):
    prompt = PROMPTS[prompt_name].format(excerpts=excerpts(chunks), question=question)
    return reply(llm, gap, prompt)
