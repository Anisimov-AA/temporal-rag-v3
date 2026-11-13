import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

GAPS = {"gemini-3.5-flash-lite": 4.5, "gemma-4-31b-it": 2.5, "llama3.1:8b": 0}
LLMS = {
    "gemini-3.5-flash-lite": lambda: ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=4, api_key=KEY),
    "gemma-4-31b-it": lambda: ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0, max_retries=4, api_key=KEY),
    "llama3.1:8b": lambda: ChatOllama(model="llama3.1:8b", temperature=0, num_ctx=8192),
}
DONT_KNOW = "I don't know. The Dupaco website does not say this. Please contact Dupaco and a member services rep can help."

PLAIN = """You are the member assistant of Dupaco Community Credit Union.
You answer only from the website excerpts you are given.

Website excerpts:

{excerpts}

Member question: {question}

Reply as JSON: {{"analysis": …, "answer": …, "source": <excerpt number>, "known": true/false}}"""

COT = """You are the member assistant of Dupaco Community Credit Union.
You answer only from the website excerpts you are given.

Before the final answer, think out loud, step by step. Pay special attention to the wording of the question.
- The excerpt with the answer may use other words than the question.
- A value can look like the answer but belong to a similar product, another date or another branch: check that it is the one asked for.

Website excerpts:

{excerpts}

Member question: {question}

Reply as JSON: {{"analysis": <step by step, at least 5 steps and 150 words>, "summary": <the reasoning in about 50 words>, "answer": …, "source": <excerpt number>, "known": true/false}}"""

PROMPTS = {"plain": PLAIN, "cot": COT}


class Reply(BaseModel):
    analysis: str
    summary: str = ""
    answer: str
    source: int
    known: bool


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
