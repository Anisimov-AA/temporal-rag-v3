import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, field_validator

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
GCP = os.getenv("GCP_OLLAMA", "http://localhost:11435")
THREADS = int(os.getenv("OLLAMA_THREADS", "0")) or None

GAPS = {"gcp:gemma4:26b": 0, "gcp:qwen3:30b-a3b": 0, "gcp:gemma4:31b-qat": 0, "gcp:qwen3.6:27b": 0, "gcp:qwen3:32b": 0}
LLMS = {
    "gcp:gemma4:26b": lambda: ChatOllama(model="gemma4:26b-a4b-it-q4_K_M", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=8192, num_predict=1024, reasoning=False),
    "gcp:qwen3:30b-a3b": lambda: ChatOllama(model="qwen3:30b-a3b-instruct-2507-q4_K_M", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=8192, num_predict=1024),
    "gcp:gemma4:31b-qat": lambda: ChatOllama(model="gemma4:31b-it-qat", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=8192, num_predict=1024, reasoning=False),
    "gcp:qwen3:32b": lambda: ChatOllama(model="qwen3:32b", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=8192, num_predict=1024, reasoning=False),
    "gcp:qwen3.6:27b": lambda: ChatOllama(model="qwen3.6:27b-q4_K_M", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=8192, num_predict=1024, reasoning=False),
}


class Reply(BaseModel):
    answer: str
    known: bool
    pages: list[int] = []

    @field_validator("answer", mode="before")
    @classmethod
    def as_text(cls, value):
        if isinstance(value, list):
            return "\n".join(str(x) for x in value)
        return "" if value is None else str(value)

    @field_validator("pages", mode="before")
    @classmethod
    def as_numbers(cls, value):
        if not isinstance(value, list):
            return []
        return [int(x) for x in value if str(x).strip().isdigit()]


BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


REFUSAL = re.compile(r"^\W*i (?:don['’]?t|do not) know\b", re.I)


def parse_reply(text, reply_format="json"):
    if reply_format == "text":
        return Reply(answer=text.strip(), known=not REFUSAL.match(text))
    try:
        return Reply(**JsonOutputParser().parse(BAD_ESCAPE.sub("", text)))
    except (OutputParserException, ValueError, TypeError):
        return Reply(answer=text.strip(), known=True)


def reply_text(message):
    content = message.content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content


last_call = [0.0]


def ask_the_llm(llm, gap, prompt, reply_format="json"):
    wait = gap - (time.time() - last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_call[0] = time.time()
    return parse_reply(reply_text(llm.invoke(prompt)), reply_format)
