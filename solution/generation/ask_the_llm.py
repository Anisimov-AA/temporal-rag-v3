import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, field_validator

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GCP = os.getenv("GCP_OLLAMA", "http://localhost:11435")
THREADS = int(os.getenv("OLLAMA_THREADS", "0")) or None
NVIDIA = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

GAPS = {"gemini-3.5-flash-lite": 4.5, "gcp:gemma4:26b": 0, "gcp:qwen3:32b": 0, "nvidia:nemotron-3-ultra": 1.6}
LLMS = {
    "gemini-3.5-flash-lite": lambda: ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=0, api_key=KEY),
    "gcp:gemma4:26b": lambda: ChatOllama(model="gemma4:26b-a4b-it-q4_K_M", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=16384, num_predict=1024, reasoning=False),
    "gcp:qwen3:32b": lambda: ChatOllama(model="qwen3:32b", base_url=GCP, temperature=0, seed=42, num_thread=THREADS, num_ctx=16384, reasoning=False),
    "nvidia:nemotron-3-ultra": lambda: ChatOpenAI(model="nvidia/nemotron-3-ultra-550b-a55b", base_url=NVIDIA, api_key=NVIDIA_KEY, temperature=0, seed=42, max_tokens=2048,
                                                  extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
}


class Reply(BaseModel):
    answer: str
    known: bool

    @field_validator("answer", mode="before")
    @classmethod
    def as_text(cls, value):
        if isinstance(value, list):
            return "\n".join(str(x) for x in value)
        return "" if value is None else str(value)


BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def parse_reply(text):
    try:
        return Reply(**JsonOutputParser().parse(BAD_ESCAPE.sub("", text)))
    except (OutputParserException, ValueError, TypeError):
        return Reply(answer=text.strip(), known=True)


last_call = [0.0]


def ask_the_llm(llm, gap, prompt):
    wait = gap - (time.time() - last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_call[0] = time.time()
    return parse_reply(llm.invoke(prompt).content)
