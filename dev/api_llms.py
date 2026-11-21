import os

from generation.ask_the_llm import GAPS, LLMS

NVIDIA = "https://integrate.api.nvidia.com/v1"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
NVIDIA_MODELS = {
    "nvidia:gemma-4-31b": "google/gemma-4-31b-it",
    "nvidia:nemotron-3.5-bolt": "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia:nemotron-3.5-lightning": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia:nemotron-3.5-super": "nvidia/nemotron-3.5-super-vl-120b-a12b",
}


def nvidia(model):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, base_url=NVIDIA, api_key=os.getenv("NVIDIA_API_KEY"), temperature=0, seed=42,
                      max_tokens=2048, extra_body=NO_THINKING, timeout=120, max_retries=2)


def gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=0, api_key=key)


for name, model in NVIDIA_MODELS.items():
    LLMS[name] = lambda model=model: nvidia(model)
    GAPS[name] = 1.6
LLMS["gemini-3.5-flash-lite"] = gemini
GAPS["gemini-3.5-flash-lite"] = 4.5
