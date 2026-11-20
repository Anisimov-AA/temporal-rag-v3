import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from contract import Answer, Snapshot  # noqa: E402
from find_conflicts import find_conflicts, rate_table  # noqa: E402
from generation.ask_the_llm import GAPS, LLMS, ask_the_llm  # noqa: E402
from generation.check_the_answer import check_the_answer  # noqa: E402
from generation.cite_the_sources import cite_the_sources  # noqa: E402
from generation.write_the_prompt import FORMATS, write_the_prompt  # noqa: E402
from indexing.clean_pages import clean_pages  # noqa: E402
from indexing.index_chunks import EMBEDDINGS, Keywords, build_index, fingerprint, index_docs  # noqa: E402
from indexing.load_pages import load_pages  # noqa: E402
from indexing.split_into_chunks import split_into_chunks  # noqa: E402
from retrieval.drop_old_versions import drop_old_versions  # noqa: E402
from retrieval.keep_top_k import keep_top_k  # noqa: E402
from retrieval.reorder_by_date import reorder_by_date  # noqa: E402
from retrieval.rerank_chunks import rerank_chunks  # noqa: E402
from retrieval.retrieve_chunks import retrieve_chunks  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parents[1] / "index"
DONT_KNOW = "I don't know. The Dupaco website does not say this. Please contact Dupaco and a member services rep can help."
CONFIG = {
    "clean": [
        "images",
        "links",
        "markdown_noise",
        "html_noise",
        "blocks",
        "repeated_lines",
        "tidy",
        "head",
        "app_pages",
        "empty_pages",
        "copies",
    ],
    "chunk_size": 4000,
    "embedding": "bge-m3",
    "search": "vectors + bm25 + rerank",
    "candidates": 20,
    "top_k": 5,
    "dates": "newest effective first",
    "prompt": "json_known_pages",
    "llm": "gcp:gemma4:26b",
}


class Knowledge:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.embeddings = EMBEDDINGS[self.config["embedding"]]()
        self.llm = LLMS[self.config["llm"]]()
        self.stores = {}
        self.keywords = {}
        self.dates = {}
        self.rates = []

    def ingest(self, snapshot: Snapshot):
        c = self.config
        pages = load_pages(snapshot)
        self.rates += rate_table(pages)
        cleaned = clean_pages(pages, c["clean"])
        chunks = split_into_chunks(cleaned, c["chunk_size"])
        docs = index_docs(chunks)
        name = f"{snapshot.snapshot_id}__{c['embedding']}__{fingerprint(docs)}"
        store, how = build_index(docs, name, self.embeddings, INDEX_DIR)
        self.stores[snapshot.snapshot_id] = store
        self.keywords[snapshot.snapshot_id] = Keywords(docs)
        self.dates[snapshot.snapshot_id] = snapshot.taken_on
        print(
            f"{snapshot.snapshot_id}: {len(pages)} pages loaded, "
            f"{len(cleaned)} after cleaning, {len(chunks)} chunks, index {how}"
        )
        self.warm_up(store, docs)

    def warm_up(self, store, docs):
        store.similarity_search("warm up", k=1)
        rerank_chunks("warm up", docs[:2])
        self.llm.invoke("Reply OK")

    def pick(self, as_of):
        days = {sid: day for sid, day in self.dates.items() if as_of is None or day <= as_of}
        return max(days, key=days.get) if days else min(self.dates, key=self.dates.get)

    def search(self, question, snapshot_id):
        c = self.config
        chunks = retrieve_chunks(self.stores[snapshot_id], self.keywords[snapshot_id], question, c["candidates"], c["search"])
        if "rerank" in c["search"]:
            chunks = rerank_chunks(question, chunks)
        if c.get("old_versions") == "drop":
            chunks = drop_old_versions(chunks)
        chunks = keep_top_k(chunks, c["top_k"])
        if c["dates"] == "newest effective first":
            chunks = reorder_by_date(chunks)
        return chunks

    def answer(self, question, as_of=None):
        chunks = self.search(question, self.pick(as_of))
        prompt = write_the_prompt(self.config["prompt"], question, chunks)
        reply = ask_the_llm(self.llm, GAPS[self.config["llm"]], prompt, FORMATS[self.config["prompt"]])
        reply = check_the_answer(question, reply)
        if not reply.known:
            return Answer(text=DONT_KNOW, confident=False)
        return Answer(text=reply.answer, sources=cite_the_sources(chunks, reply.pages))

    def conflicts(self):
        return find_conflicts(self.rates)


def build(config=None):
    return Knowledge(config)
