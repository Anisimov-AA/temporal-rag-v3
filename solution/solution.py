import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from contract import Answer, Snapshot, Source  # noqa: E402
from steps.answer import DONT_KNOW, GAPS, LLMS, ask, with_value  # noqa: E402
from steps.clean import clean  # noqa: E402
from steps.index import EMBEDDINGS, build_index, fingerprint, index_docs  # noqa: E402
from steps.load import to_documents  # noqa: E402
from steps.search import Keywords, fuse, rerank  # noqa: E402
from steps.split import split_into_chunks  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parents[1] / "index"
SPLITS = {"sections": split_into_chunks}
SEARCHES = ("vectors + bm25", "vectors", "bm25", "vectors + bm25 + rerank")
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
    "split": "sections",
    "chunk_size": 4000,
    "embedding": "bge-m3",
    "search": "vectors + bm25 + rerank",
    "candidates": 20,
    "top_k": 5,
    "llm": "gcp:qwen3:30b-a3b",
    "prompt": "closest",
    "sources": "with_value",
}


class Knowledge:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.embeddings = EMBEDDINGS[self.config["embedding"]]()
        self.llm = LLMS[self.config["llm"]]()
        self.stores = {}
        self.keywords = {}
        self.dates = {}

    def ingest(self, snapshot: Snapshot):
        c = self.config
        loaded = to_documents(snapshot)
        cleaned = clean(loaded, c["clean"])
        chunks = SPLITS[c["split"]](cleaned, c["chunk_size"])
        docs = index_docs(chunks)
        name = f"{snapshot.snapshot_id}__{c['embedding']}__{fingerprint(docs)}"
        store, how = build_index(docs, name, self.embeddings, INDEX_DIR)
        self.stores[snapshot.snapshot_id] = store
        self.keywords[snapshot.snapshot_id] = Keywords(docs)
        self.dates[snapshot.snapshot_id] = snapshot.taken_on
        print(
            f"{snapshot.snapshot_id}: {len(loaded)} pages loaded, "
            f"{len(cleaned)} after cleaning, {len(chunks)} chunks, index {how}"
        )

    def search(self, question, snapshot_id):
        c = self.config
        lists = []
        if "vectors" in c["search"]:
            lists.append(self.stores[snapshot_id].similarity_search(question, k=c["candidates"]))
        if "bm25" in c["search"]:
            lists.append(self.keywords[snapshot_id].search(question, c["candidates"]))
        found = fuse(lists)
        if "rerank" in c["search"]:
            found = rerank(question, found)
        return found[: c["top_k"]]

    def pick(self, as_of):
        days = {sid: day for sid, day in self.dates.items() if as_of is None or day <= as_of}
        return max(days, key=days.get) if days else min(self.dates, key=self.dates.get)

    def answer(self, question, as_of=None):
        snapshot_id = self.pick(as_of)
        chunks = self.search(question, snapshot_id)
        reply = ask(self.llm, GAPS[self.config["llm"]], self.config["prompt"], question, chunks)
        if not reply.known:
            return Answer(text=DONT_KNOW, confident=False)
        used = chunks[reply.source - 1] if 1 <= reply.source <= len(chunks) else chunks[0]
        urls = [used.metadata["url"]]
        if self.config.get("sources") == "with_value":
            pages = [(c.metadata["url"], c.metadata["url"] + "\n" + c.page_content) for c in chunks]
            urls += [u for u in with_value(reply.answer, pages) if u not in urls]
        return Answer(text=reply.answer, sources=[Source(url=u, snapshot_id=snapshot_id) for u in urls])

    def conflicts(self):
        return []


def build(config=None):
    return Knowledge(config)
