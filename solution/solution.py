import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from contract import Answer, Snapshot  # noqa: E402
from steps.clean import clean  # noqa: E402
from steps.index import EMBEDDINGS, build_index  # noqa: E402
from steps.load import to_documents  # noqa: E402
from steps.split import split_into_chunks  # noqa: E402

CLEAN_STEPS = [
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
]
CHUNK_SIZE = 4000
EMBEDDING = "bge-m3"
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"


class Knowledge:
    def __init__(self):
        self.embeddings = EMBEDDINGS[EMBEDDING]()
        self.stores = {}

    def ingest(self, snapshot: Snapshot):
        loaded = to_documents(snapshot)
        cleaned = clean(loaded, CLEAN_STEPS)
        chunks = split_into_chunks(cleaned, CHUNK_SIZE)
        name = f"{snapshot.snapshot_id}__{EMBEDDING}"
        store, how = build_index(chunks, name, self.embeddings, INDEX_DIR)
        self.stores[snapshot.snapshot_id] = store
        print(
            f"{snapshot.snapshot_id}: {len(loaded)} pages loaded, "
            f"{len(cleaned)} after cleaning, {len(chunks)} chunks, index {how}"
        )

    def answer(self, question, as_of=None):
        return Answer(text="I don't know.", confident=False)

    def conflicts(self):
        return []


def build():
    return Knowledge()
