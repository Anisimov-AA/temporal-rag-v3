import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from contract import Answer, Snapshot  # noqa: E402
from steps.clean import clean  # noqa: E402
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


class Knowledge:
    def __init__(self):
        self.chunks = {}

    def ingest(self, snapshot: Snapshot):
        loaded = to_documents(snapshot)
        cleaned = clean(loaded, CLEAN_STEPS)
        chunks = split_into_chunks(cleaned, CHUNK_SIZE)
        self.chunks[snapshot.snapshot_id] = chunks
        print(
            f"{snapshot.snapshot_id}: {len(loaded)} pages loaded, "
            f"{len(cleaned)} after cleaning, {len(chunks)} chunks"
        )

    def answer(self, question, as_of=None):
        return Answer(text="I don't know.", confident=False)

    def conflicts(self):
        return []


def build():
    return Knowledge()
