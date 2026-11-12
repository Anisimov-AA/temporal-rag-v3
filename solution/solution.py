import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from contract import Answer, Snapshot  # noqa: E402
from steps.clean import clean  # noqa: E402
from steps.load import to_documents  # noqa: E402

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


class Knowledge:
    def __init__(self):
        self.docs = {}

    def ingest(self, snapshot: Snapshot):
        loaded = to_documents(snapshot)
        cleaned = clean(loaded, CLEAN_STEPS)
        self.docs[snapshot.snapshot_id] = cleaned
        print(
            f"{snapshot.snapshot_id}: {len(loaded)} pages loaded, {len(cleaned)} after cleaning"
        )

    def answer(self, question, as_of=None):
        return Answer(text="I don't know.", confident=False)

    def conflicts(self):
        return []


def build():
    return Knowledge()
