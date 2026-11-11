import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from contract import Answer, Snapshot  # noqa: E402
from steps.load import to_documents  # noqa: E402


class Knowledge:
    def __init__(self):
        self.docs = {}

    def ingest(self, snapshot: Snapshot):
        self.docs[snapshot.snapshot_id] = to_documents(snapshot)
        print(
            f"loaded {snapshot.snapshot_id}: {len(self.docs[snapshot.snapshot_id])} documents"
        )

    def answer(self, question, as_of=None):
        return Answer(text="I don't know.", confident=False)

    def conflicts(self):
        return []


def build():
    return Knowledge()
