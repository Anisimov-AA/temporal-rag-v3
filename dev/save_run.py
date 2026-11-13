import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "solution")]
import solution  # noqa: E402

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"


def settings():
    return dict(solution.CONFIG)


def main():
    name, changed = sys.argv[1], sys.argv[2]
    numbers = [int(run.name[:2]) for run in RUNS.glob("*")] if RUNS.exists() else []
    folder = RUNS / f"{max(numbers, default=0) + 1:02d}_{name}"
    folder.mkdir(parents=True)
    for file in ("report.json", "per_question.jsonl"):
        shutil.copy(RESULTS / file, folder / file)
    info = {"changed": changed, "date": date.today().isoformat(), "settings": settings()}
    (folder / "settings.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("saved", folder)


if __name__ == "__main__":
    main()
