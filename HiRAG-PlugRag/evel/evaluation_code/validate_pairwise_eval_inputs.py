"""Verify that a pairwise evaluation receives aligned questions and answers."""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def get_query(row: dict) -> str:
    return str(row.get("query") or row.get("question") or row.get("input") or "").strip()


def get_answer(row: dict) -> str:
    return str(row.get("answer") or row.get("result") or "").strip()


def validate(label: str, queries: list[str], answers: list[dict], expected_count: int | None) -> None:
    if expected_count is not None and len(answers) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} rows, found {len(answers)}")
    if len(answers) != len(queries):
        raise ValueError(f"{label}: answer/query count mismatch: {len(answers)} vs {len(queries)}")
    for index, (query, row) in enumerate(zip(queries, answers), start=1):
        if not get_answer(row):
            raise ValueError(f"{label}: empty answer at row {index}")
        if get_query(row) != query:
            raise ValueError(f"{label}: query mismatch at row {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ordered pairwise evaluation inputs.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--answer1-file", required=True)
    parser.add_argument("--answer2-file", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()

    query_rows = load_jsonl(Path(args.query_file))
    queries = [get_query(row) for row in query_rows]
    if not queries or any(not query for query in queries):
        raise ValueError("Query file has no usable ordered queries")

    first = load_jsonl(Path(args.answer1_file))
    second = load_jsonl(Path(args.answer2_file))
    validate("answer1", queries, first, args.expected_count)
    validate("answer2", queries, second, args.expected_count)
    print(json.dumps({"status": "ok", "queries": len(queries), "answer1": len(first), "answer2": len(second)}))


if __name__ == "__main__":
    main()
