import argparse
import csv
import json
import re
from pathlib import Path


METRICS = [
    "Comprehensiveness",
    "Empowerment",
    "Diversity",
    "Overall Winner",
]

DATASETS = ["mix", "cs", "legal", "agriculture"]


def strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(text):
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found")


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                rows.append({"_error": f"line {line_no}: {exc}"})
                continue
            rows.append(item)
    return rows


def get_custom_id_index(item):
    custom_id = item.get("custom_id", "")
    match = re.search(r"(\d+)$", str(custom_id))
    return int(match.group(1)) if match else None


def normalize_eval_item(item):
    if not isinstance(item, dict):
        return None

    if all(metric in item for metric in METRICS):
        return item

    content = None
    try:
        content = item["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        pass

    if content is None:
        for key in ("content", "message", "text", "output"):
            if isinstance(item.get(key), str):
                content = item[key]
                break

    if content is None:
        return None

    try:
        parsed = extract_json_object(content)
    except Exception:
        return None

    if isinstance(parsed, dict) and any(metric in parsed for metric in METRICS):
        return parsed
    return None


def winner_side(winner):
    winner = str(winner or "").strip().lower()
    winner = winner.replace("answer1", "answer 1").replace("answer2", "answer 2")
    if "answer 1" in winner:
        return "answer1"
    if "answer 2" in winner:
        return "answer2"
    if "tie" in winner or "draw" in winner:
        return "tie"
    return "unknown"


def infer_dataset(path):
    text = str(path).lower()
    for dataset in DATASETS:
        if re.search(rf"(^|[\\/_.-]){re.escape(dataset)}([\\/_.-]|$)", text):
            return dataset
    return ""


def clean_stem(path):
    stem = path.stem
    suffixes = [
        "_result_openai",
        "_result_deepseek",
        "_result_glm",
        "_result",
        "_openai",
        "_deepseek",
        "_glm",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def infer_comparison(path):
    stem = clean_stem(path)
    lowered = stem.lower()
    for token in ("_vs_", "-vs-", " vs "):
        if token in lowered:
            index = lowered.index(token)
            return stem[:index], stem[index + len(token) :]
    return stem, ""


def summarize_file(path, left_label=None, right_label=None):
    raw_rows = load_jsonl(path)
    sortable = [(get_custom_id_index(item), item) for item in raw_rows if isinstance(item, dict)]
    if sortable and all(index is not None for index, _ in sortable):
        raw_rows = [item for _, item in sorted(sortable, key=lambda pair: pair[0])]

    eval_rows = []
    parse_errors = 0
    for item in raw_rows:
        eval_item = normalize_eval_item(item)
        if eval_item is None:
            parse_errors += 1
        eval_rows.append(eval_item)

    if not any(row is not None for row in eval_rows):
        return None

    total = len(raw_rows)
    midpoint = total // 2
    guessed_left, guessed_right = infer_comparison(path)
    left = left_label or guessed_left or "SystemA"
    right = right_label or guessed_right or "SystemB"

    rows = []
    for metric in METRICS:
        counts = {left: 0, right: 0, "Tie": 0, "Unknown": 0}
        for index, item in enumerate(eval_rows):
            if item is None:
                counts["Unknown"] += 1
                continue
            side = winner_side(item.get(metric, {}).get("Winner", ""))
            swapped = index >= midpoint
            if side == "answer1":
                counts[right if swapped else left] += 1
            elif side == "answer2":
                counts[left if swapped else right] += 1
            elif side == "tie":
                counts["Tie"] += 1
            else:
                counts["Unknown"] += 1

        rows.append(
            {
                "file": str(path),
                "dataset": infer_dataset(path),
                "comparison": f"{left} vs {right}" if right else left,
                "left_label": left,
                "right_label": right,
                "metric": metric,
                "total": total,
                "midpoint": midpoint,
                "left_wins": counts[left],
                "left_pct": pct(counts[left], total),
                "right_wins": counts[right],
                "right_pct": pct(counts[right], total),
                "tie": counts["Tie"],
                "tie_pct": pct(counts["Tie"], total),
                "unknown": counts["Unknown"],
                "parse_errors": parse_errors,
            }
        )
    return rows


def pct(count, total):
    if total == 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def candidate_files(root, include_all=False):
    for path in root.rglob("*.jsonl"):
        name = path.name.lower()
        if include_all or "result_" in name or name.endswith("_result.jsonl"):
            yield path


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["file"], []).append(row)

    lines = ["# Evaluation Result Summary", ""]
    for file_path, items in grouped.items():
        head = items[0]
        lines.append(f"## {Path(file_path).name}")
        lines.append("")
        lines.append(f"- File: `{file_path}`")
        lines.append(f"- Dataset: `{head['dataset'] or 'unknown'}`")
        lines.append(f"- Comparison: `{head['comparison']}`")
        lines.append(f"- Total eval rows: `{head['total']}`")
        lines.append("")
        lines.append("| Metric | Left % (n) | Right % (n) | Tie % (n) | Unknown |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in items:
            lines.append(
                "| {metric} | {left_pct:.2f}% ({left_wins}) | {right_pct:.2f}% ({right_wins}) | "
                "{tie_pct:.2f}% ({tie}) | {unknown} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_target(text):
    parts = text.split(":")
    if len(parts) == 2:
        dataset, values = parts
        metric = "Overall Winner"
    elif len(parts) == 3:
        dataset, metric, values = parts
    else:
        raise argparse.ArgumentTypeError(
            "target format: dataset:left_pct,right_pct or dataset:metric:left_pct,right_pct"
        )

    try:
        left_pct, right_pct = [float(x.strip()) for x in values.split(",", 1)]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"bad target percentages: {exc}")

    return {
        "dataset": dataset.strip().lower(),
        "metric": metric.strip(),
        "left_pct": left_pct,
        "right_pct": right_pct,
    }


def match_targets(rows, targets, allow_reverse=True):
    matches = []
    for target in targets:
        for row in rows:
            if target["dataset"] not in ("", "any") and row["dataset"] != target["dataset"]:
                continue
            if row["metric"].lower() != target["metric"].lower():
                continue

            direct = abs(row["left_pct"] - target["left_pct"]) + abs(row["right_pct"] - target["right_pct"])
            reverse = abs(row["right_pct"] - target["left_pct"]) + abs(row["left_pct"] - target["right_pct"])
            if allow_reverse and reverse < direct:
                score = reverse
                orientation = "reversed"
            else:
                score = direct
                orientation = "direct"

            match = dict(row)
            match.update(
                {
                    "target_dataset": target["dataset"],
                    "target_metric": target["metric"],
                    "target_left_pct": target["left_pct"],
                    "target_right_pct": target["right_pct"],
                    "match_score": round(score, 4),
                    "orientation": orientation,
                }
            )
            matches.append(match)
    return sorted(matches, key=lambda item: item["match_score"])


def main():
    parser = argparse.ArgumentParser(
        description="Collect LLM-as-a-judge result files and recover real winner rates."
    )
    parser.add_argument("--root", required=True, help="Directory to scan recursively.")
    parser.add_argument("--out", default="eval_result_summary.csv", help="CSV summary path.")
    parser.add_argument("--md", default="", help="Optional Markdown summary path.")
    parser.add_argument("--match-out", default="", help="Optional CSV path for target matching.")
    parser.add_argument("--target", action="append", type=parse_target, default=[])
    parser.add_argument("--left-label", default="", help="Override left/system-1 label for every file.")
    parser.add_argument("--right-label", default="", help="Override right/system-2 label for every file.")
    parser.add_argument("--include-all", action="store_true", help="Try every JSONL file, not only result-like names.")
    parser.add_argument("--top", type=int, default=20, help="How many target matches to print.")
    parser.add_argument("--no-reverse", action="store_true", help="Do not match targets with reversed left/right order.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    all_rows = []
    scanned = 0
    usable = 0
    for path in candidate_files(root, include_all=args.include_all):
        scanned += 1
        rows = summarize_file(
            path,
            left_label=args.left_label or None,
            right_label=args.right_label or None,
        )
        if rows is None:
            continue
        usable += 1
        all_rows.extend(rows)

    out_path = Path(args.out)
    write_csv(all_rows, out_path)
    print(f"Scanned JSONL files: {scanned}")
    print(f"Usable eval result files: {usable}")
    print(f"Summary CSV: {out_path.resolve()}")

    if args.md:
        md_path = Path(args.md)
        write_markdown(all_rows, md_path)
        print(f"Summary Markdown: {md_path.resolve()}")

    if args.target:
        matches = match_targets(all_rows, args.target, allow_reverse=not args.no_reverse)
        if args.match_out:
            write_csv(matches, Path(args.match_out))
            print(f"Target match CSV: {Path(args.match_out).resolve()}")
        print("\nTop target matches:")
        for item in matches[: args.top]:
            print(
                "{score:>6.2f} | {dataset:<11} | {metric:<18} | {left:.2f}/{right:.2f} | "
                "{orientation:<8} | {file}".format(
                    score=item["match_score"],
                    dataset=item["dataset"] or "unknown",
                    metric=item["metric"],
                    left=item["left_pct"],
                    right=item["right_pct"],
                    orientation=item["orientation"],
                    file=item["file"],
                )
            )


if __name__ == "__main__":
    main()
