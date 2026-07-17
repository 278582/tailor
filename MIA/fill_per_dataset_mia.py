from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


DATASETS = ["adult", "beijing", "default", "diabetes", "magic", "news", "shoppers"]
DATASET_TITLES = {
    "adult": "Adult",
    "beijing": "Beijing",
    "default": "Default",
    "diabetes": "Diabetes",
    "magic": "Magic",
    "news": "News",
    "shoppers": "Shoppers",
}

MODEL_ROWS = [
    ("tabdiff", "tabdiff", "no_theta_0_1_random_full"),
    ("tabdiff theta guided pareto-like", "tabdiff", "single_0_1_pareto"),
    ("tabsyn", "tabsyn", "no_theta_0_1_random_full"),
    ("tabsyn theta guided pareto-like", "tabsyn", "single_0_1_pareto"),
    ("great", "great", "no_theta_0_1_random_full"),
    ("great theta guided pareto-like", "great", "single_0_1_pareto"),
]
MIXED_ROW = ("mixed theta guided pareto-like", "mixed", "mixed_theta_0_1_pareto")

METRIC_COLUMNS = [
    ("Auroc", "auroc"),
    ("Attack Advantage", "attack_advantage"),
    ("TPR@FPR=1%", "tpr_at_fpr_1pct"),
    ("TPR@FPR=5%", "tpr_at_fpr_5pct"),
    ("TPR@FPR=10%", "tpr_at_fpr_10pct"),
]
TABLE_HEADERS = ["", *[display_column for display_column, _ in METRIC_COLUMNS]]

DEFAULT_RESULTS_DIR = Path("MIA/results")
DEFAULT_OUTPUT = Path("md/per-dataset-mia.md")
MISSING_VALUE = "—"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill md/per-dataset-mia.md from MIA metrics.csv files.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Print generated markdown instead of writing it.")
    return parser.parse_args()


def metrics_path(results_dir: Path, dataset: str, row: tuple[str, str, str]) -> Path:
    _, model, run_name = row
    return results_dir / model / dataset / run_name / "metrics.csv"


def summarize_metric(path: Path, column: str) -> str:
    value = summarize_metric_value(path, column)
    if value is None:
        return MISSING_VALUE
    metric_mean, metric_std = value
    return f"{format_number(metric_mean)} ± {format_number(metric_std)}"


def summarize_metric_value(path: Path, column: str) -> tuple[float, float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    metric_mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return metric_mean, std


def format_number(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def render_markdown_table(rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(TABLE_HEADERS[index]), *[len(row[index]) for row in rows])
        for index in range(len(TABLE_HEADERS))
    ]
    header = "| " + " | ".join(TABLE_HEADERS[index].ljust(widths[index]) for index in range(len(widths))) + " |"
    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(widths))) + " |" for row in rows]
    return [header, separator, *body]


def build_row_values(results_dir: Path, dataset: str, row: tuple[str, str, str]) -> list[str]:
    label, _, _ = row
    path = metrics_path(results_dir, dataset, row)
    values = [summarize_metric(path, source_column) for _, source_column in METRIC_COLUMNS]
    return [label, *values]


def build_row_values_with_cache(
    results_dir: Path,
    dataset: str,
    row: tuple[str, str, str],
    cache: dict[str, dict[str, list[float]]],
) -> list[str]:
    label, _, _ = row
    path = metrics_path(results_dir, dataset, row)
    values: list[str] = []
    for display_column, source_column in METRIC_COLUMNS:
        metric_value = summarize_metric_value(path, source_column)
        if metric_value is None:
            values.append(MISSING_VALUE)
            continue
        metric_mean, metric_std = metric_value
        values.append(f"{format_number(metric_mean)} ± {format_number(metric_std)}")
        cache.setdefault(label, {}).setdefault(display_column, []).append(round(metric_mean, 3))
    return [label, *values]


def build_table(
    results_dir: Path,
    dataset: str,
    cache: dict[str, dict[str, list[float]]] | None = None,
) -> list[str]:
    lines = [
        f"## 表 {DATASET_TITLES[dataset]} 数据集的 MIA 指标。",
        "",
    ]
    rows: list[list[str]] = []
    for row in MODEL_ROWS:
        if cache is None:
            rows.append(build_row_values(results_dir, dataset, row))
        else:
            rows.append(build_row_values_with_cache(results_dir, dataset, row, cache))
    if cache is None:
        rows.append(build_row_values(results_dir, dataset, MIXED_ROW))
    else:
        rows.append(build_row_values_with_cache(results_dir, dataset, MIXED_ROW, cache))
    lines.extend(render_markdown_table(rows))
    return lines


def summarize_all_metric(values: list[float]) -> str:
    if not values:
        return MISSING_VALUE
    metric_mean = mean(values)
    metric_std = stdev(values) if len(values) > 1 else 0.0
    return f"{format_number(metric_mean)} ± {format_number(metric_std)}"


def build_all_table(cache: dict[str, dict[str, list[float]]]) -> list[str]:
    lines = [
        "## 表 All 数据集的 MIA 指标。",
        "",
    ]
    rows: list[list[str]] = []
    for row in [*MODEL_ROWS, MIXED_ROW]:
        label = row[0]
        values = [summarize_all_metric(cache.get(label, {}).get(display_column, [])) for display_column, _ in METRIC_COLUMNS]
        rows.append([label, *values])
    lines.extend(render_markdown_table(rows))
    return lines


def build_document(results_dir: Path) -> str:
    sections: list[str] = []
    all_cache: dict[str, dict[str, list[float]]] = {}
    for dataset in DATASETS:
        sections.extend(build_table(results_dir, dataset, cache=all_cache))
        sections.extend(["", ""])
    sections.extend(build_all_table(all_cache))
    sections.extend(["", ""])
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    document = build_document(args.results_dir)
    if args.dry_run:
        print(document, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
