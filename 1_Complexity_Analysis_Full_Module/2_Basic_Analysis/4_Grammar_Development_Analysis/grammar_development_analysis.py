import csv
import gzip
import math
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0_Feature_Dataframe"))
from create_dataframe import DataframeSource, dataframe_inputs, dataframe_rows


COMMON_USAGE_PERCENT = 5.0
COMPARISON_THRESHOLDS = (2.0, 10.0)
MIN_TEXTS_PER_LEVEL = 100
MIN_FEATURE_COVERAGE_PERCENT = 95.0
MIN_LATER_LEVELS = 2
SUMMARY_LIMIT = 20
PROGRESS_EVERY = 10_000

LEVELS = tuple(range(1, 16))
FEATURE_PATTERN = re.compile(r"polke_(\d+)_per_100_words", re.IGNORECASE)
LEVEL_PATTERN = re.compile(r"_level_(\d+)(?:_|\.|$)", re.IGNORECASE)
ID_PATTERN = re.compile(r"(\d+)(?:\.0+)?")

FEATURE_COLUMNS = ("feature", "construct_id", "description", "reference_cefr")
USAGE_COLUMNS = FEATURE_COLUMNS + (
    "level",
    "total_texts_at_level",
    "texts_analyzed",
    "missing_or_invalid_values",
    "texts_with_structure",
    "percentage_of_texts_with_structure",
    "occurrences_per_100_words",
    "common_usage",
    "evidence_status",
)
PROGRESSION_COLUMNS = FEATURE_COLUMNS + (
    "common_threshold_percent",
    "first_observed_level",
    "first_common_level",
    "sustained_usage_from_level",
    "later_below_threshold_levels",
    "insufficient_data_levels",
    "status",
)


@dataclass
class Usage:
    texts: int = 0
    texts_with_structure: int = 0
    words: float = 0.0
    weighted_rate: float = 0.0


def open_csv(path: Path):
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


@contextmanager
def usage_reader(path):
    if isinstance(path, DataframeSource):
        with dataframe_rows(path) as reader:
            yield reader
    else:
        with open_csv(path) as handle:
            yield csv.reader(handle)


def read_header(reader, path: Path) -> list[str]:
    header = [value.strip().casefold() for value in next(reader, [])]
    if not header or any(not value for value in header):
        raise ValueError(f"{path.name}: the CSV header is missing or contains empty names.")
    if len(header) != len(set(header)):
        raise ValueError(f"{path.name}: duplicate column names after trimming spaces.")
    return header


def number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: str) -> int | None:
    result = number(value)
    return int(result) if result is not None and result.is_integer() else None


def csv_file(folder: Path, name: str) -> Path | None:
    for path in (folder / name, folder / f"{name}.gz"):
        if path.is_file():
            return path
    return None


def find_module(analysis_dir: Path) -> Path | None:
    for parent in (analysis_dir, *analysis_dir.parents):
        if (parent / "1_Complexity_Analysis_Methods").is_dir():
            return parent
        module = parent / "1_Complexity_Analysis_Full_Module"
        if (module / "1_Complexity_Analysis_Methods").is_dir():
            return module
        if parent.name == "1_Complexity_Analysis_Full_Module":
            return parent
    return None


def find_inputs(analysis_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    module = find_module(analysis_dir)
    if module is None:
        return None, None, None
    polke = module / "1_Complexity_Analysis_Methods" / "POLKE"
    folders = (
        polke,
        polke / "polke-main",
        polke / "results",
        polke / "polke-main" / "results",
    )
    results = next(
        (path for folder in folders if (path := csv_file(folder, "polke_results.csv"))),
        None,
    )
    metadata_folders = (results.parent, *folders) if results else folders
    metadata = next(
        (
            path for folder in metadata_folders
            if (path := csv_file(folder, "polke_feature_metadata.csv"))
        ),
        None,
    )
    errant_folder = module / "2_Basic_Analysis" / "2_Errant_Analysis"
    errant = csv_file(errant_folder, "errant_basic_stats.csv")
    return dataframe_inputs(module.parent, ["POLKE"])["POLKE"], metadata, errant


def read_metadata(path: Path | None) -> dict[str, dict]:
    metadata = {}
    if path is None:
        return metadata
    with open_csv(path) as handle:
        reader = csv.reader(handle)
        header = read_header(reader, path)
        if "feature" not in header and "construct_id" not in header:
            raise ValueError(f"{path.name}: expected a feature or construct_id column.")
        for line_number, values in enumerate(reader, 2):
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError(f"{path.name}, line {line_number}: unexpected number of columns.")
            row = dict(zip(header, (value.strip() for value in values)))
            feature = row.get("feature", "").casefold()
            match = FEATURE_PATTERN.fullmatch(feature)
            construct_id = int(match.group(1)) if match else integer(row.get("construct_id", ""))
            if construct_id is None:
                raise ValueError(f"{path.name}, line {line_number}: invalid construct ID.")
            feature = f"polke_{construct_id}_per_100_words"
            item = {
                "feature": feature,
                "construct_id": construct_id,
                "description": (
                    row.get("can_do_statement")
                    or row.get("guideword")
                    or f"POLKE structure {construct_id}"
                ),
                "reference_cefr": row.get("cefr", ""),
            }
            if feature in metadata and metadata[feature] != item:
                raise ValueError(f"{path.name}: conflicting metadata for {feature}.")
            metadata[feature] = item
    return metadata


def text_id(row: list[str], columns: dict[str, int]) -> str:
    writing_id = row[columns["writing_id"]].strip() if "writing_id" in columns else ""
    if writing_id:
        match = ID_PATTERN.fullmatch(writing_id)
        return str(int(match.group(1))) if match else writing_id.casefold()
    filename = row[columns["filename"]].strip() if "filename" in columns else ""
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    match = re.match(r"(\d+)(?:_|\.txt(?:\.gz)?$|$)", basename)
    return str(int(match.group(1))) if match else basename


def text_level(row: list[str], columns: dict[str, int]) -> int | None:
    filename = row[columns["filename"]] if "filename" in columns else ""
    match = LEVEL_PATTERN.search(filename.replace("\\", "/").rsplit("/", 1)[-1])
    filename_level = int(match.group(1)) if match else None
    column_level = integer(row[columns["level"]]) if "level" in columns else None
    if filename_level is not None and column_level is not None and filename_level != column_level:
        raise ValueError(f"Level in filename disagrees with the level column: {filename}")
    level = filename_level if filename_level is not None else column_level
    return level if level in LEVELS else None


def read_usage(path: Path) -> tuple[list[str], dict[int, list[Usage]], Counter, Counter]:
    level_totals = Counter()
    exclusions = Counter()
    seen = set()
    with usage_reader(path) as reader:
        header = read_header(reader, path)
        columns = {name: index for index, name in enumerate(header)}
        features = sorted(
            (name for name in header if FEATURE_PATTERN.fullmatch(name)),
            key=lambda name: int(FEATURE_PATTERN.fullmatch(name).group(1)),
        )
        if not features:
            raise ValueError(f"{path.name}: no polke_<ID>_per_100_words columns found.")
        if not {"filename", "writing_id"}.intersection(columns):
            raise ValueError(f"{path.name}: a filename or writing_id column is required.")
        if not {"filename", "level"}.intersection(columns):
            raise ValueError(f"{path.name}: a filename or level column is required.")
        word_column = next(
            (
                columns[name]
                for name in ("word_count", "word count", "original_word_count")
                if name in columns
            ),
            None,
        )
        if word_column is None:
            raise ValueError(f"{path.name}: a word_count column is required for pooled frequency.")
        positions = [columns[feature] for feature in features]
        usage = {level: [Usage() for _ in features] for level in LEVELS}
        print(f"Reading {len(features):,} grammatical structures", flush=True)
        for line_number, row in enumerate(reader, 2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name}, line {line_number}: unexpected number of columns.")
            identity = text_id(row, columns)
            if not identity:
                raise ValueError(f"{path.name}, line {line_number}: missing text identifier.")
            if identity in seen:
                raise ValueError(f"{path.name}: duplicate text identifier {identity!r}.")
            seen.add(identity)
            level = text_level(row, columns)
            words = number(row[word_column])
            if level is None:
                exclusions["invalid or missing course level"] += 1
                continue
            if words is None or words <= 0 or not words.is_integer():
                exclusions["invalid or zero word count"] += 1
                continue
            level_totals[level] += 1
            for position, counts in zip(positions, usage[level]):
                value = number(row[position])
                if value is None or value < 0 or not math.isfinite(value * words):
                    continue
                counts.texts += 1
                counts.words += words
                if value > 0:
                    counts.texts_with_structure += 1
                    counts.weighted_rate += value * words
            if len(seen) % PROGRESS_EVERY == 0:
                print(f"  Read {len(seen):,} texts", flush=True)
    return features, usage, level_totals, exclusions


def feature_details(feature: str, metadata: dict[str, dict]) -> dict:
    if feature in metadata:
        return metadata[feature]
    construct_id = int(FEATURE_PATTERN.fullmatch(feature).group(1))
    return {
        "feature": feature,
        "construct_id": construct_id,
        "description": f"POLKE structure {construct_id}",
        "reference_cefr": "",
    }


def evidence_status(counts: Usage, total: int) -> str:
    if not total:
        return "no_texts"
    if counts.texts < MIN_TEXTS_PER_LEVEL:
        return "too_few_texts"
    if 100.0 * counts.texts / total < MIN_FEATURE_COVERAGE_PERCENT:
        return "too_many_missing_values"
    return "sufficient"


def percentage(counts: Usage) -> float | None:
    return 100.0 * counts.texts_with_structure / counts.texts if counts.texts else None


def usage_rows(features, usage, level_totals, metadata) -> list[dict]:
    rows = []
    for index, feature in enumerate(features):
        for level in LEVELS:
            counts = usage[level][index]
            status = evidence_status(counts, level_totals[level])
            prevalence = percentage(counts)
            rows.append({
                **feature_details(feature, metadata),
                "level": level,
                "total_texts_at_level": level_totals[level],
                "texts_analyzed": counts.texts,
                "missing_or_invalid_values": level_totals[level] - counts.texts,
                "texts_with_structure": counts.texts_with_structure,
                "percentage_of_texts_with_structure": prevalence,
                "occurrences_per_100_words": counts.weighted_rate / counts.words if counts.words else None,
                "common_usage": (
                    ("yes" if prevalence >= COMMON_USAGE_PERCENT else "no")
                    if status == "sufficient" else "not assessed"
                ),
                "evidence_status": status,
            })
    return rows


def progression_rows(features, usage, level_totals, metadata) -> list[dict]:
    rows = []
    thresholds = sorted(set((COMMON_USAGE_PERCENT, *COMPARISON_THRESHOLDS)))
    for index, feature in enumerate(features):
        observed = [level for level in LEVELS if usage[level][index].texts_with_structure]
        sufficient = {
            level: evidence_status(usage[level][index], level_totals[level]) == "sufficient"
            for level in LEVELS
        }
        insufficient = [level for level in LEVELS if not sufficient[level]]
        for threshold in thresholds:
            common = {
                level: sufficient[level] and percentage(usage[level][index]) >= threshold
                for level in LEVELS
            }
            first_common = next((level for level in LEVELS if common[level]), None)
            sustained = next(
                (
                    level for level in LEVELS
                    if LEVELS[-1] - level >= MIN_LATER_LEVELS
                    and all(common[later] for later in range(level, LEVELS[-1] + 1))
                ),
                None,
            )
            drops = [
                level for level in LEVELS
                if first_common is not None and level > first_common
                and sufficient[level] and not common[level]
            ]
            if sustained is not None:
                status = "sustained_usage"
            elif first_common is None:
                status = (
                    "insufficient_data" if insufficient
                    else ("not_common" if observed else "not_observed")
                )
            elif any(level >= first_common for level in insufficient):
                status = "insufficient_data"
            elif drops:
                status = "usage_not_sustained"
            else:
                status = "too_few_later_levels"
            rows.append({
                **feature_details(feature, metadata),
                "common_threshold_percent": threshold,
                "first_observed_level": observed[0] if observed else None,
                "first_common_level": first_common,
                "sustained_usage_from_level": sustained,
                "later_below_threshold_levels": "; ".join(map(str, drops)),
                "insufficient_data_levels": "; ".join(map(str, insufficient)),
                "status": status,
            })
    return rows


def read_errant_context(path: Path | None) -> tuple[list[str], str]:
    if path is None:
        return [], "ERRANT basic statistics were not found; error context was skipped."
    try:
        rows = []
        with open_csv(path) as handle:
            reader = csv.DictReader(handle)
            reader.fieldnames = [name.strip().casefold() for name in (reader.fieldnames or [])]
            required = {"group_type", "level", "text_count", "errors_per_100_words"}
            if not required.issubset(reader.fieldnames):
                raise ValueError("expected group_type, level, text_count and errors_per_100_words columns")
            for row in reader:
                if row["group_type"].strip().casefold() != "level":
                    continue
                level = integer(row["level"])
                count = integer(row["text_count"])
                rate = number(row["errors_per_100_words"])
                if level not in LEVELS or count is None or count < 0 or rate is None or rate < 0:
                    raise ValueError("invalid level, text count or error rate")
                rows.append((level, count, rate))
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("more than one summary row for the same course level")
        if not rows:
            raise ValueError("no course-level summary rows")
        lines = [
            "",
            "ERRANT ERROR TRENDS",
            f"Input: {path}",
            "These rates describe the ERRANT sample and are not matched to individual POLKE structures.",
            "Level | Texts | Errors per 100 words",
        ]
        lines.extend(f"{level:>5} | {count:>7,} | {rate:.4f}" for level, count, rate in sorted(rows))
        return lines, ""
    except (OSError, ValueError, KeyError, AttributeError, csv.Error) as error:
        return [], f"ERRANT context was skipped: {path.name}: {error}"


def summary_text(
    results_path, metadata_path, features, level_totals, exclusions,
    progressions, errant_lines, errant_note,
) -> str:
    thresholds = sorted(set((COMMON_USAGE_PERCENT, *COMPARISON_THRESHOLDS)))
    lines = [
        "GRAMMAR DEVELOPMENT ANALYSIS",
        "",
        f"POLKE results: {results_path or 'not found'}",
        f"POLKE metadata: {metadata_path or 'not found; structure IDs are used as labels'}",
    ]
    if results_path is None:
        lines.extend([
            "",
            "Analysis skipped: polke_results.csv or polke_results.csv.gz is required.",
            "Place the script in 2_Basic_Analysis/4_Grammar_Development_Analysis "
            "inside the full complexity module.",
            "The CSV outputs contain headers only.",
        ])
    else:
        lines.extend([
            f"Texts analysed: {sum(level_totals.values()):,}",
            f"Structures analysed: {len(features):,}",
            f"Excluded texts: {sum(exclusions.values()):,}",
        ])
        lines.extend(f"  {reason}: {count:,}" for reason, count in sorted(exclusions.items()))
    lines.extend([
        "",
        "ANALYSIS SETTINGS",
        f"Main common-usage threshold: {COMMON_USAGE_PERCENT:g}% of texts",
        "Thresholds compared: " + ", ".join(f"{value:g}%" for value in thresholds),
        f"Minimum valid texts per structure and level: {MIN_TEXTS_PER_LEVEL:,}",
        f"Minimum feature coverage within a level: {MIN_FEATURE_COVERAGE_PERCENT:g}%",
        "Sustained usage requires common usage at every level from the reported "
        f"start through level 15, including at least {MIN_LATER_LEVELS} later levels.",
        "First observed level can be based on one detection. "
        "Common usage requires sufficient data.",
        "The start is the earliest level supported by the available evidence; "
        "insufficient earlier levels may hide an earlier start.",
        "Missing, non-numeric, negative and non-finite feature values are excluded, "
        "not treated as absence.",
        "Frequency per 100 words is pooled by weighting each valid text's rate "
        "by its word count.",
        "These exploratory thresholds describe group-level usage, not teaching "
        "dates, grammatical accuracy or individual retention.",
    ])
    if features:
        lines.extend(["", "TEXTS AVAILABLE BY COURSE LEVEL", "Level | Texts"])
        lines.extend(f"{level:>5} | {level_totals[level]:,}" for level in LEVELS)
        lines.extend(["", "THRESHOLD COMPARISON"])
        for threshold in thresholds:
            counts = Counter(
                row["status"] for row in progressions
                if row["common_threshold_percent"] == threshold
            )
            lines.append(
                f"{threshold:g}%: " + "; ".join(
                    f"{status.replace('_', ' ')}: {count:,}"
                    for status, count in sorted(counts.items())
                )
            )
        primary = [row for row in progressions if row["common_threshold_percent"] == COMMON_USAGE_PERCENT]
        sustained = sorted(
            (row for row in primary if row["sustained_usage_from_level"] is not None),
            key=lambda row: (row["sustained_usage_from_level"], row["construct_id"]),
        )
        lines.extend(["", f"SUSTAINED USAGE AT THE MAIN THRESHOLD (up to {SUMMARY_LIMIT})"])
        lines.extend(
            f"{row['feature']} | from level {row['sustained_usage_from_level']} "
            f"| {row['description']}"
            for row in sustained[:SUMMARY_LIMIT]
        )
        if not sustained:
            lines.append("No structures meet the sustained-usage rule.")
        drops = [row for row in primary if row["later_below_threshold_levels"]]
        lines.extend(["", f"LATER DROPS AFTER FIRST COMMON USAGE (up to {SUMMARY_LIMIT})"])
        lines.extend(
            f"{row['feature']} | first common at level {row['first_common_level']} "
            f"| below threshold at levels {row['later_below_threshold_levels']} "
            f"| {row['description']}"
            for row in drops[:SUMMARY_LIMIT]
        )
        if not drops:
            lines.append("No later drops are observed in sufficiently sampled levels.")
    if errant_note:
        lines.extend(["", errant_note])
    lines.extend(errant_lines)
    return "\n".join(lines) + "\n"


def write_outputs(folder: Path, usage: list[dict], progression: list[dict], summary: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".grammar_", dir=folder) as temporary:
        staged = Path(temporary)
        tables = (
            ("grammar_usage_by_level.csv", USAGE_COLUMNS, usage),
            ("grammar_progression.csv", PROGRESSION_COLUMNS, progression),
        )
        for name, columns, rows in tables:
            with (staged / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        (staged / "grammar_development_summary.txt").write_text(summary, encoding="utf-8")
        for name in ("grammar_usage_by_level.csv", "grammar_progression.csv", "grammar_development_summary.txt"):
            try:
                os.replace(staged / name, folder / name)
            except PermissionError as error:
                raise PermissionError(f"Close {name} in Excel or another program, then rerun the script.") from error
            print(f"Saved: {folder / name}", flush=True)


def validate_settings() -> None:
    for value in (COMMON_USAGE_PERCENT, *COMPARISON_THRESHOLDS):
        if not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError("Common-usage thresholds must be greater than 0 and at most 100.")
    if MIN_TEXTS_PER_LEVEL < 1 or int(MIN_TEXTS_PER_LEVEL) != MIN_TEXTS_PER_LEVEL:
        raise ValueError("MIN_TEXTS_PER_LEVEL must be a positive integer.")
    if not math.isfinite(MIN_FEATURE_COVERAGE_PERCENT) or not 0 < MIN_FEATURE_COVERAGE_PERCENT <= 100:
        raise ValueError("MIN_FEATURE_COVERAGE_PERCENT must be greater than 0 and at most 100.")
    if not 1 <= MIN_LATER_LEVELS < len(LEVELS) or int(MIN_LATER_LEVELS) != MIN_LATER_LEVELS:
        raise ValueError("MIN_LATER_LEVELS must be an integer between 1 and 14.")
    if SUMMARY_LIMIT < 1 or PROGRESS_EVERY < 1:
        raise ValueError("SUMMARY_LIMIT and PROGRESS_EVERY must be positive.")


def main(analysis_dir: Path | None = None) -> None:
    validate_settings()
    csv.field_size_limit(16 * 1024 * 1024)
    analysis_dir = (analysis_dir or Path(__file__).resolve().parent).resolve()
    results_path, metadata_path, errant_path = find_inputs(analysis_dir)
    print("Grammar Development Analysis", flush=True)
    print(f"POLKE results: {results_path or 'not found; analysis will be skipped'}", flush=True)
    features, totals, exclusions = [], Counter(), Counter()
    rows, progressions = [], []
    if results_path is not None:
        metadata = read_metadata(metadata_path)
        features, usage, totals, exclusions = read_usage(results_path)
        if not sum(totals.values()):
            raise ValueError("No usable POLKE texts have course levels 1-15 and positive word counts.")
        print(f"Analysing usage across levels for {len(features):,} structures", flush=True)
        rows = usage_rows(features, usage, totals, metadata)
        progressions = progression_rows(features, usage, totals, metadata)
    errant_lines, errant_note = read_errant_context(errant_path)
    summary = summary_text(
        results_path, metadata_path, features, totals, exclusions,
        progressions, errant_lines, errant_note,
    )
    write_outputs(analysis_dir, rows, progressions, summary)


if __name__ == "__main__":
    main()
