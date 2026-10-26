from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError as error:
    raise SystemExit(
        "DuckDB is required. Install it with: python -m pip install duckdb"
    ) from error

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR.parent / "0_Feature_Dataframe"))
from create_dataframe import DataframeSource, dataframe_inputs
from length_lists import LENGTH_NOTE, length_features
FULL_MODULE_DIR = ANALYSIS_DIR.parents[1]
METHODS_DIR = FULL_MODULE_DIR / "1_Complexity_Analysis_Methods"

INPUT_FILES = {
    "LCA": METHODS_DIR / "LCA" / "lca_results.csv",
    "L2SCA": METHODS_DIR / "L2SCA" / "l2sca_results.csv",
    "TAASSC": METHODS_DIR / "TAASSC" / "taassc_results.csv",
    "TAALES": METHODS_DIR / "TAALES" / "taales_results_final.csv",
    "TAALES_COVERAGE": (
        METHODS_DIR / "TAALES" / "taales_results_final_index_coverage.csv"
    ),
    "POLKE": METHODS_DIR / "POLKE" / "polke_results.csv",
}

NON_FEATURE_COLUMNS = {
    "filename",
    "writing_id",
    "cefr",
    "level",
    "grade",
    "learner_id",
    "learner_id_categorical",
    "processing_mode",
}

TEXT_LENGTH_FEATURES = length_features()

THREADS = 4
MEMORY_LIMIT = "4GB"

OVERALL_COLUMNS = (
    "source",
    "feature",
    "is_text_length_feature",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
)

LEVEL_COLUMNS = (
    "source",
    "feature",
    "is_text_length_feature",
    "cefr",
    "level",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
)

OVERALL_OUTPUT_PATH = ANALYSIS_DIR / "complexity_stats.csv"
LEVEL_OUTPUT_PATH = ANALYSIS_DIR / "complexity_level_stats.csv"
SUMMARY_OUTPUT_PATH = ANALYSIS_DIR / "complexity_summary.txt"


def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def csv_scan(path: Path) -> str:
    if isinstance(path, DataframeSource):
        return path.sql()
    escaped = path.as_posix().replace("'", "''")
    return (
        f"read_csv_auto('{escaped}', header=true, all_varchar=true, "
        "sample_size=2048, null_padding=true)"
    )


def find_input(path: Path) -> Path | None:
    if path.is_file():
        return path
    compressed = Path(f"{path}.gz")
    if compressed.is_file():
        return compressed
    return None


def discover_inputs() -> tuple[dict[str, Path], list[str]]:
    return dataframe_inputs(FULL_MODULE_DIR.parent, INPUT_FILES), []


def finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def atomic_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {path}. Close it in Excel or another program and rerun."
        ) from error


def atomic_text(path: Path, content: str) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(content, encoding="utf-8")
    try:
        os.replace(temporary, path)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {path}. Close it in another program and rerun."
        ) from error


def column_name(columns: list[str], target: str) -> str | None:
    return next(
        (name for name in columns if name.strip().casefold() == target.casefold()),
        None,
    )


def is_text_length_feature(source: str, feature: str) -> bool:
    return feature.strip().casefold() in TEXT_LENGTH_FEATURES.get(source.upper(), set())


def valid_level(expression: str) -> str:
    number = f"TRY_CAST({expression} AS DOUBLE)"
    return (
        f"CASE WHEN isfinite({number}) AND {number} BETWEEN 1 AND 15 "
        f"AND {number} = FLOOR({number}) THEN CAST({number} AS INTEGER) END"
    )


def label_expressions(columns: list[str]) -> tuple[str, str]:
    filename_name = column_name(columns, "filename")
    level_name = column_name(columns, "level")
    cefr_name = column_name(columns, "cefr")

    if filename_name is None and level_name is None:
        raise ValueError(
            "A level column or filenames containing '_level_N' are required. "
            f"Available columns: {columns}"
        )

    level = "NULL::INTEGER"
    cefr = "NULL::VARCHAR"
    if filename_name is not None:
        filename = (
            f"regexp_extract(replace(TRIM({sql_identifier(filename_name)}), "
            "chr(92), '/'), '[^/]+$', 0)"
        )
        level = valid_level(
            f"regexp_extract({filename}, '_level_([0-9]+)(_|[.][a-z]|$)', 1, 'i')"
        )
        cefr = (
            f"NULLIF(UPPER(regexp_extract({filename}, "
            "'_cefr_([^_]+)_level_', 1, 'i')), '')"
        )
    if level_name is not None:
        explicit_level = valid_level(f"TRIM({sql_identifier(level_name)})")
        level = f"COALESCE({explicit_level}, {level})"
    if cefr_name is not None:
        cefr = (
            f"COALESCE(NULLIF(UPPER(TRIM({sql_identifier(cefr_name)})), ''), "
            f"{cefr})"
        )
    return cefr, level


def numeric_expression(column: str) -> str:
    value = f"TRY_CAST(TRIM({sql_identifier(column)}) AS DOUBLE)"
    return f"CASE WHEN isfinite({value}) THEN {value} ELSE NULL END"


def aggregate_expressions(features: list[str], include_counts: bool) -> str:
    expressions = []
    for index in range(len(features)):
        value = f"value_{index}"
        if include_counts:
            expressions.append(f"COUNT({value}) AS used_{index}")
        expressions.extend(
            (
                f"AVG({value}) AS mean_{index}",
                f"STDDEV_SAMP({value}) AS sd_{index}",
                f"MIN({value}) AS min_{index}",
                f"MAX({value}) AS max_{index}",
            )
        )
    return ",\n       ".join(expressions)


def analyze_source(
    connection,
    source: str,
    path: Path,
) -> tuple[list[dict], list[dict], dict]:
    scan = csv_scan(path)
    columns = [
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
    ]
    features = [
        name for name in columns if name.strip().casefold() not in NON_FEATURE_COLUMNS
    ]
    if not features:
        raise ValueError(f"{source}: no candidate feature columns in {path.name}")
    cefr, level = label_expressions(columns)
    projections = ",\n           ".join(
        f"{numeric_expression(feature)} AS value_{index}"
        for index, feature in enumerate(features)
    )
    values_query = (
        "SELECT "
        f"{cefr} AS cefr, {level} AS level,\n           {projections}\n"
        f"FROM {scan}"
    )
    overall_aggregates = aggregate_expressions(features, include_counts=True)
    level_aggregates = aggregate_expressions(features, include_counts=False)

    started = time.perf_counter()
    print(f"  {source}: checking {len(features):,} candidate columns")
    overall_values = connection.execute(
        "WITH values AS (\n"
        f"{values_query}\n"
        ")\n"
        "SELECT COUNT(*) AS rows,\n"
        "       COUNT(level) AS labelled_rows,\n"
        "       COUNT(*) FILTER (WHERE level IS NOT NULL "
        "AND (cefr IS NULL OR cefr = '')) AS missing_cefr_rows,\n"
        f"       {overall_aggregates}\n"
        "FROM values"
    ).fetchone()
    total_rows = int(overall_values[0])
    labelled_rows = int(overall_values[1])
    missing_cefr_rows = int(overall_values[2])
    if total_rows == 0:
        raise ValueError(f"{source}: no text rows in {path.name}")
    if labelled_rows == 0:
        raise ValueError(
            f"{source}: no valid course levels in {path.name}. Supply integer "
            "levels from 1 to 15 in a level column or in the filenames."
        )
    overall_rows = []
    active_indices = []
    for index, feature in enumerate(features):
        offset = 3 + index * 5
        used = int(overall_values[offset])
        if used == 0:
            continue
        active_indices.append(index)
        overall_rows.append(
            {
                "source": source,
                "feature": feature.strip(),
                "is_text_length_feature": (
                    "yes" if is_text_length_feature(source, feature) else "no"
                ),
                "mean": finite(overall_values[offset + 1]),
                "standard_deviation": finite(overall_values[offset + 2]),
                "minimum": finite(overall_values[offset + 3]),
                "maximum": finite(overall_values[offset + 4]),
            }
        )
    if not active_indices:
        raise ValueError(f"{source}: no numeric feature values in {path.name}")

    group_values = connection.execute(
        "WITH values AS (\n"
        f"{values_query}\n"
        ")\n"
        "SELECT cefr, level,\n       "
        f"{level_aggregates}\n"
        "FROM values\n"
        "WHERE level BETWEEN 1 AND 15\n"
        "GROUP BY cefr, level\n"
        "ORDER BY level, cefr"
    ).fetchall()
    level_rows = []
    for group in group_values:
        group_cefr = str(group[0] or "")
        group_level = int(group[1])
        for index in active_indices:
            offset = 2 + index * 4
            level_rows.append(
                {
                    "source": source,
                    "feature": features[index].strip(),
                    "is_text_length_feature": (
                        "yes"
                        if is_text_length_feature(source, features[index])
                        else "no"
                    ),
                    "cefr": group_cefr,
                    "level": group_level,
                    "mean": finite(group[offset]),
                    "standard_deviation": finite(group[offset + 1]),
                    "minimum": finite(group[offset + 2]),
                    "maximum": finite(group[offset + 3]),
                }
            )
    elapsed = time.perf_counter() - started
    print(
        f"    {total_rows:,} texts, {len(active_indices):,} numeric features, "
        f"{elapsed:.1f}s"
    )
    if labelled_rows < total_rows:
        print(
            f"    {total_rows - labelled_rows:,} texts excluded from level statistics: "
            "missing or invalid course level"
        )
    if missing_cefr_rows:
        print(f"    {missing_cefr_rows:,} level-labelled texts have no CEFR label")
    source_summary = {
        "source": source,
        "file": path.name,
        "texts": total_rows,
        "labelled_texts": labelled_rows,
        "unlabelled_texts": total_rows - labelled_rows,
        "missing_cefr_texts": missing_cefr_rows,
        "candidate_features": len(features),
        "numeric_features": len(active_indices),
        "text_length_features": sum(
            is_text_length_feature(source, features[index])
            for index in active_indices
        ),
        "level_groups": len(group_values),
    }
    return overall_rows, level_rows, source_summary


def write_summary(
    inputs: dict[str, Path],
    skipped: list[str],
    source_summaries: list[dict],
    overall_rows: list[dict],
    level_rows: list[dict],
) -> None:
    found_text = ", ".join(inputs)
    skipped_text = ", ".join(skipped) if skipped else "none"
    lines = [
        "COMPLEXITY ANALYSIS SUMMARY",
        "",
        "Analysis: descriptive statistics for available complexity features",
        "Proficiency grouping: EFCAMDAT levels 1-15 and CEFR labels",
        "",
        f"Sources found: {found_text}",
        f"Sources skipped (not found): {skipped_text}",
        "",
        "SOURCE DETAILS",
    ]
    for item in source_summaries:
        lines.extend(
            (
                "",
                item["source"],
                f"  Input file: {item['file']}",
                f"  Text rows: {item['texts']:,}",
                f"  Texts in level statistics: {item['labelled_texts']:,}",
                "  Texts excluded from level statistics: "
                f"{item['unlabelled_texts']:,}",
                "  Level-labelled texts without CEFR: "
                f"{item['missing_cefr_texts']:,}",
                f"  Candidate features: {item['candidate_features']:,}",
                f"  Numeric features analyzed: {item['numeric_features']:,}",
                f"  Length-dependent features: {item['text_length_features']:,}",
                f"  Level/CEFR groups analyzed: {item['level_groups']:,}",
            )
        )
    lines.extend(
        (
            "",
            "OUTPUTS",
            "",
            f"Overall feature rows: {len(overall_rows):,}",
            f"Level-specific feature rows: {len(level_rows):,}",
            f"Overall statistics: {OVERALL_OUTPUT_PATH.name}",
            f"Level statistics: {LEVEL_OUTPUT_PATH.name}",
            "",
            "NOTES",
            "",
            "Only missing source files are skipped. A source file that exists but is "
            "malformed still raises an error so corrupted data is not silently ignored.",
            "Valid label columns are used first, with filenames as a fallback. "
            "Course levels must be integers from 1 to 15.",
            "Overall statistics include all text rows. Rows without a valid course "
            "level are excluded from level statistics and counted above. Missing CEFR "
            "labels remain blank. A source with no valid course levels raises an error.",
            "The is_text_length_feature column marks the length-dependent features. "
            + LENGTH_NOTE,
        )
    )
    atomic_text(SUMMARY_OUTPUT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    started = time.perf_counter()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    inputs, skipped = discover_inputs()
    print("Complexity basic analysis")
    print(f"Feature input: {next(iter(inputs.values())).parquet}")
    print(f"Output folder: {ANALYSIS_DIR}")
    print(f"Sources found: {', '.join(inputs)}")
    print(
        "Sources skipped (not found): "
        + (", ".join(skipped) if skipped else "none")
    )

    connection = duckdb.connect(":memory:")
    connection.execute(f"SET threads={THREADS}")
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute("SET preserve_insertion_order=false")
    try:
        overall_rows = []
        level_rows = []
        source_summaries = []
        for source, path in inputs.items():
            source_overall, source_levels, source_summary = analyze_source(
                connection, source, path
            )
            overall_rows.extend(source_overall)
            level_rows.extend(source_levels)
            source_summaries.append(source_summary)
    finally:
        connection.close()

    overall_rows.sort(key=lambda row: (row["source"], row["feature"]))
    level_rows.sort(
        key=lambda row: (row["source"], row["feature"], row["level"], row["cefr"])
    )
    atomic_csv(OVERALL_OUTPUT_PATH, OVERALL_COLUMNS, overall_rows)
    atomic_csv(LEVEL_OUTPUT_PATH, LEVEL_COLUMNS, level_rows)
    write_summary(inputs, skipped, source_summaries, overall_rows, level_rows)

    print(f"\nCompleted in {time.perf_counter() - started:.1f}s")
    print(f"Overall statistics: {OVERALL_OUTPUT_PATH}")
    print(f"Level statistics:   {LEVEL_OUTPUT_PATH}")
    print(f"Summary:            {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
