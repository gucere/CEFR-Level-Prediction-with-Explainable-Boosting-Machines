import csv
import gzip
import math
import os
import time
from pathlib import Path
import duckdb

ANALYSIS_DIR = Path(__file__).resolve().parent
FULL_MODULE_DIR = ANALYSIS_DIR.parents[1]
ERRANT_DIR = FULL_MODULE_DIR / "1_Complexity_Analysis_Methods" / "ERRANT"
RESULTS_PATH = ERRANT_DIR / "errant_results.csv"
ERROR_METADATA_PATH = ERRANT_DIR / "errant_error_type_metadata.csv"
EDITS_PATH = ERRANT_DIR / "errant_edits.csv"
BASIC_STATS_OUTPUT_PATH = ANALYSIS_DIR / "errant_basic_stats.csv"
ERROR_TYPE_OUTPUT_PATH = ANALYSIS_DIR / "errant_error_type_summary.csv"
FORM_OUTPUT_PATH = ANALYSIS_DIR / "errant_specific_correction_patterns.csv"
LEGACY_OUTPUT_PATHS = (
    ANALYSIS_DIR / "errant_stats.csv",
    ANALYSIS_DIR / "errant_form_correction_patterns.csv",
    ANALYSIS_DIR / "errant_error_prevalence_by_level.csv",
    ANALYSIS_DIR / "errant_error_occurrence_by_level.csv",
)

THREADS = 4
MEMORY_LIMIT = "4GB"

BASE_COLUMNS = (
    "original_word_count",
    "corrected_word_count",
    "total_errors",
    "distinct_error_types",
    "missing_errors",
    "unnecessary_errors",
    "replacement_errors",
    "other_errors",
    "affected_original_tokens",
    "correction_token_equivalents",
    "surface_accuracy_proxy",
    "error_free",
)

STATS_COLUMNS = (
    "group_type",
    "cefr",
    "level",
    "text_count",
    "original_word_count",
    "corrected_word_count",
    "total_errors",
    "errors_per_100_words",
    "mean_errors_per_text",
    "mean_distinct_error_types",
    "missing_errors",
    "missing_errors_per_100_words",
    "unnecessary_errors",
    "unnecessary_errors_per_100_words",
    "replacement_errors",
    "replacement_errors_per_100_words",
    "other_errors",
    "other_errors_per_100_words",
    "affected_original_tokens",
    "affected_original_tokens_per_100_words",
    "correction_token_equivalents",
    "correction_token_equivalents_per_100_words",
    "error_free_texts",
    "error_free_text_percent",
    "mean_surface_accuracy_proxy",
    "token_weighted_surface_accuracy_proxy",
)

ERROR_COLUMNS = (
    "group_type",
    "cefr",
    "level",
    "rank_by_error_count_within_group",
    "rank_by_text_percentage_within_group",
    "error_type",
    "is_text_length_feature",
    "operation",
    "error_family",
    "error_count",
    "errors_per_100_words",
    "texts_with_error",
    "total_texts",
    "percentage_of_texts_with_error",
    "share_of_group_errors_percent",
)

FORM_COLUMNS = (
    "cefr",
    "level",
    "error_type",
    "is_text_length_feature",
    "error_family",
    "operation",
    "original_form",
    "corrected_form",
    "correction_count",
    "texts_with_correction",
    "share_of_error_type_percent",
)

def sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def csv_scan(path: Path) -> str:
    escaped = path.as_posix().replace("'", "''")
    return (
        f"read_csv_auto('{escaped}', header=true, all_varchar=true, "
        "sample_size=2048, null_padding=true)"
    )


def resolve_input(path: Path) -> Path:
    if path.is_file():
        return path
    compressed = Path(f"{path}.gz")
    if compressed.is_file():
        return compressed
    available = sorted(item.name for item in path.parent.glob("*.csv*"))
    listing = "\n".join(f"  {name}" for name in available) or "  none"
    raise FileNotFoundError(
        f"Input file not found: {path}\nCSV files in {path.parent}:\n{listing}"
    )


def open_csv(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def integer(value: str, field: str, row_number: int) -> int:
    try:
        number = float(value.strip())
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError
        return int(number)
    except (TypeError, ValueError):
        raise ValueError(
            f"Row {row_number:,}: {field!r} is not an integer: {value!r}"
        ) from None


def read_metadata(path: Path) -> list[dict]:
    required = {
        "feature",
        "error_type",
        "operation",
        "error_family",
        "corpus_error_count",
    }
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                {
                    "feature": row["feature"].strip(),
                    "error_type": row["error_type"].strip(),
                    "operation": row["operation"].strip(),
                    "error_family": row["error_family"].strip(),
                    "corpus_error_count": integer(
                        row["corpus_error_count"], "corpus_error_count", row_number
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No error types found in {path}")
    if len({row["feature"].casefold() for row in rows}) != len(rows):
        raise ValueError(f"Duplicate feature names in {path.name}")
    if len({row["error_type"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate error types in {path.name}")
    return rows


def rate(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


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


def remove_legacy_outputs() -> None:
    for path in LEGACY_OUTPUT_PATHS:
        if not path.is_file():
            continue
        try:
            path.unlink()
        except PermissionError as error:
            raise PermissionError(
                f"Cannot remove {path}. Close it in Excel or another program and rerun."
            ) from error


def write_form_patterns(connection, edits_path: Path) -> tuple[Path, int]:
    scan = csv_scan(edits_path)
    columns = [
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
    ]
    by_name = {name.casefold(): name for name in columns}
    required = {
        "filename",
        "cefr",
        "level",
        "error_type",
        "error_family",
        "operation",
        "original_form",
        "corrected_form",
    }
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(
            f"Missing columns in {edits_path.name}: {missing}\n"
            f"Available columns: {columns}"
        )
    def column(name: str) -> str:
        return sql_identifier(by_name[name])

    query = (
        "WITH normalized AS (\n"
        "    SELECT\n"
        f"        TRIM({column('filename')}) AS filename,\n"
        f"        UPPER(TRIM({column('cefr')})) AS cefr,\n"
        f"        TRY_CAST(TRIM({column('level')}) AS INTEGER) AS level,\n"
        f"        TRIM({column('error_type')}) AS error_type,\n"
        f"        TRIM({column('error_family')}) AS error_family,\n"
        f"        TRIM({column('operation')}) AS operation,\n"
        f"        TRIM(COALESCE({column('original_form')}, '')) AS original_form,\n"
        f"        TRIM(COALESCE({column('corrected_form')}, '')) AS corrected_form\n"
        f"    FROM {scan}\n"
        "),\n"
        "patterns AS (\n"
        "    SELECT cefr, level, error_type, error_family, operation,\n"
        "           original_form, corrected_form,\n"
        "           COUNT(*) AS correction_count,\n"
        "           COUNT(DISTINCT filename) AS texts_with_correction\n"
        "    FROM normalized\n"
        "    WHERE level BETWEEN 1 AND 15 AND error_type <> ''\n"
        "    GROUP BY cefr, level, error_type, error_family, operation,\n"
        "             original_form, corrected_form\n"
        ")\n"
        "SELECT cefr, level, error_type, 'no' AS is_text_length_feature,\n"
        "       error_family, operation,\n"
        "       original_form, corrected_form, correction_count,\n"
        "       texts_with_correction,\n"
        "       100.0 * correction_count / "
        "SUM(correction_count) OVER (PARTITION BY cefr, level, error_type) "
        "AS share_of_error_type_percent\n"
        "FROM patterns\n"
        "ORDER BY level, cefr, error_type, correction_count DESC,\n"
        "         original_form, corrected_form"
    )
    temporary = Path(f"{FORM_OUTPUT_PATH}.tmp")
    cursor = connection.execute(query)
    row_count = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FORM_COLUMNS)
        while True:
            rows = cursor.fetchmany(50_000)
            if not rows:
                break
            writer.writerows(rows)
            row_count += len(rows)
    try:
        os.replace(temporary, FORM_OUTPUT_PATH)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {FORM_OUTPUT_PATH}. Close it in Excel or another "
            "program and rerun."
        ) from error
    return FORM_OUTPUT_PATH, row_count


def group_parts(group_type: str) -> tuple[str, str, str]:
    if group_type == "overall":
        return "'' AS cefr", "NULL::INTEGER AS level", ""
    if group_type == "cefr":
        return "cefr", "NULL::INTEGER AS level", "GROUP BY cefr"
    return "cefr", "level", "GROUP BY cefr, level"


def create_values_table(connection, results_path: Path, metadata: list[dict]) -> None:
    scan = csv_scan(results_path)
    columns = [
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
    ]
    by_name = {name.casefold(): name for name in columns}
    required = {"filename", "cefr", "level", *BASE_COLUMNS}
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(
            f"Missing columns in {results_path.name}: {missing}\n"
            f"Available columns: {columns}"
        )
    missing_features = [
        row["feature"] for row in metadata if row["feature"].casefold() not in by_name
    ]
    if missing_features:
        raise ValueError(
            f"ERRANT error-type columns missing from {results_path.name}: "
            f"{missing_features}"
        )
    base_projections = [
        f"TRY_CAST(TRIM({sql_identifier(by_name[name])}) AS DOUBLE) AS {name}"
        for name in BASE_COLUMNS
    ]
    error_projections = [
        f"TRY_CAST(TRIM({sql_identifier(by_name[row['feature'].casefold()])}) "
        f"AS DOUBLE) AS error_{index}"
        for index, row in enumerate(metadata)
    ]
    cefr = sql_identifier(by_name["cefr"])
    level = sql_identifier(by_name["level"])
    connection.execute("DROP TABLE IF EXISTS errant_values")
    connection.execute(
        "CREATE TEMP TABLE errant_values AS\n"
        "SELECT UPPER(TRIM(" + cefr + ")) AS cefr,\n"
        "       TRY_CAST(TRIM(" + level + ") AS INTEGER) AS level,\n       "
        + ",\n       ".join(base_projections + error_projections)
        + f"\nFROM {scan}"
    )
    validation_columns = ", ".join(
        f"COUNT({name}) AS {name}_count" for name in BASE_COLUMNS
    )
    validation = connection.execute(
        f"SELECT COUNT(*) AS rows, {validation_columns} FROM errant_values"
    ).fetchone()
    total = int(validation[0])
    invalid = [
        name for name, count in zip(BASE_COLUMNS, validation[1:]) if count != total
    ]
    if invalid:
        raise ValueError(
            f"Missing or non-numeric values in required columns: {invalid}"
        )
    if total == 0:
        raise ValueError(f"No data rows found in {results_path}")
    missing_cefr, invalid_level = connection.execute(
        "SELECT COUNT_IF(cefr IS NULL OR cefr = ''), "
        "COUNT_IF(level IS NULL OR level NOT BETWEEN 1 AND 15) "
        "FROM errant_values"
    ).fetchone()
    if missing_cefr or invalid_level:
        raise ValueError(
            f"Invalid labels in {results_path.name}: {missing_cefr:,} missing CEFR "
            f"value(s), {invalid_level:,} missing or invalid level value(s)."
        )


def read_stats(connection) -> list[dict]:
    rows = []
    aggregates = (
        "COUNT(*) AS text_count, "
        "SUM(original_word_count), SUM(corrected_word_count), "
        "SUM(total_errors), AVG(total_errors), AVG(distinct_error_types), "
        "SUM(missing_errors), SUM(unnecessary_errors), SUM(replacement_errors), "
        "SUM(other_errors), SUM(affected_original_tokens), "
        "SUM(correction_token_equivalents), SUM(error_free), "
        "AVG(surface_accuracy_proxy)"
    )
    for group_type in ("overall", "cefr", "level"):
        cefr, level, group_by = group_parts(group_type)
        result = connection.execute(
            f"SELECT {cefr}, {level}, {aggregates} "
            f"FROM errant_values {group_by} ORDER BY level, cefr"
        ).fetchall()
        for values in result:
            original_words = int(round(values[3] or 0))
            corrected_words = int(round(values[4] or 0))
            total_errors = int(round(values[5] or 0))
            missing_errors = int(round(values[8] or 0))
            unnecessary_errors = int(round(values[9] or 0))
            replacement_errors = int(round(values[10] or 0))
            other_errors = int(round(values[11] or 0))
            affected_tokens = int(round(values[12] or 0))
            equivalents = int(round(values[13] or 0))
            error_free = int(round(values[14] or 0))
            text_count = int(values[2])
            rows.append(
                {
                    "group_type": group_type,
                    "cefr": values[0] or "",
                    "level": "" if values[1] is None else int(values[1]),
                    "text_count": text_count,
                    "original_word_count": original_words,
                    "corrected_word_count": corrected_words,
                    "total_errors": total_errors,
                    "errors_per_100_words": rate(total_errors, original_words),
                    "mean_errors_per_text": float(values[6] or 0.0),
                    "mean_distinct_error_types": float(values[7] or 0.0),
                    "missing_errors": missing_errors,
                    "missing_errors_per_100_words": rate(
                        missing_errors, original_words
                    ),
                    "unnecessary_errors": unnecessary_errors,
                    "unnecessary_errors_per_100_words": rate(
                        unnecessary_errors, original_words
                    ),
                    "replacement_errors": replacement_errors,
                    "replacement_errors_per_100_words": rate(
                        replacement_errors, original_words
                    ),
                    "other_errors": other_errors,
                    "other_errors_per_100_words": rate(other_errors, original_words),
                    "affected_original_tokens": affected_tokens,
                    "affected_original_tokens_per_100_words": rate(
                        affected_tokens, original_words
                    ),
                    "correction_token_equivalents": equivalents,
                    "correction_token_equivalents_per_100_words": rate(
                        equivalents, original_words
                    ),
                    "error_free_texts": error_free,
                    "error_free_text_percent": rate(error_free, text_count),
                    "mean_surface_accuracy_proxy": float(values[15] or 0.0),
                    "token_weighted_surface_accuracy_proxy": max(
                        0.0, 1.0 - divide(equivalents, original_words)
                    ),
                }
            )
    return rows


def read_error_types(connection, metadata: list[dict], stats: list[dict]) -> list[dict]:
    group_stats = {
        (row["group_type"], row["cefr"], row["level"]): row for row in stats
    }
    aggregate_parts = []
    for index in range(len(metadata)):
        aggregate_parts.extend(
            (
                f"SUM(error_{index} * original_word_count / 100.0) AS count_{index}",
                f"COUNT_IF(error_{index} > 0) AS texts_{index}",
            )
        )
    aggregates = ", ".join(aggregate_parts)
    output = []
    overall_counts = {}
    for group_type in ("overall", "cefr", "level"):
        cefr, level, group_by = group_parts(group_type)
        result = connection.execute(
            f"SELECT {cefr}, {level}, {aggregates} "
            f"FROM errant_values {group_by} ORDER BY level, cefr"
        ).fetchall()
        for values in result:
            group_cefr = values[0] or ""
            group_level = "" if values[1] is None else int(values[1])
            group = group_stats[(group_type, group_cefr, group_level)]
            group_rows = []
            for index, item in enumerate(metadata):
                count = int(round(values[2 + index * 2] or 0))
                texts_with_error = int(values[3 + index * 2] or 0)
                if group_type == "overall":
                    overall_counts[item["error_type"]] = count
                group_rows.append(
                    {
                        "group_type": group_type,
                        "cefr": group_cefr,
                        "level": group_level,
                        "error_type": item["error_type"],
                        "is_text_length_feature": "no",
                        "operation": item["operation"],
                        "error_family": item["error_family"],
                        "error_count": count,
                        "errors_per_100_words": rate(
                            count, group["original_word_count"]
                        ),
                        "texts_with_error": texts_with_error,
                        "total_texts": group["text_count"],
                        "percentage_of_texts_with_error": rate(
                            texts_with_error, group["text_count"]
                        ),
                        "share_of_group_errors_percent": rate(
                            count, group["total_errors"]
                        ),
                    }
                )
            by_count = sorted(
                group_rows, key=lambda row: (-row["error_count"], row["error_type"])
            )
            for rank, row in enumerate(by_count, start=1):
                row["rank_by_error_count_within_group"] = rank
            by_percentage = sorted(
                group_rows,
                key=lambda row: (
                    -row["percentage_of_texts_with_error"],
                    -row["errors_per_100_words"],
                    row["error_type"],
                ),
            )
            for rank, row in enumerate(by_percentage, start=1):
                row["rank_by_text_percentage_within_group"] = rank
            output.extend(by_count)

    for item in metadata:
        reconstructed = overall_counts[item["error_type"]]
        if reconstructed != item["corpus_error_count"]:
            raise ValueError(
                f"{item['error_type']}: reconstructed count is {reconstructed:,}, "
                f"but metadata reports {item['corpus_error_count']:,}. "
                "Regenerate ERRANT outputs with run_ERRANT.py --export-only."
            )
    overall_total = sum(overall_counts.values())
    expected_total = group_stats[("overall", "", "")]["total_errors"]
    if overall_total != expected_total:
        raise ValueError(
            f"Error-type counts total {overall_total:,}, but total_errors is "
            f"{expected_total:,}. Regenerate ERRANT outputs."
        )
    return output


def group_table(stats: list[dict], group_type: str) -> list[str]:
    selected = [row for row in stats if row["group_type"] == group_type]
    if group_type == "cefr":
        heading = "CEFR | Texts | Errors/100 words | Error-free % | Accuracy proxy"
        lines = [heading, "-" * len(heading)]
        lines.extend(
            f"{row['cefr']:>4} | {row['text_count']:>7,} | "
            f"{row['errors_per_100_words']:>16.4f} | "
            f"{row['error_free_text_percent']:>12.2f} | "
            f"{row['mean_surface_accuracy_proxy']:>14.4f}"
            for row in selected
        )
        return lines
    heading = "Level | CEFR | Texts | Errors/100 words | Error-free % | Accuracy proxy"
    lines = [heading, "-" * len(heading)]
    lines.extend(
        f"{row['level']:>5} | {row['cefr']:>4} | {row['text_count']:>7,} | "
        f"{row['errors_per_100_words']:>16.4f} | "
        f"{row['error_free_text_percent']:>12.2f} | "
        f"{row['mean_surface_accuracy_proxy']:>14.4f}"
        for row in selected
    )
    return lines


def write_summary(stats: list[dict], errors: list[dict]) -> Path:
    overall = next(row for row in stats if row["group_type"] == "overall")
    overall_errors = [
        row
        for row in errors
        if row["group_type"] == "overall" and row["error_count"] > 0
    ]
    lines = [
        "ERRANT BASIC ANALYSIS",
        "=" * 72,
        f"Original word tokens: {overall['original_word_count']:,}",
        f"Corrected word tokens: {overall['corrected_word_count']:,}",
        f"Detected edits: {overall['total_errors']:,}",
        f"Errors per 100 original words: {overall['errors_per_100_words']:.6f}",
        f"Error-free texts: {overall['error_free_texts']:,} "
        f"({overall['error_free_text_percent']:.2f}%)",
        f"Mean surface accuracy proxy: {overall['mean_surface_accuracy_proxy']:.6f}",
        "Token-weighted surface accuracy proxy: "
        f"{overall['token_weighted_surface_accuracy_proxy']:.6f}",
        "",
        "ERROR OPERATIONS",
        "-" * 72,
        f"Missing: {overall['missing_errors']:,} "
        f"({overall['missing_errors_per_100_words']:.6f}/100 words)",
        f"Unnecessary: {overall['unnecessary_errors']:,} "
        f"({overall['unnecessary_errors_per_100_words']:.6f}/100 words)",
        f"Replacement: {overall['replacement_errors']:,} "
        f"({overall['replacement_errors_per_100_words']:.6f}/100 words)",
        f"Other: {overall['other_errors']:,} "
        f"({overall['other_errors_per_100_words']:.6f}/100 words)",
        "",
        "CEFR SUMMARY",
        "-" * 72,
        *group_table(stats, "cefr"),
        "",
        "LEVEL SUMMARY",
        "-" * 72,
        *group_table(stats, "level"),
        "",
        "TOP 20 ERRANT ERROR TYPES",
        "-" * 72,
    ]
    if overall_errors:
        lines.extend(
            f"{row['rank_by_error_count_within_group']:>2}. {row['error_type']} | "
            f"{row['error_family']} | {row['operation']} | "
            f"{row['error_count']:,} "
            f"({row['share_of_group_errors_percent']:.2f}% of edits)"
            for row in overall_errors[:20]
        )
    else:
        lines.append("No ERRANT edits were detected.")
    lines.extend(["", "TOP ERROR TYPES BY LEVEL", "-" * 72])
    for group in (row for row in stats if row["group_type"] == "level"):
        top = [
            row
            for row in errors
            if row["group_type"] == "level"
            and row["cefr"] == group["cefr"]
            and row["level"] == group["level"]
            and row["error_count"] > 0
            and row["rank_by_error_count_within_group"] <= 5
        ]
        lines.append(f"Level {group['level']} ({group['cefr']}):")
        if top:
            lines.extend(
                f"  {row['rank_by_error_count_within_group']}. {row['error_type']}: "
                f"{row['error_count']:,} | "
                f"{row['errors_per_100_words']:.4f}/100 words | "
                f"{row['percentage_of_texts_with_error']:.2f}% of texts"
                for row in top
            )
        else:
            lines.append("  No ERRANT errors detected.")

    path = ANALYSIS_DIR / "errant_analysis_summary.txt"
    temporary = Path(f"{path}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.replace(temporary, path)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {path}. Close it in another program and rerun."
        ) from error
    return path


def main() -> None:
    started = time.perf_counter()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = resolve_input(RESULTS_PATH)
    metadata_path = resolve_input(ERROR_METADATA_PATH)
    edits_path = resolve_input(EDITS_PATH)
    metadata = read_metadata(metadata_path)

    print("ERRANT basic analysis")
    print(f"Input root: {ERRANT_DIR}")
    print(f"Output folder: {ANALYSIS_DIR}")
    print(f"Error types: {len(metadata):,}")

    connection = duckdb.connect(":memory:")
    connection.execute(f"SET threads={THREADS}")
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute("SET preserve_insertion_order=false")
    try:
        create_values_table(connection, results_path, metadata)
        stats = read_stats(connection)
        errors = read_error_types(connection, metadata, stats)
        form_path, form_rows = write_form_patterns(connection, edits_path)
    finally:
        connection.close()

    stats_path = BASIC_STATS_OUTPUT_PATH
    errors_path = ERROR_TYPE_OUTPUT_PATH
    atomic_csv(stats_path, STATS_COLUMNS, stats)
    atomic_csv(errors_path, ERROR_COLUMNS, errors)
    summary_path = write_summary(stats, errors)
    remove_legacy_outputs()

    print(f"\nCompleted in {time.perf_counter() - started:.1f}s")
    print(f"ERRANT statistics: {stats_path}")
    print(f"Error types:       {errors_path}")
    print(f"Form patterns:     {form_path} ({form_rows:,} rows)")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()
