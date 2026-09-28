from __future__ import annotations
import csv
import math
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import duckdb

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
    "TAALES_COVERAGE": METHODS_DIR / "TAALES" / "taales_results_final_index_coverage.csv",
    "POLKE": METHODS_DIR / "POLKE" / "polke_results.csv",
    "ERRANT": METHODS_DIR / "ERRANT" / "errant_results.csv",
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

ERRANT_BASE_FEATURES = {
    "errors_per_100_words",
    "missing_errors_per_100_words",
    "unnecessary_errors_per_100_words",
    "replacement_errors_per_100_words",
    "other_errors_per_100_words",
    "surface_accuracy_proxy",
    "error_free",
}

TEXT_LENGTH_FEATURES = length_features()

THREADS = 4
MEMORY_LIMIT = "4GB"
SPEARMAN_BATCH_SIZE = 8
FDR_ALPHA = 0.05
PAIR_FEATURES_PER_SOURCE = 50
PAIR_SAMPLE_SIZE = 20_000
PAIR_BATCH_SIZE = 200
TOP_RESULTS = 20

COMPLEXITY_OUTPUT_PATH = ANALYSIS_DIR / "complexity_feature_level_correlations.csv"
ERRANT_OUTPUT_PATH = ANALYSIS_DIR / "errant_feature_level_correlations.csv"
COMBINED_OUTPUT_PATH = ANALYSIS_DIR / "combined_feature_level_correlations.csv"
PAIR_OUTPUT_PATH = ANALYSIS_DIR / "feature_feature_correlations.csv"
LEGACY_OUTPUT_PATHS = (
    ANALYSIS_DIR / "feature_level_correlations.csv",
    ANALYSIS_DIR / "highly_correlated_feature_pairs.csv",
)

OUTPUT_COLUMNS = (
    "source",
    "feature",
    "is_text_length_feature",
    "pearson_correlation_with_level",
    "pearson_absolute_correlation",
    "pearson_direction",
    "pearson_strength",
    "pearson_asymptotic_p_value",
    "pearson_fdr_q_value",
    "pearson_significant_after_fdr",
    "spearman_correlation_with_level",
    "spearman_absolute_correlation",
    "spearman_direction",
    "spearman_strength",
    "spearman_asymptotic_p_value",
    "spearman_fdr_q_value",
    "spearman_significant_after_fdr",
)

PAIR_COLUMNS = (
    "pair_scope",
    "contains_text_length_feature",
    "source_1",
    "feature_1",
    "source_2",
    "feature_2",
    "pearson_correlation",
    "absolute_correlation",
    "direction",
    "strength",
)


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


def column_name(columns: list[str], target: str) -> str | None:
    return next((name for name in columns if name.casefold() == target), None)


def feature_columns(source: str, columns: list[str]) -> list[str]:
    if source == "ERRANT":
        return [
            name
            for name in columns
            if name.casefold() in ERRANT_BASE_FEATURES
            or (
                name.casefold().startswith("errant_")
                and name.casefold().endswith("_per_100_words")
            )
        ]
    return [
        name for name in columns if name.casefold() not in NON_FEATURE_COLUMNS
    ]


def is_text_length_feature(source: str, feature: str) -> bool:
    return feature.casefold() in TEXT_LENGTH_FEATURES.get(source, set())


def level_expression(columns: list[str]) -> str:
    filename_name = column_name(columns, "filename")
    if filename_name is None:
        raise ValueError(f"No filename column. Available columns: {columns}")
    filename = sql_identifier(filename_name)
    parsed = (
        f"TRY_CAST(regexp_extract({filename}, '_level_([0-9]+)_', 1) AS DOUBLE)"
    )
    level_name = column_name(columns, "level")
    if level_name is None:
        return parsed
    return (
        f"COALESCE(TRY_CAST(TRIM({sql_identifier(level_name)}) AS DOUBLE), {parsed})"
    )


def text_id_expression(columns: list[str]) -> str:
    filename_name = column_name(columns, "filename")
    if filename_name is None:
        raise ValueError(f"No filename column. Available columns: {columns}")
    filename = sql_identifier(filename_name)
    parsed = (
        "NULLIF(regexp_extract(LOWER(CAST("
        f"{filename} AS VARCHAR)), '([0-9]+)_cefr_', 1), '')"
    )
    writing_id_name = column_name(columns, "writing_id")
    if writing_id_name is None:
        return parsed
    writing_id = sql_identifier(writing_id_name)
    normalized = f"CAST(TRY_CAST(TRIM({writing_id}) AS BIGINT) AS VARCHAR)"
    return f"COALESCE({normalized}, {parsed})"


def numeric_expression(column: str) -> str:
    value = f"TRY_CAST(TRIM({sql_identifier(column)}) AS DOUBLE)"
    return f"CASE WHEN isfinite({value}) THEN {value} ELSE NULL END"


def strength(absolute: float | None) -> str:
    if absolute is None:
        return "not calculable"
    if absolute < 0.20:
        return "very weak"
    if absolute < 0.40:
        return "weak"
    if absolute < 0.60:
        return "moderate"
    if absolute < 0.80:
        return "strong"
    return "very strong"


def correlation_details(value) -> tuple[float | None, float | None, str, str]:
    if value is None:
        return None, None, "not calculable", "not calculable"
    correlation = float(value)
    if not math.isfinite(correlation):
        return None, None, "not calculable", "not calculable"
    correlation = max(-1.0, min(1.0, correlation))
    absolute = abs(correlation)
    direction = (
        "positive"
        if correlation > 0
        else "negative"
        if correlation < 0
        else "none"
    )
    return correlation, absolute, direction, strength(absolute)


def asymptotic_p_value(correlation: float | None, observations: int) -> float | None:
    if correlation is None or observations < 4:
        return None
    absolute = abs(correlation)
    if absolute >= 1.0:
        return 0.0
    statistic = math.atanh(absolute) * math.sqrt(observations - 3)
    return math.erfc(statistic / math.sqrt(2.0))


def apply_fdr(rows: list[dict], method: str) -> None:
    p_key = f"{method}_asymptotic_p_value"
    q_key = f"{method}_fdr_q_value"
    significant_key = f"{method}_significant_after_fdr"
    available = sorted(
        (
            (row[p_key], index)
            for index, row in enumerate(rows)
            if row[p_key] is not None
        ),
        key=lambda item: item[0],
    )
    adjusted = [None] * len(rows)
    previous = 1.0
    tests = len(available)
    for rank in range(tests, 0, -1):
        p_value, index = available[rank - 1]
        previous = min(previous, p_value * tests / rank)
        adjusted[index] = previous
    for index, row in enumerate(rows):
        q_value = adjusted[index]
        row[q_key] = q_value
        row[significant_key] = (
            "yes" if q_value is not None and q_value <= FDR_ALPHA else "no"
        )


def memory_error(error: Exception) -> bool:
    message = str(error).casefold()
    return "out of memory" in message or "allocation failure" in message


def fetch_with_fallback(connection, query: str):
    options = tuple(dict.fromkeys((THREADS, 2, 1)))
    for position, threads in enumerate(options):
        connection.execute(f"SET threads={threads}")
        try:
            return connection.execute(query).fetchone()
        except Exception as error:
            if not memory_error(error) or position == len(options) - 1:
                raise
            next_threads = options[position + 1]
            print(f"    RAM limit reached; retrying with {next_threads} thread(s)")


def spearman_query(indices: list[int], complete_indices: set[int]) -> str:
    ranks = []
    correlations = []
    for index in indices:
        value = f"value_{index}"
        if index in complete_indices:
            ranks.extend(
                (
                    f"RANK() OVER (ORDER BY {value}) + "
                    f"(COUNT(*) OVER (PARTITION BY {value}) - 1) / 2.0 "
                    f"AS feature_rank_{index}",
                    f"level_rank_all AS level_rank_{index}",
                )
            )
        else:
            valid = f"({value} IS NOT NULL)"
            ranks.extend(
                (
                    f"CASE WHEN {valid} THEN RANK() OVER "
                    f"(PARTITION BY {valid} ORDER BY {value}) + "
                    f"(COUNT(*) OVER (PARTITION BY {valid}, {value}) - 1) / 2.0 "
                    f"END AS feature_rank_{index}",
                    f"CASE WHEN {valid} THEN RANK() OVER "
                    f"(PARTITION BY {valid} ORDER BY level) + "
                    f"(COUNT(*) OVER (PARTITION BY {valid}, level) - 1) / 2.0 "
                    f"END AS level_rank_{index}",
                )
            )
        correlations.append(
            f"corr(feature_rank_{index}, level_rank_{index}) AS rho_{index}"
        )
    return (
        "WITH ranks AS (\n"
        "    SELECT\n        "
        + ",\n        ".join(ranks)
        + "\n    FROM numeric_values\n"
        ")\nSELECT "
        + ", ".join(correlations)
        + " FROM ranks"
    )


def calculate_spearman(
    connection, active_indices: list[int], complete_indices: set[int]
) -> dict[int, float | None]:
    def batch(indices: list[int]) -> list[float | None]:
        try:
            values = fetch_with_fallback(
                connection, spearman_query(indices, complete_indices)
            )
            return [
                float(value)
                if value is not None and math.isfinite(float(value))
                else None
                for value in values
            ]
        except Exception as error:
            if not memory_error(error) or len(indices) == 1:
                raise
            midpoint = len(indices) // 2
            print(f"    Splitting a {len(indices)}-feature Spearman batch")
            return batch(indices[:midpoint]) + batch(indices[midpoint:])

    output = {}
    completed = 0
    next_progress = 100
    for start in range(0, len(active_indices), SPEARMAN_BATCH_SIZE):
        indices = active_indices[start : start + SPEARMAN_BATCH_SIZE]
        values = batch(indices)
        output.update(zip(indices, values))
        completed += len(indices)
        if completed >= next_progress or completed == len(active_indices):
            print(
                f"    Spearman progress: {completed:,}/{len(active_indices):,} features"
            )
            next_progress = ((completed // 100) + 1) * 100
    return output


def create_feature_sample(
    connection,
    source: str,
    source_index: int,
    features: list[str],
    rows: list[dict],
) -> dict:
    feature_indices = {feature: index for index, feature in enumerate(features)}
    candidates = [row for row in rows if strongest(row) is not None]
    candidates.sort(
        key=lambda row: (
            row["spearman_absolute_correlation"] is None,
            -(row["spearman_absolute_correlation"] or 0.0),
            -(row["pearson_absolute_correlation"] or 0.0),
            row["feature"],
        )
    )
    indices = [
        feature_indices[row["feature"]]
        for row in candidates[:PAIR_FEATURES_PER_SOURCE]
    ]
    if source_index == 0:
        connection.execute("DROP TABLE IF EXISTS pair_sample_ids")
        connection.execute(
            "CREATE TEMP TABLE pair_sample_ids AS "
            "SELECT filename FROM numeric_values "
            f"ORDER BY hash(filename) LIMIT {PAIR_SAMPLE_SIZE}"
        )
    table = f"pair_source_{source_index}"
    selected = ", ".join(f"n.value_{index}" for index in indices)
    selection = "n.filename" + (f", {selected}" if selected else "")
    connection.execute(
        f"CREATE TEMP TABLE {table} AS "
        f"SELECT {selection} FROM numeric_values n "
        "INNER JOIN pair_sample_ids USING (filename)"
    )
    matched = int(
        fetch_with_fallback(connection, f"SELECT COUNT(*) FROM {table}")[0]
    )
    if matched == 0:
        raise ValueError(
            f"{source}: no normalized writing IDs match the shared feature-"
            "correlation sample."
        )
    print(f"    matched feature-pair sample: {matched:,} texts")
    return {
        "source": source,
        "table": table,
        "features": [(features[index], f"value_{index}") for index in indices],
    }


def calculate_feature_pairs(connection, samples: list[dict]) -> list[dict]:
    metadata = []
    projections = []
    for source_index, sample in enumerate(samples):
        alias = f"s{source_index}"
        for feature, column in sample["features"]:
            combined_column = f"pair_value_{len(metadata)}"
            projections.append(f"{alias}.{column} AS {combined_column}")
            metadata.append((sample["source"], feature, combined_column))
    if len(metadata) < 2:
        return []

    joins = [f"{samples[0]['table']} s0"]
    joins.extend(
        f"INNER JOIN {sample['table']} s{index} USING (filename)"
        for index, sample in enumerate(samples[1:], start=1)
    )
    connection.execute("DROP TABLE IF EXISTS feature_pair_values")
    connection.execute(
        "CREATE TEMP TABLE feature_pair_values AS SELECT s0.filename, "
        + ", ".join(projections)
        + " FROM "
        + " ".join(joins)
    )
    sample_rows = int(
        fetch_with_fallback(
            connection, "SELECT COUNT(*) FROM feature_pair_values"
        )[0]
    )
    if sample_rows == 0:
        raise ValueError(
            "The available result files do not share any normalized writing IDs "
            "across the complete feature-correlation sample."
        )
    print(f"  Matched texts used for feature pairs: {sample_rows:,}")

    pairs = [
        (left, right)
        for left in range(len(metadata))
        for right in range(left + 1, len(metadata))
    ]
    output = []
    completed = 0
    next_progress = 5_000
    for start in range(0, len(pairs), PAIR_BATCH_SIZE):
        batch = pairs[start : start + PAIR_BATCH_SIZE]
        values = fetch_with_fallback(
            connection,
            "SELECT "
            + ", ".join(
                f"corr({metadata[left][2]}, {metadata[right][2]})"
                for left, right in batch
            )
            + " FROM feature_pair_values",
        )
        for (left, right), value in zip(batch, values):
            details = correlation_details(value)
            if details[1] is None:
                continue
            source_1, feature_1, _ = metadata[left]
            source_2, feature_2, _ = metadata[right]
            output.append(
                {
                    "pair_scope": (
                        "within_source" if source_1 == source_2 else "cross_source"
                    ),
                    "source_1": source_1,
                    "feature_1": feature_1,
                    "source_2": source_2,
                    "feature_2": feature_2,
                    "pearson_correlation": details[0],
                    "absolute_correlation": details[1],
                    "direction": details[2],
                    "strength": details[3],
                }
            )
        completed += len(batch)
        if completed >= next_progress or completed == len(pairs):
            print(f"  Feature-pair progress: {completed:,}/{len(pairs):,}")
            next_progress = ((completed // 5_000) + 1) * 5_000
    connection.execute("DROP TABLE feature_pair_values")
    output.sort(
        key=lambda row: (
            -row["absolute_correlation"],
            row["source_1"],
            row["feature_1"],
            row["source_2"],
            row["feature_2"],
        )
    )
    return output


def analyze_source(
    connection, source: str, source_index: int, path: Path
) -> tuple[list[dict], dict]:
    scan = csv_scan(path)
    columns = [
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
    ]
    features = feature_columns(source, columns)
    if not features:
        raise ValueError(f"{source}: no candidate feature columns in {path.name}")
    level = level_expression(columns)
    text_id = text_id_expression(columns)
    projections = ",\n           ".join(
        f"{numeric_expression(feature)} AS value_{index}"
        for index, feature in enumerate(features)
    )

    print(f"  {source}: {len(features):,} candidate features from {path.name}")
    started = time.perf_counter()
    connection.execute(f"SET threads={THREADS}")
    connection.execute("DROP TABLE IF EXISTS numeric_values")
    connection.execute(
        "CREATE TABLE numeric_values AS\n"
        "WITH converted AS (\n"
        f"    SELECT {text_id} AS filename, "
        f"{level} AS level,\n           {projections}\n"
        f"    FROM {scan}\n"
        ")\n"
        "SELECT *, RANK() OVER (ORDER BY level) + "
        "(COUNT(*) OVER (PARTITION BY level) - 1) / 2.0 AS level_rank_all\n"
        "FROM converted\n"
        "WHERE level BETWEEN 1 AND 15"
    )

    counts = fetch_with_fallback(
        connection,
        "SELECT COUNT(*)"
        + "".join(
            f", COUNT(value_{index})" for index in range(len(features))
        )
        + " FROM numeric_values",
    )
    total_rows = int(counts[0])
    if total_rows == 0:
        raise ValueError(f"{source}: no rows with levels 1-15 in {path.name}")
    identifier_count, distinct_identifier_count = fetch_with_fallback(
        connection,
        "SELECT COUNT(filename), COUNT(DISTINCT filename) FROM numeric_values",
    )
    missing_identifiers = total_rows - int(identifier_count)
    duplicate_identifiers = int(identifier_count) - int(distinct_identifier_count)
    if missing_identifiers or duplicate_identifiers:
        raise ValueError(
            f"{source}: normalized writing IDs are invalid in {path.name}: "
            f"{missing_identifiers:,} missing and {duplicate_identifiers:,} duplicate."
        )
    active_indices = [
        index for index, count in enumerate(counts[1:]) if int(count) > 0
    ]
    if not active_indices:
        raise ValueError(f"{source}: no numeric feature values in {path.name}")
    complete_indices = {
        index for index in active_indices if int(counts[index + 1]) == total_rows
    }

    print("    calculating Pearson correlations")
    pearson_values = fetch_with_fallback(
        connection,
        "SELECT "
        + ", ".join(
            f"corr(value_{index}, level) AS r_{index}" for index in active_indices
        )
        + " FROM numeric_values",
    )
    pearson = dict(zip(active_indices, pearson_values))

    print("    calculating exact Spearman correlations")
    spearman = calculate_spearman(connection, active_indices, complete_indices)

    rows = []
    for index in active_indices:
        pearson_details = correlation_details(pearson[index])
        spearman_details = correlation_details(spearman[index])
        observations = int(counts[index + 1])
        rows.append(
            {
                "source": source,
                "feature": features[index],
                "pearson_correlation_with_level": pearson_details[0],
                "pearson_absolute_correlation": pearson_details[1],
                "pearson_direction": pearson_details[2],
                "pearson_strength": pearson_details[3],
                "pearson_asymptotic_p_value": asymptotic_p_value(
                    pearson_details[0], observations
                ),
                "spearman_correlation_with_level": spearman_details[0],
                "spearman_absolute_correlation": spearman_details[1],
                "spearman_direction": spearman_details[2],
                "spearman_strength": spearman_details[3],
                "spearman_asymptotic_p_value": asymptotic_p_value(
                    spearman_details[0], observations
                ),
            }
        )
    sample = create_feature_sample(
        connection, source, source_index, features, rows
    )
    connection.execute("DROP TABLE numeric_values")
    print(
        f"    {total_rows:,} texts, {len(rows):,} numeric features, "
        f"{time.perf_counter() - started:.1f}s"
    )
    return rows, sample


def strongest(row: dict) -> float | None:
    values = (
        row["pearson_absolute_correlation"],
        row["spearman_absolute_correlation"],
    )
    available = [value for value in values if value is not None]
    return max(available) if available else None


def add_text_length_flags(results: list[dict], pairs: list[dict]) -> None:
    for row in results:
        row["is_text_length_feature"] = (
            "yes"
            if is_text_length_feature(row["source"], row["feature"])
            else "no"
        )

    for row in pairs:
        contains_length_feature = is_text_length_feature(
            row["source_1"], row["feature_1"]
        ) or is_text_length_feature(row["source_2"], row["feature_2"])
        row["contains_text_length_feature"] = (
            "yes" if contains_length_feature else "no"
        )


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


def write_summary(results: list[dict], pairs: list[dict]) -> Path:
    results_without_length = [
        row for row in results if row["is_text_length_feature"] == "no"
    ]
    pairs_without_length = [
        row for row in pairs if row["contains_text_length_feature"] == "no"
    ]
    lines = [
        "LEVEL CORRELATION SUMMARY",
        "=" * 72,
        f"Features checked: {len(results):,}",
        f"Features that are not length-dependent: "
        f"{len(results_without_length):,}",
        f"Length-dependent features: "
        f"{len(results) - len(results_without_length):,}",
        "Target: EFCAMDAT level (1-15)",
        "Pearson: linear association",
        "Spearman: monotonic association using average ranks for ties",
        "Benjamini-Hochberg FDR alpha: 0.05",
        LENGTH_NOTE,
    ]

    for method, symbol in (("spearman", "rho"), ("pearson", "r")):
        correlation_key = f"{method}_correlation_with_level"
        absolute_key = f"{method}_absolute_correlation"
        strength_key = f"{method}_strength"
        q_key = f"{method}_fdr_q_value"
        significant_key = f"{method}_significant_after_fdr"
        name = method.upper()
        lines.extend(["", f"{name} STRENGTH COUNTS", "-" * 72])
        for label in (
            "very strong",
            "strong",
            "moderate",
            "weak",
            "very weak",
            "not calculable",
        ):
            lines.append(
                f"{label}: {sum(row[strength_key] == label for row in results):,}"
            )
        lines.append(
            "Significant after FDR: "
            f"{sum(row[significant_key] == 'yes' for row in results):,}"
        )

        calculable = [row for row in results if row[correlation_key] is not None]
        filtered_calculable = [
            row for row in calculable if row["is_text_length_feature"] == "no"
        ]

        for scope, candidates in (
            ("ALL FEATURES", calculable),
            ("WITHOUT LENGTH-DEPENDENT FEATURES", filtered_calculable),
        ):
            sections = (
                (
                    f"TOP {TOP_RESULTS} {name} CORRELATIONS ({scope})",
                    sorted(
                        candidates,
                        key=lambda row: row[absolute_key],
                        reverse=True,
                    )[:TOP_RESULTS],
                ),
                (
                    f"TOP {TOP_RESULTS} POSITIVE {name} CORRELATIONS ({scope})",
                    sorted(
                        (
                            row
                            for row in candidates
                            if row[correlation_key] > 0
                        ),
                        key=lambda row: row[correlation_key],
                        reverse=True,
                    )[:TOP_RESULTS],
                ),
                (
                    f"TOP {TOP_RESULTS} NEGATIVE {name} CORRELATIONS ({scope})",
                    sorted(
                        (
                            row
                            for row in candidates
                            if row[correlation_key] < 0
                        ),
                        key=lambda row: row[correlation_key],
                    )[:TOP_RESULTS],
                ),
            )
            for heading, section_rows in sections:
                lines.extend(["", heading, "-" * len(heading)])
                if section_rows:
                    lines.extend(
                        f"{row['source']} | {row['feature']}: "
                        f"{symbol}={row[correlation_key]:.4f} "
                        f"({row[strength_key]}, q={row[q_key]:.3g})"
                        for row in section_rows
                    )
                else:
                    lines.append("No calculable correlations.")

    lines.extend(
        [
            "",
            "FEATURE-FEATURE CORRELATIONS",
            "-" * 72,
            f"Candidates: top {PAIR_FEATURES_PER_SOURCE} Spearman-ranked "
            "features per source",
            f"Deterministic matched sample: up to {PAIR_SAMPLE_SIZE:,} texts",
            "All calculable within-source and cross-source pairs are retained.",
            f"Within-source pairs: "
            f"{sum(row['pair_scope'] == 'within_source' for row in pairs):,}",
            f"Cross-source pairs: "
            f"{sum(row['pair_scope'] == 'cross_source' for row in pairs):,}",
            f"Pairs without length-dependent features: "
            f"{len(pairs_without_length):,}",
        ]
    )
    for label in ("very strong", "strong", "moderate", "weak", "very weak"):
        lines.append(
            f"{label}: {sum(row['strength'] == label for row in pairs):,}"
        )
    for scope, selected_pairs in (
        ("ALL FEATURES", pairs),
        ("WITHOUT LENGTH-DEPENDENT FEATURES", pairs_without_length),
    ):
        heading = f"TOP {TOP_RESULTS} FEATURE-FEATURE CORRELATIONS ({scope})"
        lines.extend(["", heading, "-" * len(heading)])
        if selected_pairs:
            lines.extend(
                f"{row['source_1']}:{row['feature_1']} <> "
                f"{row['source_2']}:{row['feature_2']}: "
                f"r={row['pearson_correlation']:.4f}"
                for row in selected_pairs[:TOP_RESULTS]
            )
        else:
            lines.append("No calculable feature pairs.")

    lines.extend(
        [
            "",
            "STRENGTH RULE",
            "-" * 72,
            "|correlation| < 0.20: very weak",
            "0.20 <= |correlation| < 0.40: weak",
            "0.40 <= |correlation| < 0.60: moderate",
            "0.60 <= |correlation| < 0.80: strong",
            "|correlation| >= 0.80: very strong",
            "",
            "Positive means the measure tends to increase at higher levels.",
            "Negative means the measure tends to decrease at higher levels.",
            "ERRANT uses normalized rates, surface accuracy, and error-free status.",
            "FDR correction is applied separately across all Pearson tests and all "
            "Spearman tests.",
            "With a very large corpus, a tiny correlation can be statistically "
            "significant; interpret correlation strength as the practical effect.",
            "Highly correlated feature pairs are redundancy candidates, not automatic "
            "deletion decisions.",
            "Correlation does not prove causation or measure multivariable prediction.",
        ]
    )
    path = ANALYSIS_DIR / "correlation_summary.txt"
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
    inputs = dataframe_inputs(FULL_MODULE_DIR.parent, INPUT_FILES)
    skipped = []
    if not inputs:
        checked = "\n".join(
            f"  {source}: {path}" for source, path in INPUT_FILES.items()
        )
        raise FileNotFoundError(
            "None of the supported result files were found. Checked:\n" + checked
        )

    print("Complexity and ERRANT level correlation analysis")
    print(f"Feature input: {next(iter(inputs.values())).parquet}")
    print(f"Output folder: {ANALYSIS_DIR}")
    print(f"Sources found: {', '.join(inputs)}")
    if skipped:
        print(f"Sources skipped (not found): {', '.join(skipped)}")

    with TemporaryDirectory(prefix="correlation_analysis_") as temporary_dir:
        database = Path(temporary_dir) / "correlations.duckdb"
        connection = duckdb.connect(str(database))
        connection.execute(f"SET threads={THREADS}")
        connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
        connection.execute("SET preserve_insertion_order=false")
        try:
            results = []
            samples = []
            for source_index, (source, path) in enumerate(inputs.items()):
                source_results, sample = analyze_source(
                    connection, source, source_index, path
                )
                results.extend(source_results)
                samples.append(sample)
            print("Calculating within-source and cross-source feature correlations")
            pairs = calculate_feature_pairs(connection, samples)
        finally:
            connection.close()

    apply_fdr(results, "pearson")
    apply_fdr(results, "spearman")
    results.sort(
        key=lambda row: (
            row["spearman_absolute_correlation"] is None,
            -(row["spearman_absolute_correlation"] or 0.0),
            -(row["pearson_absolute_correlation"] or 0.0),
            row["source"],
            row["feature"],
        )
    )
    pairs.sort(
        key=lambda row: (
            -row["absolute_correlation"],
            row["source_1"],
            row["feature_1"],
            row["source_2"],
            row["feature_2"],
        )
    )
    add_text_length_flags(results, pairs)
    complexity_results = [row for row in results if row["source"] != "ERRANT"]
    errant_results = [row for row in results if row["source"] == "ERRANT"]
    atomic_csv(COMPLEXITY_OUTPUT_PATH, OUTPUT_COLUMNS, complexity_results)
    atomic_csv(ERRANT_OUTPUT_PATH, OUTPUT_COLUMNS, errant_results)
    atomic_csv(COMBINED_OUTPUT_PATH, OUTPUT_COLUMNS, results)
    atomic_csv(PAIR_OUTPUT_PATH, PAIR_COLUMNS, pairs)
    summary_path = write_summary(results, pairs)
    for path in LEGACY_OUTPUT_PATHS:
        if not path.is_file():
            continue
        try:
            path.unlink()
        except PermissionError as error:
            raise PermissionError(
                f"Cannot remove {path}. Close it in Excel or another program and "
                "rerun."
            ) from error

    print(f"\nCompleted in {time.perf_counter() - started:.1f}s")
    print(f"Complexity correlations: {COMPLEXITY_OUTPUT_PATH}")
    print(f"ERRANT correlations:     {ERRANT_OUTPUT_PATH}")
    print(f"Combined correlations:   {COMBINED_OUTPUT_PATH}")
    print(f"Feature-feature pairs:   {PAIR_OUTPUT_PATH}")
    print(f"Summary:                 {summary_path}")


if __name__ == "__main__":
    main()
