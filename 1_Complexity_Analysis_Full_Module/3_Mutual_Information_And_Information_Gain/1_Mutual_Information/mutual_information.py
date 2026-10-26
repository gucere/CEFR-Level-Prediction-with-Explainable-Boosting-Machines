import argparse
import csv
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import duckdb
import numpy as np
from sklearn.feature_selection import mutual_info_classif


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
# Grammar descriptions and corpus context used by this stage.
import csv as _grammar_csv
import gzip as _grammar_gzip
import math as _grammar_math
import re as _grammar_re
from collections import Counter as _grammar_Counter
from pathlib import Path as _grammar_Path
_grammar_GRAMMAR_COLUMNS = ('grammar_construct_id', 'grammar_description', 'grammar_reference_cefr', 'corpus_first_observed_level', 'corpus_first_common_level', 'corpus_sustained_usage_from_level', 'corpus_grammar_threshold_percent', 'corpus_grammar_status')
_grammar_FEATURE_PATTERN = _grammar_re.compile('polke_(\\d+)_per_100_words', _grammar_re.IGNORECASE)
_grammar_DEFAULT_SETTINGS = {'common_usage_percent': 5.0, 'comparison_thresholds': [2.0, 10.0], 'min_texts_per_level': 100, 'min_feature_coverage_percent': 95.0, 'min_later_levels': 2}

def _grammar_feature_id(source, feature):
    if str(source).strip().upper() != 'POLKE':
        return None
    match = _grammar_FEATURE_PATTERN.fullmatch(str(feature).strip())
    return int(match.group(1)) if match else None

def _grammar_find_csv(folder, name):
    for path in (folder / name, folder / f'{name}.gz'):
        if path.is_file():
            return path
    return None

def _grammar_read_csv(path):
    opener = _grammar_gzip.open if path.suffix.lower() == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        reader = _grammar_csv.DictReader(handle)
        names = [name.strip().casefold() for name in reader.fieldnames or []]
        if not names or len(names) != len(set(names)):
            raise ValueError('missing or duplicate column names')
        reader.fieldnames = names
        rows = list(reader)
    if any((None in row or any((value is None for value in row.values())) for row in rows)):
        raise ValueError('a row does not match the CSV header')
    return [{key: value.strip() for (key, value) in row.items()} for row in rows]

def _grammar_finite_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if _grammar_math.isfinite(result) else None

def _grammar_course_level(value):
    if value in (None, ''):
        return None
    result = _grammar_finite_number(value)
    if result is None or not result.is_integer() or (not 1 <= result <= 15):
        raise ValueError(f'invalid course level: {value}')
    return int(result)

def with_grammar_columns(columns):
    columns = tuple(columns)
    if {'source', 'feature'}.issubset(columns):
        return columns + tuple((name for name in _grammar_GRAMMAR_COLUMNS if name not in columns))
    return columns

class GrammarContext:

    def __init__(self):
        self.files = {}
        self.notices = []
        self.settings = {**_grammar_DEFAULT_SETTINGS, 'comparison_thresholds': [2.0, 10.0]}
        self.descriptions = {}
        self.progressions = {}
        self.threshold = None

    def load(self, thesis_root):
        self.__init__()
        module = _grammar_Path(thesis_root) / '1_Complexity_Analysis_Full_Module'
        directory = module / '2_Basic_Analysis' / '4_Grammar_Development_Analysis'
        summary = directory / 'grammar_development_summary.txt'
        if summary.is_file():
            self.files['summary'] = summary
            try:
                self.read_settings(summary.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError) as error:
                self.notices.append(f'Grammar settings skipped: {error}')
        candidates = {'metadata': module / '1_Complexity_Analysis_Methods' / 'POLKE' / 'polke_feature_metadata.csv', 'usage': directory / 'grammar_usage_by_level.csv', 'progression': directory / 'grammar_progression.csv'}
        for (kind, expected) in candidates.items():
            path = _grammar_find_csv(expected.parent, expected.name)
            if path is None:
                continue
            self.files[kind] = path
            try:
                rows = _grammar_read_csv(path)
                if kind == 'progression':
                    self.read_progressions(rows)
                self.read_descriptions(rows)
            except (OSError, ValueError, KeyError, _grammar_csv.Error) as error:
                self.notices.append(f'Grammar {kind} skipped: {path.name}: {error}')
        return self

    def read_settings(self, text):
        patterns = {'common_usage_percent': 'Main common-usage threshold:\\s*([\\d.]+)%', 'min_texts_per_level': 'Minimum valid texts per structure and level:\\s*([\\d,]+)', 'min_feature_coverage_percent': 'Minimum feature coverage within a level:\\s*([\\d.]+)%', 'min_later_levels': 'including at least\\s+(\\d+)\\s+later levels'}
        settings = {**self.settings}
        for (key, pattern) in patterns.items():
            match = _grammar_re.search(pattern, text)
            if match:
                value = float(match.group(1).replace(',', ''))
                settings[key] = int(value) if key in {'min_texts_per_level', 'min_later_levels'} else value
        match = _grammar_re.search('Thresholds compared:\\s*([^\\n]+)', text)
        if match:
            settings['comparison_thresholds'] = [float(value) for value in _grammar_re.findall('([\\d.]+)%', match.group(1)) if float(value) != settings['common_usage_percent']]
        if not all((0 < value <= 100 for value in (settings['common_usage_percent'], *settings['comparison_thresholds']))):
            raise ValueError('invalid common-usage thresholds')
        if settings['min_texts_per_level'] < 1 or not 1 <= settings['min_later_levels'] <= 14:
            raise ValueError('invalid minimum evidence settings')
        if not 0 < settings['min_feature_coverage_percent'] <= 100:
            raise ValueError('invalid feature coverage threshold')
        self.settings = settings

    def read_descriptions(self, rows):
        descriptions = {}
        for row in rows:
            identity = _grammar_feature_id('POLKE', row.get('feature', ''))
            if identity is None:
                continue
            previous = self.descriptions.get(identity, {})
            description = row.get('description') or row.get('can_do_statement') or row.get('guideword') or f'POLKE structure {identity}'
            if description == f'POLKE structure {identity}':
                description = previous.get('grammar_description') or description
            descriptions[identity] = {'grammar_construct_id': identity, 'grammar_description': description, 'grammar_reference_cefr': row.get('reference_cefr') or row.get('cefr') or previous.get('grammar_reference_cefr', '')}
        self.descriptions.update(descriptions)

    def read_progressions(self, rows):
        if not rows:
            return
        required = {'feature', 'common_threshold_percent', 'first_observed_level', 'first_common_level', 'sustained_usage_from_level', 'status'}
        if not required.issubset(rows[0]):
            raise ValueError('required progression columns are missing')
        thresholds = {_grammar_finite_number(row['common_threshold_percent']) for row in rows}
        if None in thresholds or any((not 0 < value <= 100 for value in thresholds)):
            raise ValueError('invalid prevalence threshold')
        requested = self.settings['common_usage_percent']
        threshold = requested if requested in thresholds else next(iter(thresholds)) if len(thresholds) == 1 else None
        if threshold is None:
            raise ValueError('main threshold could not be identified; keep the grammar summary with its tables')
        selected = {}
        for row in rows:
            if _grammar_finite_number(row['common_threshold_percent']) != threshold:
                continue
            identity = _grammar_feature_id('POLKE', row['feature'])
            if identity is None:
                raise ValueError(f"invalid POLKE feature: {row['feature']}")
            if identity in selected:
                raise ValueError(f'duplicate progression for POLKE structure {identity}')
            selected[identity] = {'corpus_first_observed_level': _grammar_course_level(row['first_observed_level']), 'corpus_first_common_level': _grammar_course_level(row['first_common_level']), 'corpus_sustained_usage_from_level': _grammar_course_level(row['sustained_usage_from_level']), 'corpus_grammar_threshold_percent': threshold, 'corpus_grammar_status': row['status']}
        self.progressions = selected
        self.threshold = threshold

    def fields(self, row):
        result = dict.fromkeys(_grammar_GRAMMAR_COLUMNS, '')
        identity = _grammar_feature_id(row.get('source', ''), row.get('feature', ''))
        if identity is None:
            return result
        result.update({'grammar_construct_id': identity, 'grammar_description': f'POLKE structure {identity}'})
        result.update(self.descriptions.get(identity, {}))
        result.update(self.progressions.get(identity, {}))
        return result

    def annotate(self, row):
        return {**row, **self.fields(row)}

    def summary_lines(self, rows, title='GRAMMAR DEVELOPMENT CONTEXT', limit=10):
        unique = {}
        for row in rows:
            identity = _grammar_feature_id(row.get('source', ''), row.get('feature', ''))
            if identity is not None:
                unique.setdefault(identity, self.annotate(row))
        lines = ['', title, f'POLKE structures in this feature set: {len(unique):,}', f'With corpus progression information: {sum((key in self.progressions for key in unique)):,}', 'Corpus progression is descriptive context; predictor values, rankings and weights do not use it.']
        if not self.progressions:
            lines.append('Corpus progression is unavailable; preparation continues using the available per-text features.')
        else:
            lines.append(f'Corpus common-usage threshold: {self.threshold:g}% of texts')
            counts = _grammar_Counter((row['corpus_grammar_status'] or 'unavailable' for row in unique.values()))
            lines.extend((f"  {status.replace('_', ' ')}: {count:,}" for (status, count) in sorted(counts.items())))
        lines.extend(self.notices)
        for row in list(unique.values())[:limit]:
            common = row['corpus_first_common_level'] or 'unavailable'
            sustained = row['corpus_sustained_usage_from_level'] or 'not established'
            lines.append(f"  {row['feature']} | {row['grammar_description']} | first common: {common} | sustained from: {sustained}")
        return lines
sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "1_Complexity_Analysis_Full_Module" / "2_Basic_Analysis" / "0_Feature_Dataframe"))
from create_dataframe import DataframeSource, dataframe_inputs, validate_ranking_checkpoint
from length_lists import LENGTH_NOTE, length_features

GRAMMAR = GrammarContext()
OUTPUT_PATH = SCRIPT_DIR / "mutual_information_scores.csv"
SUMMARY_PATH = SCRIPT_DIR / "mutual_information_summary.txt"

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

OUTPUT_COLUMNS = (
    "source",
    "feature",
    "is_text_length_feature",
    "mutual_information",
    "normalized_mutual_information",
    "rank_within_source",
    "rank_overall",
)

THREADS = max(1, min(4, os.cpu_count() or 1))
MEMORY_LIMIT = "4GB"
BATCH_SIZE = 8
NEIGHBORS = 3
RANDOM_STATE = 42
CHECKPOINT_EVERY = 20
TOP_RESULTS = 25


def find_thesis_root() -> Path:
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        methods = (
            candidate
            / "1_Complexity_Analysis_Full_Module"
            / "1_Complexity_Analysis_Methods"
        )
        if methods.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the thesis root. Place this script inside "
        "Thesis\\1_Complexity_Analysis_Full_Module\\3_Mutual_Information_And_Information_Gain\\1_Mutual_Information."
    )


def input_files(thesis_root: Path) -> dict[str, Path]:
    methods = (
        thesis_root
        / "1_Complexity_Analysis_Full_Module"
        / "1_Complexity_Analysis_Methods"
    )
    return {
        "LCA": methods / "LCA" / "lca_results.csv",
        "L2SCA": methods / "L2SCA" / "l2sca_results.csv",
        "TAASSC": methods / "TAASSC" / "taassc_results.csv",
        "TAALES": methods / "TAALES" / "taales_results_final.csv",
        "TAALES_COVERAGE": (
            methods / "TAALES" / "taales_results_final_index_coverage.csv"
        ),
        "POLKE": methods / "POLKE" / "polke_results.csv",
        "ERRANT": methods / "ERRANT" / "errant_results.csv",
    }


def find_input(path: Path) -> Path | None:
    if path.is_file():
        return path
    compressed = Path(f"{path}.gz")
    return compressed if compressed.is_file() else None


def discover_inputs(thesis_root: Path) -> tuple[dict[str, Path], list[str]]:
    return dataframe_inputs(thesis_root), []


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


def column_name(columns: list[str], target: str) -> str | None:
    target = target.casefold()
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
    level_name = column_name(columns, "level")
    parsed = "NULL"
    if filename_name is not None:
        filename = sql_identifier(filename_name)
        parsed = (
            "TRY_CAST(regexp_extract(LOWER(CAST("
            f"{filename} AS VARCHAR)), '_level_([0-9]+)_', 1) AS INTEGER)"
        )
    if level_name is None:
        if filename_name is None:
            raise ValueError(
                f"No level or filename column. Available columns: {columns}"
            )
        return parsed
    explicit = f"TRY_CAST(TRIM({sql_identifier(level_name)}) AS INTEGER)"
    return f"COALESCE({explicit}, {parsed})"


def numeric_expression(column: str) -> str:
    value = f"TRY_CAST(TRIM({sql_identifier(column)}) AS DOUBLE)"
    return f"CASE WHEN isfinite({value}) THEN {value} ELSE NULL END"


def target_entropy(levels: np.ndarray) -> float:
    _, counts = np.unique(levels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def calculate_mutual_information(
    values: np.ndarray,
    levels: np.ndarray,
    workers: int,
) -> np.ndarray:
    arguments = {
        "discrete_features": False,
        "n_neighbors": NEIGHBORS,
        "random_state": RANDOM_STATE,
    }
    try:
        return mutual_info_classif(values, levels, n_jobs=workers, **arguments)
    except TypeError as error:
        if "n_jobs" not in str(error):
            raise
        return mutual_info_classif(values, levels, **arguments)


def array(data: dict, name: str, dtype) -> np.ndarray:
    values = data[name]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=dtype)


def assign_competition_ranks(
    rows: list[dict],
    score_column: str,
    rank_column: str,
) -> None:
    previous_score = None
    current_rank = 0

    for position, row in enumerate(rows, start=1):
        score = float(row[score_column])
        if previous_score is None or not math.isclose(
            score, previous_score, abs_tol=1e-15
        ):
            current_rank = position
            previous_score = score
        row[rank_column] = current_rank


def assign_ranks(rows: list[dict]) -> None:
    rows.sort(
        key=lambda row: (
            -float(row["normalized_mutual_information"]),
            -float(row["mutual_information"]),
            row["source"],
            row["feature"],
        )
    )
    assign_competition_ranks(
        rows, "normalized_mutual_information", "rank_overall"
    )

    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)

    for source_rows in by_source.values():
        source_rows.sort(
            key=lambda row: (
                -float(row["normalized_mutual_information"]),
                -float(row["mutual_information"]),
                row["feature"],
            )
        )
        assign_competition_ranks(
            source_rows,
            "normalized_mutual_information",
            "rank_within_source",
        )

    rows.sort(
        key=lambda row: (
            int(row["rank_overall"]),
            row["source"],
            row["feature"],
        )
    )


def atomic_csv(path: Path, rows: list[dict]) -> None:
    for row in rows:
        row["is_text_length_feature"] = (
            "yes"
            if is_text_length_feature(row["source"], row["feature"])
            else "no"
        )
    assign_ranks(rows)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=with_grammar_columns(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(GRAMMAR.annotate(row) for row in rows)
    try:
        os.replace(temporary, path)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {path}. Close it in Excel and rerun."
        ) from error


def load_checkpoint(path: Path, sources: set[str]) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(OUTPUT_COLUMNS) - {"is_text_length_feature"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"{path.name} has an incompatible format. Rerun with --restart."
            )
        rows = []
        for row in reader:
            if row["source"] not in sources:
                continue
            rows.append(
                {
                    "source": row["source"],
                    "feature": row["feature"],
                    "is_text_length_feature": (
                        "yes"
                        if is_text_length_feature(row["source"], row["feature"])
                        else "no"
                    ),
                    "mutual_information": float(row["mutual_information"]),
                    "normalized_mutual_information": float(
                        row["normalized_mutual_information"]
                    ),
                    "rank_within_source": int(row["rank_within_source"]),
                    "rank_overall": int(row["rank_overall"]),
                }
            )
    return rows


def source_table(
    connection,
    path: Path,
    columns: list[str],
    features: list[str],
) -> list[str]:
    aliases = [f"value_{index}" for index in range(len(features))]
    projections = ",\n        ".join(
        f"{numeric_expression(feature)} AS {alias}"
        for feature, alias in zip(features, aliases)
    )
    level = level_expression(columns)
    connection.execute("DROP TABLE IF EXISTS source_values")
    connection.execute(
        "CREATE TABLE source_values AS\n"
        "WITH converted AS (\n"
        f"    SELECT {level} AS level,\n        {projections}\n"
        f"    FROM {csv_scan(path)}\n"
        ")\n"
        "SELECT * FROM converted WHERE level BETWEEN 1 AND 15"
    )
    return aliases


def feature_counts(connection, aliases: list[str]) -> list[tuple[int, bool]]:
    results = []
    for start in range(0, len(aliases), 100):
        batch = aliases[start : start + 100]
        query = "SELECT " + ", ".join(
            expression
            for alias in batch
            for expression in (
                f"COUNT({alias})",
                f"MIN({alias})",
                f"MAX({alias})",
            )
        ) + " FROM source_values"
        values = connection.execute(query).fetchone()
        results.extend(
            (
                int(values[index]),
                values[index + 1] is not None
                and values[index + 2] is not None
                and float(values[index + 1]) != float(values[index + 2]),
            )
            for index in range(0, len(values), 3)
        )
    return results


def empty_score(source: str, feature: str) -> dict:
    return {
        "source": source,
        "feature": feature,
        "mutual_information": 0.0,
        "normalized_mutual_information": 0.0,
        "rank_within_source": 0,
        "rank_overall": 0,
    }


def complete_batch(
    connection,
    source: str,
    features: list[str],
    aliases: list[str],
    indices: list[int],
    workers: int,
) -> list[dict]:
    selected = [aliases[index] for index in indices]
    data = connection.execute(
        "SELECT level, " + ", ".join(selected) + " FROM source_values"
    ).fetchnumpy()
    levels = array(data, "level", np.int32)
    values = np.column_stack(
        [array(data, alias, np.float64) for alias in selected]
    )
    scores = calculate_mutual_information(values, levels, workers)
    entropy = target_entropy(levels)
    rows = []
    for index, score in zip(indices, scores):
        score = float(max(0.0, score))
        rows.append(
            {
                "source": source,
                "feature": features[index],
                "mutual_information": score,
                "normalized_mutual_information": (
                    min(1.0, score / entropy) if entropy else 0.0
                ),
                "rank_within_source": 0,
                "rank_overall": 0,
            }
        )
    return rows


def incomplete_feature(
    connection,
    source: str,
    feature: str,
    alias: str,
    workers: int,
) -> dict:
    data = connection.execute(
        f"SELECT level, {alias} FROM source_values WHERE {alias} IS NOT NULL"
    ).fetchnumpy()
    levels = array(data, "level", np.int32)
    values = array(data, alias, np.float64).reshape(-1, 1)
    score = float(calculate_mutual_information(values, levels, workers)[0])
    entropy = target_entropy(levels)
    score = max(0.0, score)
    return {
        "source": source,
        "feature": feature,
        "mutual_information": score,
        "normalized_mutual_information": (
            min(1.0, score / entropy) if entropy else 0.0
        ),
        "rank_within_source": 0,
        "rank_overall": 0,
    }


def process_source(
    connection,
    source: str,
    path: Path,
    completed: set[tuple[str, str]],
    rows: list[dict],
    workers: int,
    since_checkpoint: int,
) -> int:
    columns = [
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM {csv_scan(path)}"
        ).fetchall()
    ]
    features = [
        feature
        for feature in feature_columns(source, columns)
        if (source, feature) not in completed
    ]

    if not features:
        print(f"  {source}: already complete or no feature columns found")
        return since_checkpoint

    print(
        f"  {source}: importing {len(features):,} unfinished features "
        f"from {path.name}"
    )
    aliases = source_table(connection, path, columns, features)
    total_rows = int(
        connection.execute("SELECT COUNT(*) FROM source_values").fetchone()[0]
    )
    if total_rows == 0:
        raise ValueError(f"{source}: no rows with levels 1-15")

    counts = feature_counts(connection, aliases)
    usable = [index for index, (count, _) in enumerate(counts) if count >= 4]
    constants = [index for index in usable if not counts[index][1]]

    rows.extend(empty_score(source, features[index]) for index in constants)
    since_checkpoint += len(constants)

    varying = [index for index in usable if counts[index][1]]
    complete = [index for index in varying if counts[index][0] == total_rows]
    incomplete = [index for index in varying if counts[index][0] != total_rows]
    processed = len(constants)
    total_features = len(constants) + len(varying)

    for start in range(0, len(complete), BATCH_SIZE):
        indices = complete[start : start + BATCH_SIZE]
        rows.extend(
            complete_batch(
                connection,
                source,
                features,
                aliases,
                indices,
                workers,
            )
        )
        processed += len(indices)
        since_checkpoint += len(indices)
        if since_checkpoint >= CHECKPOINT_EVERY:
            atomic_csv(OUTPUT_PATH, rows)
            since_checkpoint = 0
        print(f"    {processed:,}/{total_features:,} features scored")

    for index in incomplete:
        rows.append(
            incomplete_feature(
                connection,
                source,
                features[index],
                aliases[index],
                workers,
            )
        )
        processed += 1
        since_checkpoint += 1
        if since_checkpoint >= CHECKPOINT_EVERY:
            atomic_csv(OUTPUT_PATH, rows)
            since_checkpoint = 0
        if processed % 10 == 0 or processed == total_features:
            print(f"    {processed:,}/{total_features:,} features scored")

    unusable = len(features) - len(usable)
    if unusable:
        print(
            f"    skipped {unusable:,} columns with insufficient numeric data"
        )

    connection.execute("DROP TABLE source_values")
    return since_checkpoint


def write_summary(
    rows: list[dict],
    filtered_rows: list[dict],
    inputs: dict[str, Path],
    skipped: list[str],
) -> None:
    ranked = sorted(rows, key=lambda row: int(row["rank_overall"]))
    filtered_ranked = sorted(
        filtered_rows, key=lambda row: int(row["rank_overall"])
    )
    lines = [
        "MUTUAL INFORMATION SUMMARY",
        "=" * 72,
        f"Sources analyzed: {', '.join(inputs)}",
        f"Sources skipped: {', '.join(skipped) if skipped else 'none'}",
        f"Features scored: {len(rows):,}",
        f"Features that are not length-dependent: {len(filtered_rows):,}",
        f"Length-dependent features removed: {len(rows) - len(filtered_rows):,}",
        f"Nearest neighbours: {NEIGHBORS}",
        f"Random state: {RANDOM_STATE}",
        "",
        "Normalized MI = MI divided by the entropy of the available level labels.",
        "A higher score indicates stronger dependence but gives no direction.",
        "Scores are univariate and do not measure feature interactions.",
        "The filtered version removes the length-dependent features.",
        LENGTH_NOTE,
    ]

    for label, selected_rows in (
        ("ALL FEATURES", ranked),
        ("WITHOUT LENGTH-DEPENDENT FEATURES", filtered_ranked),
    ):
        heading = f"TOP {min(TOP_RESULTS, len(selected_rows))} FEATURES ({label})"
        lines.extend(["", heading, "-" * len(heading)])
        if selected_rows:
            lines.extend(
                f"{row['rank_overall']}. {row['source']} | {row['feature']}: "
                f"MI={float(row['mutual_information']):.6f}, "
                f"normalized={float(row['normalized_mutual_information']):.6f}"
                for row in selected_rows[:TOP_RESULTS]
            )
        else:
            lines.append("No features available.")

    lines.extend(
        [
            "",
            "For unbiased model evaluation, feature selection must be repeated using",
            "training data only inside each cross-validation fold.",
        ]
    )
    lines.extend(GRAMMAR.summary_lines(ranked, limit=TOP_RESULTS))
    temporary = Path(f"{SUMMARY_PATH}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.replace(temporary, SUMMARY_PATH)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {SUMMARY_PATH}. Close it and rerun."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank available complexity and ERRANT features by Mutual Information."
        )
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard existing scores and calculate every feature again.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=THREADS,
        help=f"Parallel scikit-learn workers (default: {THREADS}).",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help=(
            "Refresh only the length flags and the summary from the saved scores, "
            "for example after a length-list change. No scores are recalculated."
        ),
    )
    return parser.parse_args()


def score_features(
    args: argparse.Namespace,
    thesis_root: Path,
    inputs: dict[str, Path],
    skipped: list[str],
) -> list[dict]:
    if args.restart:
        for path in (OUTPUT_PATH, SUMMARY_PATH):
            if path.is_file():
                path.unlink()

    validate_ranking_checkpoint(OUTPUT_PATH, inputs, args.restart)
    rows = load_checkpoint(OUTPUT_PATH, set(inputs))
    completed = {(row["source"], row["feature"]) for row in rows}

    print("Mutual Information feature ranking")
    print(f"Thesis root: {thesis_root}")
    print(f"Output folder: {SCRIPT_DIR}")
    print(f"Sources found: {', '.join(inputs)}")
    if skipped:
        print(f"Sources skipped: {', '.join(skipped)}")
    print(f"Already scored: {len(rows):,}")
    print(f"Workers: {args.workers}")

    with TemporaryDirectory(prefix="mutual_information_") as temporary_dir:
        database = Path(temporary_dir) / "mutual_information.duckdb"
        connection = duckdb.connect(str(database))
        connection.execute(f"SET threads={THREADS}")
        connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
        connection.execute("SET preserve_insertion_order=false")
        try:
            since_checkpoint = 0
            for source, path in inputs.items():
                since_checkpoint = process_source(
                    connection,
                    source,
                    path,
                    completed,
                    rows,
                    args.workers,
                    since_checkpoint,
                )
        finally:
            connection.close()
    return rows


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.relabel and args.restart:
        raise ValueError("Choose either --relabel or --restart.")

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    thesis_root = find_thesis_root()
    GRAMMAR.load(thesis_root)
    inputs, skipped = discover_inputs(thesis_root)

    if args.relabel:
        # Scores do not depend on the length lists; only the flags and filtered rankings change.
        rows = load_checkpoint(OUTPUT_PATH, set(inputs))
        if not rows:
            raise FileNotFoundError(f"No saved scores to relabel in {OUTPUT_PATH}")
        print(f"Relabelling {len(rows):,} saved scores with the current length lists")
    else:
        rows = score_features(args, thesis_root, inputs, skipped)

    atomic_csv(OUTPUT_PATH, rows)
    filtered_rows = [
        row.copy()
        for row in rows
        if not is_text_length_feature(row["source"], row["feature"])
    ]
    assign_ranks(filtered_rows)
    write_summary(rows, filtered_rows, inputs, skipped)
    print()
    print(f"Scores:  {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Repeat feature selection inside each training fold before evaluation.")


if __name__ == "__main__":
    main()
