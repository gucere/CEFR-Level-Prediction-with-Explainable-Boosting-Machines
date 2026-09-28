from __future__ import annotations
import argparse
import csv
import gc
import gzip
import importlib.metadata
import importlib.util
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_MAX_FAILURE_ATTEMPTS = 3
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_SAVE_BATCH_SIZE = 50
DEFAULT_INFLIGHT_PER_WORKER = 2
DEFAULT_TUNE_SAMPLE_SIZE = 60
MAX_AUTO_WORKERS = 8
MIN_TUNING_FREE_MEMORY_GIB = 0.6
AUTOTUNED_WORKERS_KEY = "autotuned_workers_live_memory"

BASE_RESULT_COLUMNS = [
    "filename",
    "writing_id",
    "cefr",
    "level",
    "grade",
    "original_word_count",
    "corrected_word_count",
    "total_errors",
    "errors_per_100_words",
    "distinct_error_types",
    "missing_errors",
    "missing_errors_per_100_words",
    "unnecessary_errors",
    "unnecessary_errors_per_100_words",
    "replacement_errors",
    "replacement_errors_per_100_words",
    "other_errors",
    "other_errors_per_100_words",
    "affected_original_tokens",
    "correction_token_equivalents",
    "surface_accuracy_proxy",
    "error_free",
]


_ANNOTATOR = None


@dataclass(frozen=True)
class FileLabels:
    writing_id: str
    cefr: str
    level: int | None
    grade: str


@dataclass(frozen=True)
class EditRecord:
    edit_index: int
    original_start: int
    original_end: int
    corrected_start: int
    corrected_end: int
    error_type: str
    operation: str
    error_family: str
    original_form: str
    corrected_form: str


@dataclass(frozen=True)
class WorkerOutcome:
    filename: str
    success: bool
    labels: FileLabels | None = None
    original_word_count: int = 0
    corrected_word_count: int = 0
    total_errors: int = 0
    distinct_error_types: int = 0
    missing_errors: int = 0
    unnecessary_errors: int = 0
    replacement_errors: int = 0
    other_errors: int = 0
    affected_original_tokens: int = 0
    affected_corrected_tokens: int = 0
    correction_token_equivalents: int = 0
    surface_accuracy_proxy: float = 0.0
    edits: tuple[EditRecord, ...] = ()
    error_message: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_thesis_root(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "0_Data").is_dir():
            return candidate
    if len(script_dir.parents) >= 2:
        return script_dir.parents[1]
    return script_dir


def parse_filename(filename: str) -> FileLabels:
    parts = Path(filename).stem.split("_")
    try:
        if (
            len(parts) >= 7
            and parts[1].casefold() == "cefr"
            and parts[3].casefold() == "level"
            and parts[5].casefold() == "grade"
        ):
            return FileLabels(
                writing_id=parts[0],
                cefr=parts[2].upper(),
                level=int(parts[4]),
                grade="_".join(parts[6:]),
            )
    except (TypeError, ValueError):
        pass
    return FileLabels("", "", None, "")


def read_text(path: Path) -> str:
    last_error: UnicodeError | None = None
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return path.read_text(encoding=encoding).replace("\x00", " ")
        except UnicodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not read {path}")


def count_word_tokens(document) -> int:
    return sum(1 for token in document if not token.is_space and not token.is_punct)


def covered_token_count(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def error_parts(error_type: str) -> tuple[str, str]:
    parts = error_type.split(":")
    operation = parts[0].upper() if parts and parts[0] else "OTHER"
    if operation not in {"M", "U", "R"}:
        operation = "OTHER"
    family = parts[1].upper() if len(parts) > 1 and parts[1] else "OTHER"
    return operation, family


def initialise_worker() -> None:
    global _ANNOTATOR
    import errant
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    _ANNOTATOR = errant.load("en", nlp)


def process_pair(raw_path_text: str, corrected_path_text: str) -> WorkerOutcome:
    raw_path = Path(raw_path_text)
    corrected_path = Path(corrected_path_text)
    filename = raw_path.name
    try:
        if _ANNOTATOR is None:
            raise RuntimeError("ERRANT worker was not initialized")

        original_text = read_text(raw_path).strip()
        corrected_text = read_text(corrected_path).strip()
        if not original_text:
            raise ValueError("The original text is empty")
        if not corrected_text:
            raise ValueError("The corrected text is empty")

        original_doc = _ANNOTATOR.parse(original_text, tokenise=True)
        if original_text == corrected_text:
            corrected_doc = original_doc
            raw_edits = []
        else:
            corrected_doc = _ANNOTATOR.parse(corrected_text, tokenise=True)
            raw_edits = _ANNOTATOR.annotate(original_doc, corrected_doc)

        records: list[EditRecord] = []
        operation_counts: Counter[str] = Counter()
        type_counts: Counter[str] = Counter()
        token_equivalents = 0

        for index, edit in enumerate(raw_edits):
            error_type = str(edit.type or "UNK")
            operation, family = error_parts(error_type)
            original_length = max(0, int(edit.o_end) - int(edit.o_start))
            corrected_length = max(0, int(edit.c_end) - int(edit.c_start))
            token_equivalents += max(original_length, corrected_length, 1)
            operation_counts[operation] += 1
            type_counts[error_type] += 1
            records.append(
                EditRecord(
                    edit_index=index,
                    original_start=int(edit.o_start),
                    original_end=int(edit.o_end),
                    corrected_start=int(edit.c_start),
                    corrected_end=int(edit.c_end),
                    error_type=error_type,
                    operation=operation,
                    error_family=family,
                    original_form=str(edit.o_str),
                    corrected_form=str(edit.c_str),
                )
            )

        original_word_count = count_word_tokens(original_doc)
        corrected_word_count = count_word_tokens(corrected_doc)
        denominator = max(original_word_count, 1)
        accuracy_proxy = max(0.0, 1.0 - (token_equivalents / denominator))

        return WorkerOutcome(
            filename=filename,
            success=True,
            labels=parse_filename(filename),
            original_word_count=original_word_count,
            corrected_word_count=corrected_word_count,
            total_errors=len(records),
            distinct_error_types=len(type_counts),
            missing_errors=operation_counts["M"],
            unnecessary_errors=operation_counts["U"],
            replacement_errors=operation_counts["R"],
            other_errors=operation_counts["OTHER"],
            affected_original_tokens=covered_token_count(
                (record.original_start, record.original_end) for record in records
            ),
            affected_corrected_tokens=covered_token_count(
                (record.corrected_start, record.corrected_end) for record in records
            ),
            correction_token_equivalents=token_equivalents,
            surface_accuracy_proxy=accuracy_proxy,
            edits=tuple(records),
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return WorkerOutcome(filename=filename, success=False, error_message=message[:4_000])


def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS texts (
            filename TEXT PRIMARY KEY,
            writing_id TEXT NOT NULL,
            cefr TEXT NOT NULL,
            level INTEGER,
            grade TEXT NOT NULL,
            original_word_count INTEGER NOT NULL,
            corrected_word_count INTEGER NOT NULL,
            total_errors INTEGER NOT NULL,
            distinct_error_types INTEGER NOT NULL,
            missing_errors INTEGER NOT NULL,
            unnecessary_errors INTEGER NOT NULL,
            replacement_errors INTEGER NOT NULL,
            other_errors INTEGER NOT NULL,
            affected_original_tokens INTEGER NOT NULL,
            affected_corrected_tokens INTEGER NOT NULL,
            correction_token_equivalents INTEGER NOT NULL,
            surface_accuracy_proxy REAL NOT NULL,
            processed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS edits (
            filename TEXT NOT NULL,
            edit_index INTEGER NOT NULL,
            original_start INTEGER NOT NULL,
            original_end INTEGER NOT NULL,
            corrected_start INTEGER NOT NULL,
            corrected_end INTEGER NOT NULL,
            error_type TEXT NOT NULL,
            operation TEXT NOT NULL,
            error_family TEXT NOT NULL,
            original_form TEXT NOT NULL,
            corrected_form TEXT NOT NULL,
            PRIMARY KEY (filename, edit_index),
            FOREIGN KEY (filename) REFERENCES texts(filename) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS error_counts (
            filename TEXT NOT NULL,
            error_type TEXT NOT NULL,
            operation TEXT NOT NULL,
            error_family TEXT NOT NULL,
            error_count INTEGER NOT NULL,
            PRIMARY KEY (filename, error_type),
            FOREIGN KEY (filename) REFERENCES texts(filename) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS failures (
            filename TEXT PRIMARY KEY,
            attempts INTEGER NOT NULL,
            last_error TEXT NOT NULL,
            last_failed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS unpaired (
            filename TEXT PRIMARY KEY,
            missing_side TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_edits_type ON edits(error_type);
        CREATE INDEX IF NOT EXISTS idx_edits_family ON edits(error_family);
        CREATE INDEX IF NOT EXISTS idx_counts_type ON error_counts(error_type);
        """
    )
    return connection


def set_and_validate_metadata(
    connection: sqlite3.Connection,
    raw_dir: Path,
    corrected_dir: Path,
) -> None:
    expected = {
        "raw_dir": str(raw_dir.resolve()),
        "corrected_dir": str(corrected_dir.resolve()),
    }
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    completed = connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
    if completed:
        for key in ("raw_dir", "corrected_dir"):
            if existing.get(key) not in (None, expected[key]):
                raise RuntimeError(
                    f"Checkpoint {key!r} is {existing.get(key)!r}, but this run uses "
                    f"{expected[key]!r}. Use a different --output-dir for a different corpus."
                )
    connection.executemany(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        expected.items(),
    )
    connection.commit()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def save_outcomes(connection: sqlite3.Connection, outcomes: Sequence[WorkerOutcome]) -> None:
    if not outcomes:
        return
    now = utc_now()
    try:
        connection.execute("BEGIN")
        for outcome in outcomes:
            if not outcome.success:
                connection.execute(
                    """
                    INSERT INTO failures(filename, attempts, last_error, last_failed_at)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(filename) DO UPDATE SET
                        attempts = failures.attempts + 1,
                        last_error = excluded.last_error,
                        last_failed_at = excluded.last_failed_at
                    """,
                    (outcome.filename, outcome.error_message, now),
                )
                continue

            labels = outcome.labels or FileLabels("", "", None, "")
            connection.execute("DELETE FROM edits WHERE filename = ?", (outcome.filename,))
            connection.execute("DELETE FROM error_counts WHERE filename = ?", (outcome.filename,))
            connection.execute(
                """
                INSERT INTO texts(
                    filename, writing_id, cefr, level, grade,
                    original_word_count, corrected_word_count, total_errors,
                    distinct_error_types, missing_errors, unnecessary_errors,
                    replacement_errors, other_errors, affected_original_tokens,
                    affected_corrected_tokens, correction_token_equivalents,
                    surface_accuracy_proxy, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    writing_id = excluded.writing_id,
                    cefr = excluded.cefr,
                    level = excluded.level,
                    grade = excluded.grade,
                    original_word_count = excluded.original_word_count,
                    corrected_word_count = excluded.corrected_word_count,
                    total_errors = excluded.total_errors,
                    distinct_error_types = excluded.distinct_error_types,
                    missing_errors = excluded.missing_errors,
                    unnecessary_errors = excluded.unnecessary_errors,
                    replacement_errors = excluded.replacement_errors,
                    other_errors = excluded.other_errors,
                    affected_original_tokens = excluded.affected_original_tokens,
                    affected_corrected_tokens = excluded.affected_corrected_tokens,
                    correction_token_equivalents = excluded.correction_token_equivalents,
                    surface_accuracy_proxy = excluded.surface_accuracy_proxy,
                    processed_at = excluded.processed_at
                """,
                (
                    outcome.filename,
                    labels.writing_id,
                    labels.cefr,
                    labels.level,
                    labels.grade,
                    outcome.original_word_count,
                    outcome.corrected_word_count,
                    outcome.total_errors,
                    outcome.distinct_error_types,
                    outcome.missing_errors,
                    outcome.unnecessary_errors,
                    outcome.replacement_errors,
                    outcome.other_errors,
                    outcome.affected_original_tokens,
                    outcome.affected_corrected_tokens,
                    outcome.correction_token_equivalents,
                    outcome.surface_accuracy_proxy,
                    now,
                ),
            )
            if outcome.edits:
                connection.executemany(
                    """
                    INSERT INTO edits(
                        filename, edit_index, original_start, original_end,
                        corrected_start, corrected_end, error_type, operation,
                        error_family, original_form, corrected_form
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            outcome.filename,
                            edit.edit_index,
                            edit.original_start,
                            edit.original_end,
                            edit.corrected_start,
                            edit.corrected_end,
                            edit.error_type,
                            edit.operation,
                            edit.error_family,
                            edit.original_form,
                            edit.corrected_form,
                        )
                        for edit in outcome.edits
                    ],
                )
                counts = Counter(edit.error_type for edit in outcome.edits)
                examples = {edit.error_type: edit for edit in outcome.edits}
                connection.executemany(
                    """
                    INSERT INTO error_counts(
                        filename, error_type, operation, error_family, error_count
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            outcome.filename,
                            error_type,
                            examples[error_type].operation,
                            examples[error_type].error_family,
                            count,
                        )
                        for error_type, count in counts.items()
                    ],
                )
            connection.execute("DELETE FROM failures WHERE filename = ?", (outcome.filename,))
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def sync_unpaired(
    connection: sqlite3.Connection,
    raw_only: Sequence[str],
    corrected_only: Sequence[str],
) -> None:
    now = utc_now()
    connection.execute("DELETE FROM unpaired")
    connection.executemany(
        "INSERT INTO unpaired(filename, missing_side, recorded_at) VALUES (?, ?, ?)",
        [(name, "corrected_text", now) for name in raw_only]
        + [(name, "raw_text", now) for name in corrected_only],
    )
    connection.commit()


def memory_status_gib() -> tuple[float | None, float | None]:
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return (
                status.total_physical / (1024**3),
                status.available_physical / (1024**3),
            )
        total_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (
            total_pages * page_size / (1024**3),
            available_pages * page_size / (1024**3),
        )
    except (AttributeError, OSError, ValueError):
        return None, None


def available_memory_gib() -> float | None:
    return memory_status_gib()[1]


def safe_worker_cap(total_memory_gib: float | None) -> int:
    cpu_count = os.cpu_count() or 1
    cpu_limit = max(1, cpu_count - 1)
    if total_memory_gib is None:
        memory_limit = MAX_AUTO_WORKERS
    else:
        memory_limit = max(1, int(max(0.0, total_memory_gib - 1.0) / 1.5))
    return max(1, min(cpu_limit, memory_limit, MAX_AUTO_WORKERS))


def representative_sample(names: Sequence[str], sample_size: int) -> list[str]:
    if sample_size >= len(names):
        return list(names)
    return [
        names[min(len(names) - 1, int((index + 0.5) * len(names) / sample_size))]
        for index in range(sample_size)
    ]


def worker_candidates(cap: int, sample_size: int) -> list[int]:
    cap = max(1, min(cap, sample_size))
    candidates = [value for value in (1, 2, 3, 4, 6, 8) if value <= cap]
    if cap not in candidates:
        candidates.append(cap)
    return sorted(set(candidates))


def benchmark_candidate(
    raw_dir: Path,
    corrected_dir: Path,
    sample_names: Sequence[str],
    workers: int,
) -> tuple[float, list[WorkerOutcome], float | None]:
    raw_paths = [str(raw_dir / name) for name in sample_names]
    corrected_paths = [str(corrected_dir / name) for name in sample_names]
    executor = ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker)
    try:
        warmup = [
            executor.submit(process_pair, raw_paths[index], corrected_paths[index])
            for index in range(min(workers, len(sample_names)))
        ]
        for future in warmup:
            future.result()
        free_memory_gib = available_memory_gib()
        if (
            workers > 1
            and free_memory_gib is not None
            and free_memory_gib < MIN_TUNING_FREE_MEMORY_GIB
        ):
            raise MemoryError(
                f"only {free_memory_gib:.2f} GiB remained after worker initialization"
            )
        started = time.monotonic()
        outcomes = list(
            executor.map(
                process_pair,
                raw_paths,
                corrected_paths,
                chunksize=1,
            )
        )
        elapsed = max(time.monotonic() - started, 0.001)
        successful = sum(outcome.success for outcome in outcomes)
        return successful / elapsed, outcomes, free_memory_gib
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        gc.collect()


def tune_workers(
    raw_dir: Path,
    corrected_dir: Path,
    selected_names: Sequence[str],
    sample_size: int,
    cap: int,
) -> tuple[int, list[WorkerOutcome]]:
    sample_names = representative_sample(selected_names, min(sample_size, len(selected_names)))
    candidates = worker_candidates(cap, len(sample_names))
    best_workers = 1
    best_score = (-1, -1.0)
    best_outcomes: list[WorkerOutcome] = []
    print(f"Experimental worker tuning on {len(sample_names):,} representative files")
    print("Candidates: " + ", ".join(str(value) for value in candidates))
    for workers in candidates:
        try:
            rate, outcomes, free_memory_gib = benchmark_candidate(
                raw_dir,
                corrected_dir,
                sample_names,
                workers,
            )
        except KeyboardInterrupt:
            raise
        except MemoryError as error:
            print(
                f"  {workers} worker{'s' if workers != 1 else ''}: stopped ({error})",
                flush=True,
            )
            break
        except Exception as error:
            print(
                f"  {workers} worker{'s' if workers != 1 else ''}: failed "
                f"({type(error).__name__}: {error})",
                flush=True,
            )
            break
        successful = sum(outcome.success for outcome in outcomes)
        memory_text = (
            f", {free_memory_gib:.2f} GiB free"
            if free_memory_gib is not None
            else ""
        )
        print(
            f"  {workers} worker{'s' if workers != 1 else ''}: "
            f"{rate:.3f} texts/s ({successful}/{len(outcomes)} successful{memory_text})",
            flush=True,
        )
        score = (successful, rate)
        if score > best_score:
            best_score = score
            best_workers = workers
            best_outcomes = outcomes
    if not best_outcomes:
        raise RuntimeError("Every experimental worker configuration failed")
    print(f"Selected fastest configuration: {best_workers} worker{'s' if best_workers != 1 else ''}")
    return best_workers, best_outcomes


def metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def store_tuning_result(
    connection: sqlite3.Connection,
    workers: int,
) -> None:
    connection.execute(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (AUTOTUNED_WORKERS_KEY, str(workers)),
    )
    connection.commit()


def verify_dependencies() -> None:
    missing = [
        name for name in ("errant", "spacy") if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing Python package(s): "
            + ", ".join(missing)
            + "\nInstall with: python -m pip install errant"
        )
    import spacy

    try:
        spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as error:
        raise RuntimeError(
            "The spaCy model en_core_web_sm is not installed.\n"
            "Install with: python -m spacy download en_core_web_sm"
        ) from error


def list_txt_names(folder: Path) -> set[str]:
    return {
        entry.name
        for entry in os.scandir(folder)
        if entry.is_file() and entry.name.casefold().endswith(".txt")
    }


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def feature_token(value: str) -> str:
    token = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in token.split("_") if part) or "other"


def rate_per_100(count: int, word_count: int) -> float:
    return (100.0 * count / word_count) if word_count > 0 else 0.0


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    try:
        os.replace(temp_path, final_path)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace {final_path}. Close it in Excel or another program, "
            "then run this script with --export-only."
        ) from error


def error_taxonomy(connection: sqlite3.Connection) -> tuple[list[str], list[str]]:
    error_types = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT error_type FROM error_counts ORDER BY error_type"
        )
    ]
    families = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT error_family FROM error_counts ORDER BY error_family"
        )
    ]
    return error_types, families


def export_results(connection: sqlite3.Connection, output_dir: Path) -> Path:
    final_path = output_dir / "errant_results.csv"
    temp_path = output_dir / "errant_results.csv.tmp"
    error_types, families = error_taxonomy(connection)
    family_columns = {
        family: f"errant_family_{feature_token(family)}_per_100_words"
        for family in families
    }
    type_columns = {
        error_type: f"errant_{feature_token(error_type)}_per_100_words"
        for error_type in error_types
    }
    fieldnames = BASE_RESULT_COLUMNS + list(family_columns.values()) + list(type_columns.values())

    base_query = """
        SELECT filename, writing_id, cefr, level, grade,
               original_word_count, corrected_word_count, total_errors,
               distinct_error_types, missing_errors, unnecessary_errors,
               replacement_errors, other_errors, affected_original_tokens,
               correction_token_equivalents, surface_accuracy_proxy
        FROM texts
        ORDER BY filename
    """
    count_cursor = connection.execute(
        """
        SELECT filename, error_type, error_family, error_count
        FROM error_counts
        ORDER BY filename, error_type
        """
    )
    next_count = count_cursor.fetchone()

    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in connection.execute(base_query):
            (
                filename,
                writing_id,
                cefr,
                level,
                grade,
                original_words,
                corrected_words,
                total_errors,
                distinct_types,
                missing,
                unnecessary,
                replacement,
                other,
                affected_original,
                token_equivalents,
                accuracy_proxy,
            ) = row
            family_counts: Counter[str] = Counter()
            type_counts: Counter[str] = Counter()
            while next_count is not None and next_count[0] == filename:
                _, error_type, family, count = next_count
                type_counts[error_type] += count
                family_counts[family] += count
                next_count = count_cursor.fetchone()

            result = {
                "filename": filename,
                "writing_id": writing_id,
                "cefr": cefr,
                "level": "" if level is None else level,
                "grade": grade,
                "original_word_count": original_words,
                "corrected_word_count": corrected_words,
                "total_errors": total_errors,
                "errors_per_100_words": rate_per_100(total_errors, original_words),
                "distinct_error_types": distinct_types,
                "missing_errors": missing,
                "missing_errors_per_100_words": rate_per_100(missing, original_words),
                "unnecessary_errors": unnecessary,
                "unnecessary_errors_per_100_words": rate_per_100(unnecessary, original_words),
                "replacement_errors": replacement,
                "replacement_errors_per_100_words": rate_per_100(replacement, original_words),
                "other_errors": other,
                "other_errors_per_100_words": rate_per_100(other, original_words),
                "affected_original_tokens": affected_original,
                "correction_token_equivalents": token_equivalents,
                "surface_accuracy_proxy": accuracy_proxy,
                "error_free": int(total_errors == 0),
            }
            result.update(
                {
                    column: rate_per_100(family_counts[family], original_words)
                    for family, column in family_columns.items()
                }
            )
            result.update(
                {
                    column: rate_per_100(type_counts[error_type], original_words)
                    for error_type, column in type_columns.items()
                }
            )
            writer.writerow(result)
    atomic_replace(temp_path, final_path)
    return final_path


def export_edits(connection: sqlite3.Connection, output_dir: Path) -> Path:
    final_path = output_dir / "errant_edits.csv.gz"
    temp_path = output_dir / "errant_edits.csv.gz.tmp"
    fieldnames = [
        "filename",
        "writing_id",
        "cefr",
        "level",
        "grade",
        "edit_index",
        "original_start",
        "original_end",
        "corrected_start",
        "corrected_end",
        "operation",
        "error_family",
        "error_type",
        "original_form",
        "corrected_form",
    ]
    query = """
        SELECT e.filename, t.writing_id, t.cefr, t.level, t.grade,
               e.edit_index, e.original_start, e.original_end,
               e.corrected_start, e.corrected_end, e.operation,
               e.error_family, e.error_type, e.original_form, e.corrected_form
        FROM edits AS e
        JOIN texts AS t ON t.filename = e.filename
        ORDER BY e.filename, e.edit_index
    """
    with gzip.open(temp_path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(connection.execute(query))
    atomic_replace(temp_path, final_path)
    return final_path


def export_metadata(connection: sqlite3.Connection, output_dir: Path) -> Path:
    final_path = output_dir / "errant_error_type_metadata.csv"
    temp_path = output_dir / "errant_error_type_metadata.csv.tmp"
    query = """
        SELECT error_type, operation, error_family, SUM(error_count) AS corpus_count
        FROM error_counts
        GROUP BY error_type, operation, error_family
        ORDER BY error_type
    """
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature",
                "error_type",
                "operation",
                "error_family",
                "corpus_error_count",
            ]
        )
        for error_type, operation, family, count in connection.execute(query):
            writer.writerow(
                [
                    f"errant_{feature_token(error_type)}_per_100_words",
                    error_type,
                    operation,
                    family,
                    count,
                ]
            )
    atomic_replace(temp_path, final_path)
    return final_path


def export_failures(connection: sqlite3.Connection, output_dir: Path) -> Path:
    final_path = output_dir / "errant_failed.csv"
    temp_path = output_dir / "errant_failed.csv.tmp"
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "stage", "attempts", "error", "last_failed_at"])
        for filename, attempts, error, failed_at in connection.execute(
            """
            SELECT filename, attempts, last_error, last_failed_at
            FROM failures ORDER BY filename
            """
        ):
            writer.writerow([filename, "ERRANT processing", attempts, error, failed_at])
        for filename, missing_side, recorded_at in connection.execute(
            "SELECT filename, missing_side, recorded_at FROM unpaired ORDER BY filename"
        ):
            writer.writerow(
                [
                    filename,
                    "file pairing",
                    0,
                    f"Missing {missing_side}",
                    recorded_at,
                ]
            )
    atomic_replace(temp_path, final_path)
    return final_path


def export_summary(connection: sqlite3.Connection, output_dir: Path) -> Path:
    final_path = output_dir / "errant_summary.txt"
    temp_path = output_dir / "errant_summary.txt.tmp"
    completed, total_words, total_errors, error_free, mean_accuracy = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(original_word_count), 0),
               COALESCE(SUM(total_errors), 0),
               COALESCE(SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END), 0),
               COALESCE(AVG(surface_accuracy_proxy), 0)
        FROM texts
        """
    ).fetchone()
    failed = connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    unpaired = connection.execute("SELECT COUNT(*) FROM unpaired").fetchone()[0]
    corpus_rate = rate_per_100(total_errors, total_words)
    top_types = list(
        connection.execute(
            """
            SELECT error_type, SUM(error_count) AS total
            FROM error_counts
            GROUP BY error_type
            ORDER BY total DESC, error_type
            LIMIT 20
            """
        )
    )
    lines = [
        "ERRANT ANALYSIS SUMMARY",
        "=" * 72,
        f"Generated: {utc_now()}",
        f"ERRANT version: {package_version('errant')}",
        f"Successfully processed text pairs: {completed:,}",
        f"Processing failures currently recorded: {failed:,}",
        f"Unpaired files: {unpaired:,}",
        f"Original word tokens: {total_words:,}",
        f"Detected edits: {total_errors:,}",
        f"Corpus error rate per 100 words: {corpus_rate:.6f}",
        f"Error-free texts: {error_free:,}",
        f"Mean surface accuracy proxy: {mean_accuracy:.6f}",
        "",
        "TOP ERRANT ERROR TYPES",
        "-" * 72,
    ]
    if top_types:
        lines.extend(f"{position:>2}. {name}: {count:,}" for position, (name, count) in enumerate(top_types, 1))
    else:
        lines.append("No edits were detected.")
    lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "-" * 72,
            "surface_accuracy_proxy = max(0, 1 - correction_token_equivalents /",
            "original_word_count). It is an edit-based surface accuracy proxy, not a",
            "construction-specific estimate of grammatical mastery.",
            "",
            "When a grammatical form was not needed, ERRANT provides no attempt",
            "denominator. Therefore, error rates and this proxy should not be described",
            "as correct-attempt / all-attempt accuracy for individual constructions.",
        ]
    )
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_replace(temp_path, final_path)
    return final_path


def export_all(
    connection: sqlite3.Connection,
    output_dir: Path,
    export_long_edits: bool,
) -> list[Path]:
    paths = [
        export_results(connection, output_dir),
        export_metadata(connection, output_dir),
        export_failures(connection, output_dir),
        export_summary(connection, output_dir),
    ]
    if export_long_edits:
        paths.append(export_edits(connection, output_dir))
    return paths


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    thesis_root = find_thesis_root(script_dir)
    data_dir = thesis_root / "0_Data"
    output_dir = (
        script_dir if script_dir.name.casefold() == "errant" else script_dir / "ERRANT"
    )
    parser = argparse.ArgumentParser(
        description="Extract ERRANT errors from identically named raw/corrected text pairs."
    )
    parser.add_argument("--raw-dir", type=Path, default=data_dir / "raw_texts")
    parser.add_argument("--corrected-dir", type=Path, default=data_dir / "corrected_texts")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Fixed worker count; 0 benchmarks worker counts automatically (default: 0).",
    )
    parser.add_argument(
        "--retune-workers",
        action="store_true",
        help="Run worker experiments again instead of reusing the checkpointed choice.",
    )
    parser.add_argument(
        "--retry-exhausted",
        action="store_true",
        help=f"Retry files that have already failed {DEFAULT_MAX_FAILURE_ATTEMPTS} times.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Do not process texts; regenerate outputs from the checkpoint.",
    )
    parser.add_argument(
        "--no-long-export",
        action="store_true",
        help="Do not create the potentially large errant_edits.csv.gz file.",
    )
    return parser


def process_corpus(
    connection: sqlite3.Connection,
    raw_dir: Path,
    corrected_dir: Path,
    selected_names: Sequence[str],
    workers: int,
) -> tuple[int, int]:
    total = len(selected_names)
    if total == 0:
        return 0, 0
    inflight_limit = max(workers, workers * DEFAULT_INFLIGHT_PER_WORKER)
    names: Iterator[str] = iter(selected_names)
    pending = {}
    save_batch: list[WorkerOutcome] = []
    attempted = successful = failed = 0
    started = time.monotonic()
    executor = ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker)

    def submit_next() -> bool:
        try:
            filename = next(names)
        except StopIteration:
            return False
        future = executor.submit(
            process_pair,
            str(raw_dir / filename),
            str(corrected_dir / filename),
        )
        pending[future] = filename
        return True

    try:
        for _ in range(min(inflight_limit, total)):
            submit_next()

        while pending:
            completed_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                filename = pending.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = WorkerOutcome(
                        filename=filename,
                        success=False,
                        error_message=f"WorkerFailure: {type(error).__name__}: {error}"[:4_000],
                    )
                save_batch.append(outcome)
                attempted += 1
                if outcome.success:
                    successful += 1
                else:
                    failed += 1

                if len(save_batch) >= DEFAULT_SAVE_BATCH_SIZE:
                    save_outcomes(connection, save_batch)
                    save_batch.clear()

                elapsed = max(time.monotonic() - started, 0.001)
                if (
                    attempted == 1
                    or attempted == total
                    or attempted % DEFAULT_PROGRESS_EVERY == 0
                ):
                    speed = attempted / elapsed
                    eta = (total - attempted) / speed if speed > 0 else math.inf
                    print(
                        f"Attempted {attempted:,}/{total:,} | successful {successful:,} | "
                        f"failed {failed:,} | {speed:.3f} texts/s | ETA {format_duration(eta)}",
                        flush=True,
                    )
                submit_next()
        save_outcomes(connection, save_batch)
        executor.shutdown(wait=True)
        return successful, failed
    except KeyboardInterrupt:
        for future in pending:
            future.cancel()
        save_outcomes(connection, save_batch)
        executor.shutdown(wait=False, cancel_futures=True)
        print("\nInterrupted safely. Saved results will be skipped on the next run.")
        raise
    except Exception:
        for future in pending:
            future.cancel()
        save_outcomes(connection, save_batch)
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = args.raw_dir.resolve()
    corrected_dir = args.corrected_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "errant_checkpoint.sqlite3"

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw-text folder not found: {raw_dir}")
    if not corrected_dir.is_dir():
        raise FileNotFoundError(f"Corrected-text folder not found: {corrected_dir}")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")

    connection = connect_database(checkpoint_path)
    try:
        set_and_validate_metadata(connection, raw_dir, corrected_dir)
        print(f"Raw texts:       {raw_dir}")
        print(f"Corrected texts: {corrected_dir}")
        print(f"Output folder:   {output_dir}")
        print(f"Checkpoint:      {checkpoint_path}")

        if not args.export_only:
            verify_dependencies()
            print("Scanning filenames...", flush=True)
            raw_names = list_txt_names(raw_dir)
            corrected_names = list_txt_names(corrected_dir)
            paired_names = sorted(raw_names & corrected_names)
            raw_only = sorted(raw_names - corrected_names)
            corrected_only = sorted(corrected_names - raw_names)
            sync_unpaired(connection, raw_only, corrected_only)

            completed_names = {
                row[0] for row in connection.execute("SELECT filename FROM texts")
            }
            failure_attempts = dict(
                connection.execute("SELECT filename, attempts FROM failures")
            )
            selected_names = []
            exhausted = 0
            for filename in paired_names:
                if filename in completed_names:
                    continue
                attempts = failure_attempts.get(filename, 0)
                if attempts >= DEFAULT_MAX_FAILURE_ATTEMPTS and not args.retry_exhausted:
                    exhausted += 1
                    continue
                selected_names.append(filename)

            total_memory_gib, memory_gib = memory_status_gib()
            worker_cap = safe_worker_cap(total_memory_gib)
            print(f"Raw .txt files:       {len(raw_names):,}")
            print(f"Corrected .txt files: {len(corrected_names):,}")
            print(f"Paired filenames:     {len(paired_names):,}")
            print(f"Unpaired files:       {len(raw_only) + len(corrected_only):,}")
            print(f"Already completed:    {len(completed_names):,}")
            print(f"Exhausted failures:   {exhausted:,}")
            print(f"Selected this run:    {len(selected_names):,}")
            if total_memory_gib is not None:
                print(f"Total RAM:            {total_memory_gib:.1f} GiB")
            if memory_gib is not None:
                print(f"Available RAM:        {memory_gib:.1f} GiB")
            print("Minimum word filtering: disabled")

            tuning_successful = 0
            tuning_failed = 0
            if args.workers > 0:
                workers = args.workers
                print(f"Parallel workers:     {workers} (manual override)")
            elif not selected_names:
                workers = 1
                print("Parallel workers:     1")
            else:
                cached_value = metadata_value(connection, AUTOTUNED_WORKERS_KEY)
                try:
                    cached_workers = int(cached_value) if cached_value else 0
                except ValueError:
                    cached_workers = 0
                if (
                    cached_workers > 0
                    and cached_workers <= worker_cap
                    and not args.retune_workers
                ):
                    workers = min(cached_workers, len(selected_names))
                    print(f"Parallel workers:     {workers} (reused experimental result)")
                else:
                    workers, tuning_outcomes = tune_workers(
                        raw_dir,
                        corrected_dir,
                        selected_names,
                        DEFAULT_TUNE_SAMPLE_SIZE,
                        worker_cap,
                    )
                    save_outcomes(connection, tuning_outcomes)
                    store_tuning_result(connection, workers)
                    tuned_names = {
                        outcome.filename for outcome in tuning_outcomes if outcome.success
                    }
                    selected_names = [
                        filename for filename in selected_names if filename not in tuned_names
                    ]
                    tuning_successful = sum(outcome.success for outcome in tuning_outcomes)
                    tuning_failed = len(tuning_outcomes) - tuning_successful
                    print(f"Experimental files saved: {len(tuning_outcomes):,}")
                    print(f"Remaining after tuning:   {len(selected_names):,}")

            corpus_successful, corpus_failed = process_corpus(
                connection,
                raw_dir,
                corrected_dir,
                selected_names,
                workers,
            )
            successful = tuning_successful + corpus_successful
            failed_attempts = tuning_failed + corpus_failed
            current_failures = connection.execute(
                "SELECT COUNT(*) FROM failures"
            ).fetchone()[0]
            print(f"Successful this run: {successful:,}")
            print(f"Failed attempts:     {failed_attempts:,}")
            print(f"Files still failed:  {current_failures:,}")

        print("Exporting result tables from the checkpoint...", flush=True)
        paths = export_all(connection, output_dir, not args.no_long_export)
        print("ERRANT run completed.")
        for path in paths:
            print(f"  {path.name}: {path}")
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
