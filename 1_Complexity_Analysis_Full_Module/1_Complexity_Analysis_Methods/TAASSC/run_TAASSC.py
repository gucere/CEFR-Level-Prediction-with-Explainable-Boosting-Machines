from __future__ import annotations

from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import csv
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import traceback
import warnings


for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")


MAX_WORKERS = 0
MAX_AUTO_WORKERS = 6
FILES_PER_TASK = 24
SPACY_BATCH_SIZE = 64
PENDING_BATCHES_PER_WORKER = 2
AUTO_TUNE = True
FORCE_RETUNE = False
AUTOTUNE_PROFILE_VERSION = 1
AUTO_TUNE_CANDIDATES = [1, 2, 3, 4, 5, 6]
AUTO_TUNE_SAMPLE_SIZE = 1024
AUTO_TUNE_WARMUP_FILES_PER_WORKER = 2
AUTO_TUNE_CLOSE_RATE_PERCENT = 2.0
WORKER_MEMORY_BUDGET_MB = 750
AUTO_TUNE_MIN_AVAILABLE_RAM_MB = 1500
RUNTIME_MIN_AVAILABLE_RAM_MB = 1200
RUNTIME_LOW_RAM_CHECKS = 3
RUNTIME_RESOURCE_CHECK_SECONDS = 10
CHECKPOINT_EVERY = 25
PROGRESS_EVERY = 500
MAX_POOL_RESTARTS = 3


script_dir = Path(__file__).resolve().parent
thesis_root = script_dir.parent.parent.parent

input_dir = thesis_root / "0_Data" / "raw_texts"
taassc_dir = (
    script_dir
    / "TAASSC_by_kristopherkyle"
    / "pub_versions"
    / "TAASSC 2.0.0.58"
)
core_file = taassc_dir / "TAASSC_2.0.0.58.py"
index_file = taassc_dir / "lists_BTR" / "btr_index_list_5-26-20.txt"

output_file = script_dir / "taassc_results.csv"
failed_file = script_dir / "taassc_failed.csv"
autotune_file = script_dir / "taassc_autotune.json"

CORE_CUT_MARKER = "### Sample Data analyses here"


_worker_analyze = None
_worker_nlp = None
_worker_index_list = None
_worker_cats = None


def clean_text_exactly_as_taassc(text: str) -> str:
    """Mirror the nested clean_text function in TAASSC 2.0.0.58."""
    if "[" in text and "]" in text:
        text = re.sub(r"\[.*?\]", "", text)
    if "1:" in text:
        text = re.sub(r"\n[0-9]:", "", text)
    if " " in text:
        if "\n" not in text:
            text = " ".join(text.split())
        else:
            text = "\n".join(
                " ".join(line.split()) for line in text.split("\n")
            )
    return text


def patched_core_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if CORE_CUT_MARKER not in source:
        raise RuntimeError(
            "TAASSC sample marker was not found; the core version is not "
            "the expected TAASSC 2.0.0.58 file."
        )
    source = source.split(CORE_CUT_MARKER, 1)[0]

    old_signature = (
        "def BTR_Analysis(text,indices_dict,cats_d,output = False):"
    )
    new_signature = (
        "def BTR_Analysis(text,indices_dict,cats_d,output = False,"
        "_parsed_doc = None):"
    )
    old_parse = "\tdoc = nlp(clean_text(text))"
    new_parse = (
        "\tdoc = _parsed_doc if _parsed_doc is not None "
        "else nlp(clean_text(text))"
    )
    old_model = 'nlp = spacy.load("en_core_web_sm")'
    new_model = 'nlp = spacy.load("en_core_web_sm", disable=["ner"])'
    old_two_token_lookahead = (
        '" ".join([doc_text[token.head.i + 1].text,'
        'doc_text[token.head.i + 2].text])'
    )
    new_two_token_lookahead = (
        '" ".join(item.text for item in '
        'doc_text[token.head.i + 1:token.head.i + 3])'
    )

    for old, new, description in (
        (old_signature, new_signature, "BTR_Analysis signature"),
        (old_parse, new_parse, "spaCy parse call"),
        (old_model, new_model, "spaCy model load"),
        (
            old_two_token_lookahead,
            new_two_token_lookahead,
            "end-of-document token lookahead",
        ),
    ):
        if source.count(old) != 1:
            raise RuntimeError(
                f"Could not safely patch the {description}. "
                "The TAASSC core file differs from the expected version."
            )
        source = source.replace(old, new, 1)

    return source


def worker_initialize(core_path: str, working_directory: str) -> None:
    """Load TAASSC and spaCy once inside each persistent process."""
    global _worker_analyze, _worker_nlp, _worker_index_list, _worker_cats

    os.chdir(working_directory)
    path = Path(core_path)
    source = patched_core_source(path)
    namespace = {
        "__file__": str(path),
        "__name__": "embedded_taassc_core",
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        warnings.filterwarnings(
            "ignore", message="pkg_resources is deprecated.*"
        )
        exec(compile(source, str(path), "exec"), namespace)

    _worker_analyze = namespace["BTR_Analysis"]
    _worker_nlp = namespace["nlp"]
    _worker_index_list = namespace["index_list"]
    _worker_cats = namespace["cats"]


def output_row(stem: str, output: dict) -> list[str]:
    nwords = output["nwords"]
    if float(nwords) == 0.0:
        raise ValueError("TAASSC found zero countable words.")

    row = [stem]
    for feature in _worker_index_list:
        if feature in ("nwords", "wrd_length"):
            row.append(str(output[feature]))
        else:
            row.append(str((output[feature] / nwords) * 10000))
    return row


def analyze_prepared_file(
    path_text: str, original_text: str, parsed_doc,
) -> tuple[str, list[str] | str, str, str]:
    path = Path(path_text)
    try:
        output = _worker_analyze(
            original_text,
            _worker_index_list,
            _worker_cats,
            _parsed_doc=parsed_doc,
        )
        return "success", output_row(path.stem, output), "", ""
    except Exception:
        return (
            "failure",
            path.stem,
            "analysis",
            traceback.format_exc(),
        )


def worker_analyze_batch(
    path_texts: list[str],
) -> list[tuple[str, list[str] | str, str, str]]:
    prepared = []
    results = []

    for path_text in path_texts:
        path = Path(path_text)
        try:
            original = path.read_text(encoding="utf-8", errors="replace")
            prepared.append((path_text, original, clean_text_exactly_as_taassc(original)))
        except Exception:
            results.append((
                "failure", path.stem, "reading", traceback.format_exc()
            ))

    if not prepared:
        return results

    cleaned = [item[2] for item in prepared]
    try:
        docs = list(_worker_nlp.pipe(
            cleaned,
            batch_size=min(SPACY_BATCH_SIZE, len(cleaned)),
        ))
        for (path_text, original, _), doc in zip(prepared, docs):
            results.append(analyze_prepared_file(path_text, original, doc))
    except Exception:
        # A single unusual document should not discard the rest of its batch.
        # Fall back to isolated parsing to identify and log only that file.
        for path_text, original, cleaned_text in prepared:
            try:
                doc = _worker_nlp(cleaned_text)
                results.append(
                    analyze_prepared_file(path_text, original, doc)
                )
            except Exception:
                results.append((
                    "failure",
                    Path(path_text).stem,
                    "spaCy parsing",
                    traceback.format_exc(),
                ))

    return results


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def validate_installation() -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not core_file.exists():
        raise FileNotFoundError(f"TAASSC core file not found: {core_file}")
    if not index_file.exists():
        raise FileNotFoundError(f"TAASSC index list not found: {index_file}")

    missing = [
        name for name in ("spacy", "lexical_diversity", "en_core_web_sm")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        command = f'& "{sys.executable}" -m pip install '
        packages = []
        if "spacy" in missing:
            packages.append("spacy")
        if "lexical_diversity" in missing:
            packages.append("lexical-diversity")
        raise RuntimeError(
            "Missing Python packages: " + ", ".join(missing) + ".\n"
            + (command + " ".join(packages) if packages else "")
            + ("\nThen run: " + f'& "{sys.executable}" -m spacy download en_core_web_sm'
               if "en_core_web_sm" in missing else "")
        )

    # Validate that the in-memory batching patch matches this exact core.
    patched_core_source(core_file)


def load_index_list() -> list[str]:
    features = index_file.read_text(encoding="utf-8").splitlines()
    features = [feature.strip() for feature in features if feature.strip()]
    if not features:
        raise ValueError(f"No TAASSC features found in {index_file}")
    if len(features) != len(set(features)):
        raise ValueError("The TAASSC feature list contains duplicate names.")
    return features


def repair_and_load_completed(header: list[str]) -> set[str]:
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()

    completed = set()
    malformed = 0
    with output_file.open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, None)
        if existing_header != header:
            raise ValueError(
                f"Existing results have an unexpected header: {output_file}"
            )
        for row in reader:
            if len(row) == len(header) and row[0].strip():
                completed.add(row[0].strip())
            elif row:
                malformed += 1

    if malformed == 0:
        return completed

    print(
        f"Repairing {malformed:,} incomplete result row(s) left by "
        "an interrupted write..."
    )
    temporary = output_file.with_suffix(".csv.repairing")
    with output_file.open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as source, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination)
        next(reader, None)
        writer.writerow(header)
        for row in reader:
            if len(row) == len(header) and row[0].strip():
                writer.writerow(row)
    temporary.replace(output_file)
    return completed


def available_ram_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        pass

    if os.name == "nt":
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
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical / (1024 * 1024))
    return None


def current_cpu_percent() -> float | None:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=0.5))
    except ImportError:
        pass

    if os.name == "nt":
        def system_times() -> tuple[int, int, int] | None:
            idle = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            return idle.value, kernel.value, user.value

        first = system_times()
        if first is None:
            return None
        time.sleep(0.5)
        second = system_times()
        if second is None:
            return None
        idle_delta = second[0] - first[0]
        total_delta = second[1] - first[1] + second[2] - first[2]
        if total_delta > 0:
            return max(
                0.0,
                min(100.0, 100.0 * (1.0 - idle_delta / total_delta)),
            )
    return None


def maximum_feasible_workers(available_mb: int | None) -> int:
    logical_cpus = os.cpu_count() or 1
    cpu_limit = max(1, min(MAX_AUTO_WORKERS, logical_cpus // 2))
    if available_mb is None:
        return min(cpu_limit, 4)
    ram_limit = max(
        1,
        (available_mb - AUTO_TUNE_MIN_AVAILABLE_RAM_MB)
        // WORKER_MEMORY_BUDGET_MB,
    )
    return max(1, min(cpu_limit, ram_limit, MAX_AUTO_WORKERS))


def tuning_signature() -> dict:
    return {
        "profile_version": AUTOTUNE_PROFILE_VERSION,
        "logical_cpus": os.cpu_count() or 1,
        "python": sys.version.split()[0],
        "spacy": package_version("spacy"),
        "spacy_model": package_version("en-core-web-sm"),
        "lexical_diversity": package_version("lexical-diversity"),
        "core_sha256": hashlib.sha256(core_file.read_bytes()).hexdigest(),
        "files_per_task": FILES_PER_TASK,
        "spacy_batch_size": SPACY_BATCH_SIZE,
        "candidates": AUTO_TUNE_CANDIDATES,
        "input_dir": str(input_dir),
    }


def load_cached_tuning(
    available_mb: int | None, cpu_percent: float | None,
) -> int | None:
    if FORCE_RETUNE or not autotune_file.exists():
        return None
    try:
        profile = json.loads(autotune_file.read_text(encoding="utf-8"))
        if profile.get("signature") != tuning_signature():
            return None

        workers = int(profile["selected_workers"])
        feasible_now = maximum_feasible_workers(available_mb)
        if workers < 1 or workers > feasible_now:
            return None

        tested = [
            int(item["workers"])
            for item in profile.get("benchmarks", [])
            if "workers" in item
        ]
        if max(tested, default=0) < feasible_now:
            print("More workers are feasible now; auto-tuning again.")
            return None
        if workers == 1 and feasible_now > 1:
            print("Rechecking the cached one-worker setting.")
            return None

        old_cpu = profile.get("cpu_percent_at_tuning")
        if (
            old_cpu is not None and cpu_percent is not None
            and float(old_cpu) >= 50.0
            and cpu_percent <= float(old_cpu) - 20.0
        ):
            print("The computer is less busy now; auto-tuning again.")
            return None

        print(
            f"Reusing auto-tuned setting: {workers} workers "
            f"({profile.get('measured_rate', 0):.2f} files/sec)"
        )
        return workers
    except Exception:
        return None


def process_context():
    return multiprocessing.get_context("spawn")


def new_executor(worker_count: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context(),
        initializer=worker_initialize,
        initargs=(str(core_file), str(taassc_dir)),
    )


def make_batches(paths: list[Path], size: int = FILES_PER_TASK):
    for start in range(0, len(paths), size):
        yield [str(path) for path in paths[start:start + size]]


def benchmark_worker_count(
    worker_count: int, sample_paths: list[Path],
) -> dict:
    print(f"  Benchmarking {worker_count} workers...")
    executor = new_executor(worker_count)
    memory_samples = []
    monitor_stop = threading.Event()

    def monitor_memory() -> None:
        while not monitor_stop.wait(0.25):
            value = available_ram_mb()
            if value is not None:
                memory_samples.append(value)

    try:
        warmups = []
        warmup_count = worker_count * AUTO_TUNE_WARMUP_FILES_PER_WORKER
        for index in range(warmup_count):
            path = sample_paths[index % len(sample_paths)]
            warmups.append(executor.submit(worker_analyze_batch, [str(path)]))
        for future in as_completed(warmups):
            future.result()

        monitor = threading.Thread(target=monitor_memory, daemon=True)
        monitor.start()
        started = time.perf_counter()
        futures = [
            executor.submit(worker_analyze_batch, batch)
            for batch in make_batches(sample_paths)
        ]
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        elapsed = time.perf_counter() - started
        monitor_stop.set()
        monitor.join()

        completed = len(results)
        successful = sum(item[0] == "success" for item in results)
        minimum_ram = min(memory_samples) if memory_samples else None
        rate = completed / elapsed if elapsed else 0.0
        stable = completed == len(sample_paths) and successful > 0
        if minimum_ram is not None:
            stable = (
                stable
                and minimum_ram >= AUTO_TUNE_MIN_AVAILABLE_RAM_MB
            )

        result = {
            "workers": worker_count,
            "rate": round(rate, 4),
            "elapsed_seconds": round(elapsed, 3),
            "completed": completed,
            "successful": successful,
            "errors": completed - successful,
            "minimum_available_ram_mb": minimum_ram,
            "stable": stable,
        }
        ram_text = (
            f", minimum free RAM={minimum_ram:,} MB"
            if minimum_ram is not None else ""
        )
        print(f"    {rate:.2f} files/sec, stable={stable}{ram_text}")
        return result
    finally:
        monitor_stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        time.sleep(0.5)


def save_tuning_profile(
    workers: int,
    rate: float,
    benchmarks: list[dict],
    available_mb: int | None,
    cpu_percent: float | None,
) -> None:
    profile = {
        "signature": tuning_signature(),
        "selected_workers": workers,
        "measured_rate": rate,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "available_ram_mb_at_tuning": available_mb,
        "cpu_percent_at_tuning": cpu_percent,
        "benchmarks": benchmarks,
    }
    temporary = autotune_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    temporary.replace(autotune_file)


def choose_worker_count(
    remaining: list[Path],
    available_mb: int | None,
    cpu_percent: float | None,
) -> int:
    if MAX_WORKERS > 0:
        print(f"Using manually configured workers: {MAX_WORKERS}")
        return MAX_WORKERS

    cached = load_cached_tuning(available_mb, cpu_percent)
    if cached is not None:
        return cached

    feasible = maximum_feasible_workers(available_mb)
    if not AUTO_TUNE or len(remaining) < 40:
        return feasible

    candidates = [
        count for count in AUTO_TUNE_CANDIDATES if count <= feasible
    ] or [1]
    sample_size = min(AUTO_TUNE_SAMPLE_SIZE, len(remaining))
    sample_paths = random.Random(20260720).sample(remaining, sample_size)
    print(f"Auto-tuning candidates: {candidates} on {sample_size} texts")

    benchmarks = []
    for count in candidates:
        try:
            result = benchmark_worker_count(count, sample_paths)
        except Exception as error:
            result = {
                "workers": count,
                "rate": 0.0,
                "elapsed_seconds": 0.0,
                "completed": 0,
                "successful": 0,
                "errors": 1,
                "minimum_available_ram_mb": available_ram_mb(),
                "stable": False,
                "startup_or_benchmark_error": str(error)[:2000],
            }
            print(f"    Unstable: {error}")
        benchmarks.append(result)
        if not result["stable"]:
            break

    stable = [item for item in benchmarks if item["stable"]]
    if not stable:
        details = benchmarks[-1].get("startup_or_benchmark_error", "")
        raise RuntimeError(
            "No TAASSC worker configuration completed the benchmark.\n"
            + details
        )

    fastest_rate = max(item["rate"] for item in stable)
    close_rate = fastest_rate * (
        1.0 - AUTO_TUNE_CLOSE_RATE_PERCENT / 100.0
    )
    selected = min(
        (item for item in stable if item["rate"] >= close_rate),
        key=lambda item: item["workers"],
    )
    save_tuning_profile(
        selected["workers"],
        selected["rate"],
        benchmarks,
        available_mb,
        cpu_percent,
    )
    print(
        f"Auto-tuner selected {selected['workers']} workers at "
        f"{selected['rate']:.2f} files/sec."
    )
    return selected["workers"]


def run_full_analysis(
    remaining: list[Path],
    initial_workers: int,
    result_writer,
    failure_writer,
    output_handle,
    failure_handle,
) -> tuple[int, int, int]:
    source_batches = iter(make_batches(remaining))
    retry_batches: deque[list[str]] = deque()
    source_exhausted = False
    worker_count = initial_workers
    pool_restarts = 0

    completed = 0
    successful = 0
    failed = 0
    unflushed = 0
    started = time.time()
    last_progress_bucket = 0
    last_resource_check = time.monotonic()
    consecutive_low_ram = 0

    def next_batch() -> list[str] | None:
        nonlocal source_exhausted
        if retry_batches:
            return retry_batches.popleft()
        if source_exhausted:
            return None
        try:
            return next(source_batches)
        except StopIteration:
            source_exhausted = True
            return None

    while retry_batches or not source_exhausted:
        print(f"Running with {worker_count} persistent TAASSC workers...")
        executor = new_executor(worker_count)
        pending = {}
        infrastructure_failure = None
        retire_after_pending = False

        def fill_pending() -> None:
            if retire_after_pending:
                return
            limit = max(1, worker_count * PENDING_BATCHES_PER_WORKER)
            while len(pending) < limit:
                batch = next_batch()
                if batch is None:
                    break
                future = executor.submit(worker_analyze_batch, batch)
                pending[future] = batch

        try:
            fill_pending()
            while pending:
                finished, _ = wait(
                    pending, return_when=FIRST_COMPLETED
                )
                for future in finished:
                    batch = pending.pop(future)
                    try:
                        batch_results = future.result()
                    except Exception as error:
                        retry_batches.appendleft(batch)
                        infrastructure_failure = error
                        break

                    for status, payload, stage, message in batch_results:
                        if status == "success":
                            result_writer.writerow(payload)
                            successful += 1
                        else:
                            failure_writer.writerow([
                                payload, stage, message[:10000]
                            ])
                            failed += 1
                        completed += 1
                        unflushed += 1

                        if unflushed >= CHECKPOINT_EVERY:
                            output_handle.flush()
                            failure_handle.flush()
                            unflushed = 0

                    progress_bucket = completed // PROGRESS_EVERY
                    if (
                        progress_bucket > last_progress_bucket
                        or completed == len(remaining)
                    ):
                        last_progress_bucket = progress_bucket
                        elapsed = time.time() - started
                        rate = completed / elapsed if elapsed else 0.0
                        left = len(remaining) - completed
                        eta_hours = left / rate / 3600 if rate else 0.0
                        print(
                            f"[{completed:,}/{len(remaining):,}] "
                            f"successful={successful:,}, failed={failed:,}, "
                            f"rate={rate:.2f} files/sec, "
                            f"ETA={eta_hours:.2f} hours"
                        )

                    now = time.monotonic()
                    if (
                        now - last_resource_check
                        >= RUNTIME_RESOURCE_CHECK_SECONDS
                    ):
                        last_resource_check = now
                        free_mb = available_ram_mb()
                        if (
                            free_mb is not None
                            and free_mb < RUNTIME_MIN_AVAILABLE_RAM_MB
                        ):
                            consecutive_low_ram += 1
                        else:
                            consecutive_low_ram = 0
                        if (
                            consecutive_low_ram >= RUNTIME_LOW_RAM_CHECKS
                            and worker_count > 1
                        ):
                            print(
                                "Available RAM remained below "
                                f"{RUNTIME_MIN_AVAILABLE_RAM_MB:,} MB; "
                                "retiring one worker after current jobs."
                            )
                            retire_after_pending = True
                            consecutive_low_ram = 0

                if infrastructure_failure is not None:
                    break
                fill_pending()

            if infrastructure_failure is not None:
                for future, batch in pending.items():
                    future.cancel()
                    retry_batches.appendleft(batch)
                pending.clear()
                pool_restarts += 1
                if pool_restarts > MAX_POOL_RESTARTS:
                    raise RuntimeError(
                        "The TAASSC worker pool repeatedly stopped. "
                        "Completed rows are safe; rerun to continue."
                    ) from infrastructure_failure
                if worker_count > 1:
                    worker_count -= 1
                print(
                    "A worker process stopped unexpectedly; restarting "
                    f"with {worker_count} workers."
                )
            elif retire_after_pending and worker_count > 1:
                worker_count -= 1
        except KeyboardInterrupt:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            output_handle.flush()
            failure_handle.flush()
            raise
        finally:
            if infrastructure_failure is None:
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=False, cancel_futures=True)

    output_handle.flush()
    failure_handle.flush()
    return successful, failed, worker_count


def main() -> None:
    validate_installation()
    features = load_index_list()
    header = ["filename", *features]
    completed_before = repair_and_load_completed(header)

    all_files = sorted(input_dir.glob("*.txt"), key=lambda path: path.name)
    remaining = [
        path for path in all_files if path.stem not in completed_before
    ]

    print(f"Already completed: {len(completed_before):,}")
    print(f"Remaining raw files: {len(remaining):,}")
    if not remaining:
        print("Everything is already analyzed.")
        return

    available_mb = available_ram_mb()
    cpu_percent = current_cpu_percent()
    if available_mb is not None:
        print(f"Available RAM at startup: {available_mb:,} MB")
    if cpu_percent is not None:
        print(f"CPU use at startup: {cpu_percent:.1f}%")
    print("GPU: not used; TAASSC is parallelized across CPU processes")

    worker_count = choose_worker_count(
        remaining, available_mb, cpu_percent
    )
    useful_workers = max(
        1, (len(remaining) + FILES_PER_TASK - 1) // FILES_PER_TASK
    )
    worker_count = min(worker_count, useful_workers)
    print(f"Selected TAASSC workers: {worker_count}")

    output_exists = output_file.exists() and output_file.stat().st_size > 0
    try:
        with output_file.open(
            "a", encoding="utf-8", newline="", buffering=1024 * 1024
        ) as output_handle, failed_file.open(
            "w", encoding="utf-8", newline="", buffering=1024 * 1024
        ) as failure_handle:
            result_writer = csv.writer(output_handle)
            failure_writer = csv.writer(failure_handle)
            if not output_exists:
                result_writer.writerow(header)
                output_handle.flush()
            failure_writer.writerow(["filename", "stage", "error"])
            failure_handle.flush()

            successful, failed, final_workers = run_full_analysis(
                remaining,
                worker_count,
                result_writer,
                failure_writer,
                output_handle,
                failure_handle,
            )
    except KeyboardInterrupt:
        print(
            "\nStopped safely. Run the same command again to resume from "
            "the last checkpoint."
        )
        return

    print("\nTAASSC run completed.")
    print(f"Successful this run: {successful:,}")
    print(f"Failed this run: {failed:,}")
    print(f"Final active workers: {final_workers}")
    print(f"Results: {output_file}")
    print(f"Failure log: {failed_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
