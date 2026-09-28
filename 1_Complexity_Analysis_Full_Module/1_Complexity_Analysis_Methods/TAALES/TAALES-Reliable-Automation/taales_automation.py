#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "taales_automation_config.json"

LOCATION_OF_MAIN = Path(__file__).resolve().parents[4]

DEFAULT_SOURCE = (
    LOCATION_OF_MAIN
    / "0_Data"
    / "raw_texts"
)

DEFAULT_WORK_ROOT = (
    LOCATION_OF_MAIN
    / "1_Complexity_Analysis_Full_Module"
    / "1_Complexity_Analysis_Methods"
    / "TAALES"
)

RUNTIME_DIALOG_TERMS = (
    "microsoft visual c++ runtime library",
    "runtime error",
    "application error",
    "taales_2.2.exe - application error",
    "error message",
)

_INSTANCE_LOCK_FILE = None


def require_windows() -> None:
    if os.name != "nt":
        raise SystemExit("This automation must be run on Windows.")


def import_gui_modules():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    try:
        import pyautogui
        import pygetwindow
        import pyperclip
        import psutil
    except ImportError as exc:
        raise SystemExit(
            "Missing GUI dependencies. Run Start-TAALES-Automation.ps1 "
            "instead of launching this file directly."
        ) from exc

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.20
    return pyautogui, pygetwindow, pyperclip, psutil


@dataclass
class Paths:
    work_root: Path
    parts: Path
    current: Path
    batch_input: Path
    logs: Path
    database: Path
    failures: Path
    final_results: Path
    final_coverage: Path
    compaction_journal: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        return cls(
            work_root=root,
            parts=root / "parts",
            current=root / "current",
            batch_input=root / "current" / "input",
            logs=root / "logs",
            database=root / "state.sqlite3",
            failures=root / "failed_files.txt",
            final_results=root / "taales_results_final.csv",
            final_coverage=root / "taales_results_final_index_coverage.csv",
            compaction_journal=root / "compaction_journal.json",
        )

    def create(self) -> None:
        for path in (
            self.work_root,
            self.parts,
            self.current,
            self.batch_input,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


def configure_logging(paths: Paths) -> None:
    paths.create()
    log_path = paths.logs / "taales_automation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def prompt_path(label: str, default: Path | None = None, optional: bool = False) -> Path | None:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip().strip('"')
        if not value and default:
            return default
        if not value and optional:
            return None
        path = Path(value)
        if path.exists() or label.lower().startswith("work"):
            return path
        print(f"Path does not exist: {path}")


def search_taales_exe() -> Path | None:
    home = Path.home()
    roots = [
        home / "Downloads",
        home / "Desktop",
        home / "Documents",
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for pattern in ("TAALES_2.2.exe", "TAALES*.exe"):
                candidates.extend(root.rglob(pattern))
        except (PermissionError, OSError):
            continue
    candidates = [p for p in candidates if p.is_file()]
    candidates.sort(key=lambda p: ("2.2" not in p.name, len(str(p))))
    return candidates[0] if candidates else None



class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _window_title(hwnd: int) -> str:
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def find_native_taales_hwnd() -> int:
    user32 = ctypes.windll.user32
    candidates: list[tuple[int, int, str]] = []

    enum_proc_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
    )

    def callback(hwnd, _):
        hwnd = int(hwnd)
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            title = _window_title(hwnd).strip()
            title_lower = title.lower()
            if "taales" not in title_lower:
                return True

            rect = _WinRect()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True

            width = max(0, rect.right - rect.left)
            height = max(0, rect.bottom - rect.top)
            area = width * height

            title_score = 2 if "taales version 2.2" in title_lower else 1
            candidates.append((title_score, area, hwnd))
        except Exception:
            pass
        return True

    user32.EnumWindows(enum_proc_type(callback), 0)
    if not candidates:
        return 0

    candidates.sort(reverse=True)
    return int(candidates[0][2])


def force_foreground_hwnd(hwnd: int, keep_topmost: bool = False) -> None:
    if not hwnd:
        return

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    SW_MAXIMIZE = 3
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_SHOWWINDOW = 0x0040
    flags = SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW

    foreground = int(user32.GetForegroundWindow())
    current_tid = int(kernel32.GetCurrentThreadId())
    foreground_tid = (
        int(user32.GetWindowThreadProcessId(foreground, None))
        if foreground else 0
    )
    target_tid = int(user32.GetWindowThreadProcessId(hwnd, None))

    attached_fg = False
    attached_target = False

    try:
        user32.ShowWindowAsync(hwnd, SW_MAXIMIZE)

        if foreground_tid and foreground_tid != current_tid:
            attached_fg = bool(
                user32.AttachThreadInput(current_tid, foreground_tid, True)
            )

        if target_tid and target_tid != current_tid:
            attached_target = bool(
                user32.AttachThreadInput(current_tid, target_tid, True)
            )

        user32.BringWindowToTop(hwnd)
        user32.SetActiveWindow(hwnd)
        user32.SetFocus(hwnd)
        user32.SetForegroundWindow(hwnd)

        # Force the z-order once. Keep topmost during automation when requested.
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
        if not keep_topmost:
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)

        user32.SetForegroundWindow(hwnd)
    finally:
        if attached_target:
            user32.AttachThreadInput(current_tid, target_tid, False)
        if attached_fg:
            user32.AttachThreadInput(current_tid, foreground_tid, False)

    time.sleep(1.0)


def show_calibration_message(label: str, seconds: int) -> None:
    message = (
        f"Next control:\n\n{label}\n\n"
        f"After clicking OK, move the mouse over that control and keep it still.\n"
        f"The position will be captured after {seconds} seconds.\n\n"
        "Do not click the TAALES button."
    )
    ctypes.windll.user32.MessageBoxW(
        0,
        message,
        "TAALES Automation Calibration",
        0x00000040 | 0x00040000,  # information icon + topmost
    )


class _WinRect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def native_window_rect(win) -> tuple[int, int, int, int]:
    """Return the live physical-pixel rectangle for a Windows window."""
    try:
        hwnd = int(win._hWnd)
        rect = _WinRect()
        if hwnd and ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass

    left = int(getattr(win, "left", 0))
    top = int(getattr(win, "top", 0))
    width = int(getattr(win, "width", 0))
    height = int(getattr(win, "height", 0))
    return left, top, left + width, top + height


def rect_area(win) -> int:
    left, top, right, bottom = native_window_rect(win)
    return max(0, right - left) * max(0, bottom - top)


def taales_window_containing_point(pygetwindow, x: int, y: int):
    """Find the visible TAALES window underneath a captured mouse point."""
    matches = []
    for candidate in window_candidates(pygetwindow):
        left, top, right, bottom = native_window_rect(candidate)
        if left <= x <= right and top <= y <= bottom:
            matches.append(candidate)
    matches.sort(key=rect_area, reverse=True)
    return matches[0] if matches else None


def window_candidates(pygetwindow):
    result = []
    for win in pygetwindow.getAllWindows():
        title = (win.title or "").strip()
        if "taales" in title.lower() and win.width >= 300 and win.height >= 200:
            result.append(win)
    result.sort(
        key=lambda w: (
            "version 2.2" in ((w.title or "").lower()),
            rect_area(w),
        ),
        reverse=True,
    )
    return result


def wait_for_main_window(pygetwindow, timeout: int = 180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = find_native_taales_hwnd()
        if hwnd:
            wins = window_candidates(pygetwindow)
            for win in wins:
                try:
                    if int(win._hWnd) == hwnd:
                        force_foreground_hwnd(hwnd, keep_topmost=False)
                        return win
                except Exception:
                    continue

            if wins:
                force_foreground_hwnd(hwnd, keep_topmost=False)
                return wins[0]

        time.sleep(1)
    raise TimeoutError("Could not find the TAALES main window.")


def activate_maximize(win) -> None:
    hwnd = find_native_taales_hwnd()
    if not hwnd:
        try:
            hwnd = int(win._hWnd)
        except Exception:
            hwnd = 0

    if hwnd:
        force_foreground_hwnd(hwnd, keep_topmost=False)
        return

    try:
        win.maximize()
        win.activate()
    except Exception:
        pass
    time.sleep(1.0)


def capture_relative_point(
    pyautogui,
    win,
    label: str,
    seconds: int = 7,
    pygetwindow=None,
) -> list[float]:
    show_calibration_message(label, seconds)

    hwnd = find_native_taales_hwnd()
    if not hwnd:
        try:
            hwnd = int(win._hWnd)
        except Exception:
            hwnd = 0

    if not hwnd:
        raise RuntimeError("Could not find the TAALES GUI window.")
    force_foreground_hwnd(hwnd, keep_topmost=True)

    for _ in range(seconds):
        try:
            ctypes.windll.kernel32.Beep(900, 90)
        except Exception:
            pass
        time.sleep(0.91)

    x, y = pyautogui.position()

    rect = _WinRect()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("Could not read the live TAALES window rectangle.")

    left, top, right, bottom = (
        int(rect.left),
        int(rect.top),
        int(rect.right),
        int(rect.bottom),
    )
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid TAALES window rectangle: ({left}, {top})-({right}, {bottom})."
        )

    tolerance = 12
    if not (
        left - tolerance <= x <= right + tolerance
        and top - tolerance <= y <= bottom + tolerance
    ):
        raise RuntimeError(
            f"Captured point ({x}, {y}) was outside the maximized TAALES "
            f"window ({left}, {top})-({right}, {bottom})."
        )

    rx = min(1.0, max(0.0, (x - left) / width))
    ry = min(1.0, max(0.0, (y - top) / height))

    try:
        ctypes.windll.kernel32.Beep(1350, 350)
    except Exception:
        pass

    print(
        f"Captured {label}: screen=({x}, {y}), "
        f"window=({left}, {top})-({right}, {bottom}), "
        f"relative=({rx:.4f}, {ry:.4f})"
    )

    # Return TAALES to normal topmost behavior, while keeping it maximized.
    force_foreground_hwnd(hwnd, keep_topmost=False)
    return [rx, ry]


def click_relative(pyautogui, win, point: Sequence[float]) -> None:
    hwnd = find_native_taales_hwnd()
    if not hwnd:
        try:
            hwnd = int(win._hWnd)
        except Exception:
            hwnd = 0

    if not hwnd:
        raise RuntimeError("Cannot click: TAALES GUI window was not found.")

    force_foreground_hwnd(hwnd, keep_topmost=True)

    rect = _WinRect()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("Cannot click: failed to read TAALES window rectangle.")

    left, top, right, bottom = (
        int(rect.left),
        int(rect.top),
        int(rect.right),
        int(rect.bottom),
    )
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise RuntimeError("Cannot click: invalid TAALES window rectangle.")

    x = int(left + width * float(point[0]))
    y = int(top + height * float(point[1]))

    pyautogui.moveTo(x, y, duration=0.15)
    pyautogui.click(x, y)
    time.sleep(0.45)

    force_foreground_hwnd(hwnd, keep_topmost=False)


def launch_taales(exe: Path) -> subprocess.Popen:
    """
    Launch TAALES with a real Windows console encoding.

    Older frozen Python GUI applications can report:
        LookupError: unknown encoding: cp0

    when stdout/stderr are redirected to DEVNULL or Windows reports console
    code page 0. Inherit the launcher's streams and explicitly provide a valid
    encoding instead.
    """
    logging.info("Launching TAALES: %s", exe)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    env["PYTHONUTF8"] = "1"

    creationflags = 0
    if os.name == "nt":
        # Give console-based frozen builds a valid console/code page. A GUI
        # subsystem build simply ignores the visible console behavior.
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    return subprocess.Popen(
        [str(exe)],
        cwd=str(exe.parent),
        env=env,
        stdin=None,
        stdout=None,
        stderr=None,
        creationflags=creationflags,
    )



def automation_state_exists(root: Path) -> bool:
    """Return True if a folder already contains state from an older automation run."""
    markers = (
        root / "state.sqlite3",
        root / "parts",
        root / "current",
        root / "taales_results_final.csv",
        root / "taales_results_final_index_coverage.csv",
        root / "failed_files.txt",
    )
    return any(path.exists() for path in markers)


def make_unused_fresh_work_root(requested: Path) -> Path:
    """
    Never mix a new from-zero run with an older automation state.

    If the selected folder already contains automation artifacts, create a new
    timestamped sibling folder and store that exact path in the configuration.
    """
    if not automation_state_exists(requested):
        return requested

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = requested.with_name(f"{requested.name}_{timestamp}")
    counter = 1
    while candidate.exists():
        candidate = requested.with_name(
            f"{requested.name}_{timestamp}_{counter:02d}"
        )
        counter += 1

    print()
    print("The selected work folder already contains automation state.")
    print("Older results and state will NOT be imported.")
    print("A new fresh work folder will be used instead:")
    print(f"  {candidate}")
    return candidate


def setup() -> dict:
    require_windows()
    pyautogui, pygetwindow, _, _ = import_gui_modules()

    print("\nTAALES 2.2 automation setup")
    print("This is a one-time calibration. Later runs are unattended.\n")

    # Do not recursively scan Documents/Downloads: the corpus may contain hundreds of thousands of files.
    taales_exe = prompt_path("TAALES_2.2.exe")
    source_dir = prompt_path("Folder containing all source .txt files", DEFAULT_SOURCE)
    requested_work_root = prompt_path("Work/output folder", DEFAULT_WORK_ROOT)
    work_root = make_unused_fresh_work_root(requested_work_root)

    print()
    print("Fresh-run mode is enabled.")
    print("Earlier/manual results.csv files will be ignored.")
    print("Only progress created inside this new work folder will be used for resume.")

    raw_batch_size = input("Normal batch size [1000]: ").strip()
    batch_size = int(raw_batch_size or "1000")
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    process = launch_taales(taales_exe)
    win = wait_for_main_window(pygetwindow)
    activate_maximize(win)

    print("\nThis automation is configured to use ALL TAALES features.")
    print("You do not need to manually select features before every batch.")
    print("The automation will click all FOUR Select All buttons after every TAALES restart and before each run.")
    print("Calibration uses Windows popups and keeps TAALES maximized for stable button positions.")
    input("Press Enter to begin button calibration...")

    clicks: dict[str, object] = {}
    clicks["select_all_buttons"] = [
        capture_relative_point(
            pyautogui, win,
            "Select All under Options (Frequency, Academic Language, Other Index Types)",
            pygetwindow=pygetwindow,
        ),
        capture_relative_point(
            pyautogui, win,
            "Select All under COCA Word Frequency and Range",
            pygetwindow=pygetwindow,
        ),
        capture_relative_point(
            pyautogui, win,
            "Select All under COCA Bigram Frequency, Range, and Association Strength",
            pygetwindow=pygetwindow,
        ),
        capture_relative_point(
            pyautogui, win,
            "Select All under COCA Trigram Frequency, Range, and Association Strength",
            pygetwindow=pygetwindow,
        ),
    ]

    clicks["input"] = capture_relative_point(
        pyautogui, win, "the Select Input Folder button", pygetwindow=pygetwindow
    )
    clicks["output"] = capture_relative_point(
        pyautogui, win, "the Select Output Filename button", pygetwindow=pygetwindow
    )
    clicks["run"] = capture_relative_point(
        pyautogui, win, "the button that starts the analysis", pygetwindow=pygetwindow
    )

    config = {
        "taales_exe": str(taales_exe),
        "source_dir": str(source_dir),
        "work_root": str(work_root),
        "automation_version": 13,
        "external_results_mode": "ignore",
        "batch_size": batch_size,
        "feature_mode": "all",
        "single_file_retries": 2,
        "stuck_reclick_minutes": 3,
        "stuck_reclick_max_attempts": 3,
        "stuck_cpu_seconds_per_check": 1.0,
        "stall_timeout_minutes": 60,
        "batch_timeout_hours": 24,
        "reuse_taales_between_batches": True,
        "clicks": clicks,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nSaved configuration to:\n  {CONFIG_PATH}")
    print("Close TAALES before starting the unattended run.")
    try:
        process.terminate()
    except Exception:
        pass
    return config


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"Configuration not found: {CONFIG_PATH}\n"
            "Run Start-TAALES-Automation.ps1 once and choose setup."
        )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    # External/manual CSVs are intentionally ignored, even if an older config
    # still contains these keys.
    config.pop("initial_results", None)
    config.pop("initial_coverage", None)
    config["external_results_mode"] = "ignore"
    clicks = config.get("clicks", {})
    if "select_all_buttons" not in clicks:
        raise SystemExit(
            "This configuration was created by an older one-button calibration. "
            "Delete taales_automation_config.json and run setup again with the "
            "four-Select-All version."
        )
    required = ("taales_exe", "source_dir", "work_root", "batch_size", "clicks")
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Configuration is missing: {', '.join(missing)}")
    return config


def connect_database(path: Path) -> sqlite3.Connection:
    """
    Open the durable progress database.

    FULL synchronous mode makes committed filename checkpoints resistant to a
    sudden Python termination or VM power loss.
    """
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA busy_timeout=60000")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS completed (
            filename TEXT PRIMARY KEY,
            part_number INTEGER NOT NULL,
            added_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS retry_attempts (
            filename TEXT PRIMARY KEY,
            attempts INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS failed (
            filename TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            added_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_number INTEGER PRIMARY KEY,
            job_dir TEXT NOT NULL,
            status TEXT,
            reason TEXT,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS compacted (
            filename TEXT PRIMARY KEY,
            compacted_at TEXT NOT NULL
        )
        """
    )
    con.commit()
    return con


def acquire_single_instance(work_root: Path) -> None:
    """
    Prevent two automation instances from using the same work folder.

    Uses a Windows byte-range file lock rather than a named Win32 mutex.
    The operating system releases this lock automatically if Python exits,
    crashes, or the VM reboots, so stale lock files do not block recovery.
    """
    global _INSTANCE_LOCK_FILE

    import msvcrt

    work_root.mkdir(parents=True, exist_ok=True)
    lock_path = work_root / ".taales_automation.lock"

    lock_file = lock_path.open("a+b")
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
        os.fsync(lock_file.fileno())

    lock_file.seek(0)

    try:
        msvcrt.locking(
            lock_file.fileno(),
            msvcrt.LK_NBLCK,
            1,
        )
    except OSError:
        lock_file.close()
        raise SystemExit(10)

    _INSTANCE_LOCK_FILE = lock_file


def release_single_instance(work_root: Path) -> None:
    """Release the byte-range lock and remove its harmless marker file."""
    global _INSTANCE_LOCK_FILE

    handle = _INSTANCE_LOCK_FILE
    _INSTANCE_LOCK_FILE = None

    if handle is not None:
        try:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    try:
        (work_root / ".taales_automation.lock").unlink(missing_ok=True)
    except OSError:
        # A racing process may already have opened the lock file. In that case,
        # leaving the zero-byte marker is safe; Windows still enforces the lock.
        logging.warning(
            "Could not remove the harmless lock marker: %s",
            work_root / ".taales_automation.lock",
        )


def get_completed(con: sqlite3.Connection) -> set[str]:
    return {row[0] for row in con.execute("SELECT filename FROM completed")}


def get_failed(con: sqlite3.Connection) -> set[str]:
    return {row[0] for row in con.execute("SELECT filename FROM failed")}


def next_part_number(paths: Paths, con: sqlite3.Connection) -> int:
    """Avoid reusing a part number even after a crash between file and DB writes."""
    numbers: list[int] = []

    db_value = con.execute("SELECT MAX(part_number) FROM completed").fetchone()[0]
    if db_value is not None:
        numbers.append(int(db_value))

    for pattern in ("results_part_*.csv", "coverage_part_*.csv"):
        for path in paths.parts.glob(pattern):
            match = re.search(r"_(\d+)\.csv$", path.name)
            if match:
                numbers.append(int(match.group(1)))

    return (max(numbers) + 1) if numbers else 0


def next_job_number(paths: Paths, con: sqlite3.Connection) -> int:
    numbers: list[int] = []

    db_value = con.execute("SELECT MAX(job_number) FROM jobs").fetchone()[0]
    if db_value is not None:
        numbers.append(int(db_value))

    for job_dir in paths.current.glob("job_*"):
        if not job_dir.is_dir():
            continue
        match = re.fullmatch(r"job_(\d+)", job_dir.name)
        if match:
            numbers.append(int(match.group(1)))

    return (max(numbers) + 1) if numbers else 1


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def load_json_safely(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def manifest_path(job_dir: Path) -> Path:
    return job_dir / "job_manifest.json"


def create_job_manifest(
    job_dir: Path,
    job_number: int,
    files: Sequence[Path],
    expected_result: Path,
) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "manifest_version": 1,
        "job_number": job_number,
        "status": "prepared",
        "created_at": now,
        "updated_at": now,
        "expected_result": str(expected_result),
        "files": [path.name for path in files],
    }
    atomic_write_json(manifest_path(job_dir), payload)
    return payload


def update_job_manifest(job_dir: Path, **changes) -> None:
    path = manifest_path(job_dir)
    payload = load_json_safely(path)
    payload.update(changes)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(path, payload)


def register_job(
    con: sqlite3.Connection,
    job_number: int,
    job_dir: Path,
    status: str,
    reason: str | None = None,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.execute(
            """
            INSERT INTO jobs(job_number, job_dir, status, reason, started_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_number) DO UPDATE SET
                job_dir=excluded.job_dir,
                status=excluded.status,
                reason=excluded.reason,
                updated_at=excluded.updated_at
            """,
            (job_number, str(job_dir), status, reason, now, now),
        )


def update_job_record(
    con: sqlite3.Connection,
    job_number: int,
    status: str,
    reason: str | None = None,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.execute(
            """
            UPDATE jobs
            SET status=?, reason=?, updated_at=?
            WHERE job_number=?
            """,
            (status, reason, now, job_number),
        )


def increment_retry_attempt(con: sqlite3.Connection, filename: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.execute(
            """
            INSERT INTO retry_attempts(filename, attempts, updated_at)
            VALUES(?, 1, ?)
            ON CONFLICT(filename) DO UPDATE SET
                attempts=retry_attempts.attempts + 1,
                updated_at=excluded.updated_at
            """,
            (filename, now),
        )
    row = con.execute(
        "SELECT attempts FROM retry_attempts WHERE filename=?",
        (filename,),
    ).fetchone()
    return int(row[0])


def mark_failed(
    con: sqlite3.Connection,
    filename: str,
    reason: str,
    attempts: int,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.execute(
            """
            INSERT INTO failed(filename, reason, attempts, added_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                reason=excluded.reason,
                attempts=excluded.attempts,
                added_at=excluded.added_at
            """,
            (filename, reason, attempts, now),
        )



def read_csv_rows(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], {}
        rows: dict[str, list[str]] = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            # TAALES 2.2 index-coverage output commonly writes one extra,
            # empty trailing field that is absent from its header.
            if len(row) == len(header) + 1 and row[-1] == "":
                row = row[:-1]
            if len(row) != len(header):
                logging.warning(
                    "Skipping malformed row in %s: expected %d fields, got %d",
                    path,
                    len(header),
                    len(row),
                )
                continue
            rows[row[0]] = row
    return header, rows


def write_clean_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    temp.replace(path)


def metadata_get(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def metadata_set(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    con.commit()



def read_csv_header(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def iter_valid_csv_rows(
    path: Path,
    expected_header: Sequence[str],
):
    """
    Stream valid rows without loading a complete combined CSV into memory.
    """
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(f"CSV is empty: {path}")

        if list(header) != list(expected_header):
            raise RuntimeError(f"Unexpected CSV header in {path}")

        for row in reader:
            if not row or not row[0].strip():
                continue
            if len(row) == len(header) + 1 and row[-1] == "":
                row = row[:-1]
            if len(row) != len(header):
                logging.warning(
                    "Skipping malformed row in %s: expected %d fields, got %d",
                    path,
                    len(header),
                    len(row),
                )
                continue
            yield row


def _metadata_set_in_transaction(
    con: sqlite3.Connection,
    key: str,
    value: str,
) -> None:
    con.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logging.exception("Could not delete %s", path)


def remove_tree_best_effort(
    path: Path,
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 0.5,
) -> bool:
    """
    Delete a directory without allowing a temporary Windows file lock to stop
    the automation.

    TAALES may keep the just-written results.csv open briefly after a batch.
    Retry with a short backoff. If the directory is still locked, leave it for
    the next cleanup pass/startup; its rows are already durable in the combined
    master files and SQLite.
    """
    if not path.exists():
        return True

    last_error: OSError | None = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            last_error = exc
            if attempt < attempts:
                delay = initial_delay_seconds * attempt
                logging.info(
                    "Job folder is temporarily locked; retrying deletion in "
                    "%.1f seconds (%d/%d): %s",
                    delay,
                    attempt,
                    attempts,
                    path,
                )
                time.sleep(delay)

    logging.warning(
        "Deferred deletion of locked job folder until a later cleanup pass: "
        "%s | %s",
        path,
        last_error,
    )
    return False


def _truncate_file(path: Path, size: int) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Cannot recover compacted output because this file is missing: {path}"
        )
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


def _delete_journal_parts(payload: dict) -> None:
    for key in ("result_part", "coverage_part"):
        value = payload.get(key)
        if value:
            _safe_unlink(Path(value))


def _query_compacted_names(
    con: sqlite3.Connection,
    names: Sequence[str],
) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(names), 800):
        chunk = list(names[start : start + 800])
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        rows = con.execute(
            f"SELECT filename FROM compacted "
            f"WHERE filename IN ({placeholders})",
            chunk,
        )
        found.update(row[0] for row in rows)
    return found


def _set_master_metadata(
    con: sqlite3.Connection,
    paths: Paths,
) -> None:
    compacted_count = int(
        con.execute("SELECT COUNT(*) FROM compacted").fetchone()[0]
    )
    with con:
        _metadata_set_in_transaction(
            con,
            "master_result_size",
            str(paths.final_results.stat().st_size),
        )
        _metadata_set_in_transaction(
            con,
            "master_coverage_size",
            str(paths.final_coverage.stat().st_size),
        )
        _metadata_set_in_transaction(
            con,
            "master_row_count",
            str(compacted_count),
        )


def _resolve_master_headers(
    paths: Paths,
    con: sqlite3.Connection,
) -> tuple[list[str], list[str]] | None:
    result_json = metadata_get(con, "result_header")
    coverage_json = metadata_get(con, "coverage_header")

    if result_json and coverage_json:
        return json.loads(result_json), json.loads(coverage_json)

    if paths.final_results.exists() and paths.final_coverage.exists():
        result_header = read_csv_header(paths.final_results)
        coverage_header = read_csv_header(paths.final_coverage)
        if result_header and coverage_header:
            metadata_set(
                con,
                "result_header",
                json.dumps(result_header, ensure_ascii=False),
            )
            metadata_set(
                con,
                "coverage_header",
                json.dumps(coverage_header, ensure_ascii=False),
            )
            return result_header, coverage_header

    result_parts = sorted(paths.parts.glob("results_part_*.csv"))
    for result_part in result_parts:
        suffix = result_part.name.removeprefix("results_part_")
        coverage_part = paths.parts / f"coverage_part_{suffix}"
        if not coverage_part.exists():
            continue

        result_header = read_csv_header(result_part)
        coverage_header = read_csv_header(coverage_part)
        if result_header and coverage_header:
            metadata_set(
                con,
                "result_header",
                json.dumps(result_header, ensure_ascii=False),
            )
            metadata_set(
                con,
                "coverage_header",
                json.dumps(coverage_header, ensure_ascii=False),
            )
            return result_header, coverage_header

    return None


def _ensure_master_pair(
    paths: Paths,
    result_header: Sequence[str],
    coverage_header: Sequence[str],
) -> None:
    result_exists = paths.final_results.exists()
    coverage_exists = paths.final_coverage.exists()

    if result_exists != coverage_exists:
        existing = (
            paths.final_results if result_exists else paths.final_coverage
        )
        existing_header = read_csv_header(existing)
        data_rows = count_data_rows(existing)

        # A header-only half-created pair can be safely recreated.
        if data_rows == 0 and existing_header:
            _safe_unlink(paths.final_results)
            _safe_unlink(paths.final_coverage)
            result_exists = coverage_exists = False
        else:
            raise RuntimeError(
                "Only one combined master file exists. Refusing to delete or "
                "overwrite data. Restore the missing paired file before continuing."
            )

    if not result_exists:
        write_clean_csv(paths.final_results, result_header, ())
        write_clean_csv(paths.final_coverage, coverage_header, ())
        return

    if read_csv_header(paths.final_results) != list(result_header):
        raise RuntimeError(
            f"Unexpected header in combined results file: {paths.final_results}"
        )
    if read_csv_header(paths.final_coverage) != list(coverage_header):
        raise RuntimeError(
            f"Unexpected header in combined coverage file: {paths.final_coverage}"
        )


def _master_index_is_current(
    paths: Paths,
    con: sqlite3.Connection,
) -> bool:
    stored_result_size = metadata_get(con, "master_result_size")
    stored_coverage_size = metadata_get(con, "master_coverage_size")
    stored_count = metadata_get(con, "master_row_count")

    if not stored_result_size or not stored_coverage_size or stored_count is None:
        return False

    actual_count = int(
        con.execute("SELECT COUNT(*) FROM compacted").fetchone()[0]
    )
    return (
        int(stored_result_size) == paths.final_results.stat().st_size
        and int(stored_coverage_size) == paths.final_coverage.stat().st_size
        and int(stored_count) == actual_count
    )


def _rebuild_compacted_index(
    paths: Paths,
    con: sqlite3.Connection,
    result_header: Sequence[str],
    coverage_header: Sequence[str],
) -> int:
    """
    Rebuild the compacted filename index from an already-existing master pair.

    Temporary SQLite tables store only filenames, avoiding loading the full
    high-dimensional result files into RAM.
    """
    logging.info(
        "Scanning pre-made combined result files to rebuild the compacted index."
    )

    con.execute("DROP TABLE IF EXISTS temp_master_results")
    con.execute("DROP TABLE IF EXISTS temp_master_coverage")
    con.execute(
        "CREATE TEMP TABLE temp_master_results("
        "filename TEXT PRIMARY KEY)"
    )
    con.execute(
        "CREATE TEMP TABLE temp_master_coverage("
        "filename TEXT PRIMARY KEY)"
    )

    def load_names(
        path: Path,
        header: Sequence[str],
        table: str,
    ) -> None:
        batch: list[tuple[str]] = []
        for row in iter_valid_csv_rows(path, header):
            batch.append((row[0],))
            if len(batch) >= 5000:
                con.executemany(
                    f"INSERT OR IGNORE INTO {table}(filename) VALUES(?)",
                    batch,
                )
                batch.clear()
        if batch:
            con.executemany(
                f"INSERT OR IGNORE INTO {table}(filename) VALUES(?)",
                batch,
            )

    with con:
        load_names(
            paths.final_results,
            result_header,
            "temp_master_results",
        )
        load_names(
            paths.final_coverage,
            coverage_header,
            "temp_master_coverage",
        )

    result_only = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM temp_master_results r
            LEFT JOIN temp_master_coverage c
              ON c.filename = r.filename
            WHERE c.filename IS NULL
            """
        ).fetchone()[0]
    )
    coverage_only = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM temp_master_coverage c
            LEFT JOIN temp_master_results r
              ON r.filename = c.filename
            WHERE r.filename IS NULL
            """
        ).fetchone()[0]
    )

    if result_only or coverage_only:
        raise RuntimeError(
            "The pre-made combined result and coverage files do not contain "
            "the same filenames. No chunks were deleted. "
            f"Results-only={result_only}, coverage-only={coverage_only}"
        )

    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.execute("DELETE FROM compacted")
        con.execute(
            """
            INSERT INTO compacted(filename, compacted_at)
            SELECT r.filename, ?
            FROM temp_master_results r
            INNER JOIN temp_master_coverage c
              ON c.filename = r.filename
            """,
            (now,),
        )

    con.execute("DROP TABLE IF EXISTS temp_master_results")
    con.execute("DROP TABLE IF EXISTS temp_master_coverage")

    _set_master_metadata(con, paths)
    count = int(con.execute("SELECT COUNT(*) FROM compacted").fetchone()[0])
    logging.info("Pre-made combined files contain %d safe rows.", count)
    return count


def recover_compaction_journal(
    paths: Paths,
    con: sqlite3.Connection,
) -> None:
    payload = load_json_safely(paths.compaction_journal)
    if not payload:
        return

    status = str(payload.get("status", "prepared"))
    names = [
        str(name)
        for name in payload.get("filenames", [])
        if str(name).strip()
    ]
    old_result_size = int(payload.get("old_result_size", 0))
    old_coverage_size = int(payload.get("old_coverage_size", 0))

    logging.warning(
        "Recovering interrupted result compaction: status=%s rows=%d",
        status,
        len(names),
    )

    already_compacted = _query_compacted_names(con, names)
    database_committed = bool(names) and len(already_compacted) == len(names)

    if status == "committed" or database_committed:
        _delete_journal_parts(payload)
        _safe_unlink(paths.compaction_journal)
        if paths.final_results.exists() and paths.final_coverage.exists():
            _set_master_metadata(con, paths)
        logging.info("Finished the previously committed compaction.")
        return

    # The database was not committed. Restore both masters to their exact
    # pre-append byte sizes, then retry the intact part pair normally.
    _truncate_file(paths.final_results, old_result_size)
    _truncate_file(paths.final_coverage, old_coverage_size)
    _safe_unlink(paths.compaction_journal)
    _set_master_metadata(con, paths)
    logging.info("Rolled back the interrupted master append safely.")


def _append_part_pair_to_master(
    result_part: Path,
    coverage_part: Path,
    paths: Paths,
    con: sqlite3.Connection,
    result_header: Sequence[str],
    coverage_header: Sequence[str],
) -> int:
    part_result_header, result_rows = read_csv_rows(result_part)
    part_coverage_header, coverage_rows = read_csv_rows(coverage_part)

    if part_result_header != list(result_header):
        raise RuntimeError(f"Unexpected result header in {result_part}")
    if part_coverage_header != list(coverage_header):
        raise RuntimeError(f"Unexpected coverage header in {coverage_part}")

    result_names = set(result_rows)
    coverage_names = set(coverage_rows)
    if result_names != coverage_names:
        raise RuntimeError(
            "A result part and its coverage part have different filenames. "
            f"No files were deleted: {result_part.name}"
        )

    ordered_names = list(result_rows)
    already = _query_compacted_names(con, ordered_names)
    new_names = [name for name in ordered_names if name not in already]

    if not new_names:
        _safe_unlink(result_part)
        _safe_unlink(coverage_part)
        return 0

    old_result_size = paths.final_results.stat().st_size
    old_coverage_size = paths.final_coverage.stat().st_size

    payload = {
        "journal_version": 1,
        "status": "prepared",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_part": str(result_part),
        "coverage_part": str(coverage_part),
        "filenames": new_names,
        "old_result_size": old_result_size,
        "old_coverage_size": old_coverage_size,
    }
    atomic_write_json(paths.compaction_journal, payload)

    with paths.final_results.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(result_rows[name] for name in new_names)
        output.flush()
        os.fsync(output.fileno())

    with paths.final_coverage.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(coverage_rows[name] for name in new_names)
        output.flush()
        os.fsync(output.fileno())

    payload["status"] = "appended"
    payload["new_result_size"] = paths.final_results.stat().st_size
    payload["new_coverage_size"] = paths.final_coverage.stat().st_size
    atomic_write_json(paths.compaction_journal, payload)

    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.executemany(
            "INSERT OR IGNORE INTO compacted(filename, compacted_at) "
            "VALUES(?, ?)",
            ((name, now) for name in new_names),
        )
        compacted_count = int(
            con.execute("SELECT COUNT(*) FROM compacted").fetchone()[0]
        )
        _metadata_set_in_transaction(
            con,
            "master_result_size",
            str(paths.final_results.stat().st_size),
        )
        _metadata_set_in_transaction(
            con,
            "master_coverage_size",
            str(paths.final_coverage.stat().st_size),
        )
        _metadata_set_in_transaction(
            con,
            "master_row_count",
            str(compacted_count),
        )

    payload["status"] = "committed"
    atomic_write_json(paths.compaction_journal, payload)

    _safe_unlink(result_part)
    _safe_unlink(coverage_part)
    _safe_unlink(paths.compaction_journal)

    logging.info(
        "Compacted %d rows into the permanent master files and deleted %s.",
        len(new_names),
        result_part.name,
    )
    return len(new_names)


def cleanup_compacted_storage(paths: Paths) -> tuple[int, int]:
    """
    Remove duplicate raw job output after startup recovery and compaction.

    The shared current/input folder is preserved, but its old hard links are
    cleared. Old logs are already truncated by configure_logging().
    """
    deleted_jobs = 0
    deleted_temp_files = 0

    for job_dir in sorted(paths.current.glob("job_*")):
        if not job_dir.is_dir():
            continue
        if remove_tree_best_effort(job_dir):
            deleted_jobs += 1

    clear_batch_input(paths.batch_input)

    stale_temps = (
        paths.final_results.with_suffix(".csv.tmp"),
        paths.final_coverage.with_suffix(".csv.tmp"),
    )
    for temp_path in stale_temps:
        if temp_path.exists():
            _safe_unlink(temp_path)
            deleted_temp_files += 1

    return deleted_jobs, deleted_temp_files


def compact_pending_outputs(
    paths: Paths,
    con: sqlite3.Connection,
    *,
    clean_job_folders: bool = True,
) -> tuple[int, int]:
    """
    Merge all checkpointed chunks into the permanent combined CSV pair.

    Existing pre-made master files are recognized and indexed. Each new part
    pair is appended through a durable rollback journal and deleted only after
    both master files and SQLite have been committed.
    """
    headers = _resolve_master_headers(paths, con)
    if headers is None:
        logging.info(
            "Storage compaction: no completed CSV rows exist yet."
        )
        return 0, 0

    result_header, coverage_header = headers

    # If a previous append was interrupted, the master files already exist.
    # Recover it before validating/indexing.
    if paths.compaction_journal.exists():
        recover_compaction_journal(paths, con)

    _ensure_master_pair(
        paths,
        result_header,
        coverage_header,
    )

    if not _master_index_is_current(paths, con):
        _rebuild_compacted_index(
            paths,
            con,
            result_header,
            coverage_header,
        )

    result_parts = sorted(paths.parts.glob("results_part_*.csv"))
    pending_pairs: list[tuple[Path, Path]] = []

    for result_part in result_parts:
        suffix = result_part.name.removeprefix("results_part_")
        coverage_part = paths.parts / f"coverage_part_{suffix}"
        if not coverage_part.exists():
            raise RuntimeError(
                f"Missing paired coverage chunk for {result_part}"
            )
        pending_pairs.append((result_part, coverage_part))

    # A coverage chunk without a result chunk must never be deleted.
    result_names = {path.name for path in result_parts}
    for coverage_part in paths.parts.glob("coverage_part_*.csv"):
        expected_result = (
            "results_part_"
            + coverage_part.name.removeprefix("coverage_part_")
        )
        if expected_result not in result_names:
            raise RuntimeError(
                f"Missing paired result chunk for {coverage_part}"
            )

    compacted_now = 0
    for result_part, coverage_part in pending_pairs:
        compacted_now += _append_part_pair_to_master(
            result_part,
            coverage_part,
            paths,
            con,
            result_header,
            coverage_header,
        )

    missing_from_master = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM completed c
            LEFT JOIN compacted m
              ON m.filename = c.filename
            WHERE m.filename IS NULL
            """
        ).fetchone()[0]
    )
    if missing_from_master:
        raise RuntimeError(
            f"{missing_from_master} completed filenames are not present in "
            "the combined master files and no recoverable part remains. "
            "No further cleanup was performed."
        )

    deleted_jobs = 0
    if clean_job_folders:
        deleted_jobs, _ = cleanup_compacted_storage(paths)

    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.DatabaseError:
        logging.warning("Could not truncate the SQLite WAL file.")

    total = int(con.execute("SELECT COUNT(*) FROM compacted").fetchone()[0])
    logging.info(
        "Storage compaction complete: master rows=%d newly compacted=%d "
        "deleted job folders=%d",
        total,
        compacted_now,
        deleted_jobs,
    )
    return total, compacted_now



def salvage_pair(
    result_path: Path,
    coverage_path: Path,
    requested: set[str],
    paths: Paths,
    con: sqlite3.Connection,
    label: str,
) -> set[str]:
    result_header, result_rows = read_csv_rows(result_path)
    coverage_header, coverage_rows = read_csv_rows(coverage_path)

    if not result_header or not coverage_header:
        logging.warning(
            "No salvageable output pair for %s. Results=%s Coverage=%s",
            label,
            result_path.exists(),
            coverage_path.exists(),
        )
        return set()

    expected_result_header = metadata_get(con, "result_header")
    expected_coverage_header = metadata_get(con, "coverage_header")
    result_header_json = json.dumps(result_header, ensure_ascii=False)
    coverage_header_json = json.dumps(coverage_header, ensure_ascii=False)

    if expected_result_header and expected_result_header != result_header_json:
        raise RuntimeError(
            "The result columns changed. The TAALES index selection does not "
            "match the earlier output. Recalibrate with the exact same indices."
        )
    if expected_coverage_header and expected_coverage_header != coverage_header_json:
        raise RuntimeError(
            "The index-coverage columns changed. The TAALES index selection does "
            "not match the earlier output."
        )

    if not expected_result_header:
        metadata_set(con, "result_header", result_header_json)
    if not expected_coverage_header:
        metadata_set(con, "coverage_header", coverage_header_json)

    already = get_completed(con)
    common = (
        set(result_rows)
        .intersection(coverage_rows)
        .intersection(requested)
        .difference(already)
    )
    if not common:
        return set()

    # Preserve the order in which TAALES emitted the rows.
    ordered = [name for name in result_rows if name in common]
    part_number = next_part_number(paths, con)
    part_result = paths.parts / f"results_part_{part_number:06d}.csv"
    part_coverage = paths.parts / f"coverage_part_{part_number:06d}.csv"

    write_clean_csv(
        part_result,
        result_header,
        (result_rows[name] for name in ordered),
    )
    write_clean_csv(
        part_coverage,
        coverage_header,
        (coverage_rows[name] for name in ordered),
    )

    now = datetime.now().isoformat(timespec="seconds")
    with con:
        con.executemany(
            "INSERT OR IGNORE INTO completed(filename, part_number, added_at) "
            "VALUES(?, ?, ?)",
            ((name, part_number, now) for name in ordered),
        )

    logging.info(
        "Preserved %d rows from %s as part %06d",
        len(ordered),
        label,
        part_number,
    )
    return set(ordered)


def seed_initial_outputs(config: dict, paths: Paths, con: sqlite3.Connection) -> None:
    """
    Intentionally do nothing.

    Version 10 never imports external/manual results.csv files. Resume state is
    taken only from state.sqlite3 and parts created by this automation.
    """
    logging.info(
        "External result import is disabled; using only this automation's own state."
    )



def _requested_names_for_job(
    job_dir: Path,
    result_path: Path,
    coverage_path: Path,
) -> set[str]:
    """
    Load the durable manifest. For a pre-v12 job without a manifest, safely
    infer only filenames present in both output files.
    """
    payload = load_json_safely(manifest_path(job_dir))
    names = payload.get("files", [])
    if isinstance(names, list):
        requested = {str(name) for name in names if str(name).strip()}
        if requested:
            return requested

    _, result_rows = read_csv_rows(result_path)
    _, coverage_rows = read_csv_rows(coverage_path)
    return set(result_rows).intersection(coverage_rows)


def recover_interrupted_jobs(
    paths: Paths,
    con: sqlite3.Connection,
) -> int:
    """
    Salvage every valid row left by a Python crash, forced termination, or VM
    reboot before creating any new job or deleting any current output.
    """
    total_preserved = 0

    for job_dir in sorted(paths.current.glob("job_*"), key=lambda p: p.name):
        if not job_dir.is_dir():
            continue

        match = re.fullmatch(r"job_(\d+)", job_dir.name)
        if not match:
            continue

        job_number = int(match.group(1))
        expected_result = job_dir / f"{job_dir.name}.csv"
        result_path, coverage_path = resolve_output_pair(expected_result)
        requested = _requested_names_for_job(
            job_dir,
            result_path,
            coverage_path,
        )

        if not requested:
            continue

        try:
            preserved = salvage_pair(
                result_path,
                coverage_path,
                requested,
                paths,
                con,
                f"startup recovery for {job_dir.name}",
            )
        except Exception:
            logging.exception("Could not recover interrupted %s", job_dir.name)
            continue

        total_preserved += len(preserved)
        remaining = requested.difference(get_completed(con))

        update_job_manifest(
            job_dir,
            status="recovered_after_restart",
            recovered_rows=len(preserved),
            remaining_rows=len(remaining),
        )
        register_job(
            con,
            job_number,
            job_dir,
            "recovered_after_restart",
            f"preserved={len(preserved)} remaining={len(remaining)}",
        )

        logging.info(
            "Startup recovery %s: preserved=%d remaining=%d",
            job_dir.name,
            len(preserved),
            len(remaining),
        )

    if total_preserved:
        logging.info(
            "Startup recovery preserved %d previously uncheckpointed rows.",
            total_preserved,
        )
    else:
        logging.info("Startup recovery found no uncheckpointed complete rows.")

    return total_preserved



def clear_batch_input(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except FileNotFoundError:
            pass


def prepare_batch(files: Sequence[Path], batch_input: Path) -> None:
    clear_batch_input(batch_input)
    for source in files:
        destination = batch_input / source.name
        try:
            os.link(source, destination)
        except OSError:
            # Hard links require source and destination to be on the same volume.
            shutil.copy2(source, destination)


def coverage_name(result_path: Path) -> Path:
    return result_path.with_name(
        f"{result_path.stem}_index_coverage{result_path.suffix}"
    )


def output_pair_candidates(expected_result: Path) -> list[tuple[Path, Path]]:
    """
    Candidate output pairs in priority order.

    TAALES normally receives a unique filename such as job_000001.csv. If its
    Save dialog nevertheless keeps the built-in default results.csv, the job
    has its own private directory, so that fallback is still safe and recoverable.
    """
    expected_result = expected_result.resolve()
    candidates = [
        (expected_result, coverage_name(expected_result)),
        (
            expected_result.parent / "results.csv",
            expected_result.parent / "results_index_coverage.csv",
        ),
    ]

    unique: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for result, coverage in candidates:
        key = (str(result).lower(), str(coverage).lower())
        if key not in seen:
            unique.append((result, coverage))
            seen.add(key)
    return unique


def resolve_output_pair(expected_result: Path) -> tuple[Path, Path]:
    """Return the pair TAALES is actually writing, including default results.csv."""
    candidates = output_pair_candidates(expected_result)

    # Prefer a pair where both files exist.
    for result, coverage in candidates:
        if result.exists() and coverage.exists():
            return result, coverage

    # During writing, one file may appear before the other.
    for result, coverage in candidates:
        if result.exists() or coverage.exists():
            return result, coverage

    return candidates[0]


def clear_output_candidates(expected_result: Path) -> None:
    """Remove stale expected/default files before a job begins."""
    for result, coverage in output_pair_candidates(expected_result):
        for path in (result, coverage):
            try:
                path.unlink()
            except FileNotFoundError:
                pass



def paste_text(pyautogui, pyperclip, value: str) -> None:
    pyperclip.copy(value)
    pyautogui.hotkey("ctrl", "v")


def wait_for_dialog_or_delay(pygetwindow, main_title: str, timeout: int = 12) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        active = pygetwindow.getActiveWindow()
        title = (active.title if active else "") or ""
        if title and title != main_title:
            return
        time.sleep(0.25)
    time.sleep(1)



def set_folder_dialog_path_with_uia(folder: Path, timeout: int = 15) -> bool:
    """
    Use Windows UI Automation to replace the folder chooser's current value
    (often shown as 'Windows (C:)') with the exact batch-input path.

    Returns True when the path was entered and the dialog was confirmed.
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return False

    deadline = time.time() + timeout
    target = str(folder)

    while time.time() < deadline:
        try:
            hwnd = int(ctypes.windll.user32.GetForegroundWindow())
            if not hwnd:
                time.sleep(0.25)
                continue

            dialog = Desktop(backend="uia").window(handle=hwnd)

            # Ignore the main TAALES window; wait for the folder-selection dialog.
            title = (dialog.window_text() or "").lower()
            class_name = (dialog.friendly_class_name() or "").lower()
            if "taales" in title and "dialog" not in class_name:
                time.sleep(0.25)
                continue

            edits = [
                control
                for control in dialog.descendants(control_type="Edit")
                if control.is_visible() and control.is_enabled()
            ]

            if not edits:
                time.sleep(0.25)
                continue

            # Prefer the edit containing the initial drive label shown by this
            # TAALES folder dialog, e.g. "Windows (C:)".
            folder_edit = None
            for edit in edits:
                try:
                    current = (edit.window_text() or "").strip().lower()
                except Exception:
                    current = ""

                if (
                    "windows (" in current
                    or current.endswith(":)")
                    or current == "windows (c:)"
                ):
                    folder_edit = edit
                    break

            # In this dialog the folder-name edit is usually the final visible
            # Edit control. Use that as the fallback rather than the address bar.
            if folder_edit is None:
                folder_edit = edits[-1]

            folder_edit.set_focus()
            try:
                folder_edit.set_edit_text(target)
            except Exception:
                folder_edit.type_keys("^a{BACKSPACE}", set_foreground=False)
                folder_edit.type_keys(
                    target,
                    with_spaces=True,
                    set_foreground=False,
                )

            time.sleep(0.5)

            # Prefer a visible button whose name means select/choose/OK.
            buttons = [
                control
                for control in dialog.descendants(control_type="Button")
                if control.is_visible() and control.is_enabled()
            ]

            preferred_terms = (
                "select folder",
                "choose folder",
                "select",
                "choose",
                "ok",
                "open",
            )
            for term in preferred_terms:
                for button in buttons:
                    try:
                        name = (button.window_text() or "").strip().lower()
                    except Exception:
                        name = ""
                    if name == term or term in name:
                        button.click_input()
                        time.sleep(1.5)
                        return True

            # Enter activates the dialog's default confirmation button.
            folder_edit.type_keys("{ENTER}", set_foreground=False)
            time.sleep(1.5)
            return True

        except Exception:
            time.sleep(0.25)

    return False


def choose_input_folder(
    pyautogui,
    pygetwindow,
    pyperclip,
    win,
    point,
    folder: Path,
) -> None:
    main_title = win.title
    click_relative(pyautogui, win, point)
    wait_for_dialog_or_delay(pygetwindow, main_title)

    # The TAALES folder dialog initially places "Windows (C:)" in its Folder
    # field. UI Automation finds that exact edit box, replaces its contents
    # with the required batch folder, and confirms the dialog.
    if set_folder_dialog_path_with_uia(folder):
        return

    # Keyboard fallback for systems where Windows UI Automation is unavailable.
    # Alt+N commonly focuses the Folder/File name field in the standard dialog.
    pyautogui.hotkey("alt", "n")
    time.sleep(0.5)
    pyautogui.hotkey("ctrl", "a")
    paste_text(pyautogui, pyperclip, str(folder))
    pyautogui.press("enter")
    time.sleep(1.5)

    # Some folder dialogs navigate on the first Enter and require a second Enter
    # to confirm "Select Folder".
    active = pygetwindow.getActiveWindow()
    active_title = ((active.title if active else "") or "").strip()
    if active_title and active_title != main_title:
        pyautogui.press("enter")
        time.sleep(1.5)



def _click_save_overwrite_confirmation(timeout: int = 5) -> None:
    """Click Yes/Replace if Windows shows an overwrite-confirmation dialog."""
    try:
        from pywinauto import Desktop
    except ImportError:
        return

    deadline = time.time() + timeout
    affirmative_names = (
        "yes",
        "&yes",
        "replace",
        "save",
        "overwrite",
        "ja",
        "&ja",
        "ersetzen",
    )

    while time.time() < deadline:
        try:
            hwnd = int(ctypes.windll.user32.GetForegroundWindow())
            if not hwnd:
                time.sleep(0.2)
                continue

            dialog = Desktop(backend="uia").window(handle=hwnd)
            title = (dialog.window_text() or "").strip().lower()

            # Once TAALES is back in front there is no confirmation to handle.
            if "taales version 2.2" in title:
                return

            buttons = [
                control
                for control in dialog.descendants(control_type="Button")
                if control.is_visible() and control.is_enabled()
            ]

            for button in buttons:
                try:
                    name = (button.window_text() or "").strip().lower()
                except Exception:
                    name = ""

                if name in affirmative_names or any(
                    term in name
                    for term in ("replace", "overwrite", "yes", "ersetzen")
                ):
                    button.click_input()
                    time.sleep(1)
                    return

        except Exception:
            pass

        time.sleep(0.2)


def set_save_dialog_output_with_uia(output: Path, timeout: int = 20) -> bool:
    """
    Enter the full output CSV path in the Save dialog's File name field and
    explicitly click the Save button.

    Returns True only after the Save action has been issued.
    """
    try:
        from pywinauto import Desktop
    except ImportError:
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    # The dialog is navigated to output.parent before this function is called,
    # so only the unique filename belongs in the File name field.
    target = output.name
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            hwnd = int(ctypes.windll.user32.GetForegroundWindow())
            if not hwnd:
                time.sleep(0.25)
                continue

            dialog = Desktop(backend="uia").window(handle=hwnd)
            title = (dialog.window_text() or "").strip().lower()

            # Wait for the Save dialog rather than acting on the TAALES window.
            if "taales version 2.2" in title:
                time.sleep(0.25)
                continue

            edits = [
                control
                for control in dialog.descendants(control_type="Edit")
                if control.is_visible() and control.is_enabled()
            ]

            combos = [
                control
                for control in dialog.descendants(control_type="ComboBox")
                if control.is_visible() and control.is_enabled()
            ]

            filename_control = None

            # Standard Windows Save As dialogs often identify the File name
            # control with AutomationId 1001 or FileNameControlHost.
            for control in [*edits, *combos]:
                try:
                    automation_id = (
                        control.element_info.automation_id or ""
                    ).strip().lower()
                    name = (
                        control.element_info.name or ""
                    ).strip().lower()
                except Exception:
                    automation_id = ""
                    name = ""

                if (
                    automation_id in {"1001", "filenamecontrolhost"}
                    or "file name" in name
                    or "filename" in name
                    or "dateiname" in name
                ):
                    filename_control = control
                    break

            # The final visible edit is normally the File name box, whereas the
            # first edit is frequently the address/search field.
            if filename_control is None and edits:
                filename_control = edits[-1]

            if filename_control is None:
                time.sleep(0.25)
                continue

            filename_control.set_focus()

            try:
                filename_control.set_edit_text(target)
            except Exception:
                filename_control.type_keys(
                    "^a{BACKSPACE}",
                    set_foreground=False,
                )
                filename_control.type_keys(
                    target,
                    with_spaces=True,
                    set_foreground=False,
                )

            time.sleep(0.5)

            # Verify that the default "results.csv" was actually replaced.
            try:
                current_value = (
                    filename_control.get_value()
                    if hasattr(filename_control, "get_value")
                    else filename_control.window_text()
                )
            except Exception:
                current_value = ""

            if output.name.lower() not in str(current_value).lower():
                filename_control.set_focus()
                filename_control.type_keys(
                    "^a{BACKSPACE}",
                    set_foreground=False,
                )
                filename_control.type_keys(
                    output.name,
                    with_spaces=True,
                    set_foreground=False,
                )
                time.sleep(0.4)

            buttons = [
                control
                for control in dialog.descendants(control_type="Button")
                if control.is_visible() and control.is_enabled()
            ]

            save_button = None
            save_names = (
                "save",
                "&save",
                "speichern",
                "&speichern",
            )

            for button in buttons:
                try:
                    name = (button.window_text() or "").strip().lower()
                    automation_id = (
                        button.element_info.automation_id or ""
                    ).strip().lower()
                except Exception:
                    name = ""
                    automation_id = ""

                if (
                    name in save_names
                    or name.replace("&", "") == "save"
                    or automation_id in {"1", "save"}
                ):
                    save_button = button
                    break

            if save_button is not None:
                save_button.click_input()
            else:
                # Standard Save-dialog keyboard accelerator.
                filename_control.type_keys("%s", set_foreground=False)

            time.sleep(1.2)
            _click_save_overwrite_confirmation()
            return True

        except Exception:
            time.sleep(0.25)

    return False


def choose_output_file(
    pyautogui,
    pygetwindow,
    pyperclip,
    win,
    point,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    main_title = win.title

    click_relative(pyautogui, win, point)
    wait_for_dialog_or_delay(pygetwindow, main_title)

    # First navigate the Save dialog to this job's private output folder.
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.4)
    pyautogui.hotkey("ctrl", "a")
    paste_text(pyautogui, pyperclip, str(output.parent.resolve()))
    pyautogui.press("enter")
    time.sleep(1.0)

    # Preferred path: replace the default results.csv with the unique job name
    # and explicitly click Save.
    if set_save_dialog_output_with_uia(output):
        time.sleep(1)
        return

    # Keyboard fallback: focus File name, replace results.csv, then click Save.
    pyautogui.hotkey("alt", "n")
    time.sleep(0.5)
    pyautogui.hotkey("ctrl", "a")
    paste_text(pyautogui, pyperclip, output.name)
    time.sleep(0.4)
    pyautogui.hotkey("alt", "s")
    time.sleep(1.5)

    active = pygetwindow.getActiveWindow()
    active_title = ((active.title if active else "") or "").strip()
    if active_title and active_title != main_title:
        pyautogui.press("enter")
        time.sleep(1)

    _click_save_overwrite_confirmation()


def taales_processes(psutil):
    matches = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            exe = (proc.info.get("exe") or "").lower()
            if "taales" in name or "taales" in exe:
                matches.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def process_cpu_total(psutil) -> float:
    total = 0.0
    for proc in taales_processes(psutil):
        try:
            times = proc.cpu_times()
            total += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def close_runtime_dialogs(pyautogui, pygetwindow) -> bool:
    found = False
    for win in pygetwindow.getAllWindows():
        title = (win.title or "").lower()
        if any(term in title for term in RUNTIME_DIALOG_TERMS):
            found = True
            try:
                win.activate()
            except Exception:
                pass
            time.sleep(0.3)
            pyautogui.press("enter")
    return found


def kill_taales(psutil) -> None:
    processes = taales_processes(psutil)
    for proc in processes:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone, alive = psutil.wait_procs(processes, timeout=8)
    for proc in alive:
        try:
            proc.kill()
        except Exception:
            pass


def count_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return max(0, count - 1)



@dataclass
class IncrementalCsvRowCounter:
    """
    Count appended CSV lines without rereading the complete file each poll.

    A short tail fingerprint detects truncation/overwrite, including when
    TAALES restarts the same output after Process Texts is clicked again.
    """

    active_path: Path | None = None
    offset: int = 0
    newline_count: int = 0
    tail: bytes = b""

    def _reset_for(self, path: Path) -> None:
        self.active_path = path
        self.offset = 0
        self.newline_count = 0
        self.tail = b""

    def count(self, path: Path) -> int:
        resolved = path.resolve()

        if self.active_path != resolved:
            self._reset_for(resolved)

        if not resolved.exists() or resolved.stat().st_size == 0:
            self._reset_for(resolved)
            return 0

        size = resolved.stat().st_size
        must_recount = size < self.offset

        with resolved.open("rb") as handle:
            if not must_recount and self.offset and self.tail:
                tail_start = self.offset - len(self.tail)
                handle.seek(tail_start)
                if handle.read(len(self.tail)) != self.tail:
                    must_recount = True

            if must_recount:
                self.offset = 0
                self.newline_count = 0
                self.tail = b""

            handle.seek(self.offset)
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                self.newline_count += chunk.count(b"\n")

            self.offset = handle.tell()
            tail_length = min(256, self.offset)
            if tail_length:
                handle.seek(self.offset - tail_length)
                self.tail = handle.read(tail_length)
            else:
                self.tail = b""

        # The first line is the header. An incomplete final line is not counted
        # until TAALES writes its newline.
        return max(0, self.newline_count - 1)



def wait_for_batch(
    expected_rows: int,
    expected_result_path: Path,
    config: dict,
    pyautogui,
    pygetwindow,
    psutil,
) -> tuple[bool, str, Path, Path]:
    """
    Wait for a TAALES batch and recover the known "Processing x of y" freeze.

    Performance changes:
    - poll every 5 seconds by default instead of every 15 seconds;
    - settle for 5 seconds after both CSVs reach the expected count instead of
      waiting at least 30 seconds;
    - count only newly appended bytes rather than rescanning the full CSVs.

    The existing stuck-job re-click, hard timeout, and validation safeguards
    remain active.
    """
    start = time.time()
    previous_cpu = process_cpu_total(psutil)
    previous_signature = (-1, -1, -1, -1)
    last_output_change = start
    last_meaningful_cpu = start
    complete_since: float | None = None
    reclick_attempts = 0

    poll_seconds = max(
        2.0,
        float(config.get("progress_poll_seconds", 5)),
    )
    settle_seconds = max(
        3.0,
        float(config.get("completion_settle_seconds", 5)),
    )

    stall_seconds = int(config.get("stall_timeout_minutes", 60)) * 60
    timeout_seconds = int(config.get("batch_timeout_hours", 24)) * 3600
    reclick_seconds = max(
        60,
        int(float(config.get("stuck_reclick_minutes", 3)) * 60),
    )
    max_reclicks = max(
        0,
        int(config.get("stuck_reclick_max_attempts", 3)),
    )

    # Scale the meaningful-CPU threshold to the shorter polling interval.
    configured_cpu_delta = max(
        0.1,
        float(config.get("stuck_cpu_seconds_per_check", 1.0)),
    )
    meaningful_cpu_delta = configured_cpu_delta * (poll_seconds / 15.0)

    result_counter = IncrementalCsvRowCounter()
    coverage_counter = IncrementalCsvRowCounter()

    active_result, active_coverage = resolve_output_pair(expected_result_path)

    while True:
        time.sleep(poll_seconds)
        now = time.time()

        runtime_error = close_runtime_dialogs(pyautogui, pygetwindow)
        active_result, active_coverage = resolve_output_pair(expected_result_path)

        result_rows = result_counter.count(active_result)
        coverage_rows = coverage_counter.count(active_coverage)
        result_size = (
            active_result.stat().st_size if active_result.exists() else 0
        )
        coverage_size = (
            active_coverage.stat().st_size if active_coverage.exists() else 0
        )
        signature = (result_rows, coverage_rows, result_size, coverage_size)

        cpu = process_cpu_total(psutil)
        cpu_delta = max(0.0, cpu - previous_cpu)
        process_alive = bool(taales_processes(psutil))

        output_changed = signature != previous_signature
        if output_changed:
            last_output_change = now
        if cpu_delta >= meaningful_cpu_delta:
            last_meaningful_cpu = now

        previous_cpu = cpu
        previous_signature = signature

        inactive_seconds = now - max(last_output_change, last_meaningful_cpu)

        logging.info(
            "Batch progress: results=%d/%d coverage=%d/%d "
            "result_file=%s CPU=%.1fs delta=%.1fs alive=%s "
            "inactive=%ds reclicks=%d/%d",
            result_rows,
            expected_rows,
            coverage_rows,
            expected_rows,
            active_result.name,
            cpu,
            cpu_delta,
            process_alive,
            int(max(0.0, inactive_seconds)),
            reclick_attempts,
            max_reclicks,
        )

        if result_rows >= expected_rows and coverage_rows >= expected_rows:
            if complete_since is None:
                complete_since = now
            if now - complete_since >= settle_seconds:
                return (
                    True,
                    "expected output rows reached",
                    active_result,
                    active_coverage,
                )
        else:
            complete_since = None

        if runtime_error:
            return (
                False,
                "Visual C++/runtime error dialog detected",
                active_result,
                active_coverage,
            )

        if not process_alive:
            return (
                False,
                "TAALES process exited before the batch completed",
                active_result,
                active_coverage,
            )

        if inactive_seconds >= reclick_seconds:
            if reclick_attempts >= max_reclicks:
                return (
                    False,
                    "TAALES stayed stuck after "
                    f"{max_reclicks} automatic Process Texts re-clicks",
                    active_result,
                    active_coverage,
                )

            reclick_attempts += 1
            logging.warning(
                "TAALES appears stuck at results=%d/%d coverage=%d/%d. "
                "Clicking Process Texts again (%d/%d).",
                result_rows,
                expected_rows,
                coverage_rows,
                expected_rows,
                reclick_attempts,
                max_reclicks,
            )

            try:
                win = wait_for_main_window(pygetwindow, timeout=30)
                activate_maximize(win)
                click_relative(pyautogui, win, config["clicks"]["run"])
            except Exception as exc:
                logging.exception(
                    "Automatic Process Texts re-click failed: %s",
                    exc,
                )

            now_after_click = time.time()
            last_output_change = now_after_click
            last_meaningful_cpu = now_after_click
            previous_cpu = process_cpu_total(psutil)

            # The incremental counters detect both normal append and output
            # truncation/rewrite after a re-click.
            continue

        if inactive_seconds >= stall_seconds:
            return (
                False,
                "no meaningful CPU or output activity for "
                f"{stall_seconds // 60} minutes",
                active_result,
                active_coverage,
            )

        if now - start >= timeout_seconds:
            return (
                False,
                f"batch exceeded {timeout_seconds // 3600} hours",
                active_result,
                active_coverage,
            )



def ensure_app(config: dict, pygetwindow, psutil):
    wins = window_candidates(pygetwindow)
    if wins and taales_processes(psutil):
        win = wins[0]
        activate_maximize(win)
        return win

    kill_taales(psutil)
    launch_taales(Path(config["taales_exe"]))
    win = wait_for_main_window(pygetwindow)
    activate_maximize(win)
    time.sleep(2)

    select_all_buttons = config["clicks"].get("select_all_buttons", [])
    gui = import_gui_modules()[0]
    for point in select_all_buttons:
        click_relative(gui, win, point)
        time.sleep(0.5)
    return win


def run_one_batch(
    files: Sequence[Path],
    job_number: int,
    config: dict,
    paths: Paths,
    con: sqlite3.Connection,
    pyautogui,
    pygetwindow,
    pyperclip,
    psutil,
) -> tuple[bool, str, Path, Path]:
    job_dir = paths.current / f"job_{job_number:06d}"
    job_dir.mkdir(parents=True, exist_ok=False)
    result_path = job_dir / f"job_{job_number:06d}.csv"

    # The manifest is committed before input preparation or GUI interaction.
    # A restarted automation can therefore identify the exact interrupted batch.
    create_job_manifest(job_dir, job_number, files, result_path)
    register_job(con, job_number, job_dir, "prepared")
    clear_output_candidates(result_path)

    update_job_manifest(job_dir, status="preparing_input")
    update_job_record(con, job_number, "preparing_input")
    prepare_batch(files, paths.batch_input)

    win = ensure_app(config, pygetwindow, psutil)
    activate_maximize(win)

    select_all_buttons = config["clicks"].get("select_all_buttons", [])
    for point in select_all_buttons:
        click_relative(pyautogui, win, point)
        time.sleep(0.5)

    choose_input_folder(
        pyautogui,
        pygetwindow,
        pyperclip,
        win,
        config["clicks"]["input"],
        paths.batch_input,
    )
    win = wait_for_main_window(pygetwindow, timeout=30)

    choose_output_file(
        pyautogui,
        pygetwindow,
        pyperclip,
        win,
        config["clicks"]["output"],
        result_path,
    )
    win = wait_for_main_window(pygetwindow, timeout=30)

    update_job_manifest(job_dir, status="running")
    update_job_record(con, job_number, "running")
    logging.info("Starting GUI analysis for %d files.", len(files))
    click_relative(pyautogui, win, config["clicks"]["run"])

    success, reason, actual_result, actual_coverage = wait_for_batch(
        len(files),
        result_path,
        config,
        pyautogui,
        pygetwindow,
        psutil,
    )
    update_job_manifest(
        job_dir,
        status="taales_finished" if success else "taales_failed",
        reason=reason,
        actual_result=str(actual_result),
        actual_coverage=str(actual_coverage),
    )
    update_job_record(
        con,
        job_number,
        "taales_finished" if success else "taales_failed",
        reason,
    )
    return success, reason, actual_result, actual_coverage



def merge_final(paths: Paths, con: sqlite3.Connection) -> tuple[int, int]:
    """
    Final files are maintained incrementally. Compact any pending part pairs,
    then report the durable master row count.
    """
    total, _ = compact_pending_outputs(
        paths,
        con,
        clean_job_folders=True,
    )
    if not paths.final_results.exists() or not paths.final_coverage.exists():
        return 0, 0

    logging.info(
        "Combined master files are current: %d result rows, %d coverage rows",
        total,
        total,
    )
    return total, total



def append_failed(path: Path, filename: str, reason: str) -> None:
    """Maintain a human-readable failure log without duplicate filenames."""
    existing: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                existing.add(line.split("\t", 1)[0])

    if filename in existing:
        return

    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{filename}\t{reason}\n")


def rewrite_failed_log(path: Path, con: sqlite3.Connection) -> None:
    """Rewrite failed_files.txt so it exactly matches the durable failed table."""
    rows = list(
        con.execute(
            "SELECT filename, reason FROM failed ORDER BY filename COLLATE NOCASE"
        )
    )
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for filename, reason in rows:
            clean_reason = str(reason).replace("\r", " ").replace("\n", " ")
            handle.write(f"{filename}\t{clean_reason}\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def reset_retry_attempts(
    con: sqlite3.Connection,
    names: Sequence[str],
) -> None:
    """
    Give a requested failed-file retry sweep a fresh singleton retry allowance.

    The rows remain in the failed table until their new output is safely
    checkpointed. This makes retry-failed mode crash-resumable: after a crash,
    unfinished files are still discoverable as failed on the next startup.
    """
    unique_names = sorted(set(names))
    if not unique_names:
        return

    with con:
        for start in range(0, len(unique_names), 800):
            chunk = unique_names[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            con.execute(
                f"DELETE FROM retry_attempts WHERE filename IN ({placeholders})",
                chunk,
            )


def clear_recovered_failures(
    con: sqlite3.Connection,
    names: Iterable[str],
) -> set[str]:
    """
    Remove failure markers only after rows are safely preserved.

    Returns the names actually cleared from the failed table.
    """
    unique_names = sorted(set(names))
    if not unique_names:
        return set()

    existing: set[str] = set()
    for start in range(0, len(unique_names), 800):
        chunk = unique_names[start : start + 800]
        placeholders = ",".join("?" for _ in chunk)
        existing.update(
            row[0]
            for row in con.execute(
                f"SELECT filename FROM failed WHERE filename IN ({placeholders})",
                chunk,
            )
        )

    if not existing:
        return set()

    with con:
        existing_list = sorted(existing)
        for start in range(0, len(existing_list), 800):
            chunk = existing_list[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            con.execute(
                f"DELETE FROM failed WHERE filename IN ({placeholders})",
                chunk,
            )
            con.execute(
                f"DELETE FROM retry_attempts WHERE filename IN ({placeholders})",
                chunk,
            )

    return existing


def run_automation(config: dict, retry_failed_only: bool = False) -> None:
    require_windows()
    pyautogui, pygetwindow, pyperclip, psutil = import_gui_modules()

    source_dir = Path(config["source_dir"])
    taales_exe = Path(config["taales_exe"])
    paths = Paths.from_root(Path(config["work_root"]))
    paths.create()
    configure_logging(paths)
    acquire_single_instance(paths.work_root)

    if not source_dir.exists():
        raise SystemExit(f"Source folder does not exist: {source_dir}")
    if not taales_exe.exists():
        raise SystemExit(f"TAALES executable does not exist: {taales_exe}")

    source_files = sorted(
        source_dir.glob("*.txt"),
        key=lambda p: p.name.lower(),
    )
    if not source_files:
        raise SystemExit(f"No .txt files found in: {source_dir}")

    con = connect_database(paths.database)
    seed_initial_outputs(config, paths, con)

    # If Python died but TAALES remained alive, stop it first so its CSV files
    # stop changing before startup recovery reads them.
    if taales_processes(psutil):
        logging.warning("Stopping an orphaned TAALES process before recovery.")
        kill_taales(psutil)
        time.sleep(3)

    # Critical ordering:
    # 1. salvage interrupted raw job output;
    # 2. append all validated chunks into the permanent combined masters;
    # 3. delete duplicate part/job files;
    # 4. only then calculate remaining work and open TAALES.
    recover_interrupted_jobs(paths, con)
    compact_pending_outputs(
        paths,
        con,
        clean_job_folders=True,
    )

    completed_names = get_completed(con)
    failed_names = get_failed(con)
    source_by_name = {path.name: path for path in source_files}
    retry_target_names: set[str] = set()

    # Keep the human-readable file synchronized with SQLite, including after an
    # interrupted retry sweep.
    rewrite_failed_log(paths.failures, con)

    if retry_failed_only:
        # Clean an impossible stale state where a file is both completed and
        # failed. Completed output always wins.
        stale_completed_failures = failed_names & completed_names
        if stale_completed_failures:
            cleared = clear_recovered_failures(con, stale_completed_failures)
            failed_names.difference_update(cleared)
            rewrite_failed_log(paths.failures, con)

        retry_target_names = {
            name
            for name in failed_names
            if name not in completed_names and name in source_by_name
        }
        unavailable_failed = {
            name
            for name in failed_names
            if name not in completed_names and name not in source_by_name
        }

        # Reset only the singleton-attempt counter. Keep every failed row until
        # its replacement result and coverage rows have been safely preserved.
        reset_retry_attempts(con, retry_target_names)

        missing = [
            source_by_name[name]
            for name in sorted(retry_target_names, key=str.lower)
        ]
        logging.info(
            "RETRY-FAILED MODE | corpus=%d | completed=%d | retrying=%d | "
            "failed source files unavailable=%d",
            len(source_files),
            len(completed_names),
            len(missing),
            len(unavailable_failed),
        )
    else:
        missing = [
            path
            for path in source_files
            if path.name not in completed_names
            and path.name not in failed_names
        ]
        logging.info(
            "Corpus=%d | completed=%d | permanently failed=%d | remaining=%d",
            len(source_files),
            len(completed_names),
            len(failed_names),
            len(missing),
        )

    initial_retry_names = set(retry_target_names)
    initial_retry_count = len(initial_retry_names)
    batch_size = int(config.get("batch_size", 1000))
    queue: deque[list[Path]] = deque(
        [
            missing[index : index + batch_size]
            for index in range(0, len(missing), batch_size)
        ]
    )

    job_number = next_job_number(paths, con)

    while queue:
        batch = [
            path
            for path in queue.popleft()
            if path.name not in completed_names
            and (
                path.name not in failed_names
                or path.name in retry_target_names
            )
        ]
        if not batch:
            continue

        logging.info(
            "Job %06d: %d files; %d queued jobs remain.",
            job_number,
            len(batch),
            len(queue),
        )

        job_dir = paths.current / f"job_{job_number:06d}"

        try:
            success, reason, result_path, coverage_path = run_one_batch(
                batch,
                job_number,
                config,
                paths,
                con,
                pyautogui,
                pygetwindow,
                pyperclip,
                psutil,
            )
        except Exception as exc:
            success = False
            reason = f"automation exception: {exc}"
            expected_result = (
                paths.current
                / f"job_{job_number:06d}"
                / f"job_{job_number:06d}.csv"
            )
            result_path, coverage_path = resolve_output_pair(expected_result)
            logging.exception("Job %06d failed during GUI automation.", job_number)

            if job_dir.exists():
                update_job_manifest(
                    job_dir,
                    status="automation_exception",
                    reason=reason,
                )
                register_job(
                    con,
                    job_number,
                    job_dir,
                    "automation_exception",
                    reason,
                )

        preserved = salvage_pair(
            result_path,
            coverage_path,
            {path.name for path in batch},
            paths,
            con,
            f"job {job_number:06d}",
        )
        completed_names.update(preserved)

        if retry_failed_only and preserved:
            recovered_failures = clear_recovered_failures(con, preserved)
            if recovered_failures:
                failed_names.difference_update(recovered_failures)
                retry_target_names.difference_update(recovered_failures)
                rewrite_failed_log(paths.failures, con)
                logging.info(
                    "Recovered %d previously failed files in job %06d.",
                    len(recovered_failures),
                    job_number,
                )

        # Move the newly checkpointed part into the permanent combined files
        # immediately, then remove all duplicate chunk/job output. This keeps
        # disk use nearly flat throughout a long run.
        compact_pending_outputs(
            paths,
            con,
            clean_job_folders=False,
        )

        remaining = [
            path
            for path in batch
            if path.name not in completed_names
            and (
                path.name not in failed_names
                or path.name in retry_target_names
            )
        ]

        if job_dir.exists():
            final_status = "completed" if not remaining else "partial"
            update_job_manifest(
                job_dir,
                status=final_status,
                success=success,
                reason=reason,
                preserved_rows=len(preserved),
                remaining_rows=len(remaining),
            )
            update_job_record(con, job_number, final_status, reason)

            # The durable master pair and SQLite now contain every valid row.
            # Sweep all compacted job folders. The current results.csv may still
            # be open inside TAALES, so a temporary Windows lock is deferred
            # instead of crashing the automation.
            deleted_jobs, _ = cleanup_compacted_storage(paths)
            if not job_dir.exists():
                logging.info(
                    "Deleted compacted raw job folder: %s",
                    job_dir.name,
                )
            else:
                logging.info(
                    "Compacted job folder remains temporarily locked and will "
                    "be retried later: %s",
                    job_dir.name,
                )
            if deleted_jobs > 1:
                logging.info(
                    "Also deleted %d older compacted job folders.",
                    deleted_jobs - (0 if job_dir.exists() else 1),
                )

        logging.info(
            "Job %06d ended: success=%s reason=%s preserved=%d remaining=%d",
            job_number,
            success,
            reason,
            len(preserved),
            len(remaining),
        )

        if not remaining:
            if not config.get("reuse_taales_between_batches", True):
                kill_taales(psutil)
            job_number += 1
            continue

        # A failed run may leave the GUI in a corrupt state.
        kill_taales(psutil)
        time.sleep(3)

        if len(remaining) == 1:
            item = remaining[0]
            attempts = increment_retry_attempt(con, item.name)
            max_attempts = int(config.get("single_file_retries", 2))
            if attempts <= max_attempts:
                logging.warning(
                    "Retrying single file %s (%d/%d)",
                    item.name,
                    attempts,
                    max_attempts,
                )
                queue.appendleft([item])
            else:
                logging.error(
                    "Skipping repeatedly failing file: %s | %s",
                    item.name,
                    reason,
                )
                mark_failed(con, item.name, reason, attempts)
                failed_names.add(item.name)
                append_failed(paths.failures, item.name, reason)
        else:
            midpoint = max(1, len(remaining) // 2)
            left = remaining[:midpoint]
            right = remaining[midpoint:]
            logging.warning(
                "Splitting failed remainder of %d files into %d and %d.",
                len(remaining),
                len(left),
                len(right),
            )
            if right:
                queue.appendleft(right)
            if left:
                queue.appendleft(left)

        job_number += 1

    kill_taales(psutil)
    result_count, coverage_count = merge_final(paths, con)
    total_source = len(source_files)
    final_failed_names = get_failed(con)
    failed_count = len(final_failed_names)
    rewrite_failed_log(paths.failures, con)

    logging.info(
        "DONE | source=%d results=%d coverage=%d failed=%d",
        total_source,
        result_count,
        coverage_count,
        failed_count,
    )
    print("\nAutomation finished.")
    if retry_failed_only:
        retry_still_failed = len(initial_retry_names & final_failed_names)
        recovered_count = max(0, initial_retry_count - retry_still_failed)
        print(f"Previously failed files retried: {initial_retry_count:,}")
        print(f"Recovered this retry sweep:     {recovered_count:,}")
        print(f"Still failed after retry:       {retry_still_failed:,}")
    print(f"Final results:  {paths.final_results}")
    print(f"Final coverage: {paths.final_coverage}")
    print(f"Failure log:    {paths.failures}")
    print(f"Run log:        {paths.logs / 'taales_automation.log'}")



def final_cleanup(config: dict) -> None:
    paths = Paths.from_root(Path(config["work_root"]))
    paths.create()
    configure_logging(paths)
    acquire_single_instance(paths.work_root)

    con: sqlite3.Connection | None = None
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "TAALES_2.2.exe"],
            capture_output=True,
            text=True,
            check=False,
        )
        time.sleep(2)

        con = connect_database(paths.database)

        # Salvage and compact anything left by an interruption before verifying.
        recover_interrupted_jobs(paths, con)
        compact_pending_outputs(paths, con, clean_job_folders=True)
        result_count, coverage_count = merge_final(paths, con)

        source_dir = Path(config["source_dir"])
        source_names = {
            path.name
            for path in source_dir.glob("*.txt")
        }
        completed_names = get_completed(con)
        failed_names = get_failed(con)

        overlap = completed_names & failed_names
        if overlap:
            raise RuntimeError(
                "Final cleanup refused: some filenames are marked both "
                f"completed and failed ({len(overlap)} files)."
            )

        unaccounted = source_names - completed_names - failed_names
        if unaccounted:
            sample = ", ".join(sorted(unaccounted)[:5])
            raise RuntimeError(
                "Final cleanup refused because source files remain unaccounted "
                f"for: {len(unaccounted)} files. Examples: {sample}"
            )

        missing_source_records = (completed_names | failed_names) - source_names
        if missing_source_records:
            logging.warning(
                "State contains %d filenames not currently present in the "
                "source folder. They are retained in SQLite.",
                len(missing_source_records),
            )

        if result_count != coverage_count:
            raise RuntimeError(
                "Final cleanup refused because result and coverage row counts "
                f"differ: {result_count} vs {coverage_count}."
            )

        if result_count != len(completed_names):
            raise RuntimeError(
                "Final cleanup refused because master row count does not match "
                f"completed SQLite state: masters={result_count}, "
                f"completed={len(completed_names)}."
            )

        rewrite_failed_log(paths.failures, con)

        with con:
            con.execute("DELETE FROM jobs")
            con.execute("DELETE FROM retry_attempts")

        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            logging.exception("SQLite WAL checkpoint failed during cleanup.")

        # VACUUM removes deleted job-history pages and leaves one compact state DB.
        try:
            con.execute("VACUUM")
        except sqlite3.DatabaseError:
            logging.exception("SQLite VACUUM failed; state remains valid.")

        con.close()
        con = None

        deleted_items: list[str] = []

        for directory in (paths.current, paths.parts):
            if directory.exists():
                if remove_tree_best_effort(
                    directory,
                    attempts=10,
                    initial_delay_seconds=0.5,
                ):
                    deleted_items.append(str(directory))
                else:
                    raise RuntimeError(
                        "Final cleanup could not remove a locked directory: "
                        f"{directory}"
                    )

        for path in (
            paths.compaction_journal,
            paths.database.with_name(paths.database.name + "-wal"),
            paths.database.with_name(paths.database.name + "-shm"),
        ):
            if path.exists():
                try:
                    path.unlink()
                    deleted_items.append(str(path))
                except OSError as exc:
                    raise RuntimeError(
                        f"Final cleanup could not delete {path}: {exc}"
                    ) from exc

        # Delete stale temporary files anywhere under the work root, without
        # touching the permanent master files, database, failure log, or log.
        protected = {
            paths.final_results.resolve(),
            paths.final_coverage.resolve(),
            paths.database.resolve(),
            paths.failures.resolve(),
            (paths.logs / "taales_automation.log").resolve(),
        }
        for temp_path in sorted(paths.work_root.rglob("*.tmp")):
            try:
                if temp_path.resolve() in protected:
                    continue
                temp_path.unlink(missing_ok=True)
                deleted_items.append(str(temp_path))
            except OSError as exc:
                raise RuntimeError(
                    f"Final cleanup could not delete temporary file "
                    f"{temp_path}: {exc}"
                ) from exc

        # Remove empty directories left below work_root, but retain logs.
        for directory in sorted(
            (p for p in paths.work_root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            if directory == paths.logs:
                continue
            try:
                directory.rmdir()
                deleted_items.append(str(directory))
            except OSError:
                pass

        logging.info(
            "FINAL CLEANUP COMPLETE | source=%d completed=%d failed=%d "
            "result_rows=%d coverage_rows=%d deleted_items=%d",
            len(source_names),
            len(completed_names),
            len(failed_names),
            result_count,
            coverage_count,
            len(deleted_items),
        )

        print()
        print("Final cleanup complete.")
        print(f"Source files:        {len(source_names):,}")
        print(f"Completed rows:      {result_count:,}")
        print(f"Still failed:        {len(failed_names):,}")
        print(f"Temporary items removed: {len(deleted_items):,}")
        print()
        print("Retained final files:")
        print(f"  {paths.final_results}")
        print(f"  {paths.final_coverage}")
        print(f"  {paths.database}")
        print(f"  {paths.failures}")
        print(f"  {paths.logs / 'taales_automation.log'}")
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        release_single_instance(paths.work_root)


def status(config: dict) -> None:
    paths = Paths.from_root(Path(config["work_root"]))
    paths.create()
    con = connect_database(paths.database)
    completed = get_completed(con)
    source_dir = Path(config["source_dir"])
    total = len(list(source_dir.glob("*.txt"))) if source_dir.exists() else 0
    failed = len(get_failed(con))
    print(f"Source texts: {total:,}")
    print(f"Completed:    {len(completed):,}")
    print(f"Remaining:    {max(0, total - len(completed) - failed):,}")
    print(f"Failed:       {failed:,}")
    print(f"Work folder:  {paths.work_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("setup", "run", "retry-failed", "cleanup", "status", "merge"),
        help="setup, run, retry-failed, cleanup, status, or merge",
    )
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    if args.setup:
        mode = "setup"
    elif args.run:
        mode = "run"
    elif args.retry_failed:
        mode = "retry-failed"

    if not mode:
        mode = "setup" if not CONFIG_PATH.exists() else "run"

    if mode == "setup":
        setup()
        return

    config = load_config()
    paths = Paths.from_root(Path(config["work_root"]))
    paths.create()
    configure_logging(paths)

    if mode == "run":
        run_automation(config)
    elif mode == "retry-failed":
        run_automation(config, retry_failed_only=True)
    elif mode == "cleanup":
        final_cleanup(config)
    elif mode == "status":
        status(config)
    elif mode == "merge":
        con = connect_database(paths.database)
        merge_final(paths, con)


if __name__ == "__main__":
    main()
