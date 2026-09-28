from __future__ import annotations
from pathlib import Path
from queue import Empty, Queue
import base64
import csv
import ctypes
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk



MAX_WORKERS = 0
MAX_AUTO_WORKERS = 6
JAVA_HEAP_MB = 1000
RAM_RESERVE_MB = 2500
AUTO_TUNE = True
FORCE_RETUNE = False
AUTOTUNE_PROFILE_VERSION = 2
AUTO_TUNE_CANDIDATES = [1, 2, 3, 4, 5, 6]
AUTO_TUNE_SAMPLE_SIZE = 96
AUTO_TUNE_WARMUP_PER_WORKER = 3
AUTO_TUNE_CLOSE_RATE_PERCENT = 2.0
AUTO_TUNE_MIN_AVAILABLE_RAM_MB = 1800
RUNTIME_MIN_AVAILABLE_RAM_MB = 1500
RUNTIME_LOW_RAM_CHECKS = 3
RUNTIME_RESOURCE_CHECK_SECONDS = 10
HARD_MAX_WORDS = 45
PARSER_MAX_LENGTH = 80
PROGRESS_EVERY = 100
TASKS_BUFFERED_PER_WORKER = 3


script_dir = Path(__file__).resolve().parent
thesis_root = script_dir.parent.parent.parent

raw_texts_dir = thesis_root / "0_Data" / "raw_texts"
l2sca_dir = script_dir / "L2SCA-2023-08-15"

output_file = script_dir / "l2sca_results.csv"
failed_file = script_dir / "l2sca_failed.csv"
autotune_file = script_dir / "l2sca_autotune.json"

HEADER = [
    "filename", "W", "S", "VP", "C", "T", "DC", "CT", "CP", "CN",
    "MLS", "MLT", "MLC", "C/S", "VP/T", "C/T", "DC/C", "DC/T",
    "T/S", "CT/T", "CP/T", "CP/C", "CN/T", "CN/C",
]


JAVA_SOURCE = r'''
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.tregex.TregexMatcher;
import edu.stanford.nlp.trees.tregex.TregexPattern;

public class FastL2SCAWorker {
    private static final String MODEL =
        "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";

    private static final String[] PATTERN_TEXT = new String[] {
        "ROOT !> __",
        "VP > S|SINV|SQ",
        "S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]",
        "S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]",
        "SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])",
        "S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))",
        "ADJP|ADVP|NP|VP < CC",
        "NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]",
        "SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]",
        "S < (VP <# VBG|TO) $+ VP",
        "FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])",
        "FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])",
        "MD|VBZ|VBP|VBD > (SQ !< VP)"
    };

    private final LexicalizedParser parser;
    private final TregexPattern[] patterns;
    private final TokenizerFactory<CoreLabel> tokenizerFactory;
    private final int maxLength;

    FastL2SCAWorker(int maxLength) {
        this.maxLength = maxLength;
        parser = LexicalizedParser.loadModel(
            MODEL, "-maxLength", Integer.toString(maxLength)
        );

        patterns = new TregexPattern[PATTERN_TEXT.length];
        for (int i = 0; i < PATTERN_TEXT.length; i++) {
            patterns[i] = TregexPattern.compile(PATTERN_TEXT[i]);
        }

        tokenizerFactory = PTBTokenizer.factory(
            new CoreLabelTokenFactory(), ""
        );
    }

    private List<CoreLabel> tokenize(String sentence) {
        return tokenizerFactory.getTokenizer(
            new StringReader(sentence)
        ).tokenize();
    }

    private String tokenText(CoreLabel token) {
        String value = token.word();
        if (value == null) {
            value = token.value();
        }
        return value == null ? "" : value;
    }

    private boolean isStrongBoundary(CoreLabel token) {
        String value = tokenText(token);
        return value.equals(".") || value.equals("!")
            || value.equals("?") || value.equals(";")
            || value.equals(":");
    }

    private boolean isCommaBoundary(CoreLabel token) {
        return tokenText(token).equals(",");
    }

    // Python's final safety split counts whitespace-delimited words. Stanford's
    // tokenizer can turn punctuation-heavy items such as weather(60%) into
    // several tokens, so a 44-word chunk can still exceed -maxLength 80. Split
    // only those over-limit chunks, preferring real punctuation boundaries.
    // Texts that already fit the parser are left completely unchanged.
    private List<List<CoreLabel>> parserSafeChunks(List<CoreLabel> words) {
        if (words.size() <= maxLength) {
            return Collections.singletonList(words);
        }

        List<List<CoreLabel>> chunks = new ArrayList<>();
        int start = 0;
        while (start < words.size()) {
            int hardEnd = Math.min(start + maxLength, words.size());
            int end = hardEnd;

            if (hardEnd < words.size()) {
                int minimumBoundary = start + Math.max(1, maxLength / 2);

                for (int index = hardEnd - 1;
                        index >= minimumBoundary; index--) {
                    if (isStrongBoundary(words.get(index))) {
                        end = index + 1;
                        break;
                    }
                }

                if (end == hardEnd) {
                    for (int index = hardEnd - 1;
                            index >= minimumBoundary; index--) {
                        if (isCommaBoundary(words.get(index))) {
                            end = index + 1;
                            break;
                        }
                    }
                }
            }

            chunks.add(new ArrayList<>(words.subList(start, end)));
            start = end;
        }
        return chunks;
    }

    // This mirrors the W regex used by the earlier Python script:
    // \([A-Z]+\$? [^\)\(-]+\)
    private boolean isWordTag(String tag) {
        if (tag == null || tag.isEmpty()) {
            return false;
        }

        int letters = tag.endsWith("$") ? tag.length() - 1 : tag.length();
        if (letters == 0) {
            return false;
        }
        for (int i = 0; i < letters; i++) {
            char value = tag.charAt(i);
            if (value < 'A' || value > 'Z') {
                return false;
            }
        }
        return true;
    }

    private int countWords(Tree tree) {
        int count = 0;
        for (Tree node : tree) {
            if (!node.isPreTerminal() || node.numChildren() != 1) {
                continue;
            }

            String tag = node.label().value();
            String word = node.getChild(0).label().value();

            if (isWordTag(tag) && word != null
                    && !word.isEmpty()
                    && word.indexOf('(') < 0
                    && word.indexOf(')') < 0
                    && word.indexOf('-') < 0) {
                count++;
            }
        }
        return count;
    }

    private static double divide(int x, int y) {
        if (x == 0 || y == 0) {
            return 0.0;
        }
        return ((double) x) / ((double) y);
    }

    private static String four(double value) {
        return String.format(Locale.ROOT, "%.4f", value);
    }

    private String analyze(String stem, String segmentedText) throws Exception {
        int w = 0;
        int[] count = new int[patterns.length];
        int treeCount = 0;

        try (BufferedReader reader = new BufferedReader(
                new StringReader(segmentedText))) {
            String sentence;
            while ((sentence = reader.readLine()) != null) {
                sentence = sentence.trim();
                if (sentence.isEmpty()) {
                    continue;
                }

                List<CoreLabel> words = tokenize(sentence);
                if (words.isEmpty()) {
                    continue;
                }

                for (List<CoreLabel> parserWords : parserSafeChunks(words)) {
                    Tree tree = parser.parse(parserWords);
                    if (tree == null || tree.label() == null
                            || !"ROOT".equals(tree.label().value())) {
                        throw new RuntimeException(
                            "Parser produced no usable ROOT tree for a "
                            + parserWords.size() + "-token chunk."
                        );
                    }

                    treeCount++;
                    w += countWords(tree);

                    for (int i = 0; i < patterns.length; i++) {
                        TregexMatcher matcher = patterns[i].matcher(tree);
                        // The original command used -o, meaning each tree node
                        // is counted only once even if a pattern matches it in
                        // multiple ways.
                        while (matcher.findNextMatchingNode()) {
                            count[i]++;
                        }
                    }
                }
            }
        }

        if (treeCount == 0) {
            throw new RuntimeException("No usable text after segmentation.");
        }

        int s = count[0];
        int vp = count[1] + count[12];
        int c = count[2] + count[10];
        int t = count[3] + count[11];
        int dc = count[4];
        int ct = count[5];
        int cp = count[6];
        int cn = count[7] + count[8] + count[9];

        String[] row = new String[] {
            stem,
            Integer.toString(w), Integer.toString(s), Integer.toString(vp),
            Integer.toString(c), Integer.toString(t), Integer.toString(dc),
            Integer.toString(ct), Integer.toString(cp), Integer.toString(cn),
            four(divide(w, s)), four(divide(w, t)), four(divide(w, c)),
            four(divide(c, s)), four(divide(vp, t)), four(divide(c, t)),
            four(divide(dc, c)), four(divide(dc, t)), four(divide(t, s)),
            four(divide(ct, t)), four(divide(cp, t)), four(divide(cp, c)),
            four(divide(cn, t)), four(divide(cn, c))
        };

        return String.join("\t", row);
    }

    private static String errorText(Throwable error) {
        StringWriter buffer = new StringWriter();
        error.printStackTrace(new PrintWriter(buffer));
        return Base64.getEncoder().encodeToString(
            buffer.toString().getBytes(StandardCharsets.UTF_8)
        );
    }

    public static void main(String[] args) throws Exception {
        int maxLength = Integer.parseInt(args[0]);
        FastL2SCAWorker worker = new FastL2SCAWorker(maxLength);

        BufferedReader input = new BufferedReader(
            new InputStreamReader(System.in, StandardCharsets.UTF_8)
        );
        PrintWriter output = new PrintWriter(
            new OutputStreamWriter(System.out, StandardCharsets.UTF_8), true
        );

        output.println("READY");

        String request;
        while ((request = input.readLine()) != null) {
            if (request.equals("STOP")) {
                break;
            }

            int separator = request.indexOf('\t');
            if (separator < 1) {
                output.println("ERROR\tunknown\t" + Base64.getEncoder()
                    .encodeToString("Invalid worker request."
                    .getBytes(StandardCharsets.UTF_8)));
                continue;
            }

            String stem = request.substring(0, separator);
            String segmentedText = new String(
                Base64.getDecoder().decode(request.substring(separator + 1)),
                StandardCharsets.UTF_8
            );

            try {
                output.println(
                    "OK\t" + analyzeSafely(worker, stem, segmentedText)
                );
            } catch (Throwable error) {
                output.println("ERROR\t" + stem + "\t" + errorText(error));
            }
        }
    }

    private static String analyzeSafely(
            FastL2SCAWorker worker, String stem, String segmentedText)
            throws Exception {
        return worker.analyze(stem, segmentedText);
    }
}
'''


# ---------------------------------------------------------------------------
# Sentence segmentation (same logic as the supplied preprocessing script)
# ---------------------------------------------------------------------------

def ensure_nltk_tokenizer() -> None:
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=False)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\u00a0]+", " ", text)
    text = re.sub(
        r"\s+(Property\s+\d+\s*[:*])", r"\n\1", text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(Option\s+\d+\s*[:*])", r"\n\1", text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s+(Choice\s+\d+\s*[:*])", r"\n\1", text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(\d+\s*[.)]\s+)", r"\n\1", text)
    text = re.sub(r"\s*[*•▪◦]\s*", "\n", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def safe_sent_tokenize(block: str) -> list[str]:
    try:
        return nltk.sent_tokenize(block)
    except Exception:
        return [block]


def split_on_soft_boundaries(sentence: str) -> list[str]:
    sentence = sentence.strip()
    if not sentence:
        return []

    sentence = re.sub(r"([;:!?])\s+", r"\1\n", sentence)
    sentence = re.sub(
        r"([,])\s+(first of all|secondly|thirdly|the second|the third|finally|however|therefore|because|but|so|and then)\b",
        r"\1\n\2",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(
        r"\s+(Property\s+\d+\b)", r"\n\1", sentence,
        flags=re.IGNORECASE,
    )
    return [part.strip() for part in sentence.split("\n") if part.strip()]


def split_long_chunk_by_commas(
    chunk: str, target_words: int = 30
) -> list[str]:
    if len(chunk.split()) <= HARD_MAX_WORDS:
        return [chunk.strip()]

    output = []
    current = ""
    for piece in re.split(r"(,)", chunk):
        current = (current + piece).strip()
        if len(current.split()) >= target_words:
            output.append(current)
            current = ""

    if current:
        output.append(current)
    return output or [chunk.strip()]


def hard_split_by_words(chunk: str) -> list[str]:
    words = chunk.split()
    if len(words) <= HARD_MAX_WORDS:
        return [chunk.strip()] if chunk.strip() else []
    return [
        " ".join(words[index:index + HARD_MAX_WORDS])
        for index in range(0, len(words), HARD_MAX_WORDS)
    ]


def custom_sentence_splitter(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    candidates = []
    for block in text.split("\n"):
        for sentence in safe_sent_tokenize(block.strip()):
            candidates.extend(split_on_soft_boundaries(sentence))

    comma_split = []
    for chunk in candidates:
        comma_split.extend(split_long_chunk_by_commas(chunk))

    output = []
    for chunk in comma_split:
        output.extend(hard_split_by_words(chunk))

    return [re.sub(r"\s+", " ", item).strip() for item in output if item.strip()]


# ---------------------------------------------------------------------------
# Resource selection and Java setup
# ---------------------------------------------------------------------------

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
        return float(psutil.cpu_percent(interval=1.0))
    except ImportError:
        pass

    if os.name == "nt":
        # GetSystemTimes is available without psutil. Two readings are needed
        # because the counters are cumulative since Windows started.
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
        total_delta = (
            second[1] - first[1] + second[2] - first[2]
        )
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))

    return None


def choose_worker_count() -> tuple[int, int | None, float | None]:
    available_mb = available_ram_mb()
    cpu_percent = current_cpu_percent()

    if MAX_WORKERS > 0:
        return MAX_WORKERS, available_mb, cpu_percent

    logical_cpus = os.cpu_count() or 1
    workers_by_cpu = max(1, logical_cpus // 4)

    if cpu_percent is not None:
        idle_fraction = max(0.10, 1.0 - cpu_percent / 100.0)
        available_threads = max(1, int(logical_cpus * idle_fraction))
        workers_by_cpu = max(1, available_threads // 3)

    if available_mb is None:
        workers_by_ram = 2
    else:
        usable_mb = max(JAVA_HEAP_MB, available_mb - RAM_RESERVE_MB)
        workers_by_ram = max(1, usable_mb // JAVA_HEAP_MB)

    workers = min(MAX_AUTO_WORKERS, workers_by_cpu, workers_by_ram)
    return max(1, workers), available_mb, cpu_percent


def require_program(name: str) -> str:
    program = shutil.which(name)
    if program is None:
        if name == "javac":
            raise RuntimeError(
                "javac was not found. Install a 64-bit JDK (not only a JRE), "
                "then reopen PowerShell so javac is on PATH."
            )
        raise RuntimeError(f"{name} was not found on PATH.")
    return program


def java_classpath(helper_dir: Path) -> str:
    parser_dir = l2sca_dir / "stanford-parser-full-2020-11-17"
    parser_jars = sorted(
        jar for jar in parser_dir.glob("*.jar")
        if not jar.name.endswith(("-sources.jar", "-javadoc.jar"))
    )
    entries = [
        str(helper_dir),
        str(l2sca_dir / "stanford-tregex-4.2.0.jar"),
        *(str(jar) for jar in parser_jars),
    ]
    return os.pathsep.join(entries)


def find_java_tools() -> tuple[str, str | None]:
    """Return a matched java/javac pair, preferring a real JDK."""
    javac_name = "javac.exe" if os.name == "nt" else "javac"
    java_name = "java.exe" if os.name == "nt" else "java"
    candidates: list[Path] = []

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / javac_name)

    path_javac = shutil.which("javac")
    if path_javac:
        candidates.append(Path(path_javac))

    if os.name == "nt":
        program_files = Path(
            os.environ.get("ProgramFiles", r"C:\Program Files")
        )
        roots = [
            program_files / "Eclipse Adoptium",
            program_files / "Java",
            program_files / "Microsoft",
            program_files / "Amazon Corretto",
            program_files / "BellSoft",
            program_files / "Zulu",
        ]
        discovered = []
        for root in roots:
            if root.exists():
                discovered.extend(root.glob(f"*/bin/{javac_name}"))
                discovered.extend(root.glob(f"*/*/bin/{javac_name}"))
        discovered.sort(
            key=lambda path: path.stat().st_mtime, reverse=True
        )
        candidates.extend(discovered)

    seen = set()
    for javac_path in candidates:
        try:
            resolved = javac_path.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        sibling_java = resolved.with_name(java_name)
        if sibling_java.is_file():
            return str(sibling_java), str(resolved)

    # A recent runtime can still expose the compiler module without a javac
    # launcher. Java 8 JREs cannot, and produce the tailored error below.
    return require_program("java"), None


def compile_java_worker(helper_dir: Path) -> tuple[str, str]:
    java, javac = find_java_tools()

    tregex_jar = l2sca_dir / "stanford-tregex-4.2.0.jar"
    parser_dir = l2sca_dir / "stanford-parser-full-2020-11-17"
    if not tregex_jar.exists():
        raise FileNotFoundError(f"Tregex JAR not found: {tregex_jar}")
    if not parser_dir.exists():
        raise FileNotFoundError(f"Stanford parser folder not found: {parser_dir}")

    source_file = helper_dir / "FastL2SCAWorker.java"
    source_file.write_text(JAVA_SOURCE, encoding="utf-8")
    classpath = java_classpath(helper_dir)

    if javac is not None:
        compiler_command = [javac]
    else:
        # Some modern Java runtimes contain the compiler module even when a
        # separate javac launcher is absent. Java 8 JRE installations do not.
        compiler_command = [
            java, "-m", "jdk.compiler/com.sun.tools.javac.Main"
        ]

    result = subprocess.run(
        [*compiler_command, "-encoding", "UTF-8", "-cp", classpath,
         "-d", str(helper_dir), str(source_file)],
        cwd=str(l2sca_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if javac is None:
            message += (
                "\n\nNo javac launcher was found. If this Java runtime does not "
                "contain the compiler module, install a 64-bit JDK and reopen "
                "PowerShell."
            )
        raise RuntimeError(f"Java helper compilation failed:\n{message}")

    return java, classpath


# ---------------------------------------------------------------------------
# Restartable output and persistent Java processes
# ---------------------------------------------------------------------------

def already_completed() -> set[str]:
    completed = set()
    if not output_file.exists():
        return completed

    with output_file.open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as file:
        reader = csv.reader(file)
        first = next(reader, None)
        if first is not None and first != HEADER:
            raise ValueError(
                f"Existing results have an unexpected header: {output_file}"
            )
        for row in reader:
            if len(row) == len(HEADER) and row[0].strip():
                completed.add(row[0].strip())
    return completed


def open_worker(
    worker_number: int,
    java: str,
    classpath: str,
    temp_dir: Path,
) -> tuple[subprocess.Popen, object]:
    stderr_handle = (temp_dir / f"java_worker_{worker_number}.log").open(
        "a", encoding="utf-8"
    )
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    process = subprocess.Popen(
        [
            java,
            "-Xms256m",
            f"-Xmx{JAVA_HEAP_MB}m",
            "-Djava.awt.headless=true",
            "-cp", classpath,
            "FastL2SCAWorker",
            str(PARSER_MAX_LENGTH),
        ],
        cwd=str(l2sca_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8",
        bufsize=1,
        creationflags=creation_flags,
    )

    ready = process.stdout.readline().strip()
    if ready != "READY":
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        stderr_handle.close()
        log_path = temp_dir / f"java_worker_{worker_number}.log"
        details = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"Java worker {worker_number} could not start.\n{details[-4000:]}"
        )

    return process, stderr_handle


def close_worker(process: subprocess.Popen, stderr_handle: object) -> None:
    try:
        if process.poll() is None:
            process.stdin.write("STOP\n")
            process.stdin.flush()
            process.wait(timeout=10)
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    finally:
        stderr_handle.close()


def decode_worker_error(encoded: str) -> str:
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        return encoded


def encode_segmented_text(sentences: list[str]) -> str:
    text = "\n".join(sentences) + "\n"
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def request_worker(
    process: subprocess.Popen, stem: str, encoded_text: str
) -> tuple[str, list[str] | str, str]:
    process.stdin.write(f"{stem}\t{encoded_text}\n")
    process.stdin.flush()
    response = process.stdout.readline().rstrip("\r\n")

    if not response:
        raise RuntimeError(f"Java worker exited with code {process.poll()}.")

    fields = response.split("\t")
    if fields[0] == "OK" and len(fields) == len(HEADER) + 1:
        return "success", fields[1:], ""
    if fields[0] == "ERROR" and len(fields) >= 3:
        return "failure", stem, decode_worker_error(fields[2])
    raise RuntimeError(f"Unexpected Java response: {response[:1000]}")


def start_worker_group(
    worker_count: int, java: str, classpath: str, temp_dir: Path
) -> list[tuple[int, subprocess.Popen, object]]:
    processes = []
    errors = []
    lock = threading.Lock()

    def start_one(number: int) -> None:
        try:
            worker = (number, *open_worker(number, java, classpath, temp_dir))
            with lock:
                processes.append(worker)
        except Exception as error:
            with lock:
                errors.append(str(error))

    threads = [
        threading.Thread(target=start_one, args=(number,), daemon=True)
        for number in range(1, worker_count + 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        for _, process, handle in processes:
            close_worker(process, handle)
        raise RuntimeError("\n\n".join(errors))

    return sorted(processes, key=lambda item: item[0])


def close_worker_group(
    processes: list[tuple[int, subprocess.Popen, object]]
) -> None:
    for _, process, handle in processes:
        close_worker(process, handle)


def tuning_signature() -> dict:
    return {
        "profile_version": AUTOTUNE_PROFILE_VERSION,
        "logical_cpus": os.cpu_count() or 1,
        "java_heap_mb": JAVA_HEAP_MB,
        "hard_max_words": HARD_MAX_WORDS,
        "parser_max_length": PARSER_MAX_LENGTH,
        "candidates": AUTO_TUNE_CANDIDATES,
        "raw_texts_dir": str(raw_texts_dir),
    }


def maximum_feasible_workers(available_mb: int | None) -> int:
    cpu_limit = max(1, min(MAX_AUTO_WORKERS, (os.cpu_count() or 1) // 2))
    if available_mb is None:
        return min(cpu_limit, 4)

    ram_limit = max(
        1,
        (available_mb - AUTO_TUNE_MIN_AVAILABLE_RAM_MB) // JAVA_HEAP_MB,
    )
    return max(1, min(cpu_limit, ram_limit, MAX_AUTO_WORKERS))


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

        benchmarks = profile.get("benchmarks", [])
        tested_counts = [
            int(item["workers"]) for item in benchmarks
            if "workers" in item
        ]
        tested_max = max(tested_counts, default=0)

        # A profile made while RAM was scarce must not permanently lock later
        # runs to too few workers. More candidates are benchmarked automatically
        # as soon as they become feasible.
        if tested_max < feasible_now:
            print(
                "More worker counts are feasible now; auto-tuning again."
            )
            return None
        if workers == 1 and feasible_now > 1:
            print(
                "The cached one-worker setting is being rechecked because "
                "multiple workers are now feasible."
            )
            return None

        old_cpu = profile.get("cpu_percent_at_tuning")
        if (
            old_cpu is not None and cpu_percent is not None
            and float(old_cpu) >= 50.0
            and cpu_percent <= float(old_cpu) - 20.0
        ):
            print(
                "The computer is substantially less busy now; "
                "auto-tuning again."
            )
            return None

        print(
            f"Reusing auto-tuned setting: {workers} workers "
            f"({profile.get('measured_rate', 0):.2f} files/sec in benchmark)"
        )
        print(
            f"Delete {autotune_file.name} or set FORCE_RETUNE=True "
            "to benchmark again."
        )
        return workers
    except Exception:
        return None


def save_tuning_profile(
    workers: int,
    rate: float,
    results: list[dict],
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
        "benchmarks": results,
    }
    temporary = autotune_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    temporary.replace(autotune_file)


def prepare_tuning_sample(raw_files: list[Path]) -> list[tuple[str, str]]:
    sample_size = min(AUTO_TUNE_SAMPLE_SIZE, len(raw_files))
    selected = random.Random(20260717).sample(raw_files, sample_size)
    tasks = []

    print(f"Preparing {sample_size} representative texts for auto-tuning...")
    for raw_file in selected:
        try:
            text = raw_file.read_text(encoding="utf-8", errors="replace")
            sentences = custom_sentence_splitter(text)
            if sentences:
                tasks.append((raw_file.stem, encode_segmented_text(sentences)))
        except Exception:
            continue

    return tasks


def benchmark_worker_count(
    worker_count: int,
    tasks: list[tuple[str, str]],
    java: str,
    classpath: str,
    temp_dir: Path,
) -> dict:
    print(f"  Benchmarking {worker_count} workers...")
    processes = start_worker_group(worker_count, java, classpath, temp_dir)

    def warm_worker(
        worker_index: int, process: subprocess.Popen
    ) -> None:
        for offset in range(AUTO_TUNE_WARMUP_PER_WORKER):
            stem, encoded = tasks[
                (worker_index * AUTO_TUNE_WARMUP_PER_WORKER + offset)
                % len(tasks)
            ]
            request_worker(process, stem, encoded)

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            warmups = [
                executor.submit(warm_worker, index, process)
                for index, (_, process, _) in enumerate(processes)
            ]
            for future in as_completed(warmups):
                future.result()

        partitions = [tasks[index::worker_count] for index in range(worker_count)]
        monitor_stop = threading.Event()
        memory_samples = []

        def monitor_memory() -> None:
            while not monitor_stop.wait(0.25):
                value = available_ram_mb()
                if value is not None:
                    memory_samples.append(value)

        monitor = threading.Thread(target=monitor_memory, daemon=True)
        monitor.start()

        def run_partition(
            process: subprocess.Popen, partition: list[tuple[str, str]]
        ) -> tuple[int, int, list[str]]:
            completed = 0
            successful = 0
            errors = []
            for stem, encoded in partition:
                try:
                    status, _, message = request_worker(process, stem, encoded)
                    completed += 1
                    if status == "success":
                        successful += 1
                    else:
                        errors.append(message[:500])
                except Exception as error:
                    errors.append(str(error)[:500])
                    break
            return completed, successful, errors

        started = time.perf_counter()
        partition_results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_partition, process, partition)
                for (_, process, _), partition in zip(processes, partitions)
            ]
            for future in as_completed(futures):
                partition_results.append(future.result())

        elapsed = time.perf_counter() - started
        monitor_stop.set()
        monitor.join()

        completed = sum(item[0] for item in partition_results)
        successful = sum(item[1] for item in partition_results)
        errors = [error for item in partition_results for error in item[2]]
        minimum_ram = min(memory_samples) if memory_samples else None
        rate = completed / elapsed if elapsed > 0 else 0.0

        stable = completed == len(tasks)
        if minimum_ram is not None:
            stable = stable and minimum_ram >= AUTO_TUNE_MIN_AVAILABLE_RAM_MB
        if any("OutOfMemory" in error for error in errors):
            stable = False

        result = {
            "workers": worker_count,
            "rate": round(rate, 4),
            "elapsed_seconds": round(elapsed, 3),
            "completed": completed,
            "successful": successful,
            "errors": len(errors),
            "minimum_available_ram_mb": minimum_ram,
            "stable": stable,
        }

        ram_text = (
            f", minimum free RAM={minimum_ram:,} MB"
            if minimum_ram is not None else ""
        )
        print(
            f"    {rate:.2f} files/sec, stable={stable}{ram_text}"
        )
        return result
    finally:
        close_worker_group(processes)
        time.sleep(1.0)


def autotune_worker_count(
    raw_files: list[Path], java: str, classpath: str, temp_dir: Path,
    available_mb: int | None, cpu_percent: float | None,
) -> int:
    if MAX_WORKERS > 0:
        print(f"Using manually configured worker count: {MAX_WORKERS}")
        return MAX_WORKERS

    cached = load_cached_tuning(available_mb, cpu_percent)
    if cached is not None:
        return cached

    fallback, _, _ = choose_worker_count()
    if not AUTO_TUNE or len(raw_files) < 20:
        return fallback

    feasible_max = maximum_feasible_workers(available_mb)
    candidates = [
        workers for workers in AUTO_TUNE_CANDIDATES
        if workers <= feasible_max
    ]
    if not candidates:
        candidates = [1]

    tasks = prepare_tuning_sample(raw_files)
    if len(tasks) < 10:
        print("Too few usable tuning texts; using resource-based selection.")
        return fallback

    print(f"Auto-tuning candidates: {candidates}")
    benchmark_results = []
    for workers in candidates:
        try:
            result = benchmark_worker_count(
                workers, tasks, java, classpath, temp_dir
            )
        except Exception as error:
            result = {
                "workers": workers,
                "rate": 0.0,
                "elapsed_seconds": 0.0,
                "completed": 0,
                "successful": 0,
                "errors": 1,
                "minimum_available_ram_mb": available_ram_mb(),
                "stable": False,
                "startup_or_benchmark_error": str(error)[:1000],
            }
            print(f"    Unstable: {error}")
        benchmark_results.append(result)

        # Higher counts will only increase memory pressure.
        if not result["stable"]:
            low_ram = (
                result["minimum_available_ram_mb"] is not None
                and result["minimum_available_ram_mb"]
                < AUTO_TUNE_MIN_AVAILABLE_RAM_MB
            )
            if low_ram or result.get("startup_or_benchmark_error"):
                break

    stable = [item for item in benchmark_results if item["stable"]]
    if not stable:
        raise RuntimeError(
            "No worker configuration completed the auto-tuning benchmark "
            "safely. Close other memory-intensive programs and run again."
        )

    fastest_rate = max(item["rate"] for item in stable)
    close_enough = fastest_rate * (1.0 - AUTO_TUNE_CLOSE_RATE_PERCENT / 100.0)
    selected = min(
        (item for item in stable if item["rate"] >= close_enough),
        key=lambda item: item["workers"],
    )

    save_tuning_profile(
        selected["workers"], selected["rate"], benchmark_results,
        available_mb, cpu_percent,
    )
    print(
        f"Auto-tuner selected {selected['workers']} workers at "
        f"{selected['rate']:.2f} benchmark files/sec."
    )
    return selected["workers"]


def worker_loop(
    worker_number: int,
    process: subprocess.Popen,
    stderr_handle: object,
    java: str,
    classpath: str,
    temp_dir: Path,
    tasks: Queue,
    results: Queue,
    runtime_control: dict,
    runtime_lock: threading.Lock,
) -> None:
    try:
        while True:
            task = tasks.get()
            if task is None:
                break

            stem, encoded_text = task
            try:
                status, payload, message = request_worker(
                    process, stem, encoded_text
                )
                if status == "success":
                    results.put(("success", payload, "", ""))
                else:
                    results.put((
                        "failure", stem, "analysis",
                        message,
                    ))

            except Exception as error:
                # A dead JVM is restarted for later files. The affected file is
                # recorded, so it can be retried simply by running this script again.
                results.put(("failure", stem, "Java worker", str(error)))
                close_worker(process, stderr_handle)
                try:
                    process, stderr_handle = open_worker(
                        worker_number, java, classpath, temp_dir
                    )
                except Exception as restart_error:
                    # Other live workers can finish the queued files. Do not
                    # emit a second result for the same input file.
                    print(
                        f"Worker {worker_number} could not restart: "
                        f"{restart_error}"
                    )
                    return

            with runtime_lock:
                if (
                    runtime_control["retire"] > 0
                    and runtime_control["active"] > 1
                ):
                    runtime_control["retire"] -= 1
                    return
    finally:
        with runtime_lock:
            runtime_control["active"] -= 1
        close_worker(process, stderr_handle)


def producer_loop(
    raw_files: list[Path], tasks: Queue, results: Queue, worker_count: int,
) -> None:
    try:
        for raw_file in raw_files:
            try:
                text = raw_file.read_text(encoding="utf-8", errors="replace")
                sentences = custom_sentence_splitter(text)
                if not sentences:
                    raise ValueError("No usable text after segmentation.")

                tasks.put((raw_file.stem, encode_segmented_text(sentences)))
            except Exception as error:
                results.put((
                    "failure", raw_file.stem, "preprocessing", str(error)
                ))
    finally:
        for _ in range(worker_count):
            tasks.put(None)


def main() -> None:
    if not raw_texts_dir.exists():
        raise FileNotFoundError(f"Raw-text folder not found: {raw_texts_dir}")
    if not l2sca_dir.exists():
        raise FileNotFoundError(f"L2SCA folder not found: {l2sca_dir}")

    ensure_nltk_tokenizer()
    done = already_completed()
    raw_files = sorted(
        file for file in raw_texts_dir.glob("*.txt")
        if file.stem not in done
    )

    print(f"Already completed: {len(done):,}")
    print(f"Remaining raw files: {len(raw_files):,}")
    if not raw_files:
        print("Everything is already analyzed.")
        return

    _, available_mb, cpu_percent = choose_worker_count()
    if available_mb is not None:
        print(f"Available RAM at startup: {available_mb:,} MB")
    if cpu_percent is not None:
        print(f"CPU use at startup: {cpu_percent:.1f}%")
    print(f"Java heap per worker: up to {JAVA_HEAP_MB:,} MB")
    print("GPU: not used by the Stanford constituency parser")

    successful = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix="fast_l2sca_") as temp_name:
        temp_dir = Path(temp_name)
        print("Compiling the optimized Java worker...")
        java, classpath = compile_java_worker(temp_dir)

        worker_count = autotune_worker_count(
            raw_files, java, classpath, temp_dir, available_mb, cpu_percent
        )
        print(f"Selected Java workers: {worker_count}")

        print("Loading Stanford models into persistent workers...")
        processes = start_worker_group(
            worker_count, java, classpath, temp_dir
        )

        start_time = time.time()
        runtime_control = {"active": worker_count, "retire": 0}
        runtime_lock = threading.Lock()

        tasks: Queue = Queue(
            maxsize=max(1, worker_count * TASKS_BUFFERED_PER_WORKER)
        )
        results: Queue = Queue()

        workers = [
            threading.Thread(
                target=worker_loop,
                args=(
                    number, process, handle, java, classpath, temp_dir,
                    tasks, results, runtime_control, runtime_lock,
                ),
                daemon=True,
            )
            for number, process, handle in processes
        ]
        for thread in workers:
            thread.start()

        producer = threading.Thread(
            target=producer_loop,
            args=(raw_files, tasks, results, worker_count),
            daemon=True,
        )
        producer.start()

        output_exists = output_file.exists()
        with output_file.open(
            "a", encoding="utf-8", newline="", buffering=1
        ) as output_handle, failed_file.open(
            "w", encoding="utf-8", newline="", buffering=1
        ) as failed_handle:
            result_writer = csv.writer(output_handle)
            failure_writer = csv.writer(failed_handle)

            if not output_exists or output_file.stat().st_size == 0:
                result_writer.writerow(HEADER)
            failure_writer.writerow(["filename", "stage", "error"])

            completed_this_run = 0
            last_resource_check = time.monotonic()
            consecutive_low_ram = 0
            while completed_this_run < len(raw_files):
                try:
                    status, payload, stage, message = results.get(timeout=30)
                except Empty:
                    if not any(thread.is_alive() for thread in workers):
                        raise RuntimeError(
                            "All Java workers stopped before the run finished. "
                            "See the last console messages for the cause."
                        )
                    continue

                if status == "success":
                    result_writer.writerow(payload)
                    output_handle.flush()
                    successful += 1
                else:
                    failure_writer.writerow([payload, stage, message[:10000]])
                    failed_handle.flush()
                    failed += 1

                completed_this_run += 1

                now = time.monotonic()
                if now - last_resource_check >= RUNTIME_RESOURCE_CHECK_SECONDS:
                    last_resource_check = now
                    current_available_mb = available_ram_mb()

                    if (
                        current_available_mb is not None
                        and current_available_mb
                        < RUNTIME_MIN_AVAILABLE_RAM_MB
                    ):
                        consecutive_low_ram += 1
                    else:
                        consecutive_low_ram = 0

                    if consecutive_low_ram >= RUNTIME_LOW_RAM_CHECKS:
                        with runtime_lock:
                            if (
                                runtime_control["active"] > 1
                                and runtime_control["retire"] == 0
                            ):
                                runtime_control["retire"] = 1
                                print(
                                    "Available RAM stayed below "
                                    f"{RUNTIME_MIN_AVAILABLE_RAM_MB:,} MB; "
                                    "retiring one Java worker safely."
                                )
                        consecutive_low_ram = 0

                if (
                    completed_this_run % PROGRESS_EVERY == 0
                    or completed_this_run == len(raw_files)
                ):
                    elapsed = time.time() - start_time
                    rate = completed_this_run / elapsed if elapsed else 0.0
                    remaining = len(raw_files) - completed_this_run
                    eta_hours = (remaining / rate / 3600) if rate else 0.0
                    print(
                        f"[{completed_this_run:,}/{len(raw_files):,}] "
                        f"successful={successful:,}, failed={failed:,}, "
                        f"rate={rate:.2f} files/sec, ETA={eta_hours:.2f} hours"
                    )

        producer.join()
        for thread in workers:
            thread.join()

    print("\nL2SCA run completed.")
    print(f"Successful this run: {successful:,}")
    print(f"Failed this run: {failed:,}")
    print(f"Results: {output_file}")
    print(f"Failure log: {failed_file}")


if __name__ == "__main__":
    main()
