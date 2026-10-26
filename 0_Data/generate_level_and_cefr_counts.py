from collections import Counter
from pathlib import Path
import os
import time
import xml.etree.ElementTree as ET


DATA_DIR = Path(__file__).resolve().parent
TEXT_DIR = DATA_DIR / "raw_texts"
OUTPUT_FILE = DATA_DIR / "level_and_cefr_counts.xml"

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def read_labels(filename: str) -> tuple[str, int]:
    """Read CEFR and level from writing_id_cefr_X_level_N_grade_G.txt."""
    stem = filename[:-4]
    parts = stem.split("_")
    if (
        len(parts) != 7
        or not parts[0].isdigit()
        or parts[1] != "cefr"
        or parts[3] != "level"
        or parts[5] != "grade"
        or not parts[4].isdigit()
    ):
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[2], int(parts[4])


def count_labels() -> Counter:
    if not TEXT_DIR.is_dir():
        raise FileNotFoundError(f"Text folder not found: {TEXT_DIR}")

    counts = Counter()
    for _, _, filenames in os.walk(TEXT_DIR):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                counts[read_labels(filename)] += 1

    if not counts:
        raise RuntimeError(f"No .txt files found in: {TEXT_DIR}")
    return counts


def write_xml(counts: Counter) -> None:
    total_texts = sum(counts.values())
    root = ET.Element("dataset_counts", total_texts=str(total_texts))

    cefr_values = sorted(
        {cefr for cefr, _ in counts},
        key=lambda cefr: (CEFR_ORDER.get(cefr, 99), cefr),
    )
    for cefr in cefr_values:
        levels = sorted(
            (level, texts)
            for (label, level), texts in counts.items()
            if label == cefr
        )
        cefr_element = ET.SubElement(
            root,
            "cefr",
            name=cefr,
            texts=str(sum(texts for _, texts in levels)),
        )
        for level, texts in levels:
            ET.SubElement(
                cefr_element,
                "level",
                number=str(level),
                texts=str(texts),
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)


def main() -> None:
    started = time.perf_counter()
    counts = count_labels()
    write_xml(counts)

    print(f"Texts counted: {sum(counts.values()):,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Completed in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
