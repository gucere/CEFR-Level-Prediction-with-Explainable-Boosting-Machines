from pathlib import Path
import csv
import io
import subprocess
import sys
import tempfile

import spacy


# Files are preprocessed and analysed in temporary batches. Increasing this can
# be faster, while decreasing it uses less temporary disk space.
BATCH_SIZE = 5000

script_dir = Path(__file__).resolve().parent
thesis_root = script_dir.parent.parent.parent

input_dir = thesis_root / "0_Data" / "raw_texts"
lca_dir = script_dir / "lca_of_Xiaofei_Lu"
lca_script = lca_dir / "folder-lc.py"

output_file = script_dir / "lca_results.csv"
failed_file = script_dir / "lca_failed.csv"


def write_log(path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def preprocess(raw_file, destination, nlp):
    text = raw_file.read_text(encoding="utf-8").strip()

    doc = nlp(text)
    tokens = [
        f"{token.lemma_.lower()}_{token.tag_}"
        for token in doc
        if not token.is_space and not token.is_punct
    ]

    (destination / f"{raw_file.stem}.lem").write_text(
        " ".join(tokens), encoding="utf-8"
    )


def run_lca(folder):
    result = subprocess.run(
        [sys.executable, str(lca_script), str(folder)],
        cwd=str(lca_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"LCA exited with code {result.returncode}")

    return [row for row in csv.reader(io.StringIO(result.stdout)) if row]


def add_lca_rows(rows, writer, header_written):
    for row in rows:
        first_cell = row[0].strip()
        is_header = first_cell.lower() in {"filename", "file", "name"}

        if is_header:
            if header_written:
                continue
            header_written = True
        else:
            # Convert a temporary path or .lem name back to the raw-text stem.
            row[0] = Path(first_cell).stem

        writer.writerow(row)

    return header_written


def retry_batch_individually(lem_files, writer, header_written, failures):
    """Retry a failed LCA batch one file at a time to identify bad files."""
    for lem_file in lem_files:
        with tempfile.TemporaryDirectory(prefix="lca_retry_") as retry_name:
            retry_dir = Path(retry_name)
            retry_file = retry_dir / lem_file.name
            retry_file.write_bytes(lem_file.read_bytes())

            try:
                rows = run_lca(retry_dir)
                header_written = add_lca_rows(rows, writer, header_written)
            except Exception as error:
                failures.append((f"{lem_file.stem}.txt", "LCA", str(error)))

    return header_written


def main():
    if not input_dir.exists():
        raise FileNotFoundError(f"Raw-text folder not found: {input_dir}")
    if not lca_script.exists():
        raise FileNotFoundError(f"LCA script not found: {lca_script}")

    raw_files = sorted(input_dir.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No .txt files found in: {input_dir}")

    print(f"Raw-text folder: {input_dir}")
    print(f"Files found: {len(raw_files)}")
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    failures = []
    processed = 0
    header_written = False

    with output_file.open("w", encoding="utf-8", newline="") as result_handle:
        writer = csv.writer(result_handle)

        for start in range(0, len(raw_files), BATCH_SIZE):
            batch = raw_files[start : start + BATCH_SIZE]

            with tempfile.TemporaryDirectory(prefix="lca_batch_") as temp_name:
                temp_dir = Path(temp_name)

                for raw_file in batch:
                    try:
                        preprocess(raw_file, temp_dir, nlp)
                    except Exception as error:
                        failures.append((raw_file.name, "preprocessing", str(error)))

                lem_files = sorted(temp_dir.glob("*.lem"))

                if lem_files:
                    try:
                        rows = run_lca(temp_dir)
                        header_written = add_lca_rows(rows, writer, header_written)
                    except Exception as batch_error:
                        print(
                            f"Batch LCA failed ({batch_error}). "
                            "Retrying its files individually..."
                        )
                        header_written = retry_batch_individually(
                            lem_files, writer, header_written, failures
                        )

                    processed += len(lem_files)

            completed = min(start + BATCH_SIZE, len(raw_files))
            print(f"Checked {completed}/{len(raw_files)} raw files")

    write_log(failed_file, failures, ["filename", "stage", "error"])

    print("\nLCA run completed.")
    print(f"Sent to LCA: {processed}")
    print(f"Failed: {len(failures)}")
    print(f"Results: {output_file}")
    print(f"Failure log: {failed_file}")


if __name__ == "__main__":
    main()