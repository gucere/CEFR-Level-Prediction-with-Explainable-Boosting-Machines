import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import duckdb
import psutil

HERE = Path(__file__).resolve().parent
SOURCES = {
    "LCA": ("LCA", "lca_results.csv"),
    "L2SCA": ("L2SCA", "l2sca_results.csv"),
    "TAASSC": ("TAASSC", "taassc_results.csv"),
    "TAALES": ("TAALES", "taales_results_final.csv"),
    "TAALES_COVERAGE": ("TAALES", "taales_results_final_index_coverage.csv"),
    "POLKE": ("POLKE", "polke_results.csv"),
    "ERRANT": ("ERRANT", "errant_results.csv"),
}
METADATA = {
    "filename", "writing_id", "text_id", "cefr", "cefr_level", "cefr_numeric",
    "level", "grade", "learner_id", "learner_id_categorical", "processing_mode",
    "text", "text_corrected", "corrected_text", "original_text", "split",
    "topic", "topic_id", "prompt_id", "task_id", "unit", "nationality", "l1",
}
ERROR_COUNTS = {
    "total_errors", "distinct_error_types", "missing_errors", "unnecessary_errors",
    "replacement_errors", "other_errors", "affected_original_tokens", "correction_token_equivalents",
}
ACCURACY = {
    "errors_per_100_words", "missing_errors_per_100_words",
    "unnecessary_errors_per_100_words", "replacement_errors_per_100_words",
    "other_errors_per_100_words", "surface_accuracy_proxy", "error_free",
}
LABELS = ("A1", "A2", "B1", "B2", "C1", "C2")


def identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def scan(path):
    return f"read_csv({literal(path)}, header=true, all_varchar=true, parallel=false, sample_size=1000)"


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def file_signature(path):
    if isinstance(path, DataframeSource):
        return path.signature()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "modified_ns": stat.st_mtime_ns}


@dataclass(frozen=True)
class DataframeSource:
    """A projection of the canonical dataframe using the existing analysis names."""

    source: str
    parquet: Path
    metadata: Path
    dictionary: Path
    features: tuple

    @property
    def name(self):
        return f"{self.parquet.name} [{self.source}]"

    def __str__(self):
        return f"{self.parquet} [{self.source}]"

    def is_file(self):
        return self.parquet.is_file() and self.metadata.is_file()

    def signature(self):
        return {
            "source": self.source,
            "files": [file_signature(p) for p in (self.parquet, self.metadata, self.dictionary)],
            "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def sql(self):
        # VARCHAR maintains compatibility with existing label parsers and TRIM.
        # DuckDB still reads only the Parquet columns requested by each analysis.
        fields = ["CAST(d.text_id AS VARCHAR) AS writing_id",
                  "d.cefr_level AS cefr", "m.filename", "CAST(m.level AS VARCHAR) AS level"]
        fields.extend(f"CAST(d.{identifier(column)} AS VARCHAR) AS {identifier(original)}"
                      for column, original in self.features)
        return ("(SELECT " + ", ".join(fields)
                + f" FROM read_parquet({literal(self.parquet)}) d"
                + f" JOIN read_parquet({literal(self.metadata)}) m ON d.text_id=m.text_id)")


def dataframe_inputs(thesis_root, sources=None):
    """Validate the shared tables before exposing any source to an analysis."""
    folder = (Path(thesis_root) / "1_Complexity_Analysis_Full_Module"
              / "2_Basic_Analysis" / "0_Feature_Dataframe")
    parquet, metadata = folder / "feature_dataframe.parquet", folder / "text_metadata.parquet"
    dictionary = folder / "feature_dictionary.csv"
    for path in (parquet, metadata, dictionary):
        if not path.is_file():
            action = "python create_dataframe.py --metadata-only" if path == metadata else "python create_dataframe.py"
            raise FileNotFoundError(f"Missing {path}. In {folder}, run: {action}")
    with dictionary.open(encoding="utf-8-sig", newline="") as handle:
        specs = list(csv.DictReader(handle))
    for spec in specs:
        expected_group = "accuracy" if spec["source"] == "ERRANT" else "complexity"
        if (spec["group"] != expected_group or
                spec["column"] != f"{expected_group}__{spec['source']}__{spec['original_feature']}"):
            raise ValueError("Inconsistent source/group/column mapping in the feature dictionary.")
    with duckdb.connect() as connection:
        actual = [r[0] for r in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({literal(parquet)})").fetchall()]
        if actual != ["text_id", "cefr_level"] + [s["column"] for s in specs]:
            raise ValueError("The feature dictionary does not match the dataframe columns/order.")
        validate_metadata(connection, parquet, metadata)
    selected = tuple(sources) if sources is not None else tuple(SOURCES)
    result = {}
    for source in selected:
        features = tuple((s["column"], s["original_feature"]) for s in specs if s["source"] == source)
        names = [original.strip().casefold() for _, original in features]
        if not features or len(names) != len(set(names)) or any(n in METADATA for n in names):
            raise ValueError(f"{source}: missing, duplicate, or metadata feature names in the dictionary.")
        result[source] = DataframeSource(source, parquet, metadata, dictionary, features)
    return result


def validate_metadata(connection, parquet, metadata):
    for path in (parquet, metadata):
        total, unique, missing = connection.execute(
            f"SELECT count(*), count(DISTINCT text_id), count(*) FILTER (WHERE text_id IS NULL) "
            f"FROM read_parquet({literal(path)})").fetchone()
        if not total or total != unique or missing:
            raise ValueError(f"{path}: text IDs must be nonempty, present and unique.")
    invalid = connection.execute(
        f"SELECT count(*) FROM read_parquet({literal(parquet)}) d FULL JOIN "
        f"read_parquet({literal(metadata)}) m ON d.text_id=m.text_id "
        "WHERE d.text_id IS NULL OR m.text_id IS NULL OR d.cefr_level IS DISTINCT FROM m.cefr_level "
        "OR d.cefr_level IS NULL OR d.cefr_level NOT IN ('A1','A2','B1','B2','C1','C2') "
        "OR m.level IS NULL OR m.level NOT BETWEEN 1 AND 15 OR m.filename IS NULL"
    ).fetchone()[0]
    if invalid:
        raise ValueError(f"Metadata disagrees with the dataframe for {invalid:,} rows. Rebuild with --metadata-only.")


def build_text_metadata(source_root, output, parquet):
    """Preserve actual course levels from LCA filenames, never infer them from CEFR."""
    path = Path(source_root) / "LCA" / "lca_results.csv"
    if not path.is_file():
        path = Path(str(path) + ".gz")
    if not path.is_file():
        raise FileNotFoundError(path)
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix=".metadata_build_", dir=output) as directory:
        temporary = Path(directory) / "text_metadata.parquet"
        with duckdb.connect() as connection:
            columns = [r[0] for r in connection.execute(f"DESCRIBE SELECT * FROM {scan(path)}").fetchall()]
            identity, cefr, conflict = label_expressions(columns)
            filename = next((identifier(c) for c in columns if c.casefold() == "filename"), None)
            if filename is None:
                raise ValueError("LCA filenames are required to preserve original course-level metadata.")
            connection.execute(
                f"CREATE TABLE metadata AS SELECT {identity} AS text_id, {cefr} AS cefr_level, "
                f"trim({filename}) AS filename, TRY_CAST(regexp_extract(lower({filename}), "
                "'_level_([0-9]+)(?:_|[.]|$)', 1) AS INTEGER) AS level, "
                f"COALESCE({conflict}, false) AS conflict FROM {scan(path)}")
            if connection.execute("SELECT count(*) FROM metadata WHERE conflict").fetchone()[0]:
                raise ValueError("LCA metadata contains conflicting text IDs or CEFR labels.")
            connection.execute(f"COPY (SELECT * EXCLUDE(conflict) FROM metadata ORDER BY text_id) "
                               f"TO {literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            validate_metadata(connection, parquet, temporary)
        os.replace(temporary, output / "text_metadata.parquet")
    print(f"Saved course-level metadata: {output / 'text_metadata.parquet'}", flush=True)


@contextmanager
def dataframe_rows(source):
    """Stream source rows for the existing grammar aggregation; NULL stays missing."""
    with duckdb.connect() as connection:
        cursor = connection.execute(f"SELECT * FROM {source.sql()}")
        def rows():
            yield [column[0] for column in cursor.description]
            while batch := cursor.fetchmany(1024):
                for row in batch:
                    yield ["" if value is None else str(value) for value in row]
        yield rows()


def validate_ranking_checkpoint(output, inputs, restart=False):
    """Do not resume legacy/full-source scores against a changed dataframe."""
    manifest = output.with_suffix(".input.json")
    signature = {name: file_signature(path) for name, path in inputs.items()}
    if output.exists() and not restart:
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError(f"{output.name} belongs to different or unrecorded inputs. Use --restart to recompute.")
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)


def feature_columns(source, columns):
    if len(columns) != len({c.casefold() for c in columns}):
        raise ValueError(f"{source}: duplicate column names.")
    result = []
    for name in columns:
        key = name.casefold()
        if key in METADATA:
            continue
        if source == "ERRANT":
            if key not in ERROR_COUNTS | ACCURACY and not (key.startswith("errant_") and key.endswith("_per_100_words")):
                continue
            group = "accuracy"
            kind = ("error_count" if key in ERROR_COUNTS else "error_rate_per_100_words"
                    if key.endswith("_per_100_words") else "accuracy_proxy" if key == "surface_accuracy_proxy" else "indicator")
        else:
            group = "complexity"
            kind = "index_coverage" if source == "TAALES_COVERAGE" else "complexity_measure"
        result.append({"column": f"{group}__{source}__{name}", "group": group,
                       "source": source, "original_feature": name, "measure_type": kind})
    if not result:
        raise ValueError(f"{source}: no numeric feature columns found.")
    return result


def label_expressions(columns):
    names = {c.casefold(): identifier(c) for c in columns}
    file_id = (f"TRY_CAST(regexp_extract({names['filename']}, '^([0-9]+)(?:_|[.]|$)', 1) AS BIGINT)"
               if "filename" in names else "NULL")
    file_cefr = (f"NULLIF(regexp_extract(upper({names['filename']}), '_CEFR_([ABC][12])_', 1), '')"
                 if "filename" in names else "NULL")
    raw_id = names.get("writing_id", names.get("text_id"))
    if raw_id:
        text = f"trim({raw_id})"
        explicit_id = f"CASE WHEN regexp_full_match({text}, '[0-9]+([.]0+)?') THEN TRY_CAST({text} AS BIGINT) ELSE NULL END"
        identity = explicit_id
    else:
        explicit_id, identity = "NULL", file_id
    explicit_cefr = f"NULLIF(upper(trim({names['cefr']})), '')" if "cefr" in names else "NULL"
    cefr = f"COALESCE({explicit_cefr}, {file_cefr})"
    conflicts = f"(({explicit_id}) <> ({file_id})) OR (({explicit_cefr}) <> ({file_cefr}))"
    return identity, cefr, conflicts


def memory_budget(requested):
    if requested.lower() != "auto":
        return requested
    ram = psutil.virtual_memory()
    mib = int(min(ram.available * .75, ram.total * .70) / 2**20)
    if mib < 256:
        raise MemoryError("Insufficient available RAM. Close memory-heavy applications and try again.")
    return f"{mib}MiB"


def export_dataframe(connection, staging, specs, batch_size=20000):
    """Bound hash-join memory by joining a range of writing IDs at a time."""
    ids = [r[0] for r in connection.execute("SELECT text_id FROM LCA ORDER BY text_id").fetchall()]
    projection = "a.text_id, a.cefr_level, " + ", ".join(identifier(s["column"]) for s in specs)
    batch_count = (len(ids) + batch_size - 1) // batch_size
    for batch, start in enumerate(range(0, len(ids), batch_size), 1):
        lower, upper = ids[start], ids[min(start + batch_size, len(ids)) - 1]
        condition = f"text_id BETWEEN {lower} AND {upper}"
        tables = {source: f"(SELECT * FROM {identifier(source)} WHERE {condition})" for source in SOURCES}
        joins = " ".join(f"JOIN {tables[source]} AS {identifier(source)} USING(text_id)" for source in SOURCES if source != "LCA")
        query = f"SELECT {projection} FROM {tables['LCA']} a {joins}"
        part = staging / f"part_{batch:04d}.parquet"
        connection.execute(f"COPY ({query}) TO {literal(part)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 4096)")
        print(f"  Exported batch {batch}/{batch_count} ({min(start + batch_size, len(ids)):,}/{len(ids):,} texts)", flush=True)
    target = staging / "feature_dataframe.parquet"
    parts = staging / "part_*.parquet"
    print("Combining batches into one Parquet file...", flush=True)
    connection.execute(f"COPY (SELECT * FROM read_parquet({literal(parts)})) TO {literal(target)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 4096)")
    return target


def build_dataframe(source_root, output, memory="auto", full_csv=False, overwrite=False):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    paths = {}
    for source, (folder, name) in SOURCES.items():
        path = source_root / folder / name
        if not path.is_file():
            path = Path(str(path) + ".gz")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source} results: {path}")
        paths[source] = path
    outputs = ["feature_dataframe.parquet", "feature_dictionary.csv", "dataframe_preview.csv", "dataframe_summary.json", "text_metadata.parquet"]
    if full_csv:
        outputs.append("feature_dataframe.csv")
    if not overwrite and any((output / name).exists() for name in outputs):
        raise FileExistsError(f"Dataframe outputs already exist in {output}. Use --overwrite to rebuild them.")
    # Avoid leaving a stale optional full CSV beside a newly rebuilt Parquet table.
    if overwrite and not full_csv and (output / "feature_dataframe.csv").exists():
        raise ValueError("An existing full CSV is present. Include --csv when rebuilding, or choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    budget = memory_budget(memory)
    print(f"Building the dataframe from existing results. DuckDB RAM limit: {budget}", flush=True)
    # All temporary files are confined to a newly created child of the output folder.
    with tempfile.TemporaryDirectory(prefix=".dataframe_build_", dir=output) as directory:
        staging = Path(directory).resolve()
        if staging.parent != output:
            raise ValueError("Temporary directory is outside the intended output folder.")
        connection = duckdb.connect(str(staging / "working.duckdb"))
        try:
            connection.execute(f"SET memory_limit={literal(budget)}")
            connection.execute("SET threads=1")
            connection.execute("SET preserve_insertion_order=false")
            specs, counts, stamps = [], {}, {}
            for source, path in paths.items():
                print(f"Reading {source}...", flush=True)
                columns = [r[0] for r in connection.execute(f"DESCRIBE SELECT * FROM {scan(path)}").fetchall()]
                features = feature_columns(source, columns)
                identity, cefr, conflict = label_expressions(columns)
                values = []
                for spec in features:
                    numeric = f"TRY_CAST({identifier(spec['original_feature'])} AS DOUBLE)"
                    values.append(f"CASE WHEN isfinite({numeric}) THEN {numeric} ELSE NULL END AS {identifier(spec['column'])}")
                connection.execute(f"CREATE TABLE {identifier(source)} AS SELECT {identity} AS text_id, {cefr} AS cefr_level, "
                                   f"COALESCE({conflict}, false) AS label_conflict, " + ", ".join(values) + f" FROM {scan(path)}")
                allowed = ",".join(literal(label) for label in LABELS)
                invalid = connection.execute(f"SELECT count(*) FROM {identifier(source)} WHERE text_id IS NULL OR cefr_level IS NULL OR cefr_level NOT IN ({allowed}) OR label_conflict").fetchone()[0]
                if invalid:
                    raise ValueError(f"{source}: {invalid:,} rows have missing, invalid, or contradictory IDs/CEFR labels.")
                total, unique = connection.execute(f"SELECT count(*), count(DISTINCT text_id) FROM {identifier(source)}").fetchone()
                if not total or total != unique:
                    raise ValueError(f"{source}: expected nonempty unique writing IDs; got {total:,} rows and {unique:,} IDs.")
                if source != "LCA":
                    matched = connection.execute(f"SELECT count(*) FROM LCA a JOIN {identifier(source)} b USING(text_id) WHERE a.cefr_level=b.cefr_level").fetchone()[0]
                    if matched != total or matched != counts["LCA"]:
                        raise ValueError(f"{source}: IDs/labels differ from LCA ({matched:,} matching rows, {total:,} source rows, {counts['LCA']:,} LCA rows). No rows were silently dropped.")
                counts[source] = total
                stat = path.stat()
                stamps[source] = {"file": path.name, "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}
                specs.extend(features)
                connection.execute("CHECKPOINT")
                print(f"  {total:,} texts; {len(features):,} features", flush=True)
            print("Saving the full dataframe...", flush=True)
            parquet = export_dataframe(connection, staging, specs)
            parquet_scan = f"read_parquet({literal(parquet)})"
            columns = [r[0] for r in connection.execute(f"DESCRIBE SELECT * FROM {parquet_scan}").fetchall()]
            row_count = connection.execute(f"SELECT count(*) FROM {parquet_scan}").fetchone()[0]
            expected_columns = ["text_id", "cefr_level"] + [s["column"] for s in specs]
            if columns != expected_columns or row_count != counts["LCA"]:
                raise ValueError("Exported dataframe does not match the expected rows or columns.")
            print("Saving the feature dictionary and 100-row preview...", flush=True)
            preview = connection.execute(f"SELECT * FROM {parquet_scan} ORDER BY text_id LIMIT 100").fetchall()
            write_csv(staging / "dataframe_preview.csv", columns, preview)
            dictionary_columns = ("column", "group", "source", "original_feature", "measure_type")
            write_csv(staging / "feature_dictionary.csv", dictionary_columns,
                      ([s[c] for c in dictionary_columns] for s in specs))
            if full_csv:
                print("Saving the optional full CSV...", flush=True)
                connection.execute(f"COPY (SELECT * FROM {parquet_scan}) TO {literal(staging / 'feature_dataframe.csv')} (FORMAT CSV, HEADER true)")
            summary = {
                "rows": row_count, "columns": len(columns), "feature_counts": dict(Counter(s["group"] for s in specs)),
                "identifier": "text_id", "target": "cefr_level", "sources": stamps,
                "source_row_counts": counts, "memory_limit": budget,
                "cefr_counts": dict(connection.execute(f"SELECT cefr_level,count(*) FROM {parquet_scan} GROUP BY cefr_level ORDER BY cefr_level").fetchall()),
                "numeric_storage": "float64; missing/non-numeric/nonfinite measurements are null, never zero-filled",
                "row_policy": "All source IDs and CEFR labels must match exactly; one row per text.",
                "feature_policy": "Existing complexity measurements, ERRANT error counts/rates and accuracy proxies; no statistical feature selection.",
                "preview": "First 100 rows ordered by text_id; same columns as the full dataframe.",
            }
            (staging / "dataframe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        finally:
            connection.close()
        build_text_metadata(source_root, staging, parquet)
        # Only publish files after all source and output checks have passed.
        for name in outputs:
            os.replace(staging / name, output / name)
    print(f"Created {row_count:,} rows and {len(columns):,} columns in {output}", flush=True)
    return summary


def export_existing_csv(folder, memory="auto", overwrite=False):
    """Stream the existing Parquet into CSV without re-extracting or joining features."""
    folder = Path(folder).resolve()
    parquet, destination = folder / "feature_dataframe.parquet", folder / "feature_dataframe.csv"
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists. Use --overwrite to replace the CSV.")
    with tempfile.TemporaryDirectory(prefix=".csv_export_", dir=folder) as directory:
        temporary = Path(directory) / destination.name
        with duckdb.connect() as connection:
            connection.execute(f"SET memory_limit={literal(memory_budget(memory))}")
            connection.execute("SET threads=1")
            query = f"SELECT * FROM read_parquet({literal(parquet)})"
            expected = [row[0] for row in connection.execute(f"DESCRIBE {query}").fetchall()]
            print("Exporting the complete dataframe to CSV. Missing measurements remain empty cells...", flush=True)
            count = connection.execute(f"COPY ({query}) TO {literal(temporary)} (FORMAT CSV, HEADER true)").fetchone()[0]
            with temporary.open(encoding="utf-8", newline="") as handle:
                if next(csv.reader(handle)) != expected:
                    raise ValueError("CSV column order differs from Parquet.")
        os.replace(temporary, destination)
    print(f"Saved {count:,} rows and {len(expected):,} columns: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=HERE.parent.parent / "1_Complexity_Analysis_Methods")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--memory", default="auto", help="Available-RAM-based limit, or an explicit value such as 6GB.")
    parser.add_argument("--csv", action="store_true", help="Also export the complete dataframe as a large CSV file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this script's existing dataframe outputs after a successful build.")
    parser.add_argument("--metadata-only", action="store_true", help="Create or refresh course-level metadata for the existing dataframe without rebuilding its features.")
    parser.add_argument("--csv-only", action="store_true", help="Export the existing Parquet to a full CSV without rebuilding the dataframe.")
    args = parser.parse_args(argv)
    if args.metadata_only and args.csv_only:
        parser.error("Choose either --metadata-only or --csv-only.")
    if args.csv_only:
        export_existing_csv(args.output, args.memory, args.overwrite)
    elif args.metadata_only:
        build_text_metadata(args.source_root, args.output, args.output / "feature_dataframe.parquet")
    else:
        build_dataframe(args.source_root, args.output, args.memory, args.csv, args.overwrite)


if __name__ == "__main__":
    main()
