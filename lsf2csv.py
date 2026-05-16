#!/usr/bin/env python3
"""
lsf2csv — Convert DUNE LSF mission logs to CSV files.

Usage examples:

  # List all message types in a log:
  python lsf2csv.py /path/to/Data.lsf.gz --list

  # Export specific message types to CSV (one CSV per message type):
  python lsf2csv.py /path/to/Data.lsf.gz -m EstimatedState -m GpsFix

  # Export all message types (one CSV per type):
  python lsf2csv.py /path/to/Data.lsf.gz --all

  # Export to a specific output directory:
  python lsf2csv.py /path/to/Data.lsf.gz -m EstimatedState -o /tmp/output

  # Use a specific IMC.xml definitions file:
  python lsf2csv.py /path/to/Data.lsf.gz --imc /path/to/IMC.xml -m GpsFix

  # Merge all selected message types into a single CSV:
  python lsf2csv.py /path/to/Data.lsf.gz -m Temperature -m Pressure --merge

  # Show human-readable timestamps:
  python lsf2csv.py /path/to/Data.lsf.gz -m EstimatedState --human-time

  # Resolve entity IDs to human-readable names:
  python lsf2csv.py /path/to/Data.lsf.gz -m EstimatedState --resolve-entities

  # Batch-process all Data.lsf.gz files under a log directory:
  python lsf2csv.py /path/to/log/dir/ --batch -m EstimatedState -m GpsFix

  # Lat/lon in degrees instead of radians:
  python lsf2csv.py /path/to/Data.lsf.gz -m EstimatedState --degrees
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from imc_parser import parse_imc_xml
from lsf_reader import IMCMessage, iter_packets

DEFAULT_IMC_XML = Path(__file__).parent.parent / "dune-software" / "NEW" / "imc" / "IMC.xml"

LAT_LON_FIELDS = {"lat", "lon"}


def flatten_fields(fields: dict, prefix: str = "") -> dict:
    """Flatten nested message fields into dot-separated keys."""
    flat = {}
    for k, v in fields.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_fields(v, key))
        elif isinstance(v, list):
            flat[key] = str(v)
        else:
            flat[key] = v
    return flat


def format_timestamp(ts: float, human: bool = False) -> str:
    if human:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"{ts:.6f}"


def collect_field_keys_and_entities(lsf_path: Path, msg_defs: dict,
                                    filter_abbrevs: set = None,
                                    resolve_entities: bool = False,
                                    ) -> tuple[dict[str, list[str]], dict[int, str], int]:
    """
    First pass: collect all unique field key names per message type
    and optionally build the entity ID map.
    Returns (field_keys_per_type, entity_map, total_message_count).
    Memory: stores only field name sets (strings), not message payloads.
    """
    field_keys_per_type = {}
    entity_map = {}
    total = 0

    for msg in iter_packets(lsf_path, msg_defs, filter_abbrevs, keep_raw=False):
        total += 1
        if resolve_entities and msg.abbrev == "EntityInfo":
            eid = msg.fields.get("id")
            label = msg.fields.get("label", "")
            if eid is not None and label:
                entity_map[eid] = label

        if msg.abbrev not in field_keys_per_type:
            field_keys_per_type[msg.abbrev] = {}
        keys = field_keys_per_type[msg.abbrev]
        flat = flatten_fields(msg.fields)
        for k in flat:
            keys[k] = None  # dict preserves insertion order (Python 3.7+)

    result = {abv: list(k.keys()) for abv, k in field_keys_per_type.items()}
    return result, entity_map, total


def _build_row(msg: IMCMessage, human_time: bool, entity_map: dict,
               degrees: bool, include_msg_type: bool = False) -> dict:
    """Build a CSV row dict from a decoded message."""
    flat = flatten_fields(msg.fields)
    if degrees:
        for k in LAT_LON_FIELDS:
            if k in flat and isinstance(flat[k], (int, float)):
                flat[k] = math.degrees(flat[k])

    row = {
        "timestamp": format_timestamp(msg.header.timestamp, human_time),
        "src": msg.header.src,
        "src_ent": msg.header.src_ent,
        "dst": msg.header.dst,
        "dst_ent": msg.header.dst_ent,
    }
    if include_msg_type:
        row["message_type"] = msg.abbrev
    if entity_map:
        row["src_ent_name"] = entity_map.get(msg.header.src_ent, "")
        row["dst_ent_name"] = entity_map.get(msg.header.dst_ent, "")
    row.update(flat)
    return row


def stream_write_csvs(lsf_path: Path, msg_defs: dict,
                      field_keys_per_type: dict[str, list[str]],
                      output_dir: Path, stem: str,
                      human_time: bool = False, entity_map: dict = None,
                      degrees: bool = False):
    """
    Second pass (non-merge): stream packets and write rows directly
    to per-type CSV files.  Opens one file handle per message type.
    Memory: one message at a time.
    """
    header_cols = ["timestamp", "src", "src_ent", "dst", "dst_ent"]
    if entity_map:
        header_cols = ["timestamp", "src", "src_ent", "src_ent_name",
                       "dst", "dst_ent", "dst_ent_name"]

    files = {}
    writers = {}
    row_counts = {}

    try:
        for abbrev, keys in field_keys_per_type.items():
            out_path = output_dir / f"{stem}_{abbrev}.csv"
            fieldnames = header_cols + keys
            f = open(out_path, "w", newline="")
            files[abbrev] = f
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writers[abbrev] = writer
            row_counts[abbrev] = 0

        for msg in iter_packets(lsf_path, msg_defs,
                                set(field_keys_per_type), keep_raw=False):
            writer = writers.get(msg.abbrev)
            if writer is None:
                continue
            row = _build_row(msg, human_time, entity_map, degrees)
            writer.writerow(row)
            row_counts[msg.abbrev] += 1

    finally:
        for f in files.values():
            f.close()

    for abbrev, count in row_counts.items():
        out_path = output_dir / f"{stem}_{abbrev}.csv"
        print(f"  -> {out_path}  ({count} rows)")


def stream_write_merged_csv(lsf_path: Path, msg_defs: dict,
                            filter_abbrevs: set, all_field_keys: list[str],
                            output_path: Path,
                            human_time: bool = False, entity_map: dict = None,
                            degrees: bool = False):
    """
    Second pass (merge): buffer rows in a temp sqlite3 database,
    then read back sorted by timestamp and write the merged CSV.
    Memory: one row at a time (sqlite3 handles the rest on disk).
    """
    header_cols = ["timestamp", "message_type", "src", "src_ent", "dst", "dst_ent"]
    if entity_map:
        header_cols = ["timestamp", "message_type", "src", "src_ent",
                       "src_ent_name", "dst", "dst_ent", "dst_ent_name"]
    fieldnames = header_cols + all_field_keys

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE msgs (ts REAL, data TEXT)")

        for msg in iter_packets(lsf_path, msg_defs, filter_abbrevs, keep_raw=False):
            row = _build_row(msg, human_time, entity_map, degrees,
                             include_msg_type=True)
            conn.execute("INSERT INTO msgs VALUES (?, ?)",
                         (msg.header.timestamp, json.dumps(row)))

        conn.commit()

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            cursor = conn.execute("SELECT data FROM msgs ORDER BY ts")
            written = 0
            for (data_json,) in cursor:
                writer.writerow(json.loads(data_json))
                written += 1

        print(f"  -> {output_path}  ({written} rows)")

    finally:
        Path(tmp.name).unlink(missing_ok=True)


def list_message_types(lsf_path: Path, msg_defs: dict):
    """Print a summary of all message types in the LSF file."""
    counts = defaultdict(int)
    for msg in iter_packets(lsf_path, msg_defs):
        counts[msg.abbrev] += 1

    print(f"\n{'Message Type':<35} {'Count':>10}")
    print("-" * 47)
    for abbrev, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {abbrev:<33} {count:>10}")
    print("-" * 47)
    print(f"  {'TOTAL':<33} {sum(counts.values()):>10}")
    print(f"\n  {len(counts)} distinct message types found.\n")


def find_imc_xml(args_imc: str = None) -> Path:
    """Locate IMC.xml, checking common paths."""
    if args_imc:
        p = Path(args_imc)
        if p.exists():
            return p
        print(f"Error: Specified IMC.xml not found: {p}", file=sys.stderr)
        sys.exit(1)

    candidates = [
        DEFAULT_IMC_XML,
        Path("/home/adallolio/dune-software/NEW/imc/IMC.xml"),
        Path("/home/adallolio/dune-software/NTNU/imc/IMC.xml"),
        Path("/home/adallolio/dune-software/imc/IMC.xml"),
    ]
    for c in candidates:
        if c.exists():
            return c

    print("Error: Could not find IMC.xml. Use --imc to specify the path.",
          file=sys.stderr)
    sys.exit(1)


def find_lsf_files(directory: Path) -> list[Path]:
    """Recursively find all Data.lsf and Data.lsf.gz files under a directory."""
    files = []
    for pattern in ("**/Data.lsf.gz", "**/Data.lsf", "**/*.lsf.gz", "**/*.lsf"):
        files.extend(directory.glob(pattern))
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def process_single_file(lsf_path: Path, msg_defs: dict, args) -> None:
    """Process a single LSF file with the given arguments (two-pass streaming)."""
    t0 = time.monotonic()

    filter_abbrevs = set(args.messages) if args.messages else None
    out_dir = args.output or lsf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: collect field key names + entity map (lightweight) ──
    print(f"Scanning {lsf_path} ...")
    field_keys_per_type, entity_map, total_msgs = collect_field_keys_and_entities(
        lsf_path, msg_defs, filter_abbrevs, args.resolve_entities,
    )
    elapsed = time.monotonic() - t0
    print(f"  Found {total_msgs} messages ({len(field_keys_per_type)} types) in {elapsed:.1f}s.")

    stem = lsf_path.stem.replace(".lsf", "")

    # ── Pass 2: stream rows to CSV(s) ──
    t1 = time.monotonic()

    if args.merge:
        all_keys = []
        key_set = set()
        for keys in field_keys_per_type.values():
            for k in keys:
                if k not in key_set:
                    key_set.add(k)
                    all_keys.append(k)
        out_path = out_dir / f"{stem}_merged.csv"
        stream_write_merged_csv(
            lsf_path, msg_defs, set(field_keys_per_type), all_keys,
            out_path, args.human_time, entity_map, args.degrees,
        )
    else:
        stream_write_csvs(
            lsf_path, msg_defs, field_keys_per_type,
            out_dir, stem, args.human_time, entity_map, args.degrees,
        )

    write_elapsed = time.monotonic() - t1
    print(f"  Wrote {total_msgs} rows in {write_elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DUNE LSF mission logs to CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("lsf_file", type=Path,
                        help="Path to .lsf/.lsf.gz file, or directory with --batch")
    parser.add_argument("-m", "--message", action="append", dest="messages",
                        metavar="MSG",
                        help="Message type abbreviation to export (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="Export all message types")
    parser.add_argument("--list", action="store_true",
                        help="List all message types in the log and exit")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: same as LSF file)")
    parser.add_argument("--imc", type=str, default=None,
                        help="Path to IMC.xml definitions file")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all selected messages into a single CSV")
    parser.add_argument("--human-time", action="store_true",
                        help="Use human-readable timestamps (UTC)")
    parser.add_argument("--resolve-entities", action="store_true",
                        help="Add entity name columns from EntityInfo messages")
    parser.add_argument("--degrees", action="store_true",
                        help="Convert lat/lon from radians to degrees")
    parser.add_argument("--batch", action="store_true",
                        help="Process all LSF files under a directory tree")

    args = parser.parse_args()

    if not args.lsf_file.exists():
        print(f"Error: Path not found: {args.lsf_file}", file=sys.stderr)
        sys.exit(1)

    imc_xml = find_imc_xml(args.imc)
    print(f"Using IMC definitions: {imc_xml}")

    t_global = time.monotonic()
    msg_defs = parse_imc_xml(str(imc_xml))
    print(f"Loaded {len(msg_defs)} message definitions.\n")

    if args.batch or args.lsf_file.is_dir():
        lsf_dir = args.lsf_file
        if not lsf_dir.is_dir():
            print(f"Error: --batch requires a directory, got: {lsf_dir}",
                  file=sys.stderr)
            sys.exit(1)

        lsf_files = find_lsf_files(lsf_dir)
        if not lsf_files:
            print(f"No LSF files found under {lsf_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(lsf_files)} LSF files under {lsf_dir}\n")

        for i, lsf_path in enumerate(lsf_files, 1):
            print(f"[{i}/{len(lsf_files)}] ", end="")

            if args.list:
                list_message_types(lsf_path, msg_defs)
            else:
                out_dir = args.output
                if out_dir:
                    rel = lsf_path.parent.relative_to(lsf_dir)
                    args.output = out_dir / rel
                else:
                    args.output = None
                process_single_file(lsf_path, msg_defs, args)
                args.output = out_dir
            print()

        elapsed = time.monotonic() - t_global
        print(f"Batch complete. {len(lsf_files)} files in {elapsed:.1f}s total.")
        return

    if args.list:
        list_message_types(args.lsf_file, msg_defs)
        return

    if not args.messages and not args.all:
        print("Error: Specify -m MSG, --all, or --list.", file=sys.stderr)
        sys.exit(1)

    if args.messages:
        known = {m.abbrev for m in msg_defs.values()}
        unknown = set(args.messages) - known
        if unknown:
            print(f"Warning: Unknown message types: {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            args.messages = [m for m in args.messages if m not in unknown]

    process_single_file(args.lsf_file, msg_defs, args)
    print(f"\nDone. Total time: {time.monotonic() - t_global:.1f}s")


if __name__ == "__main__":
    main()
