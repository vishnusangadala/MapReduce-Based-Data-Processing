#!/usr/bin/env python3
"""
AMS 598 — Project 1: Counting with Map/Reduce on SeaWulf
Single-file implementation with mapper() and reducer() plus a small driver.

- Input:  --input   path to a directory of text files
- Work:   --work    directory for intermediates (created if missing)
- Output: --output  final TSV "number \t count" (defaults under --work)
- Procs:  --mappers N (default 4), --reducers N (default 4)

Assumptions:
- Each input file is plain text; integers may appear anywhere (whitespace or mixed text).
- Integers are extracted with regex r'-?\\d+' (supports negatives).
- Intermediates are TSV files named map_{m}_to_red_{r}.tsv under --work.
- Reducers aggregate and emit reducer_{r}.tsv and a merged final output.

Author: Vishnu Sangadala (adapted template)
"""

from __future__ import annotations
import argparse
import concurrent.futures
import glob
import io
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

INT_RE = re.compile(r"-?\d+")

# ----------------------------
# Utility helpers
# ----------------------------

def list_input_files(input_dir: Path) -> List[Path]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            files.append(Path(root) / fname)
    files.sort()
    return files

def chunk(lst: List[Path], k: int) -> List[List[Path]]:
    """Split list into k nearly equal chunks (some may be empty)."""
    n = len(lst)
    out = [[] for _ in range(k)]
    for i, item in enumerate(lst):
        out[i % max(1, k)] .append(item)
    return out

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------
# Mapper
# ----------------------------

def mapper(map_id: int, files: List[Path], reducers: int, workdir: Path) -> List[Path]:
    """
    Reads assigned files, counts integers, then partitions counts by reducer id and
    writes TSV intermediates: map_{map_id}_to_red_{rid}.tsv
    Returns list of written intermediate paths.
    """
    # Local aggregation (number -> count)
    local: Dict[int, int] = defaultdict(int)

    for f in files:
        try:
            with io.open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    for tok in INT_RE.findall(line):
                        try:
                            num = int(tok)
                            local[num] += 1
                        except ValueError:
                            # Shouldn't happen due to regex, but be safe
                            continue
        except Exception as e:
            print(f"[{ts()}] [mapper {map_id}] WARN: failed to read {f}: {e}", file=sys.stderr)

    # Partition to reducers
    writers: Dict[int, io.TextIOBase] = {}
    paths_out: List[Path] = []
    try:
        for num, cnt in local.items():
            rid = (hash(num) % reducers + reducers) % reducers
            if rid not in writers:
                out_path = workdir / f"map_{map_id}_to_red_{rid}.tsv"
                # Store path once per reducer file
                paths_out.append(out_path)
                writers[rid] = io.open(out_path, "w", encoding="utf-8")
            writers[rid].write(f"{num}\t{cnt}\n")
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

    print(f"[{ts()}] mapper {map_id}: files={len(files)} unique_ints={len(local)} parts={len(paths_out)}")
    return paths_out

# ----------------------------
# Reducer
# ----------------------------

def reducer(red_id: int, mapper_parts: List[Path], outdir: Path) -> Path:
    """
    Aggregates all mapper parts for this reducer id and writes reducer_{red_id}.tsv
    (sorted by integer ascending). Returns the output path.
    """
    agg: Dict[int, int] = defaultdict(int)

    for part in mapper_parts:
        try:
            with io.open(part, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        num_str, cnt_str = line.split("\t")
                        num = int(num_str)
                        cnt = int(cnt_str)
                        agg[num] += cnt
                    except Exception:
                        continue
        except FileNotFoundError:
            # In case a mapper produced no file for this reducer
            continue

    out_path = outdir / f"reducer_{red_id}.tsv"
    with io.open(out_path, "w", encoding="utf-8") as out:
        for num in sorted(agg.keys()):
            out.write(f"{num}\t{agg[num]}\n")

    print(f"[{ts()}] reducer {red_id}: keys={len(agg)} -> {out_path.name}")
    return out_path

# ----------------------------
# Driver
# ----------------------------

def merge_final(reducer_outputs: List[Path], final_path: Path) -> None:
    """
    Merges already-sorted-by-key reducer outputs into a single TSV.
    Since each reducer owns a disjoint hash partition (not a range),
    we simply concatenate in reducer id order.
    """
    with io.open(final_path, "w", encoding="utf-8") as out:
        for p in sorted(reducer_outputs, key=lambda q: q.name):
            with io.open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    out.write(line)
    print(f"[{ts()}] final merged -> {final_path}")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Integer counter via Map/Reduce (single-file).")
    ap.add_argument("--input", required=True, help="Directory of input text files.")
    ap.add_argument("--work", required=True, help="Working directory for intermediates/logs.")
    ap.add_argument("--output", default=None, help="Final output TSV path (defaults to WORK/final_counts.tsv).")
    ap.add_argument("--mappers", type=int, default=4)
    ap.add_argument("--reducers", type=int, default=4)
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    workdir = Path(args.work).resolve()
    safe_mkdir(workdir)

    if args.output:
        final_out = Path(args.output).resolve()
        safe_mkdir(final_out.parent)
    else:
        final_out = workdir / "final_counts.tsv"

    files = list_input_files(input_dir)
    if not files:
        print(f"ERROR: No input files found under {input_dir}", file=sys.stderr)
        return 2

    print(f"[{ts()}] Starting Map/Reduce")
    print(f"  input_dir = {input_dir}")
    print(f"  workdir   = {workdir}")
    print(f"  output    = {final_out}")
    print(f"  mappers   = {args.mappers}")
    print(f"  reducers  = {args.reducers}")

    # Clean old intermediates from a previous run
    for old in glob.glob(str(workdir / "map_*_to_red_*.tsv")) + glob.glob(str(workdir / "reducer_*.tsv")):
        try:
            os.remove(old)
        except Exception:
            pass

    # --- MAP PHASE ---
    file_chunks = chunk(files, max(1, args.mappers))
    mapper_outputs: List[List[Path]] = [[] for _ in range(max(1, args.mappers))]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.mappers)) as exe:
        fut_to_id = {
            exe.submit(mapper, i, file_chunks[i], args.reducers, workdir): i
            for i in range(len(file_chunks))
        }
        for fut in concurrent.futures.as_completed(fut_to_id):
            mid = fut_to_id[fut]
            try:
                mapper_outputs[mid] = fut.result()
            except Exception as e:
                print(f"[{ts()}] ERROR mapper {mid} failed: {e}", file=sys.stderr)
                return 3

    # Build per-reducer file lists
    reducer_inputs: Dict[int, List[Path]] = {r: [] for r in range(args.reducers)}
    for mid, parts in enumerate(mapper_outputs):
        for p in parts:
            # file name pattern includes reducer id suffix
            try:
                rid = int(p.stem.split("_")[-1])
            except Exception:
                # fallback by scanning pattern
                rid = None
                m = re.search(r"_to_red_(\d+)$", p.stem)
                if m:
                    rid = int(m.group(1))
            if rid is not None and 0 <= rid < args.reducers:
                reducer_inputs[rid].append(p)

    # --- REDUCE PHASE ---
    reducer_outputs: List[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.reducers)) as exe:
        futs = {
            exe.submit(reducer, rid, reducer_inputs.get(rid, []), workdir): rid
            for rid in range(args.reducers)
        }
        for fut in concurrent.futures.as_completed(futs):
            try:
                reducer_outputs.append(fut.result())
            except Exception as e:
                rid = futs[fut]
                print(f"[{ts()}] ERROR reducer {rid} failed: {e}", file=sys.stderr)
                return 4

    # --- MERGE ---
    merge_final(reducer_outputs, final_out)

    print(f"[{ts()}] Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
