#!/usr/bin/env python3
"""Compare kv-marketplace dump directories across phases.

Usage:
    python -m kv_marketplace.scripts.compare_kv_dumps \
        --root /path/to/dumps \
        --phase1 run1_phase1 \
        --phase2 run1_phase2

The script expects each phase directory to contain subdirectories that
include a meta.json file (as produced by the --dump-kv-dir flag). It pairs
Phase 1 "export" dumps with Phase 2 "import" dumps using the prefix hash and
length recorded in meta.json, then compares every k_layer*.bin / v_layer*.bin
file byte-for-byte. Any mismatches, missing files, or extra dumps are listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, List
import filecmp


@dataclass
class DumpEntry:
    key: Tuple[str, int]
    path: Path
    meta: Dict


def _load_entries(phase_dir: Path) -> Dict[Tuple[str, int], List[DumpEntry]]:
    entries: Dict[Tuple[str, int], List[DumpEntry]] = {}
    if not phase_dir.exists():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
    for child in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"[WARN] Skipping {child}: cannot read meta.json ({exc})", file=sys.stderr)
            continue
        prefix_hash = meta.get("prefix_hash")
        length = meta.get("length")
        if not prefix_hash or length is None:
            continue
        key = (prefix_hash, int(length))
        entries.setdefault(key, []).append(DumpEntry(key=key, path=child, meta=meta))
    return entries


def _select_entry(entries: List[DumpEntry], dump_kind: str) -> DumpEntry | None:
    filtered = [entry for entry in entries if entry.meta.get("dump_kind") == dump_kind]
    if not filtered:
        return None
    # Prefer the latest timestamp if we have multiple dumps for the same key.
    filtered.sort(key=lambda e: e.meta.get("timestamp_ms", 0), reverse=True)
    return filtered[0]


def _collect_layers(entry: DumpEntry) -> Dict[str, Path]:
    layers = {}
    for bin_file in entry.path.glob("*_layer*.bin"):
        layers[bin_file.name] = bin_file
    return layers


def compare_phase_dirs(phase1_dir: Path, phase2_dir: Path) -> int:
    phase1_entries = _load_entries(phase1_dir)
    phase2_entries = _load_entries(phase2_dir)
    all_keys = set(phase1_entries.keys()) | set(phase2_entries.keys())
    mismatch_count = 0

    for key in sorted(all_keys):
        exports = _select_entry(phase1_entries.get(key, []), dump_kind="export")
        imports = _select_entry(phase2_entries.get(key, []), dump_kind="import")

        if exports is None and imports is None:
            continue
        if exports is None:
            mismatch_count += 1
            print(f"[MISS] No export found for prefix={key[0]} len={key[1]} in {phase1_dir}")
            continue
        if imports is None:
            mismatch_count += 1
            print(f"[MISS] No import found for prefix={key[0]} len={key[1]} in {phase2_dir}")
            continue

        layers1 = _collect_layers(exports)
        layers2 = _collect_layers(imports)
        layer_names = set(layers1.keys()) | set(layers2.keys())
        for layer_name in sorted(layer_names):
            path1 = layers1.get(layer_name)
            path2 = layers2.get(layer_name)
            if path1 is None or path2 is None:
                mismatch_count += 1
                print(f"[MISS] {layer_name} missing in "
                      f"{exports.path if path1 is None else imports.path}")
                continue
            if not filecmp.cmp(path1, path2, shallow=False):
                mismatch_count += 1
                print(f"[DIFF] {layer_name} differs for prefix={key[0]} len={key[1]}")

    if mismatch_count == 0:
        print("✓ All matching layers are identical between phases.")
    else:
        print(f"✗ Found {mismatch_count} mismatches. See log above.")
    return mismatch_count


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KV dump directories across phases.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Root directory passed to --dump-kv-dir.")
    parser.add_argument("--phase1", type=str, default="run1_phase1",
                        help="Subdirectory name for Phase 1 exports (default: run1_phase1).")
    parser.add_argument("--phase2", type=str, default="run1_phase2",
                        help="Subdirectory name for Phase 2 imports (default: run1_phase2).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    phase1_dir = args.root / args.phase1
    phase2_dir = args.root / args.phase2
    return compare_phase_dirs(phase1_dir, phase2_dir)


if __name__ == "__main__":
    raise SystemExit(main())

