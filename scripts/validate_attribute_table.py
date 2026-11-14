"""Validate `configs/finegym_attribute_table_v2.json`.

Checks:
  1. Entry count = 99 c-actions, ids c00..c98 contiguous.
  2. K consistent across all rows == len(attribute_names).
  3. All bits are 0 or 1.
  4. All 99 attribute vectors are unique (no two leaves collide).
  5. No "dead" attributes (every attribute has at least 1 active class
     AND at least 1 inactive class - so the bit carries information).
  6. Sparsity sanity: each leaf has between 3 and 18 active bits;
     report mean / min / max.
  7. `held_out_classes` is a list, all entries are valid c-action ids.

Exits non-zero if any hard check fails. Soft warnings print to stderr.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

ROOT = "/data3/xiaodan8/actionformer6"
PATH = os.path.join(ROOT, "configs/finegym_attribute_table_v2.json")


def _fail(msg: str):
    print(f"FAIL: {msg}", file=sys.stderr)


def main() -> int:
    if not os.path.exists(PATH):
        _fail(f"missing file: {PATH}")
        return 2

    with open(PATH) as f:
        data = json.load(f)

    errors = 0
    warnings = 0

    attribute_names = data.get("attribute_names")
    actions = data.get("actions", {})
    held_out = data.get("held_out_classes", [])

    if not isinstance(attribute_names, list) or not attribute_names:
        _fail("attribute_names must be a non-empty list")
        return 2
    K = len(attribute_names)
    print(f"INFO: K = {K} attributes")
    print(f"INFO: attribute_names = {attribute_names}")

    # 1) entry count + contiguous ids
    if len(actions) != 99:
        _fail(f"expected 99 actions, got {len(actions)}")
        errors += 1
    expected_ids = [f"c{i:02d}" for i in range(99)]
    missing = [k for k in expected_ids if k not in actions]
    extra = [k for k in actions if k not in expected_ids]
    if missing:
        _fail(f"missing action ids: {missing[:5]}... ({len(missing)} total)")
        errors += 1
    if extra:
        _fail(f"unexpected action ids: {extra[:5]}... ({len(extra)} total)")
        errors += 1

    # 2) K consistency + 3) binary
    bad_len = []
    bad_bits = []
    for k, rec in actions.items():
        v = rec.get("attrs")
        if not isinstance(v, list):
            bad_len.append((k, "not list"))
            continue
        if len(v) != K:
            bad_len.append((k, f"len={len(v)}"))
            continue
        if any(b not in (0, 1) for b in v):
            bad_bits.append(k)
    if bad_len:
        _fail(f"actions with wrong attrs length: {bad_len[:5]} ({len(bad_len)} total)")
        errors += 1
    if bad_bits:
        _fail(f"actions with non-binary entries: {bad_bits[:5]} ({len(bad_bits)} total)")
        errors += 1

    # 4) uniqueness
    vec_to_ids: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for k in expected_ids:
        if k not in actions:
            continue
        v = actions[k].get("attrs")
        if isinstance(v, list) and len(v) == K:
            vec_to_ids[tuple(v)].append(k)
    dup_groups = [ids for ids in vec_to_ids.values() if len(ids) > 1]
    if dup_groups:
        _fail(f"{len(dup_groups)} duplicate vector group(s):")
        for grp in dup_groups:
            names = ", ".join(f"{kid}({actions[kid]['name'][:50]})" for kid in grp)
            print(f"   - {grp} :: {names}", file=sys.stderr)
        errors += 1
    else:
        print("OK: all 99 vectors unique")

    # 5) dead attributes
    if not bad_len and not bad_bits:
        col_counts = [0] * K
        for k in expected_ids:
            v = actions[k]["attrs"]
            for j, b in enumerate(v):
                col_counts[j] += b
        N = 99
        dead_zero = [j for j, c in enumerate(col_counts) if c == 0]
        dead_one = [j for j, c in enumerate(col_counts) if c == N]
        if dead_zero:
            _fail(f"dead attributes (always 0): {[attribute_names[j] for j in dead_zero]}")
            errors += 1
        if dead_one:
            _fail(f"dead attributes (always 1): {[attribute_names[j] for j in dead_one]}")
            errors += 1
        print("INFO: per-attribute active counts:")
        for j, c in enumerate(col_counts):
            print(f"   {j:2d} {attribute_names[j]:25s} {c:3d}/{N} active")

    # 6) sparsity per leaf
    if not bad_len and not bad_bits:
        per_leaf = [sum(actions[k]["attrs"]) for k in expected_ids]
        mean = sum(per_leaf) / len(per_leaf)
        mn = min(per_leaf)
        mx = max(per_leaf)
        hist = Counter(per_leaf)
        print(f"INFO: bits-per-leaf  mean={mean:.2f}  min={mn}  max={mx}")
        print(f"INFO: bits-per-leaf histogram: {dict(sorted(hist.items()))}")
        soft_lo, soft_hi = 3, 18
        out_of_band = [
            (k, sum(actions[k]["attrs"])) for k in expected_ids
            if sum(actions[k]["attrs"]) < soft_lo or sum(actions[k]["attrs"]) > soft_hi
        ]
        if out_of_band:
            print(f"WARN: {len(out_of_band)} leaves outside [{soft_lo}, {soft_hi}] active bits:", file=sys.stderr)
            for kid, cnt in out_of_band[:10]:
                print(f"   {kid}: {cnt} bits - {actions[kid]['name'][:60]}", file=sys.stderr)
            warnings += 1

    # 7) held_out_classes structure
    if not isinstance(held_out, list):
        _fail("held_out_classes must be a list")
        errors += 1
    else:
        for h in held_out:
            if h not in actions:
                _fail(f"held_out_classes references unknown id: {h}")
                errors += 1
        print(f"INFO: held_out_classes = {held_out}")

    print()
    print(f"errors={errors}  warnings={warnings}")
    if errors:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
