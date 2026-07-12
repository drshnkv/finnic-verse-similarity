#!/usr/bin/env python3
"""Build the flat cross-corpus verse-pairs CSV (data deposit item 06) from the
cluster-membership export (item 05).

For every cross-corpus cluster, emit one row per (Estonian member, Finnish
member) pair. The row count equals the sum over clusters of
(et_members * fi_members) = 120,590, the "cross-corpus verse pairs" headline.

The clusters overlap (a verse may belong to several), so the same
(et_vid, fi_vid) pair can recur across clusters; rows are unique on
(cluster_id, et_vid, fi_vid). Output is ID-only (no verse text).

Usage:
    python build_verse_pairs_csv.py \
        --membership ../../supplementary_data/05_cluster_membership/cl_cluster_membership.jsonl \
        --out       ../../supplementary_data/06_verse_pairs/cl_verse_pairs.csv
"""
import argparse
import csv
import json


def build(membership_path, out_path):
    n_rows = 0
    seen = set()
    with open(membership_path, encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["cluster_id", "size", "category", "et_vid", "et_place", "fi_vid", "fi_place"])
        for line in fin:
            cluster = json.loads(line)
            cid = cluster["cluster_id"]
            size = cluster["size"]
            category = cluster["category"]
            ets = [m for m in cluster["members"] if m["lang"] == "et"]
            fis = [m for m in cluster["members"] if m["lang"] == "fi"]
            for e in ets:
                for f in fis:
                    writer.writerow([cid, size, category,
                                     e["vid"], e.get("place", ""),
                                     f["vid"], f.get("place", "")])
                    seen.add((e["vid"], f["vid"]))
                    n_rows += 1
    return n_rows, len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--membership", required=True,
                    help="path to cl_cluster_membership.jsonl (data deposit item 05)")
    ap.add_argument("--out", required=True,
                    help="output CSV path (data deposit item 06)")
    args = ap.parse_args()
    n_rows, n_distinct = build(args.membership, args.out)
    print(f"rows (co-membership pairs): {n_rows}")
    print(f"distinct (et_vid, fi_vid) pairs: {n_distinct}")


if __name__ == "__main__":
    main()
