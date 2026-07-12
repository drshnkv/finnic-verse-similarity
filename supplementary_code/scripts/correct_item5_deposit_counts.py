#!/usr/bin/env python3
"""Align the language columns of supplementary_data/04_form_meaning_classification/.

`rrf_ws4_cross_lingual_5algo.json` carries a per-cluster et_count / fi_count /
pair_count. Those three columns must be derived with the entity-safe verse-ID
join: some verse IDs store HTML entities (e.g. the degree sign &#xB0;) where the
verse metadata stores the decoded character, so a raw string join silently fails
to resolve 639 Estonian verses. Four single-distinct-word verses (3 Estonian,
1 Finnish) additionally resolve at poem level rather than being left unknown.

This script rewrites ONLY those three columns, sourcing et_count / fi_count from
supplementary_data/05_cluster_membership/cl_clusters_index.csv (produced by
build_cl_cluster_membership_export.py, which performs the entity-safe join).
Every other field — size, the four Jaccard scores, category, the example verses,
shared_translations — is left untouched, and the cluster set, the per-cluster
size and the 18,376 / 1,047 form/meaning split are asserted unchanged. The
resulting cross-corpus (undirected) member-pair total is 120,590.

The raw pipeline output under the build tree is intentionally NOT modified;
folder 05's builder re-resolves languages from the metadata and is the source of
truth for these counts.

Idempotent: re-running on an already-aligned file is a no-op (asserts still pass).
"""
import csv
import json
import os

# .../<deposit>/supplementary_code/scripts/ -> <deposit>/
DEPOSIT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEP = os.path.join(DEPOSIT_ROOT, "supplementary_data")
ITEM5 = os.path.join(DEP, "04_form_meaning_classification", "rrf_ws4_cross_lingual_5algo.json")
INDEX5 = os.path.join(DEP, "05_cluster_membership", "cl_clusters_index.csv")

EXPECT = dict(clusters=19423, form=18376, meaning=1047,
              et_slots=60647, fi_slots=49811, pairs=120590)


def main():
    # corrected per-cluster resolution (entity-safe) from folder 05
    corr = {}
    with open(INDEX5, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            corr[int(r["cluster_id"])] = (int(r["size"]), int(r["et_count"]), int(r["fi_count"]))

    with open(ITEM5, encoding="utf-8") as fh:
        doc = json.load(fh)
    clusters = doc["clusters"]
    assert len(clusters) == EXPECT["clusters"], len(clusters)
    assert set(c["cluster_id"] for c in clusters) == set(corr), "cluster set mismatch"

    before_pairs = sum(c["pair_count"] for c in clusters)
    cats0 = {}
    for c in clusters:
        cats0[c["category"]] = cats0.get(c["category"], 0) + 1

    changed = 0
    for c in clusters:
        size5, et5, fi5 = corr[c["cluster_id"]]
        assert size5 == c["size"], f"size drift cid {c['cluster_id']}: {c['size']} vs {size5}"
        new_pair = et5 * fi5
        if (c["et_count"], c["fi_count"], c["pair_count"]) != (et5, fi5, new_pair):
            changed += 1
        c["et_count"], c["fi_count"], c["pair_count"] = et5, fi5, new_pair

    # post-correction invariants
    assert sum(c["et_count"] for c in clusters) == EXPECT["et_slots"]
    assert sum(c["fi_count"] for c in clusters) == EXPECT["fi_slots"]
    assert sum(c["pair_count"] for c in clusters) == EXPECT["pairs"]
    assert all(c["pair_count"] == c["et_count"] * c["fi_count"] for c in clusters)
    cats1 = {}
    for c in clusters:
        cats1[c["category"]] = cats1.get(c["category"], 0) + 1
    assert cats1 == cats0, "category split changed"
    assert cats1.get("form_based") == EXPECT["form"] and cats1.get("meaning_based") == EXPECT["meaning"]

    tmp = ITEM5 + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)   # default separators -> matches original compact style
    os.replace(tmp, ITEM5)

    print(f"item5 corrected: {changed:,} clusters had stale et/fi/pair columns rewritten")
    print(f"  sum pair_count : {before_pairs:,} -> {EXPECT['pairs']:,}")
    print(f"  sum et_count   : -> {EXPECT['et_slots']:,}   sum fi_count -> {EXPECT['fi_slots']:,}")
    print(f"  form/meaning   : {cats1['form_based']:,} / {cats1['meaning_based']:,} (unchanged)")
    print(f"  size column    : unchanged for all {len(clusters):,} clusters")


if __name__ == "__main__":
    main()
