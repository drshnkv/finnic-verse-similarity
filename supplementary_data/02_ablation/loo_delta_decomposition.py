#!/usr/bin/env python
"""Decompose the leave-one-out ablation's net cluster-count change (Table 3 /
phase_c_table7.json, floor-2 experimental configs) into interpretable parts.

Question being answered (the article's ablation paragraph): why does
removing Jaccard LOWER the total cluster count by 45,031 while removing any
of the other four algorithms RAISES it (e.g. translation-pivot +49,932)?

Clusters are overlapping neighborhoods (cluster_verses_rrf.cluster_by_
neighborhoods: greedy, join iff neighborhood-vs-cluster Jaccard >= 0.5, else
seed a new cluster from the verse's whole neighborhood; floor 2 afterwards).
A verse belongs to ~2.25 clusters on average, so the accounting below is
co-habitation-based rather than a partition mapping.

For each LOO config, two passes against the 5-algo floor-2 baseline:

Baseline-cluster fate (for every baseline cluster c):
  - survivors   = members of c that appear in ANY LOO cluster
  - k2(c)       = number of distinct LOO clusters containing >= 2 members of c
  fate classes:
  - dissolved   : < 2 survivors (the group cannot exist at floor 2)
  - scattered   : >= 2 survivors but k2 == 0 (members live on, group gone)
  - preserved   : k2 == 1
  - fragmented  : k2 >= 2 ("pieces"); surplus = sum(k2 - 1) over fragmented
  Size stats are kept per fate class (is dissolution concentrated in the
  2-4 member tail?).

LOO-cluster origin (for every LOO cluster d):
  - piece       : some baseline cluster holds >= 2 of d's members (d continues
                  an existing grouping, possibly one of several pieces)
  - regrouped   : no 2 members of d co-habited any baseline cluster, but all
                  (>=2) were baseline-clustered somewhere (novel combination)
  - with_new    : d contains verses that were unclustered at baseline

Output: loo_delta_decomposition.json (checkpointed after each config) and a
printed per-config summary. Read-only over the membership .jsonl.gz files.
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent  # similarity/output
RD = Path(__file__).resolve().parent
RESULT_PATH = RD / "loo_delta_decomposition.json"

BASELINE = "floor2"
CONFIGS = ["loo_minus_jaccard", "loo_minus_translation", "loo_minus_tfidf",
           "loo_minus_charbigram", "loo_minus_sentence"]


def membership_path(tag):
    return OUT_DIR / f"rrf_cluster_membership_{tag}_experimental.jsonl.gz"


class Interner:
    def __init__(self):
        self.map = {}

    def get(self, s):
        m = self.map
        i = m.get(s)
        if i is None:
            i = len(m)
            m[s] = i
        return i


def load_membership(tag, interner):
    """Return (offsets, member_ids, csr) for one config.

    offsets/member_ids: cluster ci -> member_ids[offsets[ci]:offsets[ci+1]]
    csr: (sorted_verse_ids, cluster_ids_by_verse, starts) for verse->clusters
    """
    path = membership_path(tag)
    t0 = time.time()
    verse_ids = []
    offsets = [0]
    with gzip.open(path, "rt") as f:
        for line in f:
            o = json.loads(line)
            for m in o["members"]:
                verse_ids.append(interner.get(m))
            offsets.append(len(verse_ids))
    member_ids = np.asarray(verse_ids, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)
    n_clusters = len(offsets) - 1

    # verse -> clusters CSR
    cluster_of_entry = np.repeat(np.arange(n_clusters, dtype=np.int32),
                                 np.diff(offsets))
    order = np.argsort(member_ids, kind="stable")
    sorted_verses = member_ids[order]
    clusters_by_verse = cluster_of_entry[order]
    print(f"  [{tag}] {n_clusters:,} clusters, {len(member_ids):,} entries, "
          f"{len(np.unique(sorted_verses)):,} distinct verses "
          f"({time.time()-t0:.0f}s)", flush=True)
    return offsets, member_ids, (sorted_verses, clusters_by_verse)


def verse_lookup(csr, vid):
    sorted_verses, clusters_by_verse = csr
    lo = np.searchsorted(sorted_verses, vid, side="left")
    hi = np.searchsorted(sorted_verses, vid, side="right")
    return clusters_by_verse[lo:hi]


def size_stats(sizes):
    if not sizes:
        return {"n": 0}
    a = np.asarray(sizes)
    return {"n": int(len(a)), "mean_size": round(float(a.mean()), 2),
            "median_size": float(np.median(a)),
            "share_size_2_4": round(float((a <= 4).mean()), 4)}


def main():
    interner = Interner()
    print("Loading baseline (5-algo floor-2)...", flush=True)
    b_off, b_mem, b_csr = load_membership(BASELINE, interner)
    n_base = len(b_off) - 1
    base_clustered = np.zeros(0, dtype=bool)  # grown lazily below

    def clustered_mask(csr, n_verses):
        mask = np.zeros(n_verses, dtype=bool)
        mask[csr[0]] = True
        return mask

    results = {"baseline_clusters": n_base,
               "baseline_membership_entries": int(len(b_mem)),
               "configs": {}}
    if RESULT_PATH.exists():
        try:
            prev = json.loads(RESULT_PATH.read_text())
            if prev.get("baseline_clusters") == n_base:
                results["configs"].update(prev.get("configs", {}))
                print(f"Resuming: {sorted(results['configs'])} already done",
                      flush=True)
        except Exception:
            pass

    for tag in CONFIGS:
        if tag in results["configs"]:
            continue
        print(f"\n=== {tag} ===", flush=True)
        l_off, l_mem, l_csr = load_membership(tag, interner)
        n_loo = len(l_off) - 1
        n_verses = len(interner.map)
        base_mask = clustered_mask(b_csr, n_verses)
        loo_mask = clustered_mask(l_csr, n_verses)

        # ---- baseline-cluster fate ----
        t0 = time.time()
        fate_counts = {"dissolved": 0, "scattered": 0, "preserved": 0,
                       "fragmented": 0}
        fate_sizes = {k: [] for k in fate_counts}
        surplus_pieces = 0
        lost_groups_members = 0  # members of dissolved+scattered clusters
        for ci in range(n_base):
            members = b_mem[b_off[ci]:b_off[ci + 1]]
            survivors = members[loo_mask[members]]
            if len(survivors) < 2:
                fate = "dissolved"
            else:
                counts = {}
                for vid in survivors:
                    for d in verse_lookup(l_csr, vid):
                        counts[d] = counts.get(d, 0) + 1
                k2 = sum(1 for v in counts.values() if v >= 2)
                if k2 == 0:
                    fate = "scattered"
                elif k2 == 1:
                    fate = "preserved"
                else:
                    fate = "fragmented"
                    surplus_pieces += k2 - 1
            fate_counts[fate] += 1
            fate_sizes[fate].append(len(members))
            if fate in ("dissolved", "scattered"):
                lost_groups_members += len(members)
            if ci % 300_000 == 0 and ci:
                print(f"    fate pass {ci:,}/{n_base:,} ({time.time()-t0:.0f}s)",
                      flush=True)
        print(f"  fate pass done ({time.time()-t0:.0f}s)", flush=True)

        # ---- LOO-cluster origin ----
        t0 = time.time()
        origin_counts = {"piece": 0, "regrouped": 0, "with_new": 0}
        for di in range(n_loo):
            members = l_mem[l_off[di]:l_off[di + 1]]
            if not base_mask[members].all():
                origin_counts["with_new"] += 1
                continue
            counts = {}
            piece = False
            for vid in members:
                for c in verse_lookup(b_csr, vid):
                    n = counts.get(c, 0) + 1
                    counts[c] = n
                    if n >= 2:
                        piece = True
                        break
                if piece:
                    break
            origin_counts["piece" if piece else "regrouped"] += 1
            if di % 300_000 == 0 and di:
                print(f"    origin pass {di:,}/{n_loo:,} ({time.time()-t0:.0f}s)",
                      flush=True)
        print(f"  origin pass done ({time.time()-t0:.0f}s)", flush=True)

        entry = {
            "loo_clusters": n_loo,
            "delta_clusters": n_loo - n_base,
            "verses_clustered_baseline": int(base_mask.sum()),
            "verses_clustered_loo": int(loo_mask.sum()),
            "verses_dropped": int((base_mask & ~loo_mask).sum()),
            "verses_gained": int((~base_mask & loo_mask).sum()),
            "baseline_cluster_fate": fate_counts,
            "fate_size_stats": {k: size_stats(v) for k, v in fate_sizes.items()},
            "fragmentation_surplus_pieces": int(surplus_pieces),
            "groups_lost": fate_counts["dissolved"] + fate_counts["scattered"],
            "loo_cluster_origin": origin_counts,
        }
        results["configs"][tag] = entry
        RESULT_PATH.write_text(json.dumps(results, indent=2))
        print(f"  checkpoint written -> {RESULT_PATH.name}", flush=True)

        f = fate_counts
        print(f"  SUMMARY {tag}: delta={entry['delta_clusters']:+,} | "
              f"dissolved={f['dissolved']:,} scattered={f['scattered']:,} "
              f"preserved={f['preserved']:,} fragmented={f['fragmented']:,} "
              f"(surplus pieces={surplus_pieces:,}) | "
              f"origin: piece={origin_counts['piece']:,} "
              f"regrouped={origin_counts['regrouped']:,} "
              f"with_new={origin_counts['with_new']:,}", flush=True)

        del l_off, l_mem, l_csr, loo_mask

    print("\nAll configs done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
