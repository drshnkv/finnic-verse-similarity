#!/usr/bin/env python
"""Cross-lingual headline + provenance + size/coverage stats for the article's
two-member-floor clustering. Reuses build_ws4_cross_lingual_5algo.py's exact
functions (valismaa override, char-unigram-Jaccard 0.4 threshold) so the cross-
lingual identification and form/meaning classification match the rest of the
pipeline.

Reads the canonical full-membership export (rrf_cluster_membership.jsonl.gz,
written by rrf_validation_stats.py at the two-member floor) plus the four-algorithm
baseline (loo_minus_sentence) for provenance. Reads everything from .gz; writes
ONLY into output/floor2_rederive/.

Emits the cross-lingual headline + size/coverage stats for the full 5-algorithm
membership, and provenance against the 4-algorithm baseline (loo_minus_sentence).
Reconciliation gate: cross-lingual total != 19,423 -> exit 2 (the validated floor-2
count this wrapper must reproduce).

Run from the repo root (build_ws4 self-inserts the repo root for valismaa_override
via __file__). Sequential, one process (RAM).
"""
import sys, os, gzip, json, time, statistics
from pathlib import Path

SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # similarity/ (two levels up)
OUT = os.path.join(SIM, "output")
RD = os.path.join(OUT, "floor2_rederive")
sys.path.insert(0, SIM)

import build_ws4_cross_lingual_5algo as ws4
from valismaa_override import fi_override_pids, et_override_pids

META_GZ = os.path.join(OUT, "verse_metadata.jsonl.gz")
FLOOR2_MEMB = os.path.join(OUT, "rrf_cluster_membership.jsonl.gz")          # canonical full 5-algo membership (two-member floor)
FOURALGO_MEMB = os.path.join(OUT, "rrf_cluster_membership_loo_minus_sentence.jsonl.gz")  # 4-algo baseline (drop sentence)

DENOM = 4316744          # fixed corpus verse total (coverage denominator; floor-independent)
EXPECT_FLOOR2 = 19423    # floor-2 cross-lingual cluster count; reconciliation gate


def load_meta():
    """Mirror ws4.load_verse_metadata() exactly, but read the .gz. Keep p (poem id) for the pair/poem stats."""
    fi_pids = set(fi_override_pids()); et_pids = set(et_override_pids())
    meta = {}; ov_fi = ov_et = 0; t0 = time.time()
    with gzip.open(META_GZ, "rt") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            lang = m.get("l", ""); pid = m.get("p", "")
            if lang == "et" and pid in fi_pids:
                lang = "fi"; ov_fi += 1
            elif lang == "fi" and pid in et_pids:
                lang = "et"; ov_et += 1
            meta[m["v"]] = {"l": lang, "t": m.get("t", ""), "p": pid}
            if (i + 1) % 1_000_000 == 0:
                print(f"  ... {i+1:,} verses", flush=True)
    print(f"  Loaded {len(meta):,} verses in {time.time()-t0:.1f}s (override {ov_fi} et->fi, {ov_et} fi->et)", flush=True)
    return meta


def _stats(vals):
    return {"mean": round(statistics.mean(vals), 2), "median": int(statistics.median(vals)),
            "min": min(vals), "max": max(vals)}


def derive_core(tag, memb_gz, meta, want_pairs=False):
    """All 5-algo-only numbers for one membership. Frees the big cluster list before returning."""
    print(f"\n=== {tag} ===", flush=True)
    clusters = ws4.load_clusters(Path(memb_gz), compressed=True)
    n_total = len(clusters)
    # all-cluster size stats + coverage
    all_sizes = [c["size"] for c in clusters]
    covered = set(); slots = 0
    for c in clusters:
        for vid in c["members"]:
            covered.add(vid); slots += 1
    coverage = round(100 * len(covered) / DENOM, 2)
    cpv = round(slots / len(covered), 3) if covered else 0.0
    # cross-lingual identification + analysis
    cl = ws4.identify_cross_lingual(clusters, meta)
    cl_total = len(cl)
    cl_sizes = [c["size"] for c in cl]
    analyzed = [ws4.analyze_cluster(c, meta, {}) for c in cl]   # gloss-independent for category/ratios
    weighted = ws4.compute_weighted_ratios(analyzed, 0.4)
    res = {
        "tag": tag,
        "n_clusters_total": n_total,
        "total_cross_lingual_clusters": cl_total,
        "category_counts": {"form_based": weighted["by_cluster"]["form_based"],
                            "meaning_based": weighted["by_cluster"]["meaning_based"]},
        "by_cluster_form_pct": weighted["by_cluster"]["form_pct"],
        "weighted_ratios": weighted,                          # by_cluster / by_size / by_pair_count
        "coverage_pct": coverage,
        "covered_verses": len(covered),
        "clusters_per_verse": cpv,
        "all_cluster_size": _stats(all_sizes),                # all clusters
        "cl_cluster_size": _stats(cl_sizes),                  # cross-lingual clusters
    }
    if want_pairs:
        # cross-corpus (undirected) ET-FI member pairs + unique ET / FI verses + unique poems spanned by CL clusters
        directed = sum(a["pair_count"] for a in analyzed)
        uet = set(); ufi = set(); poems = set()
        for c in cl:
            for vid in c["_et_vids"]:
                uet.add(vid); poems.add(meta[vid]["p"])
            for vid in c["_fi_vids"]:
                ufi.add(vid); poems.add(meta[vid]["p"])
        res["directed_pairs"] = directed
        res["unique_et_verses"] = len(uet)
        res["unique_fi_verses"] = len(ufi)
        res["unique_poems"] = len(poems)
    # keep cl entries (small) for provenance; drop the big list
    del clusters, all_sizes, covered
    return res, cl, analyzed


def main():
    meta = load_meta()
    out = {}

    # ---- full 5-algorithm derivation (two-member floor) ----
    fl, cl_fl, analyzed_fl = derive_core("floor2", FLOOR2_MEMB, meta, want_pairs=True)

    # reconciliation gate
    if fl["total_cross_lingual_clusters"] != EXPECT_FLOOR2:
        print(f"\nFATAL: floor2 CL total {fl['total_cross_lingual_clusters']:,} != expected "
              f"{EXPECT_FLOOR2:,} (wrapper diverged from validated measure_crosslingual.py)", flush=True)
        out["floor2"] = fl
        Path(os.path.join(RD, "derive_ws4_floor2.json")).write_text(
            json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(2)

    # provenance against 4-algo floor2 baseline (loo_minus_sentence)
    if os.path.exists(FOURALGO_MEMB):
        clusters_4 = ws4.load_clusters(Path(FOURALGO_MEMB), compressed=True)
        cl_4 = ws4.identify_cross_lingual(clusters_4, meta)
        fouralgo_cl_count = len(cl_4)
        # 4-algo baseline coverage
        cov4 = set()
        for c in clusters_4:
            for vid in c["members"]:
                cov4.add(vid)
        fouralgo_coverage = round(100 * len(cov4) / DENOM, 2)
        provenance = ws4.compute_provenance(cl_fl, meta, clusters_4)
        from collections import Counter
        prov_counts = dict(Counter(provenance.values()))
        prov_ratios = ws4.provenance_ratios(analyzed_fl, provenance, 0.4)
        increase_pct = round(100 * (fl["total_cross_lingual_clusters"] - fouralgo_cl_count) / fouralgo_cl_count, 1)
        fl["provenance"] = {
            "fouralgo_floor2_cl_count": fouralgo_cl_count,
            "fouralgo_floor2_coverage_pct": fouralgo_coverage,
            "fivealgo_increase_over_fouralgo_pct": increase_pct,
            "counts": prov_counts,                                  # novel / retained / expanded
            "ratios_at_0.4": prov_ratios,                           # per-provenance form/meaning ratios
        }
        del clusters_4, cl_4, cov4
    else:
        fl["provenance"] = {"error": f"4-algo baseline not found: {FOURALGO_MEMB} (produce the loo_minus_sentence membership first)"}

    out["floor2"] = fl
    outpath = os.path.join(RD, "derive_ws4_floor2.json")
    Path(outpath).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved: {outpath}", flush=True)

    # Emit the floor2 analyzed array as {"clusters": [...]} with cluster_id per entry
    # (the per-cluster cross-lingual analysis returned by ws4.analyze_cluster) so the
    # downstream geography / song-type aggregation can consume it directly.
    ce_path = os.path.join(RD, "rrf_ws4_cross_lingual_5algo.json")
    Path(ce_path).write_text(json.dumps(
        {"total_cross_lingual_clusters": fl["total_cross_lingual_clusters"],
         "clusters": analyzed_fl}, ensure_ascii=False))
    print(f"Saved (geography/song-type input): {ce_path}", flush=True)
    print(f"\n--- floor2 headline ---", flush=True)
    print(f"  total clusters: {fl['n_clusters_total']:,}", flush=True)
    print(f"  CL clusters: {fl['total_cross_lingual_clusters']:,} "
          f"({fl['by_cluster_form_pct']}% form)", flush=True)
    print(f"  form/meaning: {fl['category_counts']['form_based']:,} / "
          f"{fl['category_counts']['meaning_based']:,}", flush=True)
    print(f"  by_size form_pct: {fl['weighted_ratios']['by_size']['form_pct']}  "
          f"by_pair_count form_pct: {fl['weighted_ratios']['by_pair_count']['form_pct']}", flush=True)
    print(f"  coverage: {fl['coverage_pct']}%  ({fl['covered_verses']:,} / {DENOM:,})  "
          f"clusters/verse: {fl['clusters_per_verse']}", flush=True)
    if "directed_pairs" in fl:
        print(f"  cross-corpus (undirected) member pairs: {fl['directed_pairs']:,}  ET: {fl['unique_et_verses']:,}  "
              f"FI: {fl['unique_fi_verses']:,}  poems: {fl['unique_poems']:,}", flush=True)
    if "counts" in fl.get("provenance", {}):
        p = fl["provenance"]
        print(f"  provenance: {p['counts']}  4algo_cl={p['fouralgo_floor2_cl_count']:,} "
              f"(+{p['fivealgo_increase_over_fouralgo_pct']}%)  "
              f"4algo_cov={p['fouralgo_floor2_coverage_pct']}%  "
              f"novel_form_pct={p['ratios_at_0.4'].get('novel', {}).get('form_pct')}", flush=True)


if __name__ == "__main__":
    main()
