#!/usr/bin/env python
"""Leave-one-out ablation (CL count + coverage per removed algorithm), floor2.

ONE python process: load the 4.3M-verse metadata ONCE, then loop the 5 loo_* memberships
SEQUENTIALLY (load -> count -> free -> next). Reads .gz only; writes only to
output/floor2_rederive/. Run from the repo root.

Ablation delta = full-5-algo floor2 (CL count + coverage) MINUS each leave-one-out ->
clusters lost and coverage lost per algorithm, quantifying each algorithm's marginal
contribution (e.g. the cost of removing translation-pivot).

Also computes the min_algos=3 comparison (stricter consensus vs. the min_algos=2 floor2
default): how many cross-corpus clusters the stricter consensus costs.
"""
import sys, os, gzip, json, time
from pathlib import Path

SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # similarity/ (two levels up)
OUT = os.path.join(SIM, "output")
RD = os.path.join(OUT, "floor2_rederive")
sys.path.insert(0, SIM)

import build_ws4_cross_lingual_5algo as ws4
from valismaa_override import fi_override_pids, et_override_pids

META_GZ = os.path.join(OUT, "verse_metadata.jsonl.gz")
DENOM = 4316744

FULL = os.path.join(OUT, "rrf_cluster_membership.jsonl.gz")  # canonical full 5-algo membership (two-member floor)
LOO = {
    "minus_sentence":   os.path.join(OUT, "rrf_cluster_membership_loo_minus_sentence.jsonl.gz"),
    "minus_jaccard":    os.path.join(OUT, "rrf_cluster_membership_loo_minus_jaccard.jsonl.gz"),
    "minus_tfidf":      os.path.join(OUT, "rrf_cluster_membership_loo_minus_tfidf.jsonl.gz"),
    "minus_charbigram": os.path.join(OUT, "rrf_cluster_membership_loo_minus_charbigram.jsonl.gz"),
    "minus_translation": os.path.join(OUT, "rrf_cluster_membership_loo_minus_translation.jsonl.gz"),
}


def load_meta():
    fi_pids = set(fi_override_pids()); et_pids = set(et_override_pids())
    meta = {}; t0 = time.time()
    with gzip.open(META_GZ, "rt") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            lang = m.get("l", ""); pid = m.get("p", "")
            if lang == "et" and pid in fi_pids: lang = "fi"
            elif lang == "fi" and pid in et_pids: lang = "et"
            meta[m["v"]] = {"l": lang, "t": m.get("t", "")}
            if (i + 1) % 1_000_000 == 0: print(f"  ... {i+1:,}", flush=True)
    print(f"  Loaded {len(meta):,} verses in {time.time()-t0:.1f}s", flush=True)
    return meta


def count_one(tag, memb_gz, meta):
    if not os.path.exists(memb_gz):
        print(f"  [{tag}] MISSING {os.path.basename(memb_gz)}", flush=True)
        return None
    t0 = time.time()
    clusters = ws4.load_clusters(Path(memb_gz), compressed=True)
    covered = set()
    for c in clusters:
        for vid in c["members"]:
            covered.add(vid)
    cl = ws4.identify_cross_lingual(clusters, meta)
    cl_count = len(cl)
    coverage = round(100 * len(covered) / DENOM, 2)
    res = {"tag": tag, "cl_clusters": cl_count, "coverage_pct": coverage,
           "covered_verses": len(covered), "n_clusters_total": len(clusters)}
    print(f"  [{tag}] CL={cl_count:,}  coverage={coverage}%  ({time.time()-t0:.1f}s)", flush=True)
    del clusters, covered, cl
    return res


def main():
    meta = load_meta()
    print("\n=== full 5-algo floor2 (reference) ===", flush=True)
    full = count_one("full_5algo", FULL, meta)
    rows = {}
    for tag, path in LOO.items():
        print(f"\n=== leave-one-out: {tag} ===", flush=True)
        r = count_one(tag, path, meta)
        if r and full:
            r["clusters_lost"] = full["cl_clusters"] - r["cl_clusters"]
            r["coverage_lost_pct"] = round(full["coverage_pct"] - r["coverage_pct"], 2)
        rows[tag] = r
    # count part: min_algos=3 (stricter consensus) vs. the floor2 default (min_algos=2 consensus,
    # 2-member size floor). NB: min_algos (# of algorithms that must agree) is distinct from the
    # 2-member size floor. "min_algos=3 costs N cross-corpus clusters."
    print("\n=== min_algos=3 (all 5 algos, stricter consensus) ===", flush=True)
    minalgos3_path = os.path.join(OUT, "rrf_cluster_membership_minalgos3.jsonl.gz")
    ma3 = count_one("minalgos3", minalgos3_path, meta)
    if ma3 and full:
        ma3["clusters_lost_vs_full"] = full["cl_clusters"] - ma3["cl_clusters"]
        ma3["coverage_lost_pct_vs_full"] = round(full["coverage_pct"] - ma3["coverage_pct"], 2)

    out = {"full_5algo": full, "leave_one_out": rows, "minalgos3": ma3}
    outpath = os.path.join(RD, "phase_c_table7.json")
    Path(outpath).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved: {outpath}", flush=True)
    print("\n--- Leave-one-out ablation (floor2): clusters lost / coverage lost by removed algorithm ---", flush=True)
    for tag, r in rows.items():
        if r:
            print(f"  {tag:18s}  CL={r['cl_clusters']:,}  lost={r.get('clusters_lost'):,}  "
                  f"cov={r['coverage_pct']}%  cov_lost={r.get('coverage_lost_pct')}", flush=True)


if __name__ == "__main__":
    main()
