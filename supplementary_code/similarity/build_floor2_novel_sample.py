#!/usr/bin/env python
"""Build the floor-2 sample of NOVEL cross-lingual clusters for the precision review.

The article screens 100 novel clusters (novel = absent from the four-algorithm
baseline) for cross-lingual parallelism; this script draws that sample at the
2-member clustering floor used throughout the deposit:
  * cross-lingual = cluster has >=1 'et' AND >=1 'fi' member (langs via valismaa override).
  * novel         = best member-set Jaccard < 0.3 against ANY 4-algo (no_sentence) CL cluster.
  * sample        = size-stratified, seed 42: sort novel desc by size, split at median,
                    draw 50 from the top half + 50 from the bottom half (the size-stratified 50+50 split used when the novel count exceeds 500).
  * enrichment    = up to 5 ET + up to 5 FI verses (vid, raw text, lang, parish); NO glosses.

GATE: the novel count MUST equal 8,363 (the value compute_provenance() wrote into
derive_ws4_floor2.json). An exact match confirms this sampler reproduces the
upstream novel-cluster definition. Exits 2 on mismatch.

Run from anywhere; reads .gz inputs, writes into output/floor2_rederive/.
"""
import sys, os, gzip, json, time
import numpy as np
from pathlib import Path

SIM = os.path.dirname(os.path.abspath(__file__))  # this script lives in similarity/
ROOT = os.path.dirname(SIM)   # repo root holds valismaa_override.py
OUT = os.path.join(SIM, "output")
RD = os.path.join(OUT, "floor2_rederive")
sys.path.insert(0, SIM)
sys.path.insert(0, ROOT)

from valismaa_override import fi_override_pids, et_override_pids

META_GZ = os.path.join(OUT, "verse_metadata.jsonl.gz")
FLOOR2_MEMB = os.path.join(OUT, "rrf_cluster_membership.jsonl.gz")          # canonical full 5-algo membership (two-member floor)
FOURALGO_MEMB = os.path.join(OUT, "rrf_cluster_membership_loo_minus_sentence.jsonl.gz")  # 4-algo baseline (drop sentence)

OUTPUT_PATH = os.path.join(RD, "rrf_5algo_new_cl_clusters_sample_floor2.json")
EXPECT_NOVEL = 8363   # gate: derive_ws4_floor2.json provenance.counts.novel


def load_meta():
    """verse_id -> {l, t, pl}. Mirrors derive_ws4_floor2.load_meta()'s valismaa override
    (keeps 'pl' parish for the enrichment, which derive_ws4 drops)."""
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
            meta[m["v"]] = {"l": lang, "t": m.get("t", ""), "pl": m.get("pl", [])}
            if (i + 1) % 1_000_000 == 0:
                print(f"  ... {i+1:,} verses", flush=True)
    print(f"  Loaded {len(meta):,} verses in {time.time()-t0:.1f}s "
          f"(override {ov_fi} et->fi, {ov_et} fi->et)", flush=True)
    return meta


def load_cl_clusters(path, verse_meta):
    """Load cross-lingual clusters from the membership .gz.
    Returns CL clusters in file order with members as a set."""
    clusters = []
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            members = set(entry['members'])
            langs = set()
            for vid in members:
                m = verse_meta.get(vid, {})
                langs.add(m.get('l', 'other'))
            if 'et' in langs and 'fi' in langs:
                clusters.append({
                    "cluster_id": entry['cluster_id'],
                    "members": members,
                    "size": entry['size'],
                })
    return clusters


def best_jaccard_bruteforce(members5, cl_nosent):
    """Max member-set Jaccard of a 5-algo cluster vs every 4-algo CL cluster."""
    best = 0.0
    for cn in cl_nosent:
        shared = len(members5 & cn['members'])
        union = len(members5 | cn['members'])
        if union > 0:
            j = shared / union
            if j > best:
                best = j
    return best


def main():
    meta = load_meta()

    print("\nLoading floor-2 memberships (CL only)...", flush=True)
    t0 = time.time()
    cl_5algo = load_cl_clusters(FLOOR2_MEMB, meta)
    cl_nosent = load_cl_clusters(FOURALGO_MEMB, meta)
    print(f"  5-algo CL clusters:      {len(cl_5algo):,}", flush=True)
    print(f"  no_sentence CL clusters: {len(cl_nosent):,}", flush=True)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Inverted index over 4-algo members -> list of 4-algo cluster indices (exact-equivalent
    # fast path for the O(n^2) brute force: clusters sharing 0 members have Jaccard 0).
    print("\nBuilding inverted index + computing novelty...", flush=True)
    t0 = time.time()
    inv = {}
    for idx, cn in enumerate(cl_nosent):
        for vid in cn['members']:
            inv.setdefault(vid, []).append(idx)

    new_cl = []
    for c5 in cl_5algo:
        members5 = c5['members']
        cand = set()
        for vid in members5:
            postings = inv.get(vid)
            if postings:
                cand.update(postings)
        best = 0.0
        for idx in cand:
            cn = cl_nosent[idx]
            shared = len(members5 & cn['members'])
            union = len(members5 | cn['members'])
            if union > 0:
                j = shared / union
                if j > best:
                    best = j
        if best < 0.3:
            c5['best_4algo_jaccard'] = round(best, 3)
            new_cl.append(c5)
    print(f"  New CL clusters (Jaccard < 0.3 with any 4-algo): {len(new_cl):,} "
          f"(computed in {time.time()-t0:.1f}s)", flush=True)

    # --- GATE: must equal the provenance novel count (8,363) ---
    if len(new_cl) != EXPECT_NOVEL:
        print(f"\nFATAL: novel count {len(new_cl):,} != expected {EXPECT_NOVEL:,} "
              f"(does not match derive_ws4_floor2.json provenance.counts.novel). "
              f"Sampler is NOT a faithful replication -- investigate before screening.",
              flush=True)
        sys.exit(2)
    print(f"  GATE PASS: novel count == {EXPECT_NOVEL:,}", flush=True)

    # Sample for review: size-stratified, seed 42 (50+50 split, applied when the novel count exceeds 500; 8,363 > 500).
    new_cl.sort(key=lambda c: -c['size'])
    half = len(new_cl) // 2
    rng = np.random.default_rng(42)
    top_idx = rng.choice(half, min(50, half), replace=False)
    bot_idx = rng.choice(range(half, len(new_cl)),
                         min(50, len(new_cl) - half), replace=False)
    sample = [new_cl[i] for i in sorted(top_idx)] + \
             [new_cl[i] for i in sorted(bot_idx)]
    print(f"\n  Sampled {len(sample)} clusters (50 top-half + 50 bottom-half, seed 42)", flush=True)

    # Cross-check: brute-force best_jaccard for the SAMPLED clusters == inverted-index value.
    for c in sample:
        bf = round(best_jaccard_bruteforce(c['members'], cl_nosent), 3)
        if bf != c['best_4algo_jaccard']:
            print(f"FATAL: best_jaccard mismatch on cluster {c['cluster_id']}: "
                  f"inv={c['best_4algo_jaccard']} vs brute={bf}", flush=True)
            sys.exit(2)
    print("  Cross-check PASS: inverted-index best_jaccard == brute force on all 100 sampled", flush=True)

    # Enrich each sampled cluster with up to 5 ET + 5 FI verses (no glosses).
    enriched = []
    for c in sample:
        et_verses = []
        fi_verses = []
        for vid in sorted(c['members']):
            m = meta.get(vid, {})
            entry = {"vid": vid, "text": m.get('t', ''), "lang": m.get('l', ''),
                     "parish": m.get('pl', [])}
            if m.get('l') == 'et':
                et_verses.append(entry)
            elif m.get('l') == 'fi':
                fi_verses.append(entry)
        enriched.append({
            "cluster_id": c['cluster_id'],
            "size": c['size'],
            "best_4algo_jaccard": c['best_4algo_jaccard'],
            "et_count": len(et_verses),
            "fi_count": len(fi_verses),
            "et_sample": et_verses[:5],
            "fi_sample": fi_verses[:5],
        })

    result = {
        "floor": 2,
        "method": "novel cross-lingual cluster sampling at min_cluster_size=2; "
                  "novel = best member-set Jaccard < 0.3 vs 4-algo (no_sentence) CL "
                  "clusters; sample seed 42, size-stratified 50+50.",
        "total_5algo_cl": len(cl_5algo),
        "total_nosent_cl": len(cl_nosent),
        "new_cl_clusters": len(new_cl),
        "sample_count": len(enriched),
        "sample": enriched,
    }
    Path(OUTPUT_PATH).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUTPUT_PATH}", flush=True)
    print(f"  total_5algo_cl={len(cl_5algo):,}  total_nosent_cl={len(cl_nosent):,}  "
          f"novel={len(new_cl):,}  sample={len(enriched)}", flush=True)


if __name__ == "__main__":
    main()
