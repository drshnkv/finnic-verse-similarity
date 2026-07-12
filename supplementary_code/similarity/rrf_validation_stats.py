#!/usr/bin/env python3
"""
Full cluster statistics, membership export, and internal validation.

Re-runs RRF clustering at threshold 0.033 to capture FULL cluster membership
(the original output only stores 20 sample_members per cluster).

Produces:
  - rrf_cluster_statistics.json   — coverage, size distribution, cross-lingual counts
  - rrf_cluster_membership.jsonl  — full membership for all clusters
  - rrf_internal_validation.json  — intra/inter cluster similarity metrics

Usage:
    python -u rrf_validation_stats.py [--threshold 0.033]
"""

import argparse
import json
import sys
import time
from collections import Counter

import numpy as np

# Import clustering functions from existing script
from cluster_verses_rrf import (
    OUTPUT_DIR,
    _open_jsonl,
    cluster_by_neighborhoods,
    compute_rrf_neighborhoods,
    load_cache,
    merge_similar_clusters,
)

METADATA_PATH = OUTPUT_DIR / "verse_metadata.jsonl"


def load_verse_metadata():
    """Load verse metadata for language/corpus info. gz-aware: the plain .jsonl
    is no longer materialized, only verse_metadata.jsonl.gz, so use the
    plain-or-gz opener rather than a bare open()."""
    print("Loading verse metadata...")
    meta = {}
    with _open_jsonl(METADATA_PATH) as f:
        for line in f:
            m = json.loads(line)
            meta[m['v']] = m
    print(f"  Loaded {len(meta):,} verse metadata entries")
    return meta


def compute_size_distribution(clusters):
    """Compute histogram of cluster sizes."""
    bins = {
        "5": 0, "6-10": 0, "11-20": 0, "21-50": 0,
        "51-100": 0, "101-500": 0, "500+": 0
    }
    for c in clusters:
        s = len(c)
        if s == 5:
            bins["5"] += 1
        elif s <= 10:
            bins["6-10"] += 1
        elif s <= 20:
            bins["11-20"] += 1
        elif s <= 50:
            bins["21-50"] += 1
        elif s <= 100:
            bins["51-100"] += 1
        elif s <= 500:
            bins["101-500"] += 1
        else:
            bins["500+"] += 1
    return bins


def compute_language_composition(clusters, id_to_vid, verse_meta):
    """Classify clusters as ET-only, FI-only, or mixed."""
    et_only = 0
    fi_only = 0
    mixed = 0
    other_only = 0

    for members in clusters:
        langs = set()
        for mid in members:
            vid = id_to_vid[mid]
            meta = verse_meta.get(vid, {})
            langs.add(meta.get('l', 'other'))

        has_et = 'et' in langs
        has_fi = 'fi' in langs

        if has_et and has_fi:
            mixed += 1
        elif has_et:
            et_only += 1
        elif has_fi:
            fi_only += 1
        else:
            other_only += 1

    return {
        "et_only": et_only,
        "fi_only": fi_only,
        "mixed_cross_lingual": mixed,
        "other_only": other_only,
    }


def fit_power_law_exponent(clusters):
    """Estimate power-law exponent from cluster size distribution."""
    sizes = np.array([len(c) for c in clusters], dtype=np.float64)
    if len(sizes) < 2:
        return None
    # Maximum likelihood estimator for discrete power law (Clauset et al. 2009)
    x_min = sizes.min()
    alpha = 1.0 + len(sizes) / np.sum(np.log(sizes / (x_min - 0.5)))
    return float(alpha)


def compute_internal_validation(neighborhoods, clusters, id_to_vid):
    """Compute intra-cluster and inter-cluster RRF similarity metrics.

    Uses the RRF neighborhood scores that are already computed.
    """
    print("Computing internal validation metrics...")

    # Build a fast lookup: vid_id -> {neighbor_id: rrf_score}
    nbr_scores = {}
    for vid_id, nbrs in neighborhoods.items():
        nbr_scores[vid_id] = {nid: score for nid, score in nbrs}

    intra_sims = []
    inter_sims = []

    # Sample up to 1000 clusters for efficiency
    sample_clusters = clusters[:min(len(clusters), 1000)]

    for ci, members in enumerate(sample_clusters):
        if ci % 200 == 0 and ci > 0:
            print(f"  Internal validation: {ci}/{len(sample_clusters)} clusters...")

        member_list = list(members)

        # Intra-cluster: avg RRF score between members
        cluster_intra = []
        for i in range(len(member_list)):
            for j in range(i + 1, min(len(member_list), i + 20)):  # cap pairwise
                a, b = member_list[i], member_list[j]
                score = nbr_scores.get(a, {}).get(b, 0.0)
                if score == 0.0:
                    score = nbr_scores.get(b, {}).get(a, 0.0)
                cluster_intra.append(score)

        if cluster_intra:
            intra_sims.append(np.mean(cluster_intra))

        # Inter-cluster: sample a few non-member neighbors
        for mid in member_list[:5]:
            for nid, score in nbr_scores.get(mid, {}).items():
                if nid not in members:
                    inter_sims.append(score)

    result = {
        "avg_intra_cluster_rrf": float(np.mean(intra_sims)) if intra_sims else 0.0,
        "median_intra_cluster_rrf": float(np.median(intra_sims)) if intra_sims else 0.0,
        "avg_inter_cluster_rrf": float(np.mean(inter_sims)) if inter_sims else 0.0,
        "median_inter_cluster_rrf": float(np.median(inter_sims)) if inter_sims else 0.0,
        "intra_inter_ratio": (
            float(np.mean(intra_sims) / np.mean(inter_sims))
            if intra_sims and inter_sims and np.mean(inter_sims) > 0
            else 0.0
        ),
        "clusters_sampled": len(sample_clusters),
        "intra_pairs_sampled": len(intra_sims),
        "inter_pairs_sampled": len(inter_sims),
    }

    print(f"  Intra-cluster avg RRF: {result['avg_intra_cluster_rrf']:.4f}")
    print(f"  Inter-cluster avg RRF: {result['avg_inter_cluster_rrf']:.4f}")
    print(f"  Intra/inter ratio: {result['intra_inter_ratio']:.2f}")

    return result


def export_membership(clusters, id_to_vid, output_path):
    """Export full cluster membership as JSONL."""
    print(f"Exporting full membership to {output_path.name}...")
    with open(output_path, 'w') as f:
        for ci, members in enumerate(clusters):
            member_vids = [id_to_vid[mid] for mid in sorted(members)]
            entry = {
                "cluster_id": ci,
                "size": len(members),
                "members": member_vids,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Exported {len(clusters):,} clusters ({size_mb:.1f} MB)")


def _tagged(base_name, tag):
    """Suffix a filename stem with a run tag, e.g.
    ('rrf_cluster_membership.jsonl', 'loo_minus_jaccard')
        -> 'rrf_cluster_membership_loo_minus_jaccard.jsonl'.
    tag=None -> the canonical name, unchanged (default-path byte-identity)."""
    if not tag:
        return base_name
    stem, dot, ext = base_name.partition('.')
    return f"{stem}_{tag}.{ext}" if dot else f"{base_name}_{tag}"


def main():
    parser = argparse.ArgumentParser(description='RRF cluster statistics and full membership export')
    parser.add_argument('--threshold', type=float, default=0.033,
                        help='RRF threshold (default: 0.033)')
    parser.add_argument('--min-algos', type=int, default=2)
    parser.add_argument('--min-overlap', type=float, default=0.3)
    parser.add_argument('--max-nbrs', type=int, default=20)
    parser.add_argument('--min-cluster-size', type=int, default=2,
                        help='Min cluster size to report (default: 2, the article floor)')
    parser.add_argument('--algos', type=str, default=None,
                        help='Comma-separated algo subset by name (e.g. '
                             'tfidf,translation,charbigram,sentence to leave one out). '
                             'Default: all cached algos (the canonical full membership). '
                             'Used to build the leave-one-out ablation memberships.')
    parser.add_argument('--out-tag', type=str, default=None,
                        help='Suffix appended to all output filenames (e.g. '
                             'loo_minus_jaccard, minalgos3) so ablation runs never '
                             'overwrite the canonical files. Default: canonical names.')
    parser.add_argument('--skip-internal-validation', action='store_true',
                        help='Skip internal validation metrics (saves time)')
    args = parser.parse_args()

    t0 = time.time()

    # ── Load cache ────────────────────────────────────────────────────────
    print("Loading cached numpy arrays...")
    cache_data = load_cache()
    if cache_data is None:
        print("ERROR: No cached data found. Run cluster_verses_rrf.py first.")
        sys.exit(1)

    id_to_vid, algo_names, algo_neighbors, N = cache_data
    top_k = algo_neighbors[0].shape[1]
    print(f"  {N:,} verses, {len(algo_names)} algorithms, top_k={top_k}")

    # ── Optional algorithm subset (leave-one-out ablation) ────────────────
    # Re-slice the cached arrays by name, then run the SAME
    # compute_rrf_neighborhoods below. RRF rank-weights and the min_algos
    # unique-algo count are both index-order-independent, so re-slicing
    # reproduces a custom-subset path exactly. --algos unset bypasses slicing
    # entirely → the default path is byte-identical to the full run.
    selected_algos = list(algo_names)
    if args.algos:
        name_to_idx = {name: i for i, name in enumerate(algo_names)}
        requested = [a.strip() for a in args.algos.split(',') if a.strip()]
        unknown = [a for a in requested if a not in name_to_idx]
        if unknown:
            print(f"ERROR: unknown algo name(s) {unknown}; available: {algo_names}")
            sys.exit(1)
        indices = sorted({name_to_idx[a] for a in requested})  # cache order, deduped
        algo_neighbors = [algo_neighbors[i] for i in indices]
        selected_algos = [algo_names[i] for i in indices]
        print(f"  Algo subset: {selected_algos} (indices {indices})")
    else:
        print(f"  Using all {len(selected_algos)} cached algos: {selected_algos}")

    # ── Compute RRF neighborhoods ─────────────────────────────────────────
    print(f"\nComputing RRF neighborhoods (threshold={args.threshold})...")
    neighborhoods = compute_rrf_neighborhoods(
        algo_neighbors, N, top_k,
        rrf_threshold=args.threshold,
        min_algos=args.min_algos,
        max_nbrs=args.max_nbrs,
    )
    del algo_neighbors  # Free ~1.5 GB

    print(f"  Vertices with neighborhoods: {len(neighborhoods):,}")

    # ── Cluster (overlapping neighborhood backbone) ───────────────────────
    print(f"\nClustering...")
    clusters = cluster_by_neighborhoods(
        neighborhoods,
        min_overlap=args.min_overlap,
        min_cluster_size=args.min_cluster_size,
    )

    print(f"\nMerging similar clusters...")
    clusters = merge_similar_clusters(clusters, min_shared_frac=0.1)

    # ── Load metadata ─────────────────────────────────────────────────────
    verse_meta = load_verse_metadata()

    # ── Compute statistics ────────────────────────────────────────────────
    print(f"\nComputing statistics for {len(clusters):,} clusters...")

    # Total unique verses in any cluster
    all_verse_ids = set()
    for members in clusters:
        all_verse_ids.update(members)

    total_verses_clustered = len(all_verse_ids)
    coverage_pct = total_verses_clustered / N * 100

    sizes = [len(c) for c in clusters]
    size_distribution = compute_size_distribution(clusters)
    lang_composition = compute_language_composition(clusters, id_to_vid, verse_meta)
    power_law_alpha = fit_power_law_exponent(clusters)

    # Top 20 summary
    top_20 = []
    for i, members in enumerate(clusters[:20]):
        member_vids = [id_to_vid[mid] for mid in list(members)[:5]]
        texts = [verse_meta.get(vid, {}).get('t', '') for vid in member_vids]
        langs = Counter()
        for mid in members:
            vid = id_to_vid[mid]
            langs[verse_meta.get(vid, {}).get('l', 'other')] += 1
        top_20.append({
            "rank": i + 1,
            "size": len(members),
            "example_texts": [t for t in texts if t][:3],
            "languages": dict(langs),
        })

    statistics = {
        "threshold": args.threshold,
        "min_algos": args.min_algos,
        "min_overlap": args.min_overlap,
        "min_cluster_size": args.min_cluster_size,
        "algos": selected_algos,
        "total_verses_in_corpus": N,
        "total_clusters": len(clusters),
        "total_verses_clustered": total_verses_clustered,
        "corpus_coverage_pct": round(coverage_pct, 2),
        "size_distribution": size_distribution,
        "size_stats": {
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": round(float(np.mean(sizes)), 1) if sizes else 0,
            "median": round(float(np.median(sizes)), 1) if sizes else 0,
            "total_verse_assignments": sum(sizes),
        },
        "language_composition": lang_composition,
        "cross_lingual_cluster_count": lang_composition["mixed_cross_lingual"],
        "power_law_exponent_alpha": round(power_law_alpha, 3) if power_law_alpha else None,
        "top_20_clusters": top_20,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RRF Cluster Statistics (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"Total clusters: {len(clusters):,}")
    print(f"Total verses clustered: {total_verses_clustered:,} / {N:,} "
          f"({coverage_pct:.1f}%)")
    print(f"Size range: {min(sizes)}-{max(sizes)}, mean={np.mean(sizes):.1f}")
    print(f"Size distribution: {size_distribution}")
    print(f"Language composition: {lang_composition}")
    print(f"Cross-lingual clusters: {lang_composition['mixed_cross_lingual']:,}")
    if power_law_alpha:
        print(f"Power-law exponent: {power_law_alpha:.3f}")
    print()

    for r in top_20[:10]:
        print(f"  #{r['rank']}: {r['size']} members, langs={r['languages']}")
        for t in r['example_texts'][:2]:
            print(f"    \"{t}\"")

    # ── Internal validation ───────────────────────────────────────────────
    internal_validation = None
    if not args.skip_internal_validation:
        internal_validation = compute_internal_validation(
            neighborhoods, clusters, id_to_vid
        )
        statistics["internal_validation"] = internal_validation

    # ── Export ─────────────────────────────────────────────────────────────
    stats_path = OUTPUT_DIR / _tagged("rrf_cluster_statistics.json", args.out_tag)
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics: {stats_path.name} ({stats_path.stat().st_size / 1024:.0f} KB)")

    membership_path = OUTPUT_DIR / _tagged("rrf_cluster_membership.jsonl", args.out_tag)
    export_membership(clusters, id_to_vid, membership_path)

    if internal_validation:
        val_path = OUTPUT_DIR / _tagged("rrf_internal_validation.json", args.out_tag)
        with open(val_path, 'w') as f:
            json.dump(internal_validation, f, indent=2)
        print(f"Internal validation: {val_path.name}")

    elapsed = time.time() - t0
    print(f"\nCluster stats complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
