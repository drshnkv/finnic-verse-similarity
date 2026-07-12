#!/usr/bin/env python3
"""
Formulaic verse clustering using Reciprocal Rank Fusion (RRF).

Instead of clustering on Jaccard alone (which chains unrelated verses through
shared function words), this computes RRF scores across the five verse-level
algorithms: word Jaccard, character-bigram cosine, TF-IDF cosine,
translation-based (L2 cosine on translation-pivoted TF-IDF), and
sentence-embedding cosine. A pair must rank well in multiple algorithms
to get a high fused score, filtering out surface-level coincidences.

Uses neighborhood-based clustering instead of Union-Find to avoid the
transitivity/chaining problem. A cluster is a group of verses that share
a high fraction of their RRF neighborhoods (mutual overlap).

Caches numpy arrays to output/rrf_cache/ for fast re-runs.

Usage:
    python -u cluster_verses_rrf.py [options]
    python -u cluster_verses_rrf.py --rrf-threshold 0.045 --min-algos 2
    python -u cluster_verses_rrf.py --rebuild-cache  # force re-read
"""

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
CACHE_DIR = OUTPUT_DIR / "rrf_cache"
METADATA_PATH = OUTPUT_DIR / "verse_metadata.jsonl"

ALGO_FILES = {
    "jaccard":     OUTPUT_DIR / "verse_similarities_jaccard.jsonl.gz",
    "tfidf":       OUTPUT_DIR / "verse_similarities_tfidf.jsonl.gz",
    "translation": OUTPUT_DIR / "verse_similarities_translation.jsonl.gz",
    "charbigram":  OUTPUT_DIR / "verse_similarities_charbigram.jsonl.gz",
    "sentence":    OUTPUT_DIR / "verse_similarities_sentence.jsonl.gz",
}

RRF_K = 60
TOP_K = 30


def _resolve_jsonl(path):
    """Resolve a .jsonl that may be stored gzipped: prefer the plain file,
    else fall back to <path>.gz. Returns the existing Path, or None."""
    p = Path(path)
    if p.exists():
        return p
    gz = Path(str(p) + ".gz")
    return gz if gz.exists() else None


def _open_jsonl(path):
    """Open a .jsonl (or its .gz sibling) in text mode. Raises if neither exists."""
    p = _resolve_jsonl(path)
    if p is None:
        raise FileNotFoundError(f"Neither {path} nor {path}.gz found")
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


# ── Cache management ───────────────────────────────────────────────────────

def build_and_cache(algo_paths, top_k):
    """Read all algorithm files, build vid mapping + numpy arrays, save to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    vid_to_id = {}
    id_to_vid = []

    def get_or_add(vid_str):
        idx = vid_to_id.get(vid_str)
        if idx is None:
            idx = len(id_to_vid)
            vid_to_id[vid_str] = idx
            id_to_vid.append(vid_str)
        return idx

    print("Phase 1: Building verse ID index...")
    for name, path in algo_paths:
        print(f"  Indexing {name} ({path.stat().st_size / 1024**2:.0f} MB)...")
        n_lines = 0
        with gzip.open(path, 'rt') as f:
            for line in f:
                entry = json.loads(line)
                if '_header' in entry:
                    continue
                get_or_add(entry['v'])
                for m in entry['m'][:top_k]:
                    get_or_add(m[0])
                n_lines += 1
                if n_lines % 1_000_000 == 0:
                    print(f"    {n_lines:,} entries, {len(id_to_vid):,} unique vids...")

    N = len(id_to_vid)
    print(f"  Total unique verse IDs: {N:,}")
    print(f"  Phase 1 time: {time.time() - t0:.1f}s")

    with open(CACHE_DIR / "id_to_vid.json", 'w') as f:
        json.dump(id_to_vid, f)

    print(f"\nPhase 2: Loading neighbor arrays (top-{top_k} per algo)...")
    EMPTY = -1
    algo_names_saved = []

    for name, path in algo_paths:
        print(f"  Loading {name}...")
        arr = np.full((N, top_k), EMPTY, dtype=np.int32)
        n_loaded = 0
        with gzip.open(path, 'rt') as f:
            for line in f:
                entry = json.loads(line)
                if '_header' in entry:
                    continue
                vid_id = vid_to_id[entry['v']]
                matches = entry['m']
                for rank in range(min(len(matches), top_k)):
                    arr[vid_id, rank] = vid_to_id[matches[rank][0]]
                n_loaded += 1
                if n_loaded % 1_000_000 == 0:
                    print(f"    {n_loaded:,}...")
        np.save(CACHE_DIR / f"neighbors_{name}.npy", arr)
        algo_names_saved.append(name)
        print(f"    {n_loaded:,} entries, {arr.nbytes / 1024**2:.0f} MB")
        del arr

    with open(CACHE_DIR / "cache_meta.json", 'w') as f:
        json.dump({"algo_names": algo_names_saved, "top_k": top_k, "N": N}, f)

    print(f"  Cache saved ({time.time() - t0:.1f}s)")
    return algo_names_saved, N


def load_cache():
    """Load cached data."""
    meta_path = CACHE_DIR / "cache_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    with open(CACHE_DIR / "id_to_vid.json") as f:
        id_to_vid = json.load(f)
    algo_neighbors = []
    for name in meta['algo_names']:
        arr = np.load(CACHE_DIR / f"neighbors_{name}.npy")
        algo_neighbors.append(arr)
        print(f"  Loaded {name}: {arr.shape}, {arr.nbytes / 1024**2:.0f} MB")
    return id_to_vid, meta['algo_names'], algo_neighbors, meta['N']


# ── RRF neighborhood computation ──────────────────────────────────────────

def compute_rrf_neighborhoods(algo_neighbors, N, top_k, rrf_threshold, min_algos,
                               max_nbrs=20):
    """
    For each verse, compute its RRF neighborhood: the set of other verses
    that are above threshold across multiple algorithms.

    Returns: dict {vid_id: [(neighbor_id, rrf_score), ...]} sorted by score desc
    """
    rrf_weights = np.array([1.0 / (RRF_K + r) for r in range(top_k)], dtype=np.float64)
    EMPTY = -1
    n_algos = len(algo_neighbors)

    neighborhoods = {}
    n_with_nbrs = 0

    for vid_id in range(N):
        if vid_id % 500_000 == 0 and vid_id > 0:
            print(f"    {vid_id:,}/{N:,} vertices, {n_with_nbrs:,} with neighborhoods...")

        neighbor_algo_data = defaultdict(list)  # nid -> [(algo_idx, rank)]

        for ai in range(n_algos):
            row = algo_neighbors[ai][vid_id]
            for rank in range(top_k):
                nid = int(row[rank])
                if nid == EMPTY:
                    break
                neighbor_algo_data[nid].append((ai, rank))

        # Compute RRF, require min_algos unique algorithms
        nbrs = []
        for nid, algo_ranks in neighbor_algo_data.items():
            unique_algos = set()
            rrf_score = 0.0
            for ai, rank in algo_ranks:
                if ai not in unique_algos:
                    unique_algos.add(ai)
                    rrf_score += rrf_weights[rank]
            if len(unique_algos) >= min_algos and rrf_score >= rrf_threshold:
                nbrs.append((nid, rrf_score))

        if nbrs:
            nbrs.sort(key=lambda x: -x[1])
            neighborhoods[vid_id] = nbrs[:max_nbrs]
            n_with_nbrs += 1

    return neighborhoods


# ── Neighborhood-based clustering ──────────────────────────────────────────

def cluster_by_neighborhoods(neighborhoods, min_overlap=0.5, min_cluster_size=3,
                             return_prefloor=False):
    """
    Cluster verses based on shared RRF neighborhoods.

    Algorithm:
    1. Sort vertices by number of RRF neighbors (hubs first)
    2. For each vertex, get its neighborhood set
    3. Try to assign to the existing cluster with highest Jaccard overlap
    4. If best overlap >= min_overlap, join that cluster
    5. Otherwise, start a new cluster

    This avoids chaining because membership requires high mutual overlap
    with the cluster's existing members, not just one pair.
    """
    print(f"  Clustering {len(neighborhoods):,} vertices (min_overlap={min_overlap})...")

    # Build neighborhood SETS for fast overlap computation
    nbr_sets = {}
    for vid, nbrs in neighborhoods.items():
        nbr_sets[vid] = frozenset(n for n, _ in nbrs)

    # Sort by degree descending (most connected vertices first = better seeds)
    sorted_vids = sorted(nbr_sets.keys(), key=lambda v: -len(nbr_sets[v]))

    clusters = []  # list of sets
    vid_to_cluster = {}  # vid -> cluster_index

    # Track cluster "profile" = union of all member neighborhoods
    cluster_member_sets = []  # list of Counter(vid -> count of members that have it as nbr)
    cluster_member_lists = []

    for i, vid in enumerate(sorted_vids):
        if i % 200_000 == 0 and i > 0:
            print(f"    {i:,}/{len(sorted_vids):,} assigned, "
                  f"{len(clusters):,} clusters...")

        my_nbrs = nbr_sets[vid]
        if not my_nbrs:
            continue

        # Find best matching existing cluster
        best_cluster = -1
        best_overlap = 0.0

        # Only check clusters that share at least one member with my neighbors
        candidate_clusters = set()
        for nbr in my_nbrs:
            ci = vid_to_cluster.get(nbr)
            if ci is not None:
                candidate_clusters.add(ci)

        for ci in candidate_clusters:
            # Jaccard between my neighborhood and the cluster's members
            cluster_members = clusters[ci]
            shared = len(my_nbrs & cluster_members)
            union_size = len(my_nbrs | cluster_members)
            if union_size > 0:
                jaccard = shared / union_size
                if jaccard > best_overlap or (jaccard == best_overlap and ci < best_cluster):
                    best_overlap = jaccard
                    best_cluster = ci

        if best_overlap >= min_overlap and best_cluster >= 0:
            # Join existing cluster
            clusters[best_cluster].add(vid)
            vid_to_cluster[vid] = best_cluster
        else:
            # Start new cluster
            new_ci = len(clusters)
            new_cluster = {vid} | my_nbrs
            clusters.append(new_cluster)
            vid_to_cluster[vid] = new_ci
            for m in new_cluster:
                if m not in vid_to_cluster:
                    vid_to_cluster[m] = new_ci

    # Filter by min size
    big_clusters = [c for c in clusters if len(c) >= min_cluster_size]
    big_clusters.sort(key=lambda c: -len(c))

    print(f"  Total clusters: {len(clusters):,}, "
          f"with >={min_cluster_size} members: {len(big_clusters):,}")

    # Optionally also return the full pre-floor cluster list (before the
    # min_cluster_size filter) so the evaluator can split bucket B into
    # B2 (floor-discarded, size 2..4) vs B1 (threshold/linkage).
    if return_prefloor:
        return big_clusters, clusters
    return big_clusters


def merge_similar_clusters(clusters, min_shared_frac=0.1, max_clusters_to_check=5000):
    """
    Post-processing: merge clusters that share members.

    Two clusters are merged if they share >= min_shared_frac of the smaller
    cluster's members. This catches splits like "mamsel"/"mampsel" where the
    greedy assignment created separate clusters for spelling variants.
    """
    if not clusters:
        return clusters

    n_check = min(len(clusters), max_clusters_to_check)
    print(f"  Checking top {n_check} clusters for mergeable pairs...")

    # Work with indices into the clusters list
    merged = list(range(n_check))  # merged[i] = canonical index for cluster i
    cluster_sets = [set(c) for c in clusters[:n_check]]

    n_merges = 0
    for i in range(n_check):
        if merged[i] != i:
            continue  # already merged into another
        for j in range(i + 1, n_check):
            if merged[j] != j:
                continue

            # Find canonical sets
            ci, cj = cluster_sets[i], cluster_sets[j]
            shared = len(ci & cj)
            smaller = min(len(ci), len(cj))

            if smaller > 0 and shared / smaller >= min_shared_frac:
                # Merge j into i
                cluster_sets[i] = ci | cj
                cluster_sets[j] = set()
                merged[j] = i
                n_merges += 1

    # Rebuild cluster list
    result = []
    for i in range(n_check):
        if merged[i] == i and cluster_sets[i]:
            result.append(cluster_sets[i])
    # Add unchecked clusters unchanged
    result.extend(clusters[n_check:])

    result.sort(key=lambda c: -len(c))
    print(f"  Merged {n_merges} cluster pairs. "
          f"Before: {n_check}, after: {len(result):,}")
    return result


def export_clusters_csv(clusters, id_to_vid, verse_meta, output_dir, top_n=200):
    """Export each cluster's verses to a separate CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, members in enumerate(clusters[:top_n]):
        member_vids = [id_to_vid[mid] for mid in members]

        # Get representative text for filename
        first_text = ""
        for vid in member_vids[:5]:
            meta = verse_meta.get(vid, {})
            if meta.get('t'):
                first_text = meta['t']
                break

        # Sanitize for filename (first 40 chars, ascii-safe)
        safe_name = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in first_text)
        safe_name = safe_name.strip()[:40].strip()
        if not safe_name:
            safe_name = "cluster"
        filename = f"{i+1:03d}_{safe_name}.csv"

        filepath = output_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['verse_id', 'text', 'language', 'corpus', 'places'])

            for vid in sorted(member_vids):
                meta = verse_meta.get(vid, {})
                writer.writerow([
                    vid,
                    meta.get('t', ''),
                    meta.get('l', ''),
                    meta.get('c', ''),
                    '; '.join(meta.get('pl', [])),
                ])

    print(f"  Exported {min(top_n, len(clusters))} cluster CSVs to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='RRF-based verse clustering')
    parser.add_argument('--rrf-threshold', type=float, default=0.033,
                        help='Min RRF score to include in neighborhood (default: 0.033, '
                             'the article/live-site canonical value)')
    parser.add_argument('--min-algos', type=int, default=2,
                        help='Min algorithms a pair must appear in (default: 2)')
    parser.add_argument('--min-overlap', type=float, default=0.3,
                        help='Min Jaccard overlap to join a cluster (default: 0.3)')
    parser.add_argument('--max-nbrs', type=int, default=20,
                        help='Max neighbors per verse in RRF neighborhood (default: 20)')
    parser.add_argument('--min-cluster-size', type=int, default=2,
                        help='Min cluster size to report (default: 2, the article floor)')
    parser.add_argument('--top-n', type=int, default=200,
                        help='Top N clusters to output (default: 200)')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help=f'Top K neighbors per algo to use (default: {TOP_K})')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Force rebuild of numpy cache')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()

    if args.output:
        output_path = Path(args.output)
    else:
        thr_str = f"{args.rrf_threshold:.3f}".replace('.', '')
        output_path = OUTPUT_DIR / f"formula_clusters_rrf_{thr_str}.json"

    print(f"RRF Clustering: threshold={args.rrf_threshold}, min_algos={args.min_algos}, "
          f"min_overlap={args.min_overlap}, max_nbrs={args.max_nbrs}")

    # ── Load or build cache ────────────────────────────────────────────────
    cache_data = None if args.rebuild_cache else load_cache()

    if cache_data is None:
        algo_paths = []
        for name, path in ALGO_FILES.items():
            if path.exists() and path.stat().st_size > 10000:
                algo_paths.append((name, path))
                print(f"  {name}: {path.stat().st_size / 1024**2:.0f} MB")
        if len(algo_paths) < 2:
            print("ERROR: Need at least 2 algorithm files.")
            sys.exit(1)
        build_and_cache(algo_paths, args.top_k)
        cache_data = load_cache()

    id_to_vid, algo_names, algo_neighbors, N = cache_data
    n_algos = len(algo_neighbors)
    top_k = algo_neighbors[0].shape[1]

    print(f"\n  {N:,} unique vids, {n_algos} algorithms, top_k={top_k}")

    # ── Phase 3: Compute RRF neighborhoods ─────────────────────────────────
    print(f"\nPhase 3: Computing RRF neighborhoods...")

    neighborhoods = compute_rrf_neighborhoods(
        algo_neighbors, N, top_k,
        rrf_threshold=args.rrf_threshold,
        min_algos=args.min_algos,
        max_nbrs=args.max_nbrs,
    )

    # Free numpy arrays
    del algo_neighbors

    print(f"  Verses with RRF neighborhoods: {len(neighborhoods):,}")
    if neighborhoods:
        sizes = [len(v) for v in neighborhoods.values()]
        print(f"  Avg neighborhood size: {sum(sizes)/len(sizes):.1f}, "
              f"max: {max(sizes)}")

    # ── Phase 4: Neighborhood-based clustering ─────────────────────────────
    print(f"\nPhase 4: Neighborhood-based clustering...")

    clusters = cluster_by_neighborhoods(
        neighborhoods,
        min_overlap=args.min_overlap,
        min_cluster_size=args.min_cluster_size,
    )

    del neighborhoods

    # ── Phase 4b: Merge similar clusters ───────────────────────────────────
    print(f"\nPhase 4b: Post-processing merge...")
    clusters = merge_similar_clusters(clusters, min_shared_frac=0.1)

    # ── Phase 5: Build reports ─────────────────────────────────────────────
    print(f"\nPhase 5: Building cluster reports (top {args.top_n})...")

    verse_meta = {}
    meta_path = _resolve_jsonl(METADATA_PATH)
    if meta_path is not None:
        print(f"  Loading verse metadata from {meta_path.name}...")
        with _open_jsonl(meta_path) as f:
            for line in f:
                m = json.loads(line)
                verse_meta[m['v']] = m
    else:
        print("  WARNING: verse_metadata.jsonl[.gz] not found — "
              "cluster reports will lack verse text", file=sys.stderr)

    cluster_reports = []
    geo_data = []

    for i, members in enumerate(clusters[:args.top_n]):
        example_texts = []
        lang_counts = Counter()
        places = set()
        corpus_counts = Counter()

        member_vids = [id_to_vid[mid] for mid in members]

        for vid in member_vids[:20]:
            meta = verse_meta.get(vid, {})
            if meta.get('t'):
                example_texts.append(meta['t'])
            lang_counts[meta.get('l', 'other')] += 1
            corpus_counts[meta.get('c', 'other')] += 1
            for p in meta.get('pl', []):
                places.add(p)

        report = {
            "rank": i + 1,
            "size": len(members),
            "example_verses": example_texts[:5],
            "languages": dict(lang_counts),
            "corpora": dict(corpus_counts),
            "distinct_places": len(places),
            "sample_members": member_vids[:20],
        }
        cluster_reports.append(report)
        geo_data.append({
            "cluster_rank": report['rank'],
            "size": report['size'],
            "distinct_places": report['distinct_places'],
            "example": example_texts[0] if example_texts else "",
            "language": max(lang_counts, key=lang_counts.get) if lang_counts else "unknown",
        })

    print(f"\n  Top 20 RRF clusters:")
    for r in cluster_reports[:20]:
        print(f"  #{r['rank']}: {r['size']} members, "
              f"places={r['distinct_places']}, langs={r['languages']}")
        for text in r['example_verses'][:3]:
            print(f"    \"{text}\"")

    with open(output_path, 'w') as f:
        json.dump(cluster_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  Clusters: {output_path.name} ({output_path.stat().st_size / 1024:.0f} KB)")

    geo_path = output_path.parent / output_path.name.replace('formula_clusters', 'geographic_spread')
    with open(geo_path, 'w') as f:
        json.dump(geo_data, f, indent=2, ensure_ascii=False)
    print(f"  Geographic: {geo_path.name}")

    # ── CSV export ──────────────────────────────────────────────────────────
    csv_dir = output_path.parent / "rrf_clusters_csv"
    print(f"\nPhase 6: Exporting cluster CSVs...")
    export_clusters_csv(clusters, id_to_vid, verse_meta, csv_dir, top_n=args.top_n)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. {len(clusters):,} total clusters, "
          f"top {min(args.top_n, len(clusters))} reported.")


if __name__ == '__main__':
    main()
