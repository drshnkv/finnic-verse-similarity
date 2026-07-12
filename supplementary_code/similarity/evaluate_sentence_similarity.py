#!/usr/bin/env python3
"""Evaluate the sentence-embedding similarity index against translation-pivot.

Sections:
  1. Score distribution & anisotropy check (random-pair cosine)
  2. Cross-lingual pair count (directed + undirected dedup)
  3. Overlap with translation-pivot (Jaccard on neighbor sets)
  4. Novel cross-lingual pairs (qualitative sample, 20 pairs)
  5. Precision estimate (50 high-score pairs sampled for review)

Single-pass I/O: one pass over sentence JSONL, one over translation JSONL.
"""

import gzip
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from verse_similarity_common import (
    OUTPUT_DIR,
    classify_match,
    detect_corpus_lang,
)

EMBEDDING_DIM = 384
TOTAL_VECTORS = 3_807_653
SAMPLE_SIZE = 100_000
SEED = 42

MEMMAP_PATH = OUTPUT_DIR / "verse_embeddings.f16.mmap"
IDS_PATH = OUTPUT_DIR / "verse_embedding_ids.json.gz"
REGISTRY_PATH = OUTPUT_DIR / "sentence_embedding_verse_registry.jsonl.gz"
SENTENCE_JSONL = OUTPUT_DIR / "verse_similarities_sentence.jsonl.gz"
TRANSLATION_JSONL = OUTPUT_DIR / "verse_similarities_translation.jsonl.gz"

REPORT_PATH = OUTPUT_DIR / "sentence_embedding_evaluation_report.txt"
NOVEL_PAIRS_PATH = OUTPUT_DIR / "sentence_embedding_novel_pairs_sample.json"
PRECISION_PATH = OUTPUT_DIR / "sentence_embedding_precision_sample.json"

RSS_LIMIT_GB = 4.0


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def stream_jsonl(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('_header'):
                continue
            yield entry


# ── Step 1: Load metadata ────────────────────────────────────────────────

def load_metadata():
    print("Step 1: Loading metadata...")
    t0 = time.time()

    print("  Loading verse embedding IDs...")
    with gzip.open(IDS_PATH, 'rt') as f:
        ids_list = json.load(f)
    print(f"  Loaded {len(ids_list):,} IDs")

    print("  Loading verse registry (translations)...")
    registry = {}
    with gzip.open(REGISTRY_PATH, 'rt') as f:
        for line in f:
            rec = json.loads(line)
            registry[rec['v']] = rec['t']
    print(f"  Registry: {len(registry):,} entries")

    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(ids_list), size=SAMPLE_SIZE, replace=False)
    sample_ids = set(ids_list[i] for i in sample_indices)
    print(f"  Pre-selected {len(sample_ids):,} sample IDs (seed={SEED})")

    print(f"  Metadata loaded in {time.time() - t0:.1f}s, RSS {rss_gb():.1f} GB")
    return ids_list, registry, sample_ids


# ── Step 2: Anisotropy check ─────────────────────────────────────────────

def anisotropy_check(ids_list):
    print("\nStep 2: Random-pair anisotropy check...")
    t0 = time.time()

    mmap = np.memmap(MEMMAP_PATH, dtype=np.float16, mode='r',
                     shape=(TOTAL_VECTORS, EMBEDDING_DIM))

    rng = np.random.default_rng(SEED)
    idx_a = rng.integers(0, TOTAL_VECTORS, size=SAMPLE_SIZE)
    idx_b = rng.integers(0, TOTAL_VECTORS, size=SAMPLE_SIZE)

    vecs_a = mmap[idx_a].astype(np.float32)
    vecs_b = mmap[idx_b].astype(np.float32)

    cosines = np.sum(vecs_a * vecs_b, axis=1)
    del vecs_a, vecs_b, mmap

    stats = {
        'mean': float(np.mean(cosines)),
        'std': float(np.std(cosines)),
        'P50': float(np.percentile(cosines, 50)),
        'P95': float(np.percentile(cosines, 95)),
        'P99': float(np.percentile(cosines, 99)),
        'min': float(np.min(cosines)),
        'max': float(np.max(cosines)),
    }

    passed = stats['mean'] < 0.10
    flag_pca = stats['mean'] > 0.20

    print(f"  Random-pair cosine: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"  P50={stats['P50']:.4f}, P95={stats['P95']:.4f}, P99={stats['P99']:.4f}")
    print(f"  PASS: {'YES' if passed else 'NO'} (threshold: mean < 0.10)")
    if flag_pca:
        print("  WARNING: mean > 0.20 — consider PCA whitening")
    print(f"  Done in {time.time() - t0:.1f}s")

    return stats, passed, flag_pca


# ── Step 3: Stream sentence JSONL ────────────────────────────────────────

def stream_sentence_pass(sample_ids):
    print("\nStep 3: Streaming sentence JSONL (single pass)...")
    t0 = time.time()

    total_matches = 0
    total_cross = 0
    total_same = 0
    total_within = 0
    verse_count = 0

    score_buckets = {
        '0.9-1.0': 0, '0.8-0.9': 0, '0.7-0.8': 0,
        '0.6-0.7': 0, '0.5-0.6': 0, '0.3-0.5': 0,
    }
    cross_score_buckets = {
        '0.9-1.0': 0, '0.8-0.9': 0, '0.7-0.8': 0,
        '0.6-0.7': 0, '0.5-0.6': 0, '0.3-0.5': 0,
    }

    nn_scores_all = []

    sentence_neighbors = {}

    precision_candidates = []
    MAX_PRECISION_CANDIDATES = 50_000

    cross_pair_hashes = set()
    dedup_fallback = False

    for entry in stream_jsonl(SENTENCE_JSONL):
        vid = entry['v']
        matches = entry['m']
        verse_count += 1

        is_sampled = vid in sample_ids

        if is_sampled:
            sentence_neighbors[vid] = set()

        for m in matches:
            target_vid, score = m[0], m[1]
            match_type = m[2] if len(m) > 2 else classify_match(vid, target_vid)

            total_matches += 1
            nn_scores_all.append(score)

            bucket = _score_bucket(score)
            if bucket:
                score_buckets[bucket] += 1

            if match_type == 'x':
                total_cross += 1
                if bucket:
                    cross_score_buckets[bucket] += 1

                if not dedup_fallback:
                    pair_key = (min(vid, target_vid), max(vid, target_vid))
                    cross_pair_hashes.add(hash(pair_key))
                    if rss_gb() > RSS_LIMIT_GB:
                        print(f"  WARNING: RSS {rss_gb():.1f} GB > {RSS_LIMIT_GB} GB limit, "
                              "dropping dedup set")
                        cross_pair_hashes.clear()
                        dedup_fallback = True

                if (score > 0.80 and
                        len(precision_candidates) < MAX_PRECISION_CANDIDATES):
                    precision_candidates.append((vid, target_vid, score))

            elif match_type == 's':
                total_same += 1
            else:
                total_within += 1

            if is_sampled:
                sentence_neighbors[vid].add(target_vid)

        if verse_count % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"  {verse_count / 1e6:.1f}M verses, "
                  f"{total_cross / 1e6:.1f}M cross-lingual, "
                  f"RSS {rss_gb():.1f} GB, "
                  f"{elapsed:.0f}s elapsed")

    nn_scores = np.array(nn_scores_all, dtype=np.float32)
    del nn_scores_all
    nn_stats = {
        'mean': float(np.mean(nn_scores)),
        'P50': float(np.percentile(nn_scores, 50)),
        'P95': float(np.percentile(nn_scores, 95)),
        'count': len(nn_scores),
    }
    del nn_scores

    unique_cross = len(cross_pair_hashes) if not dedup_fallback else None
    cross_pair_hashes.clear()

    elapsed = time.time() - t0
    print(f"  Done: {verse_count:,} verses in {elapsed:.0f}s")
    print(f"  Total matches: {total_matches:,}")
    print(f"  Cross-lingual (directed): {total_cross:,}")
    if unique_cross is not None:
        print(f"  Cross-lingual (unique undirected): {unique_cross:,}")
    print(f"  Same-language: {total_same:,}, Within-poem: {total_within:,}")
    print(f"  NN score distribution: mean={nn_stats['mean']:.4f}, "
          f"P50={nn_stats['P50']:.4f}, P95={nn_stats['P95']:.4f}")

    cross_pass = total_cross >= 5_000_000

    return {
        'verse_count': verse_count,
        'total_matches': total_matches,
        'total_cross': total_cross,
        'total_same': total_same,
        'total_within': total_within,
        'unique_cross': unique_cross,
        'dedup_fallback': dedup_fallback,
        'score_buckets': score_buckets,
        'cross_score_buckets': cross_score_buckets,
        'nn_stats': nn_stats,
        'cross_pass': cross_pass,
    }, sentence_neighbors, precision_candidates


def _score_bucket(score):
    if score >= 0.9:
        return '0.9-1.0'
    if score >= 0.8:
        return '0.8-0.9'
    if score >= 0.7:
        return '0.7-0.8'
    if score >= 0.6:
        return '0.6-0.7'
    if score >= 0.5:
        return '0.5-0.6'
    if score >= 0.3:
        return '0.3-0.5'
    return None


# ── Step 4: Select precision sample ──────────────────────────────────────

def select_precision_sample(precision_candidates, sample_ids):
    print(f"\nStep 4: Selecting precision sample from {len(precision_candidates):,} candidates...")
    rng = np.random.default_rng(SEED)

    n = min(50, len(precision_candidates))
    indices = rng.choice(len(precision_candidates), size=n, replace=False)
    selected = [precision_candidates[i] for i in indices]

    extra_ids = set()
    for vid_a, vid_b, _ in selected:
        extra_ids.add(vid_a)
        extra_ids.add(vid_b)

    augmented_sample = sample_ids | extra_ids
    print(f"  Selected {n} precision pairs, augmented sample: {len(augmented_sample):,} IDs")
    return selected, augmented_sample


# ── Step 5: Stream translation JSONL ─────────────────────────────────────

def stream_translation_pass(augmented_sample):
    print("\nStep 5: Streaming translation JSONL (single pass)...")
    t0 = time.time()

    translation_neighbors = {}
    verse_count = 0

    for entry in stream_jsonl(TRANSLATION_JSONL):
        vid = entry['v']
        if vid not in augmented_sample:
            verse_count += 1
            if verse_count % 500_000 == 0:
                print(f"  {verse_count / 1e6:.1f}M verses scanned, "
                      f"{len(translation_neighbors):,} matched, "
                      f"{time.time() - t0:.0f}s elapsed")
            continue

        neighbor_set = set()
        neighbor_scores = {}
        for m in entry['m']:
            target_vid, score = m[0], m[1]
            neighbor_set.add(target_vid)
            neighbor_scores[target_vid] = score

        translation_neighbors[vid] = {
            'set': neighbor_set,
            'scores': neighbor_scores,
        }

        verse_count += 1
        if verse_count % 500_000 == 0:
            print(f"  {verse_count / 1e6:.1f}M verses scanned, "
                  f"{len(translation_neighbors):,} matched, "
                  f"{time.time() - t0:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"  Done: {verse_count:,} verses in {elapsed:.0f}s, "
          f"{len(translation_neighbors):,} sample IDs found")
    return translation_neighbors


# ── Step 6: Compute results ──────────────────────────────────────────────

def compute_overlap(sentence_neighbors, translation_neighbors):
    print("\nStep 6a: Computing Jaccard overlap...")

    overlaps_all = []
    overlaps_cross = []

    both_count = 0
    for vid, s_set in sentence_neighbors.items():
        if vid not in translation_neighbors:
            continue
        t_set = translation_neighbors[vid]['set']
        both_count += 1

        union = len(s_set | t_set)
        if union == 0:
            continue
        jaccard = len(s_set & t_set) / union
        overlaps_all.append(jaccard)

        s_cross = {n for n in s_set if _is_cross_lingual(vid, n)}
        t_cross = {n for n in t_set if _is_cross_lingual(vid, n)}
        cross_union = len(s_cross | t_cross)
        if cross_union > 0:
            cross_jaccard = len(s_cross & t_cross) / cross_union
            overlaps_cross.append(cross_jaccard)

    overlaps_all = np.array(overlaps_all, dtype=np.float32)
    overlaps_cross = np.array(overlaps_cross, dtype=np.float32)

    def _stats(arr):
        if len(arr) == 0:
            return {}
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'P10': float(np.percentile(arr, 10)),
            'P25': float(np.percentile(arr, 25)),
            'P75': float(np.percentile(arr, 75)),
            'P90': float(np.percentile(arr, 90)),
            'count': int(len(arr)),
        }

    histogram = {}
    for lo in np.arange(0, 1.0, 0.1):
        hi = lo + 0.1
        label = f"[{lo:.1f}, {hi:.1f})"
        count = int(np.sum((overlaps_all >= lo) & (overlaps_all < hi)))
        histogram[label] = count

    overlap_stats = {
        'all': _stats(overlaps_all),
        'cross_only': _stats(overlaps_cross),
        'histogram': histogram,
        'both_count': both_count,
    }

    passed = overlap_stats['all'].get('mean', 1.0) < 0.70
    abort = overlap_stats['all'].get('mean', 0) > 0.80

    print(f"  Verses in both files: {both_count:,}")
    if overlap_stats['all']:
        print(f"  All neighbors — mean Jaccard: {overlap_stats['all']['mean']:.4f}, "
              f"median: {overlap_stats['all']['median']:.4f}")
    if overlap_stats['cross_only']:
        print(f"  Cross-lingual only — mean Jaccard: {overlap_stats['cross_only']['mean']:.4f}, "
              f"median: {overlap_stats['cross_only']['median']:.4f}")
    print(f"  Overlap PASS: {'YES' if passed else 'NO'} (threshold: mean < 0.70)")
    if abort:
        print("  NOTE: overlap > 0.80 — sentence-embedding neighbours largely overlap translation-pivot")

    return overlap_stats, passed, abort


def _is_cross_lingual(vid_a, vid_b):
    poem_a = vid_a.rsplit(':', 1)[0]
    poem_b = vid_b.rsplit(':', 1)[0]
    if poem_a == poem_b:
        return False
    _, lang_a = detect_corpus_lang(poem_a)
    _, lang_b = detect_corpus_lang(poem_b)
    return lang_a != lang_b and lang_a != 'other' and lang_b != 'other'


def find_novel_pairs(sentence_neighbors, translation_neighbors, registry):
    print("\nStep 6b: Finding novel cross-lingual pairs...")

    novel = []
    for vid, s_set in sentence_neighbors.items():
        if vid not in translation_neighbors:
            continue
        t_set = translation_neighbors[vid]['set']

        for neighbor_vid in s_set:
            if neighbor_vid in t_set:
                continue
            if not _is_cross_lingual(vid, neighbor_vid):
                continue
            novel.append((vid, neighbor_vid))

    rng = np.random.default_rng(SEED)
    if len(novel) > 20:
        indices = rng.choice(len(novel), size=20, replace=False)
        novel = [novel[i] for i in indices]

    results = []
    for vid_a, vid_b in novel:
        corpus_a, lang_a = detect_corpus_lang(vid_a.rsplit(':', 1)[0])
        corpus_b, lang_b = detect_corpus_lang(vid_b.rsplit(':', 1)[0])
        results.append({
            'verse_a': vid_a,
            'verse_b': vid_b,
            'corpus_a': corpus_a,
            'lang_a': lang_a,
            'corpus_b': corpus_b,
            'lang_b': lang_b,
            'translation_a': registry.get(vid_a, ''),
            'translation_b': registry.get(vid_b, ''),
        })

    print(f"  Found {len(results)} novel cross-lingual pairs")
    return results


def build_precision_sample(precision_pairs, translation_neighbors, registry):
    print("\nStep 6c: Building precision sample...")

    results = []
    for vid_a, vid_b, score in precision_pairs:
        corpus_a, lang_a = detect_corpus_lang(vid_a.rsplit(':', 1)[0])
        corpus_b, lang_b = detect_corpus_lang(vid_b.rsplit(':', 1)[0])

        trans_match = None
        if vid_a in translation_neighbors:
            t_data = translation_neighbors[vid_a]
            if vid_b in t_data['scores']:
                t_score = t_data['scores'][vid_b]
                t_rank = sorted(t_data['scores'].values(), reverse=True).index(t_score) + 1
                trans_match = {'score': t_score, 'rank': t_rank}

        results.append({
            'verse_a': vid_a,
            'verse_b': vid_b,
            'sentence_score': score,
            'corpus_a': corpus_a,
            'lang_a': lang_a,
            'corpus_b': corpus_b,
            'lang_b': lang_b,
            'translation_a': registry.get(vid_a, ''),
            'translation_b': registry.get(vid_b, ''),
            'translation_pivot_match': trans_match,
        })

    print(f"  Built {len(results)} precision pairs")
    return results


# ── Step 7: Write outputs ────────────────────────────────────────────────

def write_report(aniso_stats, aniso_pass, aniso_flag,
                 sentence_stats, overlap_stats, overlap_pass, overlap_abort,
                 novel_count, precision_count):
    lines = []
    lines.append("=" * 70)
    lines.append("SENTENCE EMBEDDING EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("─" * 50)
    lines.append("SECTION 1: Score Distribution & Anisotropy Check")
    lines.append("─" * 50)
    lines.append(f"  Random-pair cosine similarity (100K pairs):")
    for k, v in aniso_stats.items():
        lines.append(f"    {k:>5}: {v:.6f}")
    lines.append(f"  PASS (mean < 0.10): {'YES' if aniso_pass else 'NO'}")
    if aniso_flag:
        lines.append(f"  WARNING: mean > 0.20 — consider PCA whitening")
    lines.append("")
    lines.append(f"  NN output score distribution (from sentence JSONL):")
    nn = sentence_stats['nn_stats']
    lines.append(f"    mean: {nn['mean']:.6f}")
    lines.append(f"    P50:  {nn['P50']:.6f}")
    lines.append(f"    P95:  {nn['P95']:.6f}")
    lines.append(f"    count: {nn['count']:,}")
    lines.append("")

    lines.append("─" * 50)
    lines.append("SECTION 2: Cross-Lingual Pair Count")
    lines.append("─" * 50)
    lines.append(f"  Verses processed: {sentence_stats['verse_count']:,}")
    lines.append(f"  Total matches: {sentence_stats['total_matches']:,}")
    lines.append(f"  Cross-lingual (directed): {sentence_stats['total_cross']:,}")
    if sentence_stats['unique_cross'] is not None:
        lines.append(f"  Cross-lingual (unique undirected): {sentence_stats['unique_cross']:,}")
    else:
        lines.append(f"  Cross-lingual (unique undirected): N/A (memory fallback)")
    lines.append(f"  Same-language: {sentence_stats['total_same']:,}")
    lines.append(f"  Within-poem: {sentence_stats['total_within']:,}")
    lines.append("")
    lines.append(f"  Score buckets (all matches):")
    for bucket, count in sentence_stats['score_buckets'].items():
        lines.append(f"    {bucket}: {count:>12,}")
    lines.append("")
    lines.append(f"  Score buckets (cross-lingual only):")
    for bucket, count in sentence_stats['cross_score_buckets'].items():
        lines.append(f"    {bucket}: {count:>12,}")
    lines.append("")
    lines.append(f"  PASS (≥ 5M directed cross-lingual): "
                 f"{'YES' if sentence_stats['cross_pass'] else 'NO'}")
    lines.append("")

    lines.append("─" * 50)
    lines.append("SECTION 3: Overlap with Translation-Pivot")
    lines.append("─" * 50)
    lines.append(f"  Verses in both files: {overlap_stats['both_count']:,}")
    if overlap_stats['all']:
        s = overlap_stats['all']
        lines.append(f"  All neighbors:")
        lines.append(f"    mean:   {s['mean']:.4f}")
        lines.append(f"    median: {s['median']:.4f}")
        lines.append(f"    P10:    {s['P10']:.4f}")
        lines.append(f"    P25:    {s['P25']:.4f}")
        lines.append(f"    P75:    {s['P75']:.4f}")
        lines.append(f"    P90:    {s['P90']:.4f}")
    if overlap_stats['cross_only']:
        s = overlap_stats['cross_only']
        lines.append(f"  Cross-lingual neighbors only:")
        lines.append(f"    mean:   {s['mean']:.4f}")
        lines.append(f"    median: {s['median']:.4f}")
        lines.append(f"    P10:    {s['P10']:.4f}")
        lines.append(f"    P25:    {s['P25']:.4f}")
        lines.append(f"    P75:    {s['P75']:.4f}")
        lines.append(f"    P90:    {s['P90']:.4f}")
    lines.append("")
    lines.append(f"  Histogram:")
    for bucket, count in overlap_stats['histogram'].items():
        pct = 100 * count / overlap_stats['both_count'] if overlap_stats['both_count'] else 0
        bar = "#" * int(pct / 2)
        lines.append(f"    {bucket}: {count:>7,} ({pct:5.1f}%) {bar}")
    lines.append("")
    lines.append(f"  PASS (mean < 0.70): {'YES' if overlap_pass else 'NO'}")
    if overlap_abort:
        lines.append(f"  NOTE: overlap > 0.80 — sentence-embedding neighbours largely overlap translation-pivot")
    lines.append("")

    lines.append("─" * 50)
    lines.append("SECTION 4: Novel Cross-Lingual Pairs")
    lines.append("─" * 50)
    lines.append(f"  {novel_count} pairs written to: {NOVEL_PAIRS_PATH.name}")
    lines.append("")

    lines.append("─" * 50)
    lines.append("SECTION 5: Precision Estimate")
    lines.append("─" * 50)
    lines.append(f"  {precision_count} pairs written to: {PRECISION_PATH.name}")
    lines.append(f"  PRECISION REVIEW NEEDED: estimate % that are genuine parallels")
    lines.append(f"  Target: > 50% precision")
    lines.append("")

    lines.append("=" * 70)
    lines.append("GO / NO-GO SUMMARY")
    lines.append("=" * 70)

    criteria = [
        ("Random-pair mean < 0.10", aniso_pass),
        ("Cross-lingual pairs >= 5M (directed)", sentence_stats['cross_pass']),
        ("Overlap < 70% (mean Jaccard)", overlap_pass),
        ("Precision > 50% (review)", None),
    ]

    all_auto_pass = all(c[1] for c in criteria if c[1] is not None)

    for name, result in criteria:
        if result is None:
            lines.append(f"  [ REVIEW ] {name}")
        elif result:
            lines.append(f"  [ PASS  ] {name}")
        else:
            lines.append(f"  [ FAIL  ] {name}")

    lines.append("")
    if overlap_abort:
        lines.append("  Automated criteria: sentence-embedding neighbours largely overlap translation-pivot.")
    elif all_auto_pass:
        lines.append("  Automated criteria: all pass; the 50-pair precision sample is set aside for review.")
    else:
        lines.append("  Automated criteria: one or more fail. See details above.")
    lines.append("")

    report_text = '\n'.join(lines)

    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)
    print(f"\nReport written to {REPORT_PATH.name}")

    return report_text


def print_precision_table(precision_results):
    print("\n" + "─" * 90)
    print("PRECISION SAMPLE (50 pairs for precision review)")
    print("─" * 90)
    for i, p in enumerate(precision_results, 1):
        t_info = "not in translation-pivot"
        if p['translation_pivot_match']:
            tm = p['translation_pivot_match']
            t_info = f"translation-pivot rank={tm['rank']}, score={tm['score']:.3f}"
        print(f"\n{i:2d}. [{p['lang_a']}] {p['verse_a']}")
        print(f"    [{p['lang_b']}] {p['verse_b']}")
        print(f"    Sentence score: {p['sentence_score']:.4f} | {t_info}")
        print(f"    A: {p['translation_a'][:100]}")
        print(f"    B: {p['translation_b'][:100]}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SENTENCE EMBEDDING EVALUATION")
    print("=" * 70)
    t_total = time.time()

    for path in [MEMMAP_PATH, IDS_PATH, REGISTRY_PATH, SENTENCE_JSONL, TRANSLATION_JSONL]:
        if not path.exists():
            print(f"ERROR: missing {path}")
            sys.exit(1)

    ids_list, registry, sample_ids = load_metadata()

    aniso_stats, aniso_pass, aniso_flag = anisotropy_check(ids_list)

    sentence_stats, sentence_neighbors, precision_candidates = \
        stream_sentence_pass(sample_ids)

    precision_pairs, augmented_sample = \
        select_precision_sample(precision_candidates, sample_ids)
    del precision_candidates

    translation_neighbors = stream_translation_pass(augmented_sample)

    overlap_stats, overlap_pass, overlap_abort = \
        compute_overlap(sentence_neighbors, translation_neighbors)

    novel_pairs = find_novel_pairs(
        sentence_neighbors, translation_neighbors, registry)

    precision_results = build_precision_sample(
        precision_pairs, translation_neighbors, registry)

    del sentence_neighbors

    with open(NOVEL_PAIRS_PATH, 'w') as f:
        json.dump(novel_pairs, f, indent=2, ensure_ascii=False)
    print(f"Novel pairs written to {NOVEL_PAIRS_PATH.name} ({len(novel_pairs)} entries)")

    with open(PRECISION_PATH, 'w') as f:
        json.dump(precision_results, f, indent=2, ensure_ascii=False)
    print(f"Precision sample written to {PRECISION_PATH.name} ({len(precision_results)} entries)")

    report = write_report(
        aniso_stats, aniso_pass, aniso_flag,
        sentence_stats, overlap_stats, overlap_pass, overlap_abort,
        len(novel_pairs), len(precision_results),
    )

    print_precision_table(precision_results)

    print("\n" + report.split("GO / NO-GO SUMMARY")[1] if "GO / NO-GO SUMMARY" in report else "")

    total_time = time.time() - t_total
    print(f"\nTotal evaluation time: {total_time / 60:.1f} min")
    print(f"Peak RSS: {rss_gb():.1f} GB")


if __name__ == '__main__':
    main()
