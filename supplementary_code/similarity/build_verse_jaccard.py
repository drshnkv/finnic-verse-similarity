#!/usr/bin/env python3
"""Build verse-level similarity index using Jaccard wordform overlap.

For each verse (~4.3M total), finds the top-K most similar verses based on
shared normalized wordforms. Two-pass streaming architecture keeps peak
memory under ~1.1 GB.

Two variants:
  Standard Jaccard:     J(A,B) = |A∩B| / |A∪B|
  IDF-weighted Jaccard: wJ(A,B) = Σ_{w∈A∩B} idf(w) / Σ_{w∈A∪B} idf(w)

Usage:
  python -u build_verse_jaccard.py --limit 100          # smoke test
  python -u build_verse_jaccard.py                      # full run
  python -u build_verse_jaccard.py --idf-weighted       # IDF variant
  python -u build_verse_jaccard.py --exclude-stopwords  # remove function words
  python -u build_verse_jaccard.py --exclude-same-poem  # filter within-poem
  python -u build_verse_jaccard.py --fallback           # force inverted index
"""

import argparse
import math
import sys
import time

import numpy as np

from verse_similarity_common import (
    OUTPUT_DIR,
    normalize_word,
    build_vocabulary_streaming,
    build_sparse_matrix_streaming,
    build_compact_store_streaming,
    similarity_writer,
    classify_match,
    check_disk_space,
    log_memory,
    ensure_output_dir,
    cleanup_progress,
)

# --- Parameters ---
TOP_K = 50
MIN_SCORE = 0.2       # higher than poem-level (short docs)
MIN_SHARED = 2        # require >= 2 shared words
MIN_DF = 5
MAX_DF_FRAC = 0.05    # skip words in >5% of verses (consistent with TF-IDF)
BATCH_SIZE = 2000     # rows per batch for sparse matmul


def make_word_extractor(exclude_stopwords=False):
    """Create word extractor function for Jaccard (raw wordforms)."""
    from verse_similarity_common import RUNOSONG_STOPWORDS

    def extract(verse_text, poem_id):
        words = set()
        for token in verse_text.split():
            w = normalize_word(token)
            if w:
                if exclude_stopwords and w in RUNOSONG_STOPWORDS:
                    continue
                words.add(w)
        return words

    return extract


def compute_jaccard_sparse(mat_csr, verse_ids, idf_weights,
                            top_k, min_score, min_shared,
                            exclude_same_poem, exclude_adjacent,
                            verse_texts=None, adaptive_min_shared=False):
    """Compute Jaccard similarity using sparse matrix.

    Keeps matmul result SPARSE to avoid O(N) dense memory per batch row.
    At 4.29M verses, dense would be 32 GB per batch vs ~50 MB sparse.

    For standard Jaccard (binary matrix, cells = 1.0):
        dot(A, B) = |A ∩ B|,  union = |A| + |B| - |A ∩ B|
        J = |A ∩ B| / |A ∪ B|

    For IDF-weighted Jaccard (matrix cells = sqrt(idf)):
        dot(A, B) = Σ sqrt(idf)*sqrt(idf) = Σ idf  (weighted intersection)
        verse_weight = Σ idf  (via mat.multiply(mat).sum)
        wJ = Σ_{A∩B} idf / Σ_{A∪B} idf

    Args:
        adaptive_min_shared: If True, use min(min_shared, verse_word_count) per verse.

    Yields:
        (verse_id, group_size, matches_list)
    """
    N = mat_csr.shape[0]
    use_idf = idf_weights is not None

    # Precompute verse weights: mat.multiply(mat).sum() works for both cases
    verse_weights = np.array(mat_csr.multiply(mat_csr).sum(axis=1)).ravel()

    # Precompute per-verse word counts from CSR structure (nnz per row)
    verse_word_counts = np.diff(mat_csr.indptr)  # matrix words per verse

    if adaptive_min_shared:
        n_adapted = int(np.sum(verse_word_counts < min_shared))
        print(f"  Adaptive MIN_SHARED: {n_adapted:,} verses will use reduced threshold")

    # For IDF mode: also build a binary matrix to get true shared word counts
    if use_idf:
        bin_mat = mat_csr.copy()
        bin_mat.data[:] = 1.0
    else:
        bin_mat = None

    t_start = time.time()

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        batch = mat_csr[start:end]
        intersect_block = batch @ mat_csr.T  # keep sparse

        # For IDF mode: binary intersection for true shared counts
        if use_idf:
            bin_batch = bin_mat[start:end]
            bin_intersect_block = bin_batch @ bin_mat.T  # keep sparse

        for local_i in range(end - start):
            global_i = start + local_i
            w_a = verse_weights[global_i]

            # Get sparse row: only non-zero intersections
            row = intersect_block.getrow(local_i)
            cols = row.indices.copy()
            vals = row.data.copy()

            # Exclude self
            self_mask = cols != global_i
            cols = cols[self_mask]
            vals = vals[self_mask]

            if len(cols) == 0:
                continue

            # Compute Jaccard for non-zero entries only
            other_weights = verse_weights[cols]
            unions = w_a + other_weights - vals
            unions[unions == 0] = 1
            jaccard_vals = vals / unions

            # Filter by min_score
            score_mask = jaccard_vals >= min_score
            cols = cols[score_mask]
            jaccard_vals = jaccard_vals[score_mask]
            vals = vals[score_mask]

            if len(cols) == 0:
                continue

            # Get shared word counts
            if use_idf:
                bin_row = bin_intersect_block.getrow(local_i)
                # Look up shared count for each col in the sparse bin row
                bin_cols_set = dict(zip(bin_row.indices, bin_row.data.astype(int)))
                shared_counts = np.array([bin_cols_set.get(c, 0) for c in cols])
            else:
                shared_counts = vals.astype(int)

            # Filter by min_shared (adaptive: per-verse threshold)
            if adaptive_min_shared:
                min_shared_here = min(min_shared, int(verse_word_counts[global_i]))
            else:
                min_shared_here = min_shared
            shared_mask = shared_counts >= min_shared_here
            cols = cols[shared_mask]
            jaccard_vals = jaccard_vals[shared_mask]
            shared_counts = shared_counts[shared_mask]

            if len(cols) == 0:
                continue

            # Top-K selection (overselect for text dedup headroom)
            select_k = top_k * 5 if verse_texts else top_k
            if len(cols) <= select_k:
                order = np.argsort(-jaccard_vals)
            else:
                top_idx = np.argpartition(-jaccard_vals, select_k)[:select_k]
                order = top_idx[np.argsort(-jaccard_vals[top_idx])]

            vid = verse_ids[global_i]
            entries = []
            seen_texts = set() if verse_texts else None
            for idx in order:
                j = int(cols[idx])
                score = float(jaccard_vals[idx])
                shared = int(shared_counts[idx])

                other_vid = verse_ids[j]
                mtype = classify_match(vid, other_vid)

                if exclude_same_poem and mtype == 'w':
                    continue
                if exclude_adjacent and mtype == 'w':
                    vi_a = int(vid.rsplit(':', 1)[1])
                    vi_b = int(other_vid.rsplit(':', 1)[1])
                    if abs(vi_a - vi_b) <= 1:
                        continue

                # Text-based dedup
                if verse_texts:
                    txt = verse_texts[j]
                    if txt in seen_texts:
                        continue
                    seen_texts.add(txt)

                entries.append([other_vid, round(score, 4), shared, mtype])
                if verse_texts and len(entries) >= top_k:
                    break

            if entries:
                yield (vid, 1, entries)

        del intersect_block
        if use_idf:
            del bin_intersect_block

        elapsed = time.time() - t_start
        rate = end / elapsed if elapsed > 0 else 0
        remaining = (N - end) / rate if rate > 0 else 0
        rss = log_memory()
        print(f"  {end:,}/{N:,} ({end / N * 100:.1f}%) - "
              f"{rate:.0f}/s, ~{remaining:.0f}s left, RSS={rss:.0f} MB")
        sys.stdout.flush()

        if rss > 3000:
            print(f"  WARNING: RSS={rss:.0f} MB exceeds 3 GB safety limit!")

        if end % 100000 < BATCH_SIZE:
            check_disk_space(min_gb=3.0)


def compute_jaccard_inverted(store, inv_index, verse_ids, word_to_idx,
                              idf_weights, top_k, min_score, min_shared,
                              exclude_same_poem, exclude_adjacent,
                              verse_texts=None, adaptive_min_shared=False):
    """Compute Jaccard using inverted index (fallback, no sparse_dot_topn needed).

    Uses rare-word-first traversal and early termination.
    """
    from collections import Counter as CCounter

    N = len(verse_ids)
    use_idf = idf_weights is not None

    # Sort posting lists by length (rare words first)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    posting_lens = {idx: len(arr) for idx, arr in inv_index.items()}

    # Precompute verse weights for IDF mode
    if use_idf:
        verse_weights = np.zeros(N, dtype=np.float32)
        for row in range(N):
            word_indices = store.get_verse(row)
            verse_weights[row] = sum(
                idf_weights.get(idx_to_word.get(wi, ''), 0.0) for wi in word_indices
            )

    t_start = time.time()
    results = []

    for row_i in range(N):
        word_indices = store.get_verse(row_i)
        # Adaptive: use min(min_shared, verse_word_count)
        min_shared_here = min(min_shared, len(word_indices)) if adaptive_min_shared else min_shared
        if len(word_indices) < min_shared_here:
            continue

        # Collect candidates: count shared words per candidate verse
        candidate_counts = CCounter()
        sorted_indices = sorted(word_indices, key=lambda wi: posting_lens.get(wi, 0))

        for wi in sorted_indices:
            if wi in inv_index:
                for other_row in inv_index[wi]:
                    if other_row != row_i:
                        candidate_counts[other_row] += 1

        # Score candidates
        scored = []
        if use_idf:
            w_a = verse_weights[row_i]
            word_set_a = set(word_indices)
            for other_row, shared_count in candidate_counts.items():
                if shared_count < min_shared_here:
                    continue
                other_indices = store.get_verse(other_row)
                shared_indices = word_set_a & set(other_indices)
                intersect_w = sum(
                    idf_weights.get(idx_to_word.get(wi, ''), 0.0) for wi in shared_indices
                )
                w_b = verse_weights[other_row]
                union_w = w_a + w_b - intersect_w
                if union_w == 0:
                    continue
                score = intersect_w / union_w
                if score >= min_score:
                    scored.append((other_row, score, len(shared_indices)))
        else:
            size_a = len(word_indices)
            for other_row, shared_count in candidate_counts.items():
                if shared_count < min_shared_here:
                    continue
                size_b = store.verse_size(other_row)
                union_size = size_a + size_b - shared_count
                if union_size == 0:
                    continue
                score = shared_count / union_size
                if score >= min_score:
                    scored.append((other_row, score, shared_count))

        # Top-K (overselect for text dedup)
        scored.sort(key=lambda x: -x[1])
        select_k = top_k * 5 if verse_texts else top_k
        scored = scored[:select_k]

        vid = verse_ids[row_i]
        entries = []
        seen_texts = set() if verse_texts else None
        for other_row, score, shared in scored:
            other_vid = verse_ids[other_row]
            mtype = classify_match(vid, other_vid)

            if exclude_same_poem and mtype == 'w':
                continue
            if exclude_adjacent and mtype == 'w':
                vi_a = int(vid.rsplit(':', 1)[1])
                vi_b = int(other_vid.rsplit(':', 1)[1])
                if abs(vi_a - vi_b) <= 1:
                    continue

            if verse_texts:
                txt = verse_texts[other_row]
                if txt in seen_texts:
                    continue
                seen_texts.add(txt)

            entries.append([other_vid, round(score, 4), shared, mtype])
            if verse_texts and len(entries) >= top_k:
                break

        if entries:
            results.append((vid, 1, entries))

        # Progress
        if (row_i + 1) % 50000 == 0:
            elapsed = time.time() - t_start
            rate = (row_i + 1) / elapsed if elapsed > 0 else 0
            remaining = (N - row_i - 1) / rate if rate > 0 else 0
            rss = log_memory()
            print(f"  {row_i + 1:,}/{N:,} ({(row_i + 1) / N * 100:.1f}%) - "
                  f"{rate:.0f}/s, ~{remaining:.0f}s left, RSS={rss:.0f} MB")
            sys.stdout.flush()
            check_disk_space(min_gb=3.0)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Build verse-level Jaccard similarity index'
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Max verses to process (default: all)')
    parser.add_argument('--idf-weighted', action='store_true',
                        help='Use IDF-weighted Jaccard')
    parser.add_argument('--exclude-stopwords', action='store_true',
                        help='Remove RUNOSONG_STOPWORDS before comparison')
    parser.add_argument('--exclude-same-poem', action='store_true',
                        help='Exclude all within-poem matches')
    parser.add_argument('--exclude-adjacent', action='store_true',
                        help='Exclude only adjacent verse matches')
    parser.add_argument('--fallback', action='store_true',
                        help='Force inverted-index path (no sparse matmul)')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help=f'Max neighbors per verse (default: {TOP_K})')
    parser.add_argument('--min-score', type=float, default=MIN_SCORE,
                        help=f'Min similarity threshold (default: {MIN_SCORE})')
    parser.add_argument('--min-shared', type=str, default=str(MIN_SHARED),
                        help='Min shared words: integer or "adaptive" (default: 2)')
    parser.add_argument('--min-df', type=int, default=MIN_DF,
                        help=f'Min document frequency for vocabulary (default: {MIN_DF})')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output filename (e.g., "_config_B")')
    args = parser.parse_args()

    t_total = time.time()
    ensure_output_dir()

    # Parse min_shared: integer or "adaptive"
    adaptive_min_shared = args.min_shared.lower() == 'adaptive'
    if adaptive_min_shared:
        min_shared_val = MIN_SHARED  # base value for adaptive
    else:
        min_shared_val = int(args.min_shared)

    # Build output filename
    suffix = '_idf' if args.idf_weighted else ''
    suffix += args.output_suffix
    output_path = OUTPUT_DIR / f"verse_similarities_jaccard{suffix}.jsonl.gz"

    # Use CLI min_df (with auto-adjust for small test runs)
    min_df = args.min_df

    print(f"=== Verse-Level Jaccard Similarity ===")
    print(f"  IDF-weighted: {args.idf_weighted}")
    print(f"  Exclude stopwords: {args.exclude_stopwords}")
    print(f"  Top-K: {args.top_k}, Min score: {args.min_score}")
    print(f"  MIN_SHARED: {'adaptive' if adaptive_min_shared else min_shared_val}")
    print(f"  MIN_DF: {min_df}")
    print(f"  Limit: {args.limit or 'all'}")
    print(f"  Strategy: {'inverted index (forced)' if args.fallback else 'sparse matmul'}")
    print()

    # Check strategy
    use_sparse = not args.fallback

    # Word extractor
    word_extractor = make_word_extractor(exclude_stopwords=args.exclude_stopwords)

    # Auto-adjust min_df for small test runs
    if args.limit and args.limit < 5000 and min_df > 2:
        min_df = 2
        print(f"  (auto-lowered min_df to {min_df} for small test)")

    # Pass 1: Build vocabulary
    print("[1/4] Pass 1: Streaming vocabulary...")
    word_to_idx, verse_ids, n_verses, verse_texts, df_filtered = build_vocabulary_streaming(
        word_extractor,
        min_df=min_df,
        max_df_frac=MAX_DF_FRAC,
        limit=args.limit,
        exclude_stopwords=args.exclude_stopwords,
    )

    if n_verses == 0:
        print("  No verses found. Check poem chunk files.")
        sys.exit(1)

    # Compute IDF weights if needed (using DF from vocabulary pass — no re-scan)
    idf_weights = None
    if args.idf_weighted:
        print("  Computing IDF weights from vocabulary DF counts...")
        idf_weights = {}
        for w, count in df_filtered.items():
            if count > 0:
                val = math.log(n_verses / count)
                if val > 0:
                    idf_weights[w] = val
        print(f"  IDF weights computed for {len(idf_weights):,} terms")
    del df_filtered

    # For IDF-weighted sparse path: store sqrt(idf) in matrix so that
    # dot product gives Σ sqrt(idf)*sqrt(idf) = Σ idf (correct weighted intersection)
    idf_for_matrix = None
    if idf_weights and not args.fallback:
        idf_for_matrix = {w: math.sqrt(v) for w, v in idf_weights.items()}

    # Pass 2: Build data structure
    if use_sparse:
        print(f"\n[2/4] Pass 2: Building sparse matrix...")
        mat_csr = build_sparse_matrix_streaming(
            word_extractor, word_to_idx, n_verses,
            idf=idf_for_matrix, limit=args.limit
        )

        # Compute + stream results directly to output
        print(f"\n[3/4] Computing Jaccard similarities + writing (sparse, batch={BATCH_SIZE})...")
        params = {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "min_shared": "adaptive" if adaptive_min_shared else min_shared_val,
            "min_df": min_df,
            "max_df_frac": MAX_DF_FRAC,
            "idf_weighted": args.idf_weighted,
            "exclude_stopwords": args.exclude_stopwords,
            "exclude_same_poem": args.exclude_same_poem,
            "exclude_adjacent": args.exclude_adjacent,
            "strategy": "sparse",
        }

        n_with_matches = 0
        total_pairs = 0
        type_counts = {'w': 0, 's': 0, 'x': 0}

        with similarity_writer(output_path, "jaccard_wordform", params,
                               n_verses) as write_entry:
            for vid, group_size, matches in compute_jaccard_sparse(
                mat_csr, verse_ids, idf_weights,
                top_k=args.top_k, min_score=args.min_score,
                min_shared=min_shared_val,
                exclude_same_poem=args.exclude_same_poem,
                exclude_adjacent=args.exclude_adjacent,
                verse_texts=verse_texts,
                adaptive_min_shared=adaptive_min_shared,
            ):
                write_entry(vid, group_size, matches)
                n_with_matches += 1
                total_pairs += len(matches)
                for m in matches:
                    type_counts[m[3]] += 1

        del mat_csr
    else:
        print(f"\n[2/4] Pass 2: Building CompactVerseStore + inverted index...")
        store, inv_index = build_compact_store_streaming(
            word_extractor, word_to_idx, n_verses, limit=args.limit
        )

        # Compute similarities (inverted path returns list — keep as-is for fallback)
        print(f"\n[3/4] Computing Jaccard similarities (inverted index)...")
        results = compute_jaccard_inverted(
            store, inv_index, verse_ids, word_to_idx, idf_weights,
            top_k=args.top_k, min_score=args.min_score,
            min_shared=min_shared_val,
            exclude_same_poem=args.exclude_same_poem,
            exclude_adjacent=args.exclude_adjacent,
            verse_texts=verse_texts,
            adaptive_min_shared=adaptive_min_shared,
        )
        del store, inv_index

        # Write output
        print(f"\n[4/4] Writing {len(results):,} verses with matches...")
        params = {
            "top_k": args.top_k,
            "min_score": args.min_score,
            "min_shared": "adaptive" if adaptive_min_shared else min_shared_val,
            "min_df": min_df,
            "max_df_frac": MAX_DF_FRAC,
            "idf_weighted": args.idf_weighted,
            "exclude_stopwords": args.exclude_stopwords,
            "exclude_same_poem": args.exclude_same_poem,
            "exclude_adjacent": args.exclude_adjacent,
            "strategy": "inverted",
        }

        n_with_matches = len(results)
        total_pairs = sum(len(m) for _, _, m in results)
        type_counts = {'w': 0, 's': 0, 'x': 0}
        for _, _, matches in results:
            for m in matches:
                type_counts[m[3]] += 1

        with similarity_writer(output_path, "jaccard_wordform", params,
                               n_verses) as write_entry:
            for vid, group_size, matches in results:
                write_entry(vid, group_size, matches)

    cleanup_progress(output_path)

    print(f"\n=== Summary ===")
    print(f"  Total verses processed: {n_verses:,}")
    print(f"  Verses with matches: {n_with_matches:,} ({n_with_matches / n_verses * 100:.1f}%)")
    print(f"  Total match pairs: {total_pairs:,}")
    print(f"  Match types: within-poem={type_counts['w']:,}, "
          f"cross-poem={type_counts['s']:,}, cross-lingual={type_counts['x']:,}")
    print(f"  Total time: {(time.time() - t_total) / 60:.1f} minutes")
    print(f"  Peak RSS: {log_memory():.0f} MB")


if __name__ == "__main__":
    main()
