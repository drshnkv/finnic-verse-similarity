#!/usr/bin/env python3
"""Build verse-level similarity index using TF-IDF lemma-based cosine.

For each verse, finds the top-K most similar verses based on shared lemmas
weighted by IDF. Uses Strategy F' (corpus + DeepSeek + phonological
gate) for lemma resolution via RunoVerseLemmaResolver.

Two-pass streaming architecture keeps peak memory under ~1.2 GB.

Usage:
  python -u build_verse_tfidf.py --limit 100    # smoke test
  python -u build_verse_tfidf.py                # full run
"""

import argparse
import math
import sys
import time

from verse_similarity_common import (
    ROOT,
    OUTPUT_DIR,
    normalize_word,
    build_vocabulary_streaming,
    build_sparse_matrix_streaming,
    compute_topk_batched,
    similarity_writer,
    classify_match,
    l2_normalize,
    log_memory,
    ensure_output_dir,
    cleanup_progress,
)

# --- Parameters ---
TOP_K = 50
MIN_SCORE = 0.15      # cosine threshold
MIN_DF = 5
MAX_DF_FRAC = 0.05    # skip lemmas in >5% of verses (IDF handles the rest)
BATCH_SIZE = 2000

# Strategy F' lemma-resolution data paths (in deployment/)
DEPLOY_DIR = ROOT.parent / "deployment"
LEXICON_DATA_PATH = DEPLOY_DIR / "lexicon_data.json.gz"
DISTRIBUTION_PATH = DEPLOY_DIR / "wordform_lemma_distribution.json"


def load_runoverse_resolver():
    """Load the Strategy F' lemma resolver.

    Combines corpus-pipeline counts with DeepSeek LLM counts, a phonological
    gate, and a cross-language exemption to resolve ambiguous wordforms.
    """
    if not LEXICON_DATA_PATH.exists():
        print(f"  ERROR: {LEXICON_DATA_PATH} not found.")
        sys.exit(1)
    if not DISTRIBUTION_PATH.exists():
        print(f"  ERROR: {DISTRIBUTION_PATH} not found.")
        sys.exit(1)

    from runoverse_lemma_resolver import RunoVerseLemmaResolver
    t0 = time.time()
    resolver = RunoVerseLemmaResolver(LEXICON_DATA_PATH, DISTRIBUTION_PATH)
    print(f"  RunoVerse resolver ready ({time.time() - t0:.1f}s)")
    return resolver


def make_lemma_extractor(resolver):
    """Create word extractor that maps wordforms to lemmas via Strategy F'."""

    def extract(verse_text, poem_id):
        lemmas = set()
        for token in verse_text.split():
            w = normalize_word(token)
            if not w:
                continue
            lemma = resolver.resolve(w)
            if len(lemma) >= 2:
                lemmas.add(lemma)
        return lemmas

    return extract


def main():
    parser = argparse.ArgumentParser(
        description='Build verse-level TF-IDF cosine similarity index'
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Max verses to process (default: all)')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help=f'Max neighbors per verse (default: {TOP_K})')
    parser.add_argument('--min-score', type=float, default=MIN_SCORE,
                        help=f'Min cosine threshold (default: {MIN_SCORE})')
    parser.add_argument('--exclude-same-poem', action='store_true',
                        help='Exclude all within-poem matches')
    args = parser.parse_args()

    t_total = time.time()
    ensure_output_dir()

    output_path = OUTPUT_DIR / "verse_similarities_tfidf.jsonl.gz"

    print(f"=== Verse-Level TF-IDF Cosine Similarity ===")
    print(f"  Top-K: {args.top_k}, Min score: {args.min_score}")
    print(f"  Limit: {args.limit or 'all'}")
    print()

    # Load Strategy F' resolver
    print("[1/4] Loading RunoVerse lemma resolver (Strategy F')...")
    resolver = load_runoverse_resolver()
    word_extractor = make_lemma_extractor(resolver)

    # Auto-adjust min_df for small test runs
    min_df = MIN_DF
    if args.limit and args.limit < 5000:
        min_df = 2
        print(f"  (auto-lowered min_df to {min_df} for small test)")

    # Auto-detect whether to write metadata (in case Jaccard hasn't run first)
    meta_path = OUTPUT_DIR / "verse_metadata.jsonl"
    if not meta_path.exists():
        print("  [INFO] verse_metadata.jsonl not found, will write metadata")
        write_metadata = True
    else:
        write_metadata = False

    # Pass 1: Build vocabulary over lemmas
    print(f"\n[2/4] Pass 1: Streaming lemma vocabulary...")
    word_to_idx, verse_ids, n_verses, verse_texts, df = build_vocabulary_streaming(
        word_extractor,
        min_df=min_df,
        max_df_frac=MAX_DF_FRAC,
        limit=args.limit,
        write_metadata=write_metadata,
    )

    if n_verses == 0:
        print("  No verses found.")
        sys.exit(1)

    # Compute IDF weights from vocabulary DF counts (no re-scan needed)
    idf = {w: math.log(n_verses / count) for w, count in df.items() if count > 0}
    del df
    print(f"  IDF weights for {len(idf):,} lemmas")

    # Pass 2: Build sparse IDF matrix
    print(f"\n[3/4] Pass 2: Building sparse TF-IDF matrix...")
    mat_csr = build_sparse_matrix_streaming(
        word_extractor, word_to_idx, n_verses,
        idf=idf, limit=args.limit
    )

    # L2-normalize for cosine similarity
    print("  L2-normalizing rows...")
    norm_mat = l2_normalize(mat_csr)
    del mat_csr

    # Compute + stream results directly to output
    print(f"\n[4/4] Computing cosine similarities + writing (batch={BATCH_SIZE})...")
    params = {
        "top_k": args.top_k,
        "min_score": args.min_score,
        "min_df": min_df,
        "max_df_frac": MAX_DF_FRAC,
        "exclude_same_poem": args.exclude_same_poem,
    }

    n_with_matches = 0
    total_pairs = 0
    type_counts = {'w': 0, 's': 0, 'x': 0}

    with similarity_writer(output_path, "tfidf_cosine_lemma", params,
                           n_verses) as write_entry:
        for vid, entries in compute_topk_batched(
            norm_mat, verse_ids, args.top_k, args.min_score,
            batch_size=BATCH_SIZE, algorithm_name="tfidf",
            verse_texts=verse_texts,
        ):
            matches = []
            for other_vid, score in entries:
                mtype = classify_match(vid, other_vid)
                if args.exclude_same_poem and mtype == 'w':
                    continue
                matches.append([other_vid, score, mtype])

            if matches:
                write_entry(vid, 1, matches)
                n_with_matches += 1
                total_pairs += len(matches)
                for m in matches:
                    type_counts[m[2]] += 1

    del norm_mat
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
