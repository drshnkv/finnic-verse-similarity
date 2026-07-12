#!/usr/bin/env python3
"""Build verse-level similarity index using character bigram cosine.

Inspired by FILTER's shortsim-ngrcos (Janicki, Kallio & Sarv 2023 DSH 38(1)):
captures subword orthographic similarity that word-level algorithms miss.

Key design choices following FILTER research:
  - Top-200 most frequent character bigrams (dense vectors, like shortsim)
  - Binary weighting with L2 normalization
  - FAISS IVFFlat for approximate nearest neighbor search (CPU-efficient)
  - Higher threshold (0.6) than word-level algorithms

Uses FAISS for efficient CPU-based search. Memory-optimized: vectors are built
via memmap (avoids double RAM during index construction), then the memmap is
deleted before search. Query vectors are reconstructed from the FAISS index
itself via reconstruct_n(), keeping peak RSS to ~3.5 GB.

Usage:
  python -u build_verse_charbigram.py --limit 500    # smoke test
  python -u build_verse_charbigram.py                 # full run
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter

import faiss
import numpy as np

from verse_similarity_common import (
    OUTPUT_DIR,
    normalize_word,
    get_poem_chunks,
    similarity_writer,
    classify_match,
    log_memory,
    ensure_output_dir,
    cleanup_progress,
)

# --- Parameters ---
TOP_K = 50
MIN_SCORE = 0.6       # FILTER uses 0.7-0.8
DIM = 200             # top-N most frequent bigrams (FILTER default)
NLIST = 4096           # IVF cells
NPROBE = 32            # cells to search per query (64 was too slow on CPU)
SEARCH_BATCH = 10000   # queries per FAISS search batch
TRAIN_SAMPLE = 200000  # vectors to train IVF quantizer (>= nlist*39)


def stream_verses(limit=None):
    """Stream (poem_id, verse_idx, verse_text) from poem chunks."""
    chunks = get_poem_chunks()
    n = 0
    for chunk_path in chunks:
        with open(chunk_path) as f:
            chunk = json.load(f)
        # Handle both old format ({pid: {"v": [...]}}) and new format ({"metadata":..., "poems": {pid: {"text": "..."}}})
        poems_data = chunk.get("poems", chunk) if isinstance(chunk, dict) else chunk
        for poem_id, poem in poems_data.items():
            if poem_id == "metadata":
                continue
            if not isinstance(poem, dict):
                continue
            # Old format: "v" array of verse strings
            # New format: "text" field with "/" or newline-separated verses
            verses = poem.get("v", [])
            if not verses:
                text = poem.get("text", "")
                if text:
                    if " / " in text:
                        verses = [v.strip() for v in text.split(" / ") if v.strip()]
                    elif "\n" in text:
                        verses = [v.strip() for v in text.split("\n") if v.strip()]
                    else:
                        verses = [text.strip()] if text.strip() else []
            if not verses:
                continue
            for vi, verse_text in enumerate(verses):
                if limit and n >= limit:
                    return
                yield poem_id, vi, verse_text
                n += 1


def normalize_verse(verse_text):
    """Normalize verse text: lowercase, strip punctuation, rejoin."""
    tokens = [normalize_word(t) for t in verse_text.split()]
    return ' '.join(t for t in tokens if t)


def build_bigram_vocab(limit=None):
    """Pass 1: Count bigram frequencies and select top-DIM.

    Returns:
        bigram_to_idx: dict mapping top-DIM bigrams to column indices
        verse_ids: list of verse_id strings
        verse_texts_norm: list of normalized texts (for vectorization)
    """
    print(f"[1/5] Pass 1: Counting bigram frequencies...")
    t0 = time.time()

    bigram_freq = Counter()
    verse_ids = []
    verse_texts_norm = []
    n = 0

    for poem_id, vi, verse_text in stream_verses(limit):
        text = normalize_verse(verse_text)
        if len(text) < 2:
            continue
        vid = f"{poem_id}:{vi}"
        verse_ids.append(vid)
        verse_texts_norm.append(text)

        bigrams = set(text[i:i+2] for i in range(len(text) - 1))
        bigram_freq.update(bigrams)
        n += 1
        if n % 1_000_000 == 0:
            print(f"  ... {n:,} verses")

    top_bigrams = bigram_freq.most_common(DIM)
    bigram_to_idx = {bg: i for i, (bg, _) in enumerate(top_bigrams)}

    elapsed = time.time() - t0
    print(f"  {len(verse_ids):,} verses, {len(bigram_freq):,} unique bigrams")
    print(f"  Top-{DIM} bigrams selected (min freq: {top_bigrams[-1][1]:,})")
    print(f"  Pass 1 time: {elapsed:.1f}s")

    return bigram_to_idx, verse_ids, verse_texts_norm


def vectorize_to_memmap(verse_texts_norm, bigram_to_idx, memmap_path):
    """Pass 2: Build dense binary vectors and save to memmap file.

    Returns the memmap array (read-only after creation).
    """
    n = len(verse_texts_norm)
    dim = len(bigram_to_idx)
    print(f"\n[2/5] Pass 2: Building dense {dim}-dim binary vectors → memmap...")
    t0 = time.time()

    # Create memmap file
    vectors = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(n, dim))

    # Fill in batches
    VBATCH = 100000
    for start in range(0, n, VBATCH):
        end = min(start + VBATCH, n)
        batch = np.zeros((end - start, dim), dtype=np.float32)

        for i in range(start, end):
            text = verse_texts_norm[i]
            bigrams = set(text[j:j+2] for j in range(len(text) - 1))
            for bg in bigrams:
                idx = bigram_to_idx.get(bg)
                if idx is not None:
                    batch[i - start, idx] = 1.0

        # L2-normalize batch
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch /= norms

        vectors[start:end] = batch

        if end % 1_000_000 == 0 or end == n:
            print(f"  ... {end:,}/{n:,} vectorized")

    vectors.flush()
    mem_mb = n * dim * 4 / 1024 / 1024
    elapsed = time.time() - t0
    print(f"  Vectors: {n:,} x {dim}, {mem_mb:.0f} MB on disk")
    print(f"  Pass 2 time: {elapsed:.1f}s")

    return vectors


def build_faiss_index(vectors, n_verses):
    """Build FAISS IVFFlat index for inner product search."""
    dim = vectors.shape[1]
    print(f"\n[3/5] Building FAISS IVFFlat index (nlist={NLIST}, nprobe={NPROBE})...")
    t0 = time.time()

    if n_verses < NLIST * 40:
        print("  Using brute-force (dataset too small for IVF)")
        index = faiss.IndexFlatIP(dim)
        # Add in batches (memmap reads)
        for start in range(0, n_verses, 100000):
            end = min(start + 100000, n_verses)
            index.add(np.array(vectors[start:end]))
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, NLIST, faiss.METRIC_INNER_PRODUCT)

        # Train on random subset (read from memmap)
        train_n = min(TRAIN_SAMPLE, n_verses)
        train_idx = np.random.RandomState(42).choice(n_verses, train_n, replace=False)
        train_idx.sort()  # sequential access for memmap
        train_vecs = np.array(vectors[train_idx])  # copy into RAM
        print(f"  Training on {train_n:,} vectors...")
        index.train(train_vecs)
        del train_vecs
        gc.collect()

        # Add all vectors in batches
        print(f"  Adding {n_verses:,} vectors...")
        for start in range(0, n_verses, 100000):
            end = min(start + 100000, n_verses)
            batch = np.array(vectors[start:end])  # copy from memmap
            index.add(batch)
            if end % 500000 == 0 or end == n_verses:
                print(f"    ... {end:,}/{n_verses:,} added")

        index.nprobe = NPROBE

    # Enable vector reconstruction (for query vectors during search)
    if hasattr(index, 'make_direct_map'):
        print(f"  Building direct map for reconstruct_n()...")
        index.make_direct_map()

    elapsed = time.time() - t0
    print(f"  Index built in {elapsed:.1f}s")
    print(f"  RSS: {log_memory():.0f} MB")

    return index


def search_and_write(index, verse_ids, n_verses, top_k, min_score,
                     output_path, exclude_same_poem=False):
    """Search FAISS index and write results.

    Uses index.reconstruct_n() for query vectors (no external vector storage
    needed during search), keeping memory to just the FAISS index + direct map.
    """
    faiss_k = min(top_k + 1, n_verses)

    print(f"\n[4/5] Searching + writing (batch={SEARCH_BATCH}, k={faiss_k})...")
    t0 = time.time()

    params = {
        "top_k": top_k,
        "min_score": min_score,
        "dim": DIM,
        "nlist": NLIST,
        "nprobe": NPROBE,
        "weighting": "binary",
        "method": "FILTER-inspired (Janicki et al. 2023) + FAISS IVFFlat",
    }

    n_with_matches = 0
    total_pairs = 0
    type_counts = {'w': 0, 's': 0, 'x': 0}

    with similarity_writer(output_path, "charbigram_cosine", params,
                           n_verses) as write_entry:
        for start in range(0, n_verses, SEARCH_BATCH):
            end = min(start + SEARCH_BATCH, n_verses)
            batch = index.reconstruct_n(start, end - start)

            scores, indices = index.search(batch, faiss_k)

            for local_i in range(end - start):
                global_i = start + local_i
                vid = verse_ids[global_i]

                matches = []
                for j in range(faiss_k):
                    neighbor_idx = int(indices[local_i, j])
                    score = float(scores[local_i, j])

                    if neighbor_idx < 0 or neighbor_idx == global_i:
                        continue
                    if score < min_score:
                        continue

                    other_vid = verse_ids[neighbor_idx]
                    mtype = classify_match(vid, other_vid)
                    if exclude_same_poem and mtype == 'w':
                        continue
                    matches.append([other_vid, round(score, 4), mtype])

                if matches:
                    write_entry(vid, 1, matches[:top_k])
                    n_with_matches += 1
                    total_pairs += len(matches[:top_k])
                    for m in matches[:top_k]:
                        type_counts[m[2]] += 1

            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            remaining = (n_verses - end) / rate if rate > 0 else 0
            if end % (SEARCH_BATCH * 10) == 0 or end == n_verses:
                print(f"  {end:,}/{n_verses:,} ({end/n_verses*100:.1f}%) - "
                      f"{rate:.0f}/s, ~{remaining:.0f}s left, "
                      f"RSS={log_memory():.0f} MB")

    return n_with_matches, total_pairs, type_counts


def main():
    parser = argparse.ArgumentParser(
        description='Build verse-level character bigram cosine similarity index'
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

    output_path = OUTPUT_DIR / "verse_similarities_charbigram.jsonl.gz"

    print(f"=== Verse-Level Character Bigram Cosine Similarity (FAISS) ===")
    print(f"  Top-K: {args.top_k}, Min score: {args.min_score}")
    print(f"  Dimensions: {DIM} (top-{DIM} most frequent bigrams)")
    print(f"  FAISS: IVFFlat, nlist={NLIST}, nprobe={NPROBE}")
    print(f"  Weighting: binary (FILTER-inspired)")
    print(f"  Limit: {args.limit or 'all'}")
    print()

    # Pass 1: Build bigram vocabulary
    bigram_to_idx, verse_ids, verse_texts_norm = build_bigram_vocab(args.limit)

    n_verses = len(verse_ids)
    if n_verses == 0:
        print("  No verses found.")
        sys.exit(1)

    # Pass 2: Vectorize to memmap file (avoids double RAM for vectors + FAISS)
    memmap_path = str(OUTPUT_DIR / "_charbigram_vectors.tmp")
    vectors = vectorize_to_memmap(verse_texts_norm, bigram_to_idx, memmap_path)

    # Free verse texts and vocab (no longer needed)
    del verse_texts_norm
    del bigram_to_idx
    gc.collect()
    print(f"  RSS after cleanup: {log_memory():.0f} MB")

    # Build FAISS index (reads from memmap, stores internally)
    index = build_faiss_index(vectors, n_verses)

    # Free memmap before search — query vectors come from reconstruct_n()
    del vectors
    gc.collect()
    try:
        os.unlink(memmap_path)
        print(f"  Deleted memmap, RSS after: {log_memory():.0f} MB")
    except OSError:
        pass

    # Search + write (uses reconstruct_n for queries, no external storage)
    n_with_matches, total_pairs, type_counts = search_and_write(
        index, verse_ids, n_verses, args.top_k, args.min_score,
        output_path, args.exclude_same_poem,
    )

    # Cleanup
    del index
    gc.collect()

    cleanup_progress(output_path)

    print(f"\n[5/5] Summary")
    print(f"  Total verses processed: {n_verses:,}")
    print(f"  Verses with matches: {n_with_matches:,} ({n_with_matches / n_verses * 100:.1f}%)")
    print(f"  Total match pairs: {total_pairs:,}")
    print(f"  Match types: within-poem={type_counts['w']:,}, "
          f"cross-poem={type_counts['s']:,}, cross-lingual={type_counts['x']:,}")
    print(f"  Total time: {(time.time() - t_total) / 60:.1f} minutes")
    print(f"  Peak RSS: {log_memory():.0f} MB")


if __name__ == "__main__":
    main()
