#!/usr/bin/env python3
"""Build verse-level similarity using sentence embeddings on FVT translations.

Encodes 4.3M English verse translations with a sentence transformer model,
builds a FAISS IVF+PQ index, and finds top-50 nearest neighbors per verse.
Output key: "s" (sentence embedding) for integration into verse_sim_chunks.

Phases:
  --phase 0   : Build verse registry from FVT chunks
  --phase 1   : Encode all verses to float16 memmap + mean-center
  --phase 2   : Build FAISS IVF+PQ index
  --phase 3   : Search for top-50 neighbors, write JSONL

Usage:
  python -u build_verse_sentence_embedding.py --phase 0
  python -u build_verse_sentence_embedding.py --phase 1
  python -u build_verse_sentence_embedding.py --phase 2
  python -u build_verse_sentence_embedding.py --phase 3
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from verse_similarity_common import (
    DEPLOYMENT,
    OUTPUT_DIR,
    classify_match,
    detect_corpus_lang,
    ensure_output_dir,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FVT_DIR = DEPLOYMENT / "fullverse_translations"
SIM_INDEX_PATH = DEPLOYMENT / "verse_sim_index.json.gz"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ENCODE_BATCH_SIZE = 512
SEARCH_BATCH_SIZE = 5000
TOP_K = 50
MIN_SCORE = 0.30
NPROBE = 64
FAISS_TRAIN_SAMPLE = 200_000
FAISS_ADD_BATCH = 50_000
INDEX_SPEC = "IVF4096,PQ48"  # OPQ48 segfaults on macOS ARM FAISS 1.13.2

CHECKPOINT_INTERVAL = 1000  # batches between flush/checkpoint
RSS_LIMIT_GB = 11.0

# Output paths
REGISTRY_PATH = OUTPUT_DIR / "sentence_embedding_verse_registry.jsonl.gz"
MEMMAP_PATH = OUTPUT_DIR / "verse_embeddings.f16.mmap"
IDS_PATH = OUTPUT_DIR / "verse_embedding_ids.json.gz"
FAISS_INDEX_PATH = OUTPUT_DIR / "verse_sentence_faiss.index"
FAISS_PARTIAL_PATH = OUTPUT_DIR / "verse_sentence_faiss.partial.index"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "verse_similarities_sentence.jsonl.gz"

PROGRESS_DIR = OUTPUT_DIR / ".sentence_embedding_progress"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_rss(label=""):
    """Check RSS memory usage, abort if above limit."""
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    if rss_gb > RSS_LIMIT_GB:
        print(f"ABORT: RSS {rss_gb:.2f} GB exceeds {RSS_LIMIT_GB} GB limit [{label}]")
        sys.exit(1)
    return rss_gb


def progress_path(phase):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return PROGRESS_DIR / f"phase{phase}.json"


def save_phase_progress(phase, data):
    with open(progress_path(phase), 'w') as f:
        json.dump(data, f)


def load_phase_progress(phase):
    p = progress_path(phase)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def clear_phase_progress(phase):
    p = progress_path(phase)
    if p.exists():
        p.unlink()


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def fmt_count(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Phase 0: Data Preparation & Validation
# ---------------------------------------------------------------------------

def run_phase_0():
    """Build verse registry from FVT chunks, validating against sim_index."""
    print("=" * 60)
    print("Phase 0: Data Preparation & Validation")
    print("=" * 60)
    ensure_output_dir()

    # Load verse_sim_index for validation
    print(f"  Loading verse_sim_index from {SIM_INDEX_PATH.name}...")
    with gzip.open(SIM_INDEX_PATH, 'rt') as f:
        sim_index = json.load(f)
    sim_poem_ids = set(sim_index.keys())
    print(f"  Sim index: {len(sim_poem_ids):,} poems")

    # Find FVT chunk files
    fvt_files = sorted(FVT_DIR.glob("fvt_chunk_*.json"))
    if not fvt_files:
        print("ERROR: No fvt_chunk_*.json files found in", FVT_DIR)
        sys.exit(1)
    print(f"  Found {len(fvt_files)} FVT chunks")

    total_verses = 0
    skipped_flagged = 0
    skipped_empty = 0
    skipped_unmatched_poems = 0
    unmatched_poem_ids = []
    corpus_counts = {"et": 0, "fi": 0, "other": 0}

    t0 = time.time()
    with gzip.open(REGISTRY_PATH, 'wt', compresslevel=6) as out_f:
        for chunk_file in fvt_files:
            print(f"  Processing {chunk_file.name}...", end=" ", flush=True)
            with open(chunk_file) as f:
                chunk = json.load(f)

            poems = chunk.get("poems", {})
            chunk_verses = 0
            chunk_skipped = 0

            for poem_title, poem_data in poems.items():
                # Validate against sim_index
                if poem_title not in sim_poem_ids:
                    skipped_unmatched_poems += 1
                    if len(unmatched_poem_ids) < 20:
                        unmatched_poem_ids.append(poem_title)
                    skipped_empty += poem_data.get("vc", 0)
                    continue

                translations = poem_data.get("t", [])
                flags = poem_data.get("f", [])

                _, lang = detect_corpus_lang(poem_title)

                for i, trans in enumerate(translations):
                    # Skip flagged verses
                    if i < len(flags) and flags[i] != 0:
                        skipped_flagged += 1
                        continue
                    # Skip empty translations
                    if not trans or not trans.strip():
                        skipped_empty += 1
                        continue

                    verse_id = f"{poem_title}:{i}"
                    entry = {"v": verse_id, "t": trans.strip()}
                    out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    total_verses += 1
                    chunk_verses += 1
                    corpus_counts[lang if lang in corpus_counts else "other"] += 1

            print(f"{chunk_verses:,} verses")
            check_rss("phase0")

    elapsed = time.time() - t0
    print()
    print(f"  Registry: {REGISTRY_PATH.name}")
    print(f"  Total verses: {total_verses:,}")
    print(f"  Skipped (flagged): {skipped_flagged:,}")
    print(f"  Skipped (empty): {skipped_empty:,}")
    print(f"  Unmatched poems: {skipped_unmatched_poems:,}")
    if unmatched_poem_ids:
        print(f"    Examples: {unmatched_poem_ids[:5]}")
    print(f"  Corpus: ET={corpus_counts['et']:,} / FI={corpus_counts['fi']:,} / other={corpus_counts['other']:,}")
    print(f"  Time: {fmt_time(elapsed)}")
    print(f"  RSS: {check_rss('phase0_done'):.2f} GB")


# ---------------------------------------------------------------------------
# Phase 1: Sentence Encoding
# ---------------------------------------------------------------------------

ONNX_DIR = OUTPUT_DIR / "onnx_minilm"
ONNX_MODEL_PATH = ONNX_DIR / "model.onnx"


def _export_onnx_model():
    """Export sentence-transformers model to ONNX for 5-7x CPU speedup."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    hf_name = f"sentence-transformers/{MODEL_NAME}"

    print(f"    Exporting {hf_name} to ONNX...")
    model = AutoModel.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model.eval()

    dummy = tokenizer("hello world", return_tensors='pt', padding=True)
    torch.onnx.export(
        model,
        (dummy['input_ids'], dummy['attention_mask']),
        str(ONNX_MODEL_PATH),
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'last_hidden_state': {0: 'batch', 1: 'seq'},
        },
        opset_version=14,
    )
    tokenizer.save_pretrained(str(ONNX_DIR))
    del model
    print(f"    ONNX model exported to {ONNX_DIR.name}/")


def _load_onnx_encoder():
    """Load ONNX Runtime session + tokenizer. Returns (session, tokenizer)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    if not ONNX_MODEL_PATH.exists():
        _export_onnx_model()

    sess = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=['CPUExecutionProvider'],
    )
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{MODEL_NAME}")
    return sess, tokenizer


def _encode_batch_onnx(texts, sess, tokenizer, batch_size=256):
    """Encode texts using ONNX Runtime with mean pooling + L2 norm."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors='np'
        )
        outputs = sess.run(None, {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64),
        })
        token_embs = outputs[0]
        mask = encoded['attention_mask'][..., np.newaxis]
        pooled = (token_embs * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-8)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-8)
        all_embs.append(pooled.astype(np.float32))
    return np.vstack(all_embs)


def run_phase_1():
    """Encode all verses to float16 memmap, then mean-center."""
    print("=" * 60)
    print("Phase 1: Sentence Encoding")
    print("=" * 60)

    if not REGISTRY_PATH.exists():
        print("ERROR: Run --phase 0 first to build registry")
        sys.exit(1)

    ensure_output_dir()

    # Count total verses
    print("  Counting verses in registry...")
    total = 0
    with gzip.open(REGISTRY_PATH, 'rt') as f:
        for _ in f:
            total += 1
    print(f"  Total verses: {total:,}")

    # Check for resume
    progress = load_phase_progress(1)
    start_batch = 0
    if progress and progress.get("step") == "encoding":
        start_batch = progress["last_batch"] + 1
        encoded_so_far = progress["encoded"]
        print(f"  Resuming from batch {start_batch} ({encoded_so_far:,} already encoded)")
    else:
        encoded_so_far = 0

    # Create or open memmap
    if start_batch == 0:
        print(f"  Creating memmap: {total} x {EMBEDDING_DIM} x float16 "
              f"= {total * EMBEDDING_DIM * 2 / (1024**3):.2f} GB")
        mmap = np.memmap(
            MEMMAP_PATH, dtype=np.float16, mode='w+',
            shape=(total, EMBEDDING_DIM)
        )
    else:
        mmap = np.memmap(
            MEMMAP_PATH, dtype=np.float16, mode='r+',
            shape=(total, EMBEDDING_DIM)
        )

    # Load ONNX encoder (5-7x faster than PyTorch on this CPU)
    print("  Loading ONNX encoder...")
    sess, tokenizer = _load_onnx_encoder()
    print("  ONNX session ready")

    # Encoding pass
    print(f"  Encoding in batches of {ENCODE_BATCH_SIZE}...")
    t0 = time.time()
    batch_texts = []
    ids_list = []
    batch_idx = 0
    offset = 0

    with gzip.open(REGISTRY_PATH, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            batch_texts.append(entry["t"])
            ids_list.append(entry["v"])

            if len(batch_texts) >= ENCODE_BATCH_SIZE:
                if batch_idx >= start_batch:
                    embeddings = _encode_batch_onnx(batch_texts, sess, tokenizer)
                    mmap[offset:offset + len(embeddings)] = embeddings.astype(np.float16)
                    encoded_so_far += len(batch_texts)

                    if batch_idx % CHECKPOINT_INTERVAL == 0:
                        mmap.flush()
                        save_phase_progress(1, {
                            "step": "encoding",
                            "last_batch": batch_idx,
                            "encoded": encoded_so_far,
                        })
                        elapsed = time.time() - t0
                        rate = encoded_so_far / elapsed if elapsed > 0 else 0
                        eta = (total - encoded_so_far) / rate if rate > 0 else 0
                        rss = check_rss(f"batch_{batch_idx}")
                        print(f"    Batch {batch_idx}: {fmt_count(encoded_so_far)} encoded, "
                              f"{rate:.0f} sent/s, ETA {fmt_time(eta)}, "
                              f"RSS {rss:.1f} GB")

                offset += len(batch_texts)
                batch_texts = []
                batch_idx += 1

    # Final partial batch
    if batch_texts and batch_idx >= start_batch:
        embeddings = _encode_batch_onnx(batch_texts, sess, tokenizer)
        mmap[offset:offset + len(embeddings)] = embeddings.astype(np.float16)
        encoded_so_far += len(batch_texts)

    mmap.flush()
    elapsed = time.time() - t0
    print(f"\n  Encoding complete: {encoded_so_far:,} verses in {fmt_time(elapsed)}")
    if elapsed > 0:
        print(f"  Average: {encoded_so_far/elapsed:.0f} sent/sec")

    # Save verse IDs
    print(f"  Saving verse IDs ({len(ids_list):,})...")
    with gzip.open(IDS_PATH, 'wt', compresslevel=6) as f:
        json.dump(ids_list, f)

    # Free encoder memory before mean-centering
    del sess, tokenizer

    # Mean-centering (anisotropy mitigation)
    mean_progress = load_phase_progress("1_mean")
    if mean_progress and mean_progress.get("done"):
        print("  Mean-centering already completed (skipping)")
    else:
        print("\n  Mean-centering (anisotropy mitigation)...")
        _mean_center_memmap(mmap, total)
        save_phase_progress("1_mean", {"done": True})

    clear_phase_progress(1)
    print("\n  Phase 1 complete.")
    print(f"  Memmap: {MEMMAP_PATH.name} ({MEMMAP_PATH.stat().st_size / (1024**3):.2f} GB)")
    print(f"  IDs: {IDS_PATH.name}")


def _mean_center_memmap(mmap, total):
    """Subtract corpus mean and re-normalize all embeddings in-place."""
    MEAN_BATCH = 50_000
    t0 = time.time()

    # Pass 1: compute mean
    print("    Pass 1: computing corpus mean...")
    running_sum = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for start in range(0, total, MEAN_BATCH):
        end = min(start + MEAN_BATCH, total)
        batch = mmap[start:end].astype(np.float32)
        running_sum += batch.sum(axis=0).astype(np.float64)
    corpus_mean = (running_sum / total).astype(np.float32)
    print(f"    Mean norm: {np.linalg.norm(corpus_mean):.6f}")

    # Pass 2: subtract mean and re-normalize
    print("    Pass 2: subtracting mean and re-normalizing...")
    for start in range(0, total, MEAN_BATCH):
        end = min(start + MEAN_BATCH, total)
        batch = mmap[start:end].astype(np.float32)
        batch -= corpus_mean
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        batch /= norms
        mmap[start:end] = batch.astype(np.float16)

        if start % (MEAN_BATCH * 10) == 0 and start > 0:
            check_rss("mean_center")

    mmap.flush()
    elapsed = time.time() - t0
    print(f"    Mean-centering done in {fmt_time(elapsed)}")


# ---------------------------------------------------------------------------
# Phase 2: FAISS Index Building
# ---------------------------------------------------------------------------

def run_phase_2():
    """Build FAISS IVF+PQ index from memmap embeddings."""
    print("=" * 60)
    print("Phase 2: FAISS Index Building")
    print("=" * 60)

    import faiss

    if not MEMMAP_PATH.exists():
        print("ERROR: Run --phase 1 first to encode embeddings")
        sys.exit(1)
    if not IDS_PATH.exists():
        print("ERROR: Missing verse IDs file")
        sys.exit(1)

    ensure_output_dir()

    # Load IDs to get total count
    print("  Loading verse IDs...")
    with gzip.open(IDS_PATH, 'rt') as f:
        ids_list = json.load(f)
    total = len(ids_list)
    print(f"  Total vectors: {total:,}")
    del ids_list  # free memory

    # Open memmap
    mmap = np.memmap(MEMMAP_PATH, dtype=np.float16, mode='r', shape=(total, EMBEDDING_DIM))

    # Sample training vectors
    print(f"  Sampling {FAISS_TRAIN_SAMPLE:,} vectors for training...")
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(total, size=min(FAISS_TRAIN_SAMPLE, total), replace=False)
    sample_indices.sort()  # sort for sequential I/O locality

    train_vectors = np.empty((len(sample_indices), EMBEDDING_DIM), dtype=np.float32)
    for i, idx in enumerate(sample_indices):
        train_vectors[i] = mmap[idx].astype(np.float32)
    print(f"  Training vectors loaded: {train_vectors.shape}")
    check_rss("phase2_train_loaded")

    # Shuffle for training (FAISS expects random order)
    rng.shuffle(train_vectors)

    # Build index
    print(f"  Creating FAISS index: {INDEX_SPEC}")
    index = faiss.index_factory(EMBEDDING_DIM, INDEX_SPEC, faiss.METRIC_INNER_PRODUCT)
    print("  Training index (this may take a few minutes)...")
    t0 = time.time()
    index.train(train_vectors)
    print(f"  Training done in {fmt_time(time.time() - t0)}")
    del train_vectors
    check_rss("phase2_trained")

    # Add all vectors in batches
    print(f"  Adding {total:,} vectors in batches of {FAISS_ADD_BATCH:,}...")
    t0 = time.time()
    added = 0
    for start in range(0, total, FAISS_ADD_BATCH):
        end = min(start + FAISS_ADD_BATCH, total)
        batch = mmap[start:end].astype(np.float32).copy()
        index.add(batch)
        added += len(batch)

        if added % (FAISS_ADD_BATCH * 10) == 0:
            rss = check_rss(f"phase2_add_{added}")
            elapsed = time.time() - t0
            rate = added / elapsed
            eta = (total - added) / rate if rate > 0 else 0
            print(f"    Added {fmt_count(added)}, {rate:.0f} vec/s, "
                  f"ETA {fmt_time(eta)}, RSS {rss:.1f} GB")

        # Partial save every 500K
        if added % 500_000 == 0 and added > 0:
            faiss.write_index(index, str(FAISS_PARTIAL_PATH))

    elapsed = time.time() - t0
    print(f"  All vectors added in {fmt_time(elapsed)}")

    # Save final index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    idx_size_mb = FAISS_INDEX_PATH.stat().st_size / (1024**2)
    print(f"  Index saved: {FAISS_INDEX_PATH.name} ({idx_size_mb:.1f} MB)")

    # Clean up partial
    if FAISS_PARTIAL_PATH.exists():
        FAISS_PARTIAL_PATH.unlink()

    # Validation: recall@50 on 1K random queries
    print("\n  Validating recall@50 on 1000 random queries...")
    _validate_recall(index, mmap, total)

    print("\n  Phase 2 complete.")


def _validate_recall(index, mmap, total):
    """Check recall@50 of the ANN index vs exact search on 1K queries."""
    import faiss

    rng = np.random.default_rng(123)
    query_indices = rng.choice(total, size=1000, replace=False)
    queries = np.empty((1000, EMBEDDING_DIM), dtype=np.float32)
    for i, idx in enumerate(query_indices):
        queries[i] = mmap[idx].astype(np.float32)

    # ANN search
    index.nprobe = NPROBE
    _, ann_ids = index.search(queries, TOP_K)

    # Exact search (flat index on a small sample isn't feasible for 4.3M,
    # so we compare against the index's own exact results at high nprobe)
    index_high = faiss.clone_index(index)
    index_high.nprobe = min(4096, index.invlists.nlist)
    _, exact_ids = index_high.search(queries, TOP_K)

    # Compute recall
    recalls = []
    for i in range(1000):
        ann_set = set(ann_ids[i][ann_ids[i] >= 0])
        exact_set = set(exact_ids[i][exact_ids[i] >= 0])
        if exact_set:
            recalls.append(len(ann_set & exact_set) / len(exact_set))

    mean_recall = np.mean(recalls)
    print(f"  Recall@{TOP_K} (nprobe={NPROBE} vs nprobe={index_high.nprobe}): "
          f"{mean_recall:.4f}")
    if mean_recall < 0.90:
        print(f"  WARNING: Recall below 90%. Consider increasing nprobe.")


# ---------------------------------------------------------------------------
# Phase 3: Nearest Neighbor Search
# ---------------------------------------------------------------------------

def run_phase_3():
    """Search for top-50 neighbors with adaptive threshold, write JSONL."""
    print("=" * 60)
    print("Phase 3: Nearest Neighbor Search")
    print("=" * 60)

    import faiss

    if not FAISS_INDEX_PATH.exists():
        print("ERROR: Run --phase 2 first to build FAISS index")
        sys.exit(1)

    ensure_output_dir()

    # Load index
    print("  Loading FAISS index...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    index.nprobe = NPROBE
    print(f"  Index loaded: {index.ntotal:,} vectors, nprobe={NPROBE}")

    # Load IDs
    print("  Loading verse IDs...")
    with gzip.open(IDS_PATH, 'rt') as f:
        ids_list = json.load(f)
    total = len(ids_list)
    print(f"  Total verses: {total:,}")

    # Open memmap
    mmap = np.memmap(MEMMAP_PATH, dtype=np.float16, mode='r', shape=(total, EMBEDDING_DIM))

    # Check for resume
    progress = load_phase_progress(3)
    start_batch = 0
    written_count = 0
    if progress:
        start_batch = progress.get("last_batch", 0) + 1
        written_count = progress.get("written", 0)
        print(f"  Resuming from batch {start_batch} ({written_count:,} verses written)")

    # Write output
    output_mode = 'at' if start_batch > 0 else 'wt'
    t0 = time.time()

    with gzip.open(OUTPUT_JSONL_PATH, output_mode, compresslevel=6) as out_f:
        # Write header if starting fresh
        if start_batch == 0:
            header = {
                "_header": True,
                "algorithm": "sentence_embedding",
                "params": {
                    "model": MODEL_NAME,
                    "index": INDEX_SPEC,
                    "nprobe": NPROBE,
                    "top_k": TOP_K,
                    "min_score": MIN_SCORE,
                    "threshold": "adaptive_margin",
                },
                "total_verses": total,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            out_f.write(json.dumps(header, ensure_ascii=False) + '\n')

        batch_idx = 0
        for start in range(0, total, SEARCH_BATCH_SIZE):
            if batch_idx < start_batch:
                batch_idx += 1
                continue

            end = min(start + SEARCH_BATCH_SIZE, total)
            batch_size = end - start

            # Load and convert query batch
            queries = mmap[start:end].astype(np.float32).copy()

            # Search
            scores, indices = index.search(queries, TOP_K)

            # Process each query
            for i in range(batch_size):
                verse_idx = start + i
                query_vid = ids_list[verse_idx]

                # Get valid results (exclude self and invalid indices)
                valid_mask = (indices[i] >= 0) & (indices[i] != verse_idx)
                valid_scores = scores[i][valid_mask]
                valid_indices = indices[i][valid_mask]

                if len(valid_scores) == 0:
                    continue

                # Adaptive threshold: keep neighbors above noise floor
                matches = _apply_adaptive_threshold(
                    query_vid, valid_scores, valid_indices, ids_list
                )

                if matches:
                    entry = {"v": query_vid, "g": len(matches), "m": matches}
                    out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    written_count += 1

            # Checkpoint
            if batch_idx % 100 == 0:
                elapsed = time.time() - t0
                queries_done = end
                rate = queries_done / elapsed if elapsed > 0 else 0
                eta = (total - queries_done) / rate if rate > 0 else 0
                rss = check_rss(f"phase3_batch_{batch_idx}")
                print(f"    Batch {batch_idx}: {fmt_count(queries_done)} queried, "
                      f"{written_count:,} with matches, "
                      f"{rate:.0f} queries/s, ETA {fmt_time(eta)}, "
                      f"RSS {rss:.1f} GB")

            if batch_idx % 50 == 0:
                save_phase_progress(3, {
                    "last_batch": batch_idx,
                    "written": written_count,
                })

            batch_idx += 1

    elapsed = time.time() - t0
    clear_phase_progress(3)

    out_size_mb = OUTPUT_JSONL_PATH.stat().st_size / (1024**2)
    print(f"\n  Search complete in {fmt_time(elapsed)}")
    print(f"  Verses with matches: {written_count:,} / {total:,}")
    print(f"  Output: {OUTPUT_JSONL_PATH.name} ({out_size_mb:.1f} MB)")
    print("\n  Phase 3 complete.")


def _apply_adaptive_threshold(query_vid, scores, indices, ids_list):
    """Apply per-query adaptive filtering + hard floor.

    Strategy: keep neighbors whose score exceeds the noise floor
    (estimated from the tail of the top-50 results).
    """
    # Hard floor filter
    floor_mask = scores >= MIN_SCORE
    scores = scores[floor_mask]
    indices = indices[floor_mask]

    if len(scores) == 0:
        return []

    # Adaptive margin: use the weakest 10 surviving scores as noise estimate
    if len(scores) >= 10:
        tail_start = max(0, len(scores) - 10)
        tail = scores[tail_start:]
        noise_mean = tail.mean()
        noise_std = tail.std()
        adaptive_floor = noise_mean + 2.0 * noise_std
        # Only apply adaptive filter if it's stricter than hard floor
        if adaptive_floor > MIN_SCORE:
            adaptive_mask = scores >= adaptive_floor
            scores = scores[adaptive_mask]
            indices = indices[adaptive_mask]

    if len(scores) == 0:
        return []

    # Build match tuples
    matches = []
    for score, idx in zip(scores, indices):
        target_vid = ids_list[int(idx)]
        match_type = classify_match(query_vid, target_vid)
        # Skip within-poem matches
        if match_type == 'w':
            continue
        matches.append([target_vid, round(float(score), 4), match_type])

    return matches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Build sentence embedding verse similarity"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--phase', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--model', default=MODEL_NAME,
                        help=f"Override model name (default: {MODEL_NAME}). "
                             f"WARNING: only 384-dim models work; EMBEDDING_DIM "
                             f"is hardwired into the memmap shape and FAISS index.")
    args = parser.parse_args()

    if args.model != MODEL_NAME:
        MODEL_NAME = args.model
        print(f"  Using model: {MODEL_NAME}")

    if args.phase == 0:
        run_phase_0()
    elif args.phase == 1:
        run_phase_1()
    elif args.phase == 2:
        run_phase_2()
    elif args.phase == 3:
        run_phase_3()


if __name__ == "__main__":
    main()
