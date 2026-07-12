#!/usr/bin/env python3
"""Shared infrastructure for verse-level similarity computation.

Two-pass streaming architecture: verses are never stored as Python sets.
Pass 1 builds vocabulary + DF counts, Pass 2 builds sparse matrix directly.

Peak memory target: ~500 MB (vocabulary + verse_ids + metadata on disk).
"""

import gzip
import html
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix

try:
    from sparse_dot_topn import sp_matmul_topn
    _HAS_SPARSE_DOT_TOPN = True
except ImportError:
    _HAS_SPARSE_DOT_TOPN = False

ROOT = Path(__file__).resolve().parent          # similarity/
PROJECT_ROOT = ROOT.parent                        # finnic_lexicon_annotated/
DEPLOYMENT = PROJECT_ROOT / "deployment"
POEMS_DIR = DEPLOYMENT / "poems"
OUTPUT_DIR = ROOT / "output"

# välismaa FI-language override: 68 ERAB-archived pids whose text is
# Finnish/Ingrian. Library-level patch — every consumer of detect_corpus_lang
# inherits the override automatically.
sys.path.insert(0, str(PROJECT_ROOT))
from valismaa_override import is_fi_override, is_et_override  # noqa: E402

PUNCT_RE = re.compile(
    r'[.,;:!?"\'„\u201c\u201d\u2018\u2019\u00ab\u00bb\-\u2013\u2014()\[\]{}]'
)

MIN_WORD_LEN = 2

# Ultra-high-frequency stopwords in runosong corpus
RUNOSONG_STOPWORDS = frozenset({
    "ja", "ei", "on", "se", "et", "ka", "nii", "kui", "aga",
    "jo", "ole", "en", "eij",           # Estonian
    "niin", "kun",                       # Finnish (ja/ei/on/se/en/ole shared)
})


def normalize_word(w):
    """Normalize identically to JavaScript normalizeWord().
    NFC -> lowercase -> strip punctuation -> strip whitespace.
    Returns '' if result < MIN_WORD_LEN.
    """
    result = PUNCT_RE.sub('', unicodedata.normalize('NFC', w).lower().strip())
    if len(result) < MIN_WORD_LEN:
        return ''
    return result


def _detect_corpus_lang_raw(poem_id):
    """Prefix-only detection. Internal; do NOT call directly — use ``detect_corpus_lang``."""
    if not poem_id:
        return 'other', 'other'
    if poem_id.startswith('SKVR'):
        return 'SKVR', 'fi'
    if poem_id.startswith('JR'):
        return 'JR', 'fi'
    if poem_id.startswith('KR'):
        return 'KR', 'fi'
    if poem_id.startswith('Kalevipoeg'):
        return 'literary', 'et'
    if poem_id.startswith(('Kalevala', 'Kanteletar')):
        return 'literary', 'fi'
    for prefix in ('Lönnrot', 'Borenius', 'Europaeus', 'Reinholm', 'Ahlqvist',
                    'Caján', 'Cajan', 'Gottlund', 'Genetz', 'Porkka',
                    'Niemi', 'Krohn', 'Toppelius', 'Wanha', 'Schroeder',
                    'Castrén', 'Saxbäck', 'Warelius', 'Ganander', 'Porthan',
                    'Lencqvist', 'Sjögren', 'Renvall', 'Polén'):
        if poem_id.startswith(prefix):
            return prefix, 'fi'
    for prefix in ('ERAB', 'ERA', 'ERa', 'ERÄ', 'ERI',
                    'H ', 'H,', 'E ', 'E,', 'E.',
                    'EÜS', 'EKmS', 'EKMS', 'EEA',
                    'ERM', 'EKS', 'EKLA',
                    'K ', 'K,',
                    'AES', 'ARS',
                    'Veske', 'Leoke', 'Leske', 'TEM',
                    'B '):
        if poem_id.startswith(prefix):
            return prefix.rstrip(' ,'), 'et'
    # Case-insensitive fallback for Estonian prefixes
    pid_lower = poem_id.lower()
    for prefix in ('era', 'e,', 'e.', 'e '):
        if pid_lower.startswith(prefix):
            return prefix.upper().rstrip(' ,.'), 'et'
    # Estonian publications
    if poem_id.startswith('Vana kannel'):
        return 'Vana_kannel', 'et'
    # Numeric-only IDs (e.g. "60878/9") are typically Estonian archive refs
    if poem_id[0].isdigit():
        return 'numeric', 'et'
    return 'other', 'other'


def detect_corpus_lang(poem_id):
    """Detect corpus and language from poem ID prefix.
    Returns (corpus, lang) tuple.

    välismaa override: for the 68 Finnish/Ingrian pids archived under
    Estonian-prefixed IDs, return the *original* archive corpus (typically
    ERAB) paired with ``'fi'``. Provenance preserved, only language flips.
    """
    corpus, lang = _detect_corpus_lang_raw(poem_id)
    if poem_id and is_fi_override(poem_id):
        return corpus, 'fi'
    if poem_id and is_et_override(poem_id):
        return corpus, 'et'
    return corpus, lang


def get_poem_chunks():
    """Return sorted list of poem chunk paths. Uses glob, not poems_index.json."""
    chunks = sorted(POEMS_DIR.glob("poems_chunk_*.json"))
    if not chunks:
        print(f"ERROR: No poem chunks found in {POEMS_DIR}")
        sys.exit(1)
    return chunks


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def check_disk_space(min_gb=5.0):
    """Warn if free disk drops below threshold during long runs."""
    stat = shutil.disk_usage(OUTPUT_DIR)
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"WARNING: Only {free_gb:.1f} GB free disk remaining!")
        return False
    return True


def log_memory():
    """Log peak RSS memory usage (macOS/Linux)."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == 'darwin':
            return ru.ru_maxrss / (1024 * 1024)  # bytes → MB
        else:
            return ru.ru_maxrss / 1024            # KB → MB
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Two-Pass Architecture
# ---------------------------------------------------------------------------

def build_vocabulary_streaming(word_extractor, min_df=5, max_df_frac=0.05,
                               limit=None, exclude_stopwords=False,
                               write_metadata=True):
    """Stream all poem chunks to build vocabulary and document frequencies.

    Pass 1: does NOT store verse content — only word->count mapping.

    Args:
        word_extractor: callable(verse_text, poem_id) -> set[str] of normalized words
        min_df: minimum document frequency to keep a word
        max_df_frac: maximum document frequency fraction to keep
        limit: max number of verses to process (None = all)
        exclude_stopwords: remove RUNOSONG_STOPWORDS from vocabulary
        write_metadata: write verse_metadata.jsonl to disk

    Returns:
        word_to_idx: dict[str, int] — filtered vocabulary mapping
        verse_ids: list[str] — ordered list of verse_id strings
        n_verses: int — total verse count
        verse_texts: list[str] — lowercased verse text for each verse (for dedup)
        df_filtered: dict[str, int] — document frequency counts for kept vocabulary
    """
    ensure_output_dir()
    df = Counter()
    verse_ids = []
    verse_texts = []
    n_verses = 0
    chunks = get_poem_chunks()

    meta_path = OUTPUT_DIR / "verse_metadata.jsonl"
    meta_f = open(meta_path, 'w') if write_metadata else None

    try:
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
                poem_id = html.unescape(poem_id)  # normalize HTML-entity IDs (e.g. "EKmS 4&#xB0;" -> "EKmS 4°") to match literal-° poems space
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
                corpus, lang = detect_corpus_lang(poem_id)
                places = poem.get("p", []) if "p" in poem else poem.get("places", [])
                year = poem.get("y", "") if "y" in poem else poem.get("year", "")

                for vi, verse_text in enumerate(verses):
                    words = word_extractor(verse_text, poem_id)
                    if len(words) < 2:
                        continue
                    vid = f"{poem_id}:{vi}"
                    verse_ids.append(vid)
                    verse_texts.append(" ".join(verse_text.split()).lower())
                    for w in words:
                        df[w] += 1

                    if meta_f:
                        meta_f.write(json.dumps({
                            "v": vid, "p": poem_id, "i": vi,
                            "t": verse_text, "c": corpus, "l": lang,
                            "pl": places, "y": year
                        }, ensure_ascii=False) + '\n')

                    n_verses += 1
                    if limit and n_verses >= limit:
                        break
                if limit and n_verses >= limit:
                    break
            del chunk

            if limit and n_verses >= limit:
                break
    finally:
        if meta_f:
            meta_f.close()

    # Filter vocabulary
    max_df = int(n_verses * max_df_frac) if n_verses > 0 else 1
    word_to_idx = {}
    for word, count in df.items():
        if count < min_df or count > max_df:
            continue
        if exclude_stopwords and word in RUNOSONG_STOPWORDS:
            continue
        word_to_idx[word] = len(word_to_idx)

    too_common = sum(1 for w, c in df.items() if c > max_df)
    too_rare = sum(1 for w, c in df.items() if c < min_df)
    print(f"  Verses found: {n_verses:,}")
    print(f"  Unique words/tokens: {len(df):,}")
    print(f"  Too common (>{max_df}): {too_common}")
    print(f"  Too rare (<{min_df}): {too_rare}")
    if exclude_stopwords:
        print(f"  Stopwords excluded: {len(RUNOSONG_STOPWORDS)}")
    print(f"  Vocabulary kept: {len(word_to_idx):,}")

    df_filtered = {w: df[w] for w in word_to_idx}
    del df
    return word_to_idx, verse_ids, n_verses, verse_texts, df_filtered


def build_sparse_matrix_streaming(word_extractor, word_to_idx, n_verses,
                                   idf=None, limit=None):
    """Stream poem chunks again, building sparse matrix directly.

    Pass 2: each verse becomes a row. Column values are idf[word] if provided,
    else 1.0 (binary).

    Returns:
        mat: csr_matrix (n_verses x len(word_to_idx)), float32
    """
    vocab_size = len(word_to_idx)
    mat = lil_matrix((n_verses, vocab_size), dtype=np.float32)
    row = 0
    chunks = get_poem_chunks()

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
                words = word_extractor(verse_text, poem_id)
                if len(words) < 2:
                    continue

                for w in words:
                    if w in word_to_idx:
                        col = word_to_idx[w]
                        mat[row, col] = idf[w] if idf else 1.0
                row += 1
                if limit and row >= limit:
                    break
            if limit and row >= limit:
                break
        del chunk
        if limit and row >= limit:
            break

    if row != n_verses:
        print(f"  ERROR: Pass 2 produced {row} rows but Pass 1 counted {n_verses}")
        sys.exit(1)

    print(f"  Sparse matrix: {n_verses:,} x {vocab_size:,}, "
          f"nnz={mat.nnz:,}")
    return mat.tocsr()


# ---------------------------------------------------------------------------
# CompactVerseStore — CSR-like storage for inverted-index fallback
# ---------------------------------------------------------------------------

class CompactVerseStore:
    """CSR-like storage for verse word indices. ~105 MB for 4.35M verses.

    Instead of dict[verse_id -> set[str]] (~2.6 GB), stores:
    - indices: np.int32 array of all word indices concatenated (~88 MB)
    - indptr: np.int32 array of row pointers (~17 MB)
    """

    def __init__(self):
        self._indices = []
        self._indptr = [0]
        self._finalized = False

    def add_verse(self, word_indices):
        self._indices.extend(sorted(word_indices))
        self._indptr.append(len(self._indices))

    def finalize(self):
        self.indices = np.array(self._indices, dtype=np.int32)
        self.indptr = np.array(self._indptr, dtype=np.int32)
        del self._indices
        del self._indptr
        self._finalized = True

    def get_verse(self, row):
        return self.indices[self.indptr[row]:self.indptr[row + 1]]

    def verse_size(self, row):
        return self.indptr[row + 1] - self.indptr[row]

    def __len__(self):
        if self._finalized:
            return len(self.indptr) - 1
        return len(self._indptr) - 1


def build_compact_store_streaming(word_extractor, word_to_idx, n_verses,
                                   limit=None):
    """Stream poem chunks, building a CompactVerseStore + inverted index.

    Returns:
        store: CompactVerseStore
        inv_index: dict[int, np.ndarray] — word_idx -> sorted array of verse rows
    """
    store = CompactVerseStore()
    inv_lists = {i: [] for i in range(len(word_to_idx))}
    row = 0
    chunks = get_poem_chunks()

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
                words = word_extractor(verse_text, poem_id)
                if len(words) < 2:
                    continue

                word_indices = []
                for w in words:
                    if w in word_to_idx:
                        idx = word_to_idx[w]
                        word_indices.append(idx)
                        inv_lists[idx].append(row)

                store.add_verse(word_indices)
                row += 1
                if limit and row >= limit:
                    break
            if limit and row >= limit:
                break
        del chunk
        if limit and row >= limit:
            break

    store.finalize()

    if len(store) != n_verses:
        print(f"  ERROR: CompactVerseStore has {len(store)} rows but expected {n_verses}")
        sys.exit(1)

    # Convert inverted index lists to numpy arrays
    inv_index = {}
    for idx, rows in inv_lists.items():
        if rows:
            inv_index[idx] = np.array(rows, dtype=np.int32)
    del inv_lists

    print(f"  CompactVerseStore: {len(store):,} verses, "
          f"{len(store.indices):,} total word refs")
    print(f"  Inverted index: {len(inv_index):,} terms with postings")

    return store, inv_index


# ---------------------------------------------------------------------------
# Similarity computation helpers
# ---------------------------------------------------------------------------

def _sparse_row_topk(sparse_row, top_k, min_score, exclude_idx, overselect=1):
    """Extract top-K entries from a single sparse CSR row.

    Args:
        sparse_row: single row from sparse matmul result (1 x N CSR)
        top_k: max entries to return
        min_score: minimum value threshold
        exclude_idx: global row index to exclude (self-similarity)
        overselect: multiplier for over-selection (for later text dedup)

    Returns:
        list of (col_index, score) sorted by descending score
    """
    # Get the non-zero entries from this sparse row
    cols = sparse_row.indices
    vals = sparse_row.data

    if len(cols) == 0:
        return []

    # Filter: exclude self and below threshold
    mask = (cols != exclude_idx) & (vals >= min_score)
    cols = cols[mask]
    vals = vals[mask]

    if len(cols) == 0:
        return []

    # Top-K selection (with overselect for text dedup headroom)
    select_k = top_k * overselect
    if len(cols) <= select_k:
        order = np.argsort(-vals)
    else:
        # argpartition is O(n) vs O(n log n) for argsort
        top_idx = np.argpartition(-vals, select_k)[:select_k]
        order = top_idx[np.argsort(-vals[top_idx])]

    return [(int(cols[i]), float(vals[i])) for i in order]


def compute_topk_batched(norm_mat, verse_ids, top_k, min_score,
                          batch_size=2000, algorithm_name="",
                          verse_texts=None):
    """Compute top-K cosine neighbors via batched sparse matrix multiplication.

    Uses sparse_dot_topn when available (2-6x faster) with fallback to
    standard sparse matmul + per-row top-K extraction.

    Args:
        norm_mat: L2-normalized sparse matrix (n x vocab)
        verse_ids: list of verse_id strings (length n)
        top_k: max neighbors per verse
        min_score: minimum similarity threshold
        batch_size: rows per batch
        verse_texts: optional list of lowercased verse texts for text-based dedup

    Yields:
        (verse_id, [(other_id, score), ...])
    """
    N = norm_mat.shape[0]
    t_start = time.time()
    overselect = 5 if verse_texts else 1
    select_k = top_k * overselect

    if _HAS_SPARSE_DOT_TOPN:
        print(f"  Using sparse_dot_topn (accelerated)")
        n_threads = min(os.cpu_count() or 1, 4)
    else:
        print(f"  Using standard sparse matmul (fallback)")
        n_threads = 1

    norm_mat_t = norm_mat.T.tocsr()  # pre-transpose once; CSR avoids internal CSC→CSR in sp_matmul_topn

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = norm_mat[start:end]

        if _HAS_SPARSE_DOT_TOPN:
            sim_block = sp_matmul_topn(
                batch, norm_mat_t,
                top_n=select_k + 1,   # +1 for self-match
                threshold=min_score - 1e-9,  # sp_matmul_topn uses strict >, we want >=
                sort=True,
                n_threads=n_threads,
            )
        else:
            sim_block = batch @ norm_mat_t  # sparse @ sparse.T = sparse

        for local_i in range(end - start):
            global_i = start + local_i

            if _HAS_SPARSE_DOT_TOPN:
                # Already sorted + thresholded by sp_matmul_topn
                row = sim_block.getrow(local_i)
                cols = row.indices
                vals = row.data

                # Exclude self
                mask = cols != global_i
                cols, vals = cols[mask], vals[mask]
                if len(cols) == 0:
                    continue

                vid = verse_ids[global_i]
                if verse_texts:
                    seen = set()
                    entries = []
                    for ci in range(len(cols)):
                        txt = verse_texts[cols[ci]]
                        if txt in seen:
                            continue
                        seen.add(txt)
                        entries.append((verse_ids[int(cols[ci])], round(float(vals[ci]), 4)))
                        if len(entries) >= top_k:
                            break
                else:
                    entries = [(verse_ids[int(cols[ci])], round(float(vals[ci]), 4))
                               for ci in range(min(len(cols), top_k))]
            else:
                # Fallback: standard sparse matmul + per-row top-K
                sparse_row = sim_block.getrow(local_i)
                top_entries = _sparse_row_topk(sparse_row, top_k, min_score,
                                               global_i, overselect=overselect)
                if not top_entries:
                    continue

                vid = verse_ids[global_i]
                if verse_texts:
                    seen = set()
                    entries = []
                    for j, score in top_entries:
                        txt = verse_texts[j]
                        if txt in seen:
                            continue
                        seen.add(txt)
                        entries.append((verse_ids[j], round(score, 4)))
                        if len(entries) >= top_k:
                            break
                else:
                    entries = [(verse_ids[j], round(score, 4))
                               for j, score in top_entries]

            if entries:
                yield vid, entries

        del sim_block  # free batch result

        # Progress
        elapsed = time.time() - t_start
        rate = end / elapsed if elapsed > 0 else 0
        remaining = (N - end) / rate if rate > 0 else 0
        rss = log_memory()
        print(f"  {end:,}/{N:,} ({end / N * 100:.1f}%) - "
              f"{rate:.0f}/s, ~{remaining:.0f}s left, RSS={rss:.0f} MB")
        sys.stdout.flush()

        # Memory safety check
        if rss > 3000:
            print(f"  WARNING: RSS={rss:.0f} MB exceeds 3 GB safety limit!")

        # Disk space check every 100K verses
        if end % 100000 < batch_size:
            check_disk_space(min_gb=3.0)


def l2_normalize(mat_csr):
    """L2-normalize rows of a sparse CSR matrix (vectorized, memory-efficient)."""
    mat = mat_csr.copy()
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1.astype(np.float32)
    norms[norms == 0] = 1.0
    row_norms = np.repeat(norms, np.diff(mat.indptr))
    mat.data /= row_norms
    del row_norms
    return mat


# ---------------------------------------------------------------------------
# Match type detection
# ---------------------------------------------------------------------------

def classify_match(vid_a, vid_b):
    """Classify a verse pair as within-poem ('w'), cross-poem same-lang ('s'),
    or cross-lingual ('x').
    """
    # verse_id format: "POEM_ID:verse_index"
    poem_a = vid_a.rsplit(':', 1)[0]
    poem_b = vid_b.rsplit(':', 1)[0]

    if poem_a == poem_b:
        return 'w'

    _, lang_a = detect_corpus_lang(poem_a)
    _, lang_b = detect_corpus_lang(poem_b)

    if lang_a != lang_b and lang_a != 'other' and lang_b != 'other':
        return 'x'
    return 's'


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

@contextmanager
def similarity_writer(output_path, algorithm, params, total_verses, unique_verses=None):
    """Gzipped JSONL output with metadata header line."""
    ensure_output_dir()

    header = {
        "_header": True,
        "algorithm": algorithm,
        "params": params,
        "total_verses": total_verses,
        "unique_verses": unique_verses or total_verses,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    f = gzip.open(output_path, 'wt', compresslevel=6)
    try:
        f.write(json.dumps(header, ensure_ascii=False) + '\n')

        def write_entry(verse_id, group_size, matches):
            """Write one verse's similarity data.
            matches: list of (other_vid, score, match_type) or (other_vid, score, shared_count, match_type)
            """
            entry = {"v": verse_id, "g": group_size, "m": matches}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        yield write_entry
    finally:
        f.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Output: {output_path.name} ({size_mb:.1f} MB gzipped)")


# ---------------------------------------------------------------------------
# Checkpoint/resume
# ---------------------------------------------------------------------------

def get_progress_path(output_path):
    return output_path.with_suffix(output_path.suffix + '.progress')


def cleanup_progress(output_path):
    """Remove progress file after successful completion."""
    progress_path = get_progress_path(output_path)
    if progress_path.exists():
        progress_path.unlink()
