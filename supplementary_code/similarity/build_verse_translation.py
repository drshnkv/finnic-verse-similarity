#!/usr/bin/env python3
"""Build verse-level similarity index using English translation pivot.

Cross-lingual thematic similarity: each verse becomes a bag of English
translations from gloss_index.json. Two verses are similar if they share
translated vocabulary — regardless of source language (ET/FI).

Two-pass streaming architecture keeps peak memory under ~1.3 GB.

Usage:
  python -u build_verse_translation.py --limit 100    # smoke test
  python -u build_verse_translation.py                # full run
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from verse_similarity_common import (
    ROOT,
    DEPLOYMENT,
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
    MIN_WORD_LEN,
)

# --- Parameters ---
TOP_K = 50
MIN_SCORE = 0.15
MIN_DF = 5
MAX_DF_FRAC = 0.05    # skip translations in >5% of verses
BATCH_SIZE = 2000

GLOSS_INDEX_PATH = DEPLOYMENT / "gloss_index.json"
LEMMA_ENGLISH_PATH = DEPLOYMENT / "lemma_english.json"

# Strategy F' lemma-resolution data paths (for lemma fallback)
LEXICON_DATA_PATH = DEPLOYMENT / "lexicon_data.json.gz"
DISTRIBUTION_PATH = DEPLOYMENT / "wordform_lemma_distribution.json"

# --- Translation normalization ---

REFRAIN_RE = re.compile(
    r'^[\[\(\s]*(refrain|meaningless refrain|refrain particle|meaningless|'
    r'nonsense refrain|refrain word|nonsense word)[\]\)\s]*$',
    re.IGNORECASE
)

PREPOSITIONS = [
    'between', 'through', 'against', 'towards', 'outside', 'beneath',
    'without', 'beside', 'behind', 'before', 'inside', 'within',
    'toward', 'across', 'around', 'under', 'above', 'below',
    'along', 'about', 'among', 'upon', 'onto', 'into',
    'over', 'from', 'with', 'for', 'at', 'on', 'in',
]

PREP_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(p) for p in PREPOSITIONS) + r')\s+',
    re.IGNORECASE
)

PAREN_RE = re.compile(r'\([^)]*\)')
BRACKET_RE = re.compile(r'\[[^\]]*\]')
ARTICLE_RE = re.compile(r'^(?:the|an?)\s+', re.IGNORECASE)
TO_INF_RE = re.compile(r'^to\s+', re.IGNORECASE)

# --- English verb lemmatization ---

LEMMA_EXCEPTIONS = {
    'does': 'do', 'goes': 'go', 'has': 'have', 'is': 'be',
    'was': 'be', 'dies': 'die', 'lies': 'lie', 'ties': 'tie',
    'axes': 'axe', 'eyes': 'eye',
    # Irregular past tenses
    'brought': 'bring', 'thought': 'think', 'sang': 'sing',
    'sung': 'sing', 'came': 'come', 'knew': 'know',
    'grew': 'grow', 'threw': 'throw', 'gave': 'give',
    'took': 'take', 'made': 'make', 'said': 'say',
    'told': 'tell', 'found': 'find', 'left': 'leave',
    'felt': 'feel', 'kept': 'keep', 'held': 'hold',
    'stood': 'stand', 'sat': 'sit', 'ran': 'run',
    'fell': 'fall', 'wore': 'wear', 'wove': 'weave',
    'bore': 'bear', 'swam': 'swim', 'drank': 'drink',
    'ate': 'eat', 'spoke': 'speak', 'broke': 'break',
    'woke': 'wake', 'rode': 'ride', 'drove': 'drive',
    'wrote': 'write', 'rose': 'rise', 'chose': 'choose',
    'froze': 'freeze',
    # Past participles
    'hidden': 'hide', 'written': 'write', 'driven': 'drive',
    'given': 'give', 'taken': 'take', 'eaten': 'eat',
    'beaten': 'beat', 'fallen': 'fall', 'chosen': 'choose',
    'frozen': 'freeze', 'spoken': 'speak', 'broken': 'break',
    'woken': 'wake', 'stolen': 'steal', 'forgotten': 'forget',
    'gotten': 'get',
    # Irregular past with consonant change
    'bound': 'bind', 'ground': 'grind', 'wound': 'wind',
    'spun': 'spin', 'begun': 'begin', 'drunk': 'drink',
    'sunk': 'sink', 'rung': 'ring', 'stung': 'sting',
    'swung': 'swing', 'clung': 'cling', 'flung': 'fling',
    # -nt/-lt endings
    'bent': 'bend', 'sent': 'send', 'spent': 'spend',
    'built': 'build', 'burnt': 'burn', 'meant': 'mean',
    'dealt': 'deal', 'knelt': 'kneel', 'slept': 'sleep',
    'wept': 'weep', 'crept': 'creep', 'swept': 'sweep',
    'leapt': 'leap',
    # Very common missing forms
    'been': 'be', 'gone': 'go', 'done': 'do', 'seen': 'see',
    'had': 'have', 'did': 'do', 'went': 'go', 'got': 'get',
    # -ought/-aught
    'caught': 'catch', 'taught': 'teach', 'bought': 'buy',
    'sought': 'seek', 'fought': 'fight',
    # Other common irregulars
    'lost': 'lose', 'won': 'win', 'hung': 'hang',
    'led': 'lead', 'fed': 'feed', 'bred': 'breed', 'bled': 'bleed',
    'paid': 'pay', 'laid': 'lay',
    # Forms of "be"
    'were': 'be',
    # Past participles ending in -n
    'drawn': 'draw', 'grown': 'grow', 'known': 'know',
    'blown': 'blow', 'shown': 'show', 'thrown': 'throw',
    'worn': 'wear', 'torn': 'tear', 'born': 'bear',
    'sworn': 'swear', 'slain': 'slay',
    # Additional irregular past/participle forms
    'shook': 'shake', 'struck': 'strike', 'stuck': 'stick',
    'dug': 'dig', 'bit': 'bite', 'tore': 'tear',
    'blew': 'blow', 'drew': 'draw', 'flew': 'fly',
    'lit': 'light', 'bitten': 'bite', 'shaken': 'shake',
    'woven': 'weave', 'shrunk': 'shrink', 'sprung': 'spring',
    'ridden': 'ride',
}

NO_STRIP_S = {
    'this', 'thus', 'plus', 'lens', 'news', 'axis', 'always', 'perhaps',
    'sometimes', 'towards', 'backwards', 'forwards', 'upwards', 'downwards',
    'afterwards', 'besides', 'whereas', 'atlas', 'christmas', 'canvas',
    'basis', 'crisis', 'oasis', 'thesis', 'genus', 'corpus', 'bonus',
    'focus', 'radius', 'status', 'virus', 'bus', 'us', 'its', 'his',
}

NO_STRIP_ING = {
    'morning', 'evening', 'nothing', 'something', 'everything', 'anything',
    'spring', 'string',
    'herring', 'offering', 'offspring', 'blessing',
    'ceiling', 'dwelling', 'feeling', 'greeting',
    'hearing', 'meaning', 'meeting', 'opening', 'pudding',
    'reading', 'willing',
    'darling', 'sterling', 'shilling', 'sibling',
    'lightning', 'during',
}

NO_STRIP_ED = {
    'hundred', 'kindred', 'sacred', 'wicked', 'naked',
    'crooked', 'cursed', 'blessed', 'beloved', 'hatred',
}

DOUBLE_CONSONANT_ING = {
    'running', 'sitting', 'hitting', 'cutting', 'getting', 'putting',
    'setting', 'letting', 'beginning', 'spinning', 'winning', 'swimming',
    'digging', 'dropping', 'stopping', 'stepping', 'chopping', 'rubbing',
    'begging', 'bidding', 'tipping', 'wrapping', 'stripping', 'gripping',
    'shipping', 'fitting', 'knitting', 'slipping', 'clipping',
}


def lemmatize_english_verb(token):
    """Simple rule-based English verb lemmatization."""
    if ' ' in token:
        return token

    if token in LEMMA_EXCEPTIONS:
        return LEMMA_EXCEPTIONS[token]
    if token in NO_STRIP_S:
        return token

    # -ing stripping
    if token.endswith('ing') and len(token) >= 6 and token not in NO_STRIP_ING:
        stem = token[:-3]
        if token in DOUBLE_CONSONANT_ING:
            return stem[:-1]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            return stem[:-1]
        if (stem[-1] not in 'aeiou' and len(stem) >= 2 and stem[-2] in 'aeiou'
                and (len(stem) < 3 or stem[-3] not in 'aeiou')):
            return stem + 'e'
        return stem

    # -es stripping
    if token.endswith('es') and len(token) >= 5:
        if token.endswith('ies'):
            return token[:-3] + 'y'
        if token.endswith(('shes', 'ches', 'xes', 'zes', 'sses')):
            return token[:-2]

    # -s stripping
    if token.endswith('s') and not token.endswith('ss') and len(token) >= 4:
        return token[:-1]

    # -ed stripping
    if token.endswith('ed') and len(token) >= 5 and token not in NO_STRIP_ED:
        if token.endswith('ied'):
            return token[:-3] + 'y'
        if token.endswith('eed'):
            return token
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            return stem[:-1]
        if (stem[-1] not in 'aeiou' and len(stem) >= 2 and stem[-2] in 'aeiou'
                and (len(stem) < 3 or stem[-3] not in 'aeiou')):
            return stem + 'e'
        return stem

    return token


def strip_leading_preposition(text):
    """Strip a leading preposition, but only if result is 1-2 words."""
    m = PREP_RE.match(text)
    if m:
        remainder = text[m.end():]
        word_count = len(remainder.split())
        if 1 <= word_count <= 2:
            return remainder
    return text


def normalize_translation(english, pos=None):
    """Apply the full normalization pipeline to an English translation."""
    if not english or not english.strip():
        return None

    english = english.strip()
    if english in ('\u2014', '-'):
        return None
    if REFRAIN_RE.match(english):
        return None
    if english.lower() in ('unclear', 'unknown'):
        return None
    if english.startswith('**'):
        return None
    if pos == 'INTJ':
        return None

    t = english.lower()
    t = PAREN_RE.sub('', t).strip()
    t = BRACKET_RE.sub('', t).strip()
    if '?' in t:
        return None

    if t.endswith("'s"):
        t = t[:-2].strip()

    t = ARTICLE_RE.sub('', t).strip()
    t = TO_INF_RE.sub('', t).strip()
    t = strip_leading_preposition(t)
    t = t.strip()
    if not t:
        return None

    if ' ' not in t:
        t = lemmatize_english_verb(t)

    if len(t) < MIN_WORD_LEN:
        return None

    return t


def load_slim_gloss_index(path, include_fallback=False):
    """Load only (english, pos) per wordform. Free original immediately.
    Full gloss_index.json = ~800 MB in memory. Slim version = ~150 MB.
    """
    print("  Loading gloss_index.json (slim)...")
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)

    gloss = data.get('g', {})
    total = len(gloss)

    skip_stats = Counter()
    wf_to_trans = {}

    for wf, entry in gloss.items():
        if not isinstance(entry, list) or len(entry) < 2:
            skip_stats['malformed'] += 1
            continue
        if not include_fallback and len(entry) >= 5:
            skip_stats['fallback'] += 1
            continue

        english = entry[1] if len(entry) > 1 else ''
        pos = entry[2] if len(entry) > 2 else ''

        if not isinstance(english, str) or not english.strip():
            skip_stats['empty'] += 1
            continue
        if english in ('\u2014', '-'):
            skip_stats['em_dash'] += 1
            continue

        wf_to_trans[wf] = (english.strip(), pos)

    del data
    del gloss

    print(f"  Loaded {total:,} entries -> {len(wf_to_trans):,} usable ({time.time() - t0:.1f}s)")
    print(f"  Skipped: {dict(skip_stats)}")
    return wf_to_trans


def load_lemma_fallback():
    """Load the Strategy F' resolver + lemma->english map for translation fallback.

    Uses Strategy F' (corpus + DeepSeek + phonological gate) to resolve a
    wordform's lemma when it has no direct gloss entry.
    """
    if not LEXICON_DATA_PATH.exists() or not DISTRIBUTION_PATH.exists():
        print("  [WARN] resolver data files not found, skipping lemma resolution")
        return None, {}
    if not LEMMA_ENGLISH_PATH.exists():
        print("  [WARN] lemma_english.json not found, skipping lemma resolution")
        return None, {}
    print("  Loading RunoVerse resolver for lemma fallback...")
    t0 = time.time()
    from runoverse_lemma_resolver import RunoVerseLemmaResolver
    resolver = RunoVerseLemmaResolver(LEXICON_DATA_PATH, DISTRIBUTION_PATH)
    with open(LEMMA_ENGLISH_PATH) as f:
        le = json.load(f)
    print(f"  RunoVerse resolver + {len(le):,} lemma->english mappings "
          f"in {time.time() - t0:.1f}s")
    return resolver, le


def make_translation_extractor(wf_to_trans, resolver=None, lemma_english=None):
    """Create word extractor that maps wordforms to normalized English translations.

    Uses the Strategy F' resolver for lemma fallback when a wordform
    has no direct gloss entry.
    """
    fallback_count = [0]

    def extract(verse_text, poem_id):
        translations = set()
        for token in verse_text.split():
            w = normalize_word(token)
            if not w:
                continue
            entry = wf_to_trans.get(w)
            if not entry and resolver and lemma_english:
                lemma = resolver.resolve(w)
                if lemma != w:  # resolver found a real lemma
                    eng = lemma_english.get(lemma.lower())
                    if eng:
                        entry = (eng, '')
                        fallback_count[0] += 1
            if entry:
                raw_english, pos = entry
                normalized = normalize_translation(raw_english, pos)
                if normalized:
                    translations.add(normalized)
        return translations

    extract.fallback_count = fallback_count
    return extract


def main():
    parser = argparse.ArgumentParser(
        description='Build verse-level translation-pivot similarity index'
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Max verses to process (default: all)')
    parser.add_argument('--top-k', type=int, default=TOP_K,
                        help=f'Max neighbors per verse (default: {TOP_K})')
    parser.add_argument('--min-score', type=float, default=MIN_SCORE,
                        help=f'Min cosine threshold (default: {MIN_SCORE})')
    parser.add_argument('--exclude-same-poem', action='store_true',
                        help='Exclude all within-poem matches')
    parser.add_argument('--include-fallback', action='store_true',
                        help='Include similarity-fallback gloss entries')
    args = parser.parse_args()

    t_total = time.time()
    ensure_output_dir()

    output_path = OUTPUT_DIR / "verse_similarities_translation.jsonl.gz"

    print(f"=== Verse-Level Translation-Pivot Cosine Similarity ===")
    print(f"  Top-K: {args.top_k}, Min score: {args.min_score}")
    print(f"  Limit: {args.limit or 'all'}")
    print(f"  Include fallback glosses: {args.include_fallback}")
    print()

    # Load slim gloss index
    print("[1/5] Loading gloss index...")
    wf_to_trans = load_slim_gloss_index(GLOSS_INDEX_PATH,
                                         include_fallback=args.include_fallback)

    # Load the Strategy F' resolver for lemma fallback (wordforms without direct translation)
    resolver, lemma_english = load_lemma_fallback()
    word_extractor = make_translation_extractor(wf_to_trans, resolver=resolver,
                                                 lemma_english=lemma_english)

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

    # Pass 1: Build vocabulary over English translations
    print(f"\n[2/5] Pass 1: Streaming translation vocabulary...")
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
    print(f"  IDF weights for {len(idf):,} translations")

    # Pass 2: Build sparse IDF matrix (reuses wf_to_trans + word_extractor from Pass 1)
    print(f"\n[3/5] Pass 2: Building sparse translation-IDF matrix...")
    mat_csr = build_sparse_matrix_streaming(
        word_extractor, word_to_idx, n_verses,
        idf=idf, limit=args.limit
    )

    # Free gloss index now — no longer needed
    del wf_to_trans

    # L2-normalize for cosine similarity
    print("  L2-normalizing rows...")
    norm_mat = l2_normalize(mat_csr)
    del mat_csr

    # Compute + stream results directly to output
    print(f"\n[4/5] Computing cosine similarities + writing (batch={BATCH_SIZE})...")
    params = {
        "top_k": args.top_k,
        "min_score": args.min_score,
        "min_df": min_df,
        "max_df_frac": MAX_DF_FRAC,
        "exclude_same_poem": args.exclude_same_poem,
        "include_fallback": args.include_fallback,
    }

    n_with_matches = 0
    total_pairs = 0
    type_counts = {'w': 0, 's': 0, 'x': 0}

    with similarity_writer(output_path, "tfidf_cosine_translation_pivot", params,
                           n_verses) as write_entry:
        for vid, entries in compute_topk_batched(
            norm_mat, verse_ids, args.top_k, args.min_score,
            batch_size=BATCH_SIZE, algorithm_name="translation",
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

    cross_pct = type_counts['x'] / total_pairs * 100 if total_pairs else 0

    print(f"\n=== Summary ===")
    print(f"  Total verses processed: {n_verses:,}")
    print(f"  Verses with matches: {n_with_matches:,} ({n_with_matches / n_verses * 100:.1f}%)")
    print(f"  Total match pairs: {total_pairs:,}")
    print(f"  Match types: within-poem={type_counts['w']:,}, "
          f"cross-poem={type_counts['s']:,}, cross-lingual={type_counts['x']:,} ({cross_pct:.1f}%)")
    if hasattr(word_extractor, 'fallback_count') and word_extractor.fallback_count[0]:
        print(f"  Lemma fallback resolved {word_extractor.fallback_count[0]:,} additional word tokens")
    print(f"  Total time: {(time.time() - t_total) / 60:.1f} minutes")
    print(f"  Peak RSS: {log_memory():.0f} MB")


if __name__ == "__main__":
    main()
