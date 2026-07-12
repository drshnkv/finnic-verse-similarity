#!/usr/bin/env python3
"""Cross-lingual (Estonian-Finnish) form/meaning analysis of the 5-algo RRF clusters.

Category labels: "form_based" (char-unigram Jaccard > 0.4 orthographic
similarity) vs "meaning_based" (orthographic similarity is
not proof of cognacy). Runs a full census of all cross-lingual clusters and
computes:
  - Character bigram and trigram Jaccard (alongside unigram)
  - Provenance tracking: retained/expanded/novel vs the 4-algo baseline
  - Weighted ratios (by cluster, by size, by pair count)
  - Histogram data for threshold sensitivity analysis
  - Stratified manual validation sample (100 clusters)

Inputs:
  similarity/output/rrf_cluster_membership.jsonl          (5-algo full membership)
  similarity/output/rrf_cluster_membership_loo_minus_sentence.jsonl.gz  (4-algo baseline)
  similarity/output/verse_metadata.jsonl                  (4.3M verses)
  deployment/gloss_index.json                             (71 MB, English glosses)

Output:
  similarity/output/rrf_ws4_cross_lingual_5algo.json
  similarity/output/rrf_ws4_manual_validation_sample.json

Scope: this module provides the cross-lingual identification and form/meaning
classification functions (load_clusters, identify_cross_lingual, analyze_cluster,
compute_provenance, ...) used across the pipeline. Its main() runs a census over
the canonical full membership (rrf_cluster_membership.jsonl, at the two-member
floor) and writes a manual-validation sample. The aggregate cross-lingual counts
and form/meaning breakdowns are produced by the sibling scripts
output/floor2_rederive/derive_ws4_floor2.py and phase_c_table7.py, which import
these functions and add the provenance, cross-corpus (undirected) member-pair,
coverage, and leave-one-out breakdowns.

Usage:
  python -u similarity/build_ws4_cross_lingual_5algo.py
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
DEPLOYMENT = BASE.parent / "deployment"

MEMBERSHIP_PATH = OUTPUT_DIR / "rrf_cluster_membership.jsonl"
MEMBERSHIP_4ALGO_PATH = OUTPUT_DIR / "rrf_cluster_membership_loo_minus_sentence.jsonl.gz"
METADATA_PATH = OUTPUT_DIR / "verse_metadata.jsonl"
GLOSS_INDEX_PATH = DEPLOYMENT / "gloss_index.json"

OUTPUT_PATH = OUTPUT_DIR / "rrf_ws4_cross_lingual_5algo.json"
VALIDATION_SAMPLE_PATH = OUTPUT_DIR / "rrf_ws4_manual_validation_sample.json"

sys.path.insert(0, str(BASE.parent))
from valismaa_override import fi_override_pids as _fi_override_pids, et_override_pids as _et_override_pids  # noqa: E402

TEXT_CAP = 10


def load_verse_metadata() -> dict[str, dict]:
    fi_pids = set(_fi_override_pids())
    et_pids = set(_et_override_pids())
    meta: dict[str, dict] = {}
    overridden_fi = 0
    overridden_et = 0
    t0 = time.time()
    with METADATA_PATH.open() as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            lang = m.get("l", "")
            pid = m.get("p", "")
            if lang == "et" and pid in fi_pids:
                lang = "fi"
                overridden_fi += 1
            elif lang == "fi" and pid in et_pids:
                lang = "et"
                overridden_et += 1
            meta[m["v"]] = {"l": lang, "t": m.get("t", ""), "pl": m.get("pl", []) or []}
            if (i + 1) % 1_000_000 == 0:
                print(f"  ... {i + 1:,} verses loaded")
    dt = time.time() - t0
    print(f"  Loaded {len(meta):,} verses in {dt:.1f}s")
    print(f"    välismaa override: {overridden_fi} et→fi, {overridden_et} fi→et")
    return meta


def load_gloss_index() -> dict:
    t0 = time.time()
    with GLOSS_INDEX_PATH.open() as f:
        data = json.load(f)
    glosses = data.get("g", data)
    print(f"  Gloss index: {len(glosses):,} entries in {time.time() - t0:.1f}s")
    return glosses


def load_clusters(path: Path, compressed: bool = False) -> list[dict]:
    t0 = time.time()
    clusters = []
    opener = gzip.open if compressed else open
    mode = "rt" if compressed else "r"
    with opener(path, mode) as f:
        for line in f:
            clusters.append(json.loads(line))
    print(f"  Loaded {len(clusters):,} clusters from {path.name} in {time.time() - t0:.1f}s")
    return clusters


def char_ngram_set(text: str, n: int) -> set[str]:
    if n == 1:
        return set(text)
    return {text[i:i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def char_ngram_jaccard(et_words: set[str], fi_words: set[str], n: int) -> float:
    et_text = " ".join(sorted(et_words))
    fi_text = " ".join(sorted(fi_words))
    et_ngrams = char_ngram_set(et_text, n)
    fi_ngrams = char_ngram_set(fi_text, n)
    union = et_ngrams | fi_ngrams
    if not union:
        return 0.0
    return len(et_ngrams & fi_ngrams) / len(union)


def get_translations(words: set[str], gloss_data: dict, limit: int = 20) -> set[str]:
    translations = set()
    for w in list(words)[:limit]:
        entry = gloss_data.get(w)
        if isinstance(entry, list) and len(entry) > 1:
            eng = entry[1]
            if isinstance(eng, str) and eng:
                translations.add(eng.lower())
    return translations


def identify_cross_lingual(clusters: list[dict], verse_meta: dict) -> list[dict]:
    cl_clusters = []
    for entry in clusters:
        et_vids = []
        fi_vids = []
        for vid in entry["members"]:
            m = verse_meta.get(vid)
            if m is None:
                continue
            lang = m["l"]
            if lang == "et":
                et_vids.append(vid)
            elif lang == "fi":
                fi_vids.append(vid)
        if et_vids and fi_vids:
            entry["_et_vids"] = et_vids
            entry["_fi_vids"] = fi_vids
            cl_clusters.append(entry)
    return cl_clusters


def analyze_cluster(entry: dict, verse_meta: dict, gloss_data: dict) -> dict:
    et_vids = entry["_et_vids"]
    fi_vids = entry["_fi_vids"]

    et_texts = [verse_meta[vid]["t"] for vid in et_vids if verse_meta[vid]["t"]]
    fi_texts = [verse_meta[vid]["t"] for vid in fi_vids if verse_meta[vid]["t"]]

    et_words = set()
    fi_words = set()
    for t in et_texts[:TEXT_CAP]:
        et_words.update(t.lower().split())
    for t in fi_texts[:TEXT_CAP]:
        fi_words.update(t.lower().split())

    uni_j = char_ngram_jaccard(et_words, fi_words, 1)
    bi_j = char_ngram_jaccard(et_words, fi_words, 2)
    tri_j = char_ngram_jaccard(et_words, fi_words, 3)

    word_j = (
        len(et_words & fi_words) / len(et_words | fi_words)
        if (et_words | fi_words) else 0.0
    )

    # "form_based" = orthographic similarity (char-unigram Jaccard > 0.4).
    # Matches the article's terminology (orthographic similarity, not cognacy).
    category = "form_based" if uni_j > 0.4 else "meaning_based"

    et_trans = get_translations(et_words, gloss_data)
    fi_trans = get_translations(fi_words, gloss_data)
    shared_trans = sorted(et_trans & fi_trans)[:10]

    pair_count = len(et_vids) * len(fi_vids)

    return {
        "cluster_id": entry["cluster_id"],
        "size": entry["size"],
        "et_count": len(et_vids),
        "fi_count": len(fi_vids),
        "pair_count": pair_count,
        "et_example": et_texts[0] if et_texts else "",
        "fi_example": fi_texts[0] if fi_texts else "",
        "char_unigram_jaccard": round(uni_j, 4),
        "char_bigram_jaccard": round(bi_j, 4),
        "char_trigram_jaccard": round(tri_j, 4),
        "word_jaccard": round(word_j, 4),
        "category": category,
        "shared_translations": shared_trans,
    }


def compute_provenance(
    cl_5algo: list[dict],
    verse_meta: dict,
    clusters_4algo: list[dict],
) -> dict[int, str]:
    print("\n  Computing provenance (retained/expanded/novel)...")
    t0 = time.time()

    cl_4algo = identify_cross_lingual(clusters_4algo, verse_meta)
    print(f"    4-algo CL clusters: {len(cl_4algo):,}")

    members_4 = {}
    for entry in cl_4algo:
        members_4[entry["cluster_id"]] = set(entry["members"])

    provenance: dict[int, str] = {}
    for entry in cl_5algo:
        members_5 = set(entry["members"])
        best_overlap = 0.0
        for cid_4, mset_4 in members_4.items():
            inter = len(members_5 & mset_4)
            union = len(members_5 | mset_4)
            if union > 0:
                j = inter / union
                if j > best_overlap:
                    best_overlap = j
        if best_overlap >= 0.7:
            provenance[entry["cluster_id"]] = "retained"
        elif best_overlap >= 0.3:
            provenance[entry["cluster_id"]] = "expanded"
        else:
            provenance[entry["cluster_id"]] = "novel"

    counts = Counter(provenance.values())
    print(f"    Provenance: {dict(counts)}")
    print(f"    Computed in {time.time() - t0:.1f}s")
    return provenance


def build_histograms(analyzed: list[dict]) -> dict:
    bins = [i / 20 for i in range(21)]
    result = {}
    for metric in ("char_unigram_jaccard", "char_bigram_jaccard", "char_trigram_jaccard"):
        values = [a[metric] for a in analyzed]
        counts, _ = np.histogram(values, bins=bins)
        result[metric] = {
            "bins": [round(b, 2) for b in bins],
            "counts": counts.tolist(),
            "mean": round(float(np.mean(values)), 4),
            "median": round(float(np.median(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
        }
    return result


def try_otsu_threshold(values: list[float]) -> tuple[float | None, str]:
    """Attempt Otsu's method on a Jaccard distribution. Return (threshold, note)."""
    arr = np.array(values)
    n_bins = 50
    counts, edges = np.histogram(arr, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return None, "empty"

    centers = 0.5 * (edges[:-1] + edges[1:])

    best_thresh = None
    best_var = -1.0
    cum_w = 0
    cum_sum = 0.0
    total_sum = float((counts * centers).sum())

    for i in range(n_bins):
        cum_w += counts[i]
        if cum_w == 0:
            continue
        bg_w = total - cum_w
        if bg_w == 0:
            break
        cum_sum += counts[i] * centers[i]
        mean_fg = cum_sum / cum_w
        mean_bg = (total_sum - cum_sum) / bg_w
        between_var = cum_w * bg_w * (mean_fg - mean_bg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = edges[i + 1]

    if best_thresh is None:
        return None, "unimodal"

    fg = arr[arr <= best_thresh]
    bg = arr[arr > best_thresh]
    if len(fg) < 10 or len(bg) < 10:
        return best_thresh, "weak_bimodality"

    return best_thresh, "bimodal"


def compute_weighted_ratios(analyzed: list[dict], threshold: float) -> dict:
    by_cluster = Counter()
    by_size = Counter()
    by_pairs = Counter()
    for a in analyzed:
        cat = "form_based" if a["char_unigram_jaccard"] > threshold else "meaning_based"
        by_cluster[cat] += 1
        by_size[cat] += a["size"]
        by_pairs[cat] += a["pair_count"]

    def pct(counter: Counter, key: str) -> float:
        total = sum(counter.values())
        return round(counter[key] / total * 100, 2) if total else 0.0

    return {
        "threshold": threshold,
        "by_cluster": {
            "form_based": by_cluster["form_based"],
            "meaning_based": by_cluster["meaning_based"],
            "form_pct": pct(by_cluster, "form_based"),
        },
        "by_size": {
            "form_based": by_size["form_based"],
            "meaning_based": by_size["meaning_based"],
            "form_pct": pct(by_size, "form_based"),
        },
        "by_pair_count": {
            "form_based": by_pairs["form_based"],
            "meaning_based": by_pairs["meaning_based"],
            "form_pct": pct(by_pairs, "form_based"),
        },
    }


def draw_validation_sample(analyzed: list[dict], n: int = 100) -> list[dict]:
    """Stratified sample: high (>0.6), mid (0.3-0.6), low (<0.3) unigram Jaccard."""
    high = [a for a in analyzed if a["char_unigram_jaccard"] > 0.6]
    mid = [a for a in analyzed if 0.3 <= a["char_unigram_jaccard"] <= 0.6]
    low = [a for a in analyzed if a["char_unigram_jaccard"] < 0.3]

    random.seed(42)
    n_per = n // 3
    n_remainder = n - 3 * n_per

    sample = []
    for stratum, name, count in [
        (high, "high", n_per + n_remainder),
        (mid, "mid", n_per),
        (low, "low", n_per),
    ]:
        drawn = random.sample(stratum, min(count, len(stratum)))
        for item in drawn:
            item["validation_stratum"] = name
        sample.extend(drawn)

    random.shuffle(sample)
    return sample


def provenance_ratios(analyzed: list[dict], provenance: dict[int, str], threshold: float) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for a in analyzed:
        prov = provenance.get(a["cluster_id"], "unknown")
        groups[prov].append(a)

    result = {}
    for prov_type, items in sorted(groups.items()):
        form = sum(1 for a in items if a["char_unigram_jaccard"] > threshold)
        mng = len(items) - form
        result[prov_type] = {
            "count": len(items),
            "form_based": form,
            "meaning_based": mng,
            "form_pct": round(form / len(items) * 100, 2) if items else 0.0,
        }
    return result


def main():
    t_start = time.time()

    print("=" * 60)
    print("Cross-Lingual Analysis (5-algo, full census)")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/7] Loading verse metadata...")
    verse_meta = load_verse_metadata()

    print("\n[2/7] Loading gloss index...")
    gloss_data = load_gloss_index()

    print("\n[3/7] Loading 5-algo clusters...")
    clusters_5algo = load_clusters(MEMBERSHIP_PATH)

    # --- Identify CL clusters ---
    print("\n[4/7] Identifying cross-lingual clusters...")
    cl_5algo = identify_cross_lingual(clusters_5algo, verse_meta)
    print(f"  Cross-lingual clusters: {len(cl_5algo):,}")

    # --- Analyze all CL clusters ---
    print(f"\n[5/7] Analyzing all {len(cl_5algo):,} CL clusters...")
    analyzed = []
    t0 = time.time()
    for i, entry in enumerate(cl_5algo):
        result = analyze_cluster(entry, verse_meta, gloss_data)
        analyzed.append(result)
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1:,} / {len(cl_5algo):,} analyzed")
    print(f"  Analysis complete in {time.time() - t0:.1f}s")

    categories = Counter(a["category"] for a in analyzed)
    print(f"  Categories: {dict(categories)}")

    # --- Provenance tracking ---
    print("\n[6/7] Computing provenance against 4-algo baseline...")
    if MEMBERSHIP_4ALGO_PATH.exists():
        clusters_4algo = load_clusters(MEMBERSHIP_4ALGO_PATH, compressed=True)
        provenance = compute_provenance(cl_5algo, verse_meta, clusters_4algo)
        del clusters_4algo
    else:
        print(f"  WARNING: {MEMBERSHIP_4ALGO_PATH.name} not found — skipping provenance")
        provenance = {}

    for a in analyzed:
        a["provenance"] = provenance.get(a["cluster_id"], "unknown")

    # --- Histograms and threshold analysis ---
    print("\n[7/7] Computing histograms and threshold analysis...")
    histograms = build_histograms(analyzed)

    uni_values = [a["char_unigram_jaccard"] for a in analyzed]
    otsu_thresh, otsu_note = try_otsu_threshold(uni_values)
    print(f"  Otsu threshold (unigram): {otsu_thresh} ({otsu_note})")

    bi_values = [a["char_bigram_jaccard"] for a in analyzed]
    otsu_bi, otsu_bi_note = try_otsu_threshold(bi_values)
    print(f"  Otsu threshold (bigram): {otsu_bi} ({otsu_bi_note})")

    tri_values = [a["char_trigram_jaccard"] for a in analyzed]
    otsu_tri, otsu_tri_note = try_otsu_threshold(tri_values)
    print(f"  Otsu threshold (trigram): {otsu_tri} ({otsu_tri_note})")

    # Weighted ratios at original threshold
    weighted = compute_weighted_ratios(analyzed, 0.4)

    # Sensitivity: also compute at Otsu threshold if different
    sensitivity = {"original_0.4": weighted}
    if otsu_thresh is not None and abs(otsu_thresh - 0.4) > 0.01:
        sensitivity[f"otsu_{otsu_thresh:.3f}"] = compute_weighted_ratios(analyzed, otsu_thresh)

    # Provenance-stratified ratios
    prov_ratios = provenance_ratios(analyzed, provenance, 0.4)
    if otsu_thresh is not None and abs(otsu_thresh - 0.4) > 0.01:
        prov_ratios_otsu = provenance_ratios(analyzed, provenance, otsu_thresh)
    else:
        prov_ratios_otsu = None

    # --- Manual validation sample ---
    sample = draw_validation_sample(analyzed, 100)
    print(f"\n  Validation sample: {len(sample)} clusters "
          f"(strata: {Counter(s['validation_stratum'] for s in sample)})")

    # --- Summary ---
    meaning_examples = [a for a in analyzed if a["category"] == "meaning_based"][:5]
    print(f"\n  Meaning-based examples:")
    for a in meaning_examples:
        print(f"    Cluster {a['cluster_id']}: ET \"{a['et_example'][:50]}\" / "
              f"FI \"{a['fi_example'][:50]}\"")
        print(f"      uni_J={a['char_unigram_jaccard']:.3f} bi_J={a['char_bigram_jaccard']:.3f} "
              f"tri_J={a['char_trigram_jaccard']:.3f}")

    # --- Write main output ---
    output = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cross_lingual_clusters": len(cl_5algo),
        "analyzed_count": len(analyzed),
        "category_counts": dict(categories),
        "primary_threshold": 0.4,
        "primary_metric": "char_unigram_jaccard",
        "metrics": ["char_unigram_jaccard", "char_bigram_jaccard", "char_trigram_jaccard"],
        "text_truncation_note": f"ET and FI texts capped at {TEXT_CAP} per cluster",
        "threshold_analysis": {
            "otsu_unigram": {"threshold": otsu_thresh, "note": otsu_note},
            "otsu_bigram": {"threshold": otsu_bi, "note": otsu_bi_note},
            "otsu_trigram": {"threshold": otsu_tri, "note": otsu_tri_note},
        },
        "histograms": histograms,
        "weighted_ratios": sensitivity,
        "provenance_counts": dict(Counter(provenance.values())) if provenance else {},
        "provenance_ratios_at_0.4": prov_ratios,
        "provenance_ratios_at_otsu": prov_ratios_otsu,
        "clusters": analyzed,
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH.name} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Write validation sample ---
    sample_output = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(sample),
        "strata": {
            "high_gt_0.6": len([s for s in sample if s["validation_stratum"] == "high"]),
            "mid_0.3_0.6": len([s for s in sample if s["validation_stratum"] == "mid"]),
            "low_lt_0.3": len([s for s in sample if s["validation_stratum"] == "low"]),
        },
        "instructions": (
            "For each cluster, label as: form_based, meaning_based, or ambiguous. "
            "Compare ET and FI examples. If the words look similar across languages "
            "(shared roots, similar sounds / orthography), it's form-based. If the "
            "content is thematically similar but words are different, it's meaning-based."
        ),
        "clusters": sample,
    }

    with VALIDATION_SAMPLE_PATH.open("w") as f:
        json.dump(sample_output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {VALIDATION_SAMPLE_PATH.name}")

    total_time = time.time() - t_start
    print(f"\nAnalysis complete in {total_time:.0f}s ({total_time / 60:.1f} min)")

    # --- Verification summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  analyzed_count == total_cl: {len(analyzed)} == {len(cl_5algo)} → {'PASS' if len(analyzed) == len(cl_5algo) else 'FAIL'}")
    form_plus_mng = categories.get("form_based", 0) + categories.get("meaning_based", 0)
    print(f"  form + meaning == analyzed: {form_plus_mng} == {len(analyzed)} → {'PASS' if form_plus_mng == len(analyzed) else 'FAIL'}")
    if provenance:
        prov_total = sum(Counter(provenance.values()).values())
        print(f"  provenance entries == CL count: {prov_total} == {len(cl_5algo)} → {'PASS' if prov_total == len(cl_5algo) else 'FAIL'}")
    print(f"  validation sample size: {len(sample)}")


if __name__ == "__main__":
    main()
