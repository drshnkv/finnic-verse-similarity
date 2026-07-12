#!/usr/bin/env python3
"""
Proper gold-standard evaluation against the SKVR Verse Equivalence Gold Standard
(Janicki, Kallio & Sarv 2023; CC BY-NC-ND 4.0).

It scores co-membership correctly on two points a naive evaluation gets wrong:

  (1) WRONG SUBSET. A naive evaluation scores the 1,010-pair "common" sample, which
      the dataset README explicitly excludes from evaluation ("The common sample is
      not included in the evaluation dataset" - it exists only to measure
      inter-annotator agreement). It also used a single annotator (Kati). The
      intended evaluation set is the concatenation of the four annotator-specific
      samples = ~12,000 pairs (results-{annotator}.csv).

  (2) OVERLAP COLLAPSED. A naive evaluation reduces each verse to ONE vid and ONE
      cluster (first-seen), so it could not detect a shared cluster for a verse that
      lives in several overlapping clusters (~2.1 on average). That deflates OUR
      recall specifically. This evaluator is overlap-aware (set intersection).

Also decomposes our false negatives (gold says "same formula", we say "not together"):
  A) text not found in our corpus at all      -> corpus/text-match coverage
  B) found in corpus but not in any cluster    -> clustering coverage (singleton)
  C) both clustered, but in disjoint clusters  -> linkage miss (consensus/threshold)

Two membership paths can be scored head-to-head in ONE run, so the paired
McNemar/bootstrap test compares a candidate against a baseline over IDENTICAL
eval pairs:
  CLEAN_DIGITS=1 evaluate_gold_standard_proper.py \
      --baseline <baseline_membership>.jsonl.gz \
      --candidate <candidate_membership>.jsonl.gz \
      --out <eval_output>.json
A bare `--baseline <membership>` run (no --candidate/--out) scores a
single membership and writes output/gold_standard_proper_eval[_clean].json.

Outputs: prints a report + writes output/gold_standard_proper_eval[_clean].json
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

# CLEAN_DIGITS=1 turns on EDITORIAL-MARKER cleaning of the matching key. The gold
# sample texts are the only one of the three sources (gold / our corpus / FILTER's
# v_clust) that carry editorial transcription that defeats exact text matching:
#   * pure-digit tokens   : verse-position markers ("20 Minä kakun...")
#   * underscores         : sandhi word-joins ("Nukun_nukun_nurmilintu")
#   * #n footnote markers : ("silmin#1", "kuoppimassa."#30)
#   * [[ ]] / glottal ˀ   : uncertainty/glottal notation ("ˀ_ˀ[[#1]]")
# Our corpus (0 of 4.29M verses) and FILTER (0 of 2.9M texts) contain none of these,
# so stripping them only removes a matching artifact, symmetrically for both systems
# (the strip lives in the shared normalize_text). Verified: ~92% of the "not in our
# corpus" verses are actually present once these markers are undone.
STRIP_DIGIT_TOKENS = os.environ.get("CLEAN_DIGITS") == "1"
GLOTTAL = "ˀ"  # ˀ modifier letter glottal stop (editorial)

BASE = Path(__file__).resolve().parent
GS = BASE / "filter_comparison" / "skvr-gs" / "data"
META = BASE / "output" / "verse_metadata.jsonl.gz"
MEMB = BASE / "output" / "rrf_cluster_membership.jsonl.gz"
VCLUST = BASE.parent / "filter_exports" / "v_clust.tsv"
OUT = BASE / "output" / (
    "gold_standard_proper_eval_clean.json"
    if os.environ.get("CLEAN_DIGITS") == "1"
    else "gold_standard_proper_eval.json"
)

ANNOT = ["kati", "mari", "venla", "jukka"]
FILTER_CLUSTERING_ID = "0"


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower().strip()
    if STRIP_DIGIT_TOKENS:
        # Undo gold-only editorial markers BEFORE punctuation removal so "#1" /
        # "[[ ]]" / "_" become word boundaries instead of gluing into tokens.
        text = text.replace("_", " ").replace(GLOTTAL, " ")
        text = re.sub(r"#\d+", " ", text)
        text = text.replace("[[", " ").replace("]]", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    if STRIP_DIGIT_TOKENS:
        text = " ".join(t for t in text.split() if not t.isdigit())
    return text.strip()


def is_same_label(raw):
    return str(raw).strip().lower() in ("s", "same", "equivalent", "eq", "1", "true")


def load_annotator_pairs(sample_path, results_path):
    """Return list of (norm_text1, norm_text2, is_equivalent), joined on (id1,id2)."""
    texts = {}
    with open(sample_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts[(row["id1"], row["id2"])] = (row.get("text1", ""), row.get("text2", ""))
    pairs = []
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = texts.get((row["id1"], row["id2"]))
            if t is None:
                continue
            cls = str(row["class"]).strip().lower()
            if cls not in ("s", "d"):  # drop the 1 'not annotated' row (mari)
                continue
            pairs.append((normalize_text(t[0]), normalize_text(t[1]), is_same_label(row["class"])))
    return pairs


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def bootstrap_ci(records, n_boot=2000, seed=42):
    """records: list of (pred_bool, gold_bool). Returns CIs for P,R,F1."""
    rng = np.random.default_rng(seed)
    preds = np.array([1 if p else 0 for p, _ in records])
    golds = np.array([1 if g else 0 for _, g in records])
    n = len(records)
    ps, rs, fs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pp, gg = preds[idx], golds[idx]
        tp = int(((pp == 1) & (gg == 1)).sum())
        fp = int(((pp == 1) & (gg == 0)).sum())
        fn = int(((pp == 0) & (gg == 1)).sum())
        p, r, f = prf(tp, fp, fn)
        ps.append(p); rs.append(r); fs.append(f)
    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"precision_95ci": ci(ps), "recall_95ci": ci(rs), "f1_95ci": ci(fs)}


def _f1_arr(pred, gold):
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    _, _, f = prf(tp, fp, fn)
    return f


def paired_significance(rec_a, rec_b, n_boot=2000, seed=123):
    """rec_a (ours), rec_b (FILTER): aligned (pred,gold) over the SAME pairs.
    McNemar on gold positives (continuity-corrected) + paired bootstrap of the
    F1 difference (FILTER - ours). Same-pairs => the comparison is paired."""
    import math
    a_only = b_only = 0  # discordant catches among gold positives
    for (pa, ga), (pb, _gb) in zip(rec_a, rec_b):
        if ga:
            if pa and not pb:
                a_only += 1
            elif pb and not pa:
                b_only += 1
    nd = a_only + b_only
    chi2 = (abs(b_only - a_only) - 1) ** 2 / nd if nd else 0.0
    p_mc = math.erfc(math.sqrt(chi2 / 2)) if chi2 else 1.0
    rng = np.random.default_rng(seed)
    pa = np.array([1 if p else 0 for p, _ in rec_a])
    ga = np.array([1 if g else 0 for _, g in rec_a])
    pb = np.array([1 if p else 0 for p, _ in rec_b])
    gb = np.array([1 if g else 0 for _, g in rec_b])
    n = len(rec_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = _f1_arr(pb[idx], gb[idx]) - _f1_arr(pa[idx], ga[idx])
    return {"ours_only_catches": a_only, "filter_only_catches": b_only,
            "mcnemar_chi2_cc": round(chi2, 1),
            "mcnemar_p_approx": p_mc,
            "f1_diff_filter_minus_ours_95ci":
                [round(float(np.percentile(diffs, 2.5)), 3),
                 round(float(np.percentile(diffs, 97.5)), 3)],
            "frac_bootstrap_filter_not_better": float(np.mean(diffs <= 0))}


def relabel_paired_baseline_candidate(sig):
    """paired_significance(rec_baseline, rec_candidate) puts baseline in the
    rec_a('ours') slot and candidate in rec_b('FILTER') slot, so its F1 diff is
    candidate - baseline (positive = candidate improvement). Rename keys so the
    direction is explicit and not mislabelled ours-vs-FILTER."""
    return {
        "baseline_only_catches": sig["ours_only_catches"],
        "candidate_only_catches": sig["filter_only_catches"],
        "mcnemar_chi2_cc": sig["mcnemar_chi2_cc"],
        "mcnemar_p_approx": sig["mcnemar_p_approx"],
        "f1_diff_candidate_minus_baseline_95ci": sig["f1_diff_filter_minus_ours_95ci"],
        "frac_bootstrap_candidate_not_better": sig["frac_bootstrap_filter_not_better"],
    }


def _open_text(path):
    """Open a .jsonl or .jsonl.gz in text mode."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_membership(memb_path, Vint):
    """Stream a {cluster_id,size,members} membership JSONL(.gz) restricted to Vint.
    Returns (vid_to_clusters: vid->set(cid), vid_first_cluster: vid->first cid)."""
    vid_to_clusters = defaultdict(set)
    vid_first_cluster = {}
    with _open_text(memb_path) as f:
        for line in f:
            e = json.loads(line)
            cid = e["cluster_id"]
            for vid in e["members"]:
                if vid in Vint:
                    vid_to_clusters[vid].add(cid)
                    vid_first_cluster.setdefault(vid, cid)
    return vid_to_clusters, vid_first_cluster


def eval_rrf(pairs, text_to_vids, vid_to_clusters, vid_first_cluster,
             mode="overlap", decompose=False):
    """Score one RRF-style membership."""
    def text_clusters(n):
        cs = set()
        for v in text_to_vids.get(n, ()):
            cs |= vid_to_clusters.get(v, set())
        return cs

    tp = fp = fn = tn = 0
    buckets = {"A_not_in_corpus": 0, "B_unclustered": 0, "C_disjoint_clusters": 0}
    examples = {"A": [], "B": [], "C": []}
    fp_examples = []
    records = []
    for n1, n2, is_eq in pairs:
        if mode == "overlap":
            same = len(text_clusters(n1) & text_clusters(n2)) > 0
        else:  # 'single' = mimic the committed harness (first vid, first cluster)
            v1 = text_to_vids.get(n1, [None])[0]
            v2 = text_to_vids.get(n2, [None])[0]
            c1 = vid_first_cluster.get(v1)
            c2 = vid_first_cluster.get(v2)
            same = c1 is not None and c1 == c2
        records.append((same, is_eq))
        if is_eq:
            if same:
                tp += 1
            else:
                fn += 1
                if decompose:
                    inc1, inc2 = n1 in text_to_vids, n2 in text_to_vids
                    cl1, cl2 = bool(text_clusters(n1)), bool(text_clusters(n2))
                    if not inc1 or not inc2:
                        bk = "A"; buckets["A_not_in_corpus"] += 1
                    elif not cl1 or not cl2:
                        bk = "B"; buckets["B_unclustered"] += 1
                    else:
                        bk = "C"; buckets["C_disjoint_clusters"] += 1
                    if len(examples[bk]) < 8:
                        examples[bk].append([n1, n2])
        else:
            if same:
                fp += 1
                if decompose and len(fp_examples) < 8:
                    fp_examples.append([n1, n2])
            else:
                tn += 1
    p, r, f = prf(tp, fp, fn)
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    if decompose:
        out["fn_buckets"] = buckets
        out["fn_examples"] = examples
        out["fp_examples"] = fp_examples
    return out, records


def eval_filter(pairs, filter_map):
    tp = fp = fn = tn = 0
    records = []
    for n1, n2, is_eq in pairs:
        same = len(filter_map.get(n1, set()) & filter_map.get(n2, set())) > 0
        records.append((same, is_eq))
        if is_eq:
            tp += same; fn += (not same)
        else:
            fp += same; tn += (not same)
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}, records


def lang_breakdown(pairs, text_to_vids, vid_lang):
    counts = defaultdict(int)
    for n1, n2, is_eq in pairs:
        if not is_eq:
            continue
        l1 = next((vid_lang[v] for v in text_to_vids.get(n1, []) if v in vid_lang), "?")
        l2 = next((vid_lang[v] for v in text_to_vids.get(n2, []) if v in vid_lang), "?")
        counts["-".join(sorted([str(l1), str(l2)]))] += 1
    return dict(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=str(MEMB),
                    help="primary RRF membership .jsonl(.gz) scored as 'rrf' (default: the canonical full membership)")
    ap.add_argument("--candidate", default=None,
                    help="optional second membership; scored head-to-head vs --baseline (paired test)")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: CLEAN_DIGITS-derived gold_standard_proper_eval[_clean].json)")
    args = ap.parse_args()

    memb_path = Path(args.baseline)
    out_path = Path(args.out) if args.out else OUT

    # -----------------------------------------------------------------------
    print("Loading gold-standard pairs...")
    eval_pairs = []  # the proper ~12k evaluation set
    per_annot_counts = {}
    for a in ANNOT:
        pa = load_annotator_pairs(GS / f"sample-{a}.csv", GS / f"results-{a}.csv")
        per_annot_counts[a] = (len(pa), sum(1 for _, _, e in pa if e))
        eval_pairs += pa
    print(f"  Proper eval set: {len(eval_pairs):,} pairs, "
          f"{sum(1 for _,_,e in eval_pairs if e):,} positives")
    for a in ANNOT:
        print(f"    {a:6s}: {per_annot_counts[a][0]:4d} pairs, {per_annot_counts[a][1]:3d} positive")

    # common sample (inter-annotator agreement set) for repro + cross-check
    common_texts = {}
    with open(GS / "sample-common.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            common_texts[(row["id1"], row["id2"])] = (row["text1"], row["text2"])
    common_votes = defaultdict(dict)
    for a in ANNOT:
        with open(GS / f"results-common-{a}.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                common_votes[(row["id1"], row["id2"])][a] = is_same_label(row["class"])

    kati_common, maj_common = [], []
    for key, (t1, t2) in common_texts.items():
        n1, n2 = normalize_text(t1), normalize_text(t2)
        votes = common_votes.get(key, {})
        if "kati" in votes:
            kati_common.append((n1, n2, votes["kati"]))
        if votes:
            maj_common.append((n1, n2, sum(votes.values()) >= 2))
    print(f"  Common sample: {len(kati_common):,} pairs (Kati), "
          f"{sum(1 for _,_,e in maj_common if e):,} majority-positive")

    # gold text universe
    G = set()
    for plist in (eval_pairs, kati_common, maj_common):
        for n1, n2, _ in plist:
            if n1:
                G.add(n1)
            if n2:
                G.add(n2)
    print(f"  Distinct gold normalized texts: {len(G):,}")

    # -----------------------------------------------------------------------
    print("\nStreaming our corpus metadata (text -> vids, lang)...")
    text_to_vids = defaultdict(list)
    vid_lang = {}
    with gzip.open(META, "rt", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            n = normalize_text(e.get("t", ""))
            if n in G:
                text_to_vids[n].append(e["v"])
                vid_lang[e["v"]] = e.get("l")
    Vint = set(v for vs in text_to_vids.values() for v in vs)
    print(f"  Gold texts present in our corpus: {len(text_to_vids):,} / {len(G):,} "
          f"({100*len(text_to_vids)/len(G):.1f}%); {len(Vint):,} verses")

    print(f"Streaming our cluster membership (overlap-aware) <- {memb_path.name}")
    vid_to_clusters, vid_first_cluster = stream_membership(memb_path, Vint)

    cand_maps = None
    if args.candidate:
        print(f"Streaming CANDIDATE membership (overlap-aware) <- {Path(args.candidate).name}")
        cand_maps = stream_membership(args.candidate, Vint)

    print("Streaming FILTER clustering (gold texts only)...")
    # overlap-aware (set) to match how RRF is scored: a normalized text that collapses
    # to >1 FILTER cluster keeps all of them (17 such texts; avoids charging FILTER a
    # normalization-collapse artifact). FILTER's published clustering is a partition, so
    # this only affects a handful of normalization-merged texts.
    filter_map = defaultdict(set)
    with open(VCLUST, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or parts[0] != FILTER_CLUSTERING_ID:
                continue
            n = normalize_text(parts[1])
            if n in G:
                filter_map[n].add(parts[2])
    print(f"  Gold texts present in FILTER clustering: {len(filter_map):,} / {len(G):,} "
          f"({100*len(filter_map)/len(G):.1f}%)")

    # -----------------------------------------------------------------------
    report = {}

    print("\n" + "=" * 72)
    print("(0) REPRODUCTION CHECK: Kati common, single-cluster (mimics committed eval)")
    r_single, _ = eval_rrf(kati_common, text_to_vids, vid_to_clusters, vid_first_cluster, mode="single")
    f_kati, _ = eval_filter(kati_common, filter_map)
    print(f"    RRF (single):  P={r_single['precision']} R={r_single['recall']} "
          f"F1={r_single['f1']} (tp={r_single['tp']},fp={r_single['fp']},fn={r_single['fn']})  "
          f"[committed: P=0.833 R=0.047 F1=0.088]")
    print(f"    FILTER:        P={f_kati['precision']} R={f_kati['recall']} "
          f"F1={f_kati['f1']}  [committed: P=0.708 R=0.397 F1=0.509]")
    report["repro_kati_common_single"] = {"rrf": r_single, "filter": f_kati}

    print("\n(1) OVERLAP FIX, same Kati common sample")
    r_ov, _ = eval_rrf(kati_common, text_to_vids, vid_to_clusters, vid_first_cluster, mode="overlap")
    print(f"    RRF (overlap): P={r_ov['precision']} R={r_ov['recall']} F1={r_ov['f1']} "
          f"(tp={r_ov['tp']},fp={r_ov['fp']},fn={r_ov['fn']})")
    report["kati_common_overlap"] = {"rrf": r_ov, "filter": f_kati}

    print("\n(2) PROPER 12k EVALUATION SET (annotator-specific, overlap-aware)")
    r_12k, rec_r = eval_rrf(eval_pairs, text_to_vids, vid_to_clusters, vid_first_cluster,
                            mode="overlap", decompose=True)
    f_12k, rec_f = eval_filter(eval_pairs, filter_map)
    ci_r = bootstrap_ci(rec_r)
    ci_f = bootstrap_ci(rec_f)
    print(f"    RRF:    P={r_12k['precision']} R={r_12k['recall']} F1={r_12k['f1']} "
          f"(tp={r_12k['tp']},fp={r_12k['fp']},fn={r_12k['fn']},tn={r_12k['tn']})")
    print(f"            P95CI={ci_r['precision_95ci']} R95CI={ci_r['recall_95ci']} F1_95CI={ci_r['f1_95ci']}")
    print(f"    FILTER: P={f_12k['precision']} R={f_12k['recall']} F1={f_12k['f1']} "
          f"(tp={f_12k['tp']},fp={f_12k['fp']},fn={f_12k['fn']},tn={f_12k['tn']})")
    print(f"            P95CI={ci_f['precision_95ci']} R95CI={ci_f['recall_95ci']} F1_95CI={ci_f['f1_95ci']}")
    report["proper_12k"] = {"rrf": {**r_12k, **ci_r}, "filter": {**f_12k, **ci_f},
                            "n_pairs": len(eval_pairs),
                            "n_positives": sum(1 for _, _, e in eval_pairs if e)}

    print("\n(2b) PAIRED SIGNIFICANCE (same 12k pairs; FILTER vs ours)")
    sig = paired_significance(rec_r, rec_f)
    print(f"    Discordant gold-positive catches: FILTER-only={sig['filter_only_catches']}, "
          f"ours-only={sig['ours_only_catches']}")
    print(f"    McNemar (continuity-corrected) chi2={sig['mcnemar_chi2_cc']}, "
          f"p~={sig['mcnemar_p_approx']:.2e}")
    print(f"    Paired bootstrap F1 diff (FILTER - ours): "
          f"{sig['f1_diff_filter_minus_ours_95ci']}  "
          f"(resamples where FILTER not better: {sig['frac_bootstrap_filter_not_better']:.4f})")
    report["paired_significance"] = sig

    print("\n(3) FALSE-NEGATIVE DECOMPOSITION (our system, 12k positives)")
    b = r_12k["fn_buckets"]
    tot_fn = sum(b.values())
    tp12 = r_12k["tp"]
    pos = tp12 + tot_fn
    print(f"    Gold positives: {pos}   TP (we catch): {tp12}   FN (we miss): {tot_fn}")
    for k, v in b.items():
        print(f"      {k:24s}: {v:4d}  ({100*v/max(1,tot_fn):.1f}% of misses)")
    print(f"    Lang breakdown of gold positives: {lang_breakdown(eval_pairs, text_to_vids, vid_lang)}")

    print("\n(4) MAJORITY-VOTE common sample (>=2 of 4 annotators), overlap-aware [cross-check]")
    r_maj, _ = eval_rrf(maj_common, text_to_vids, vid_to_clusters, vid_first_cluster, mode="overlap")
    f_maj, _ = eval_filter(maj_common, filter_map)
    print(f"    RRF:    P={r_maj['precision']} R={r_maj['recall']} F1={r_maj['f1']}")
    print(f"    FILTER: P={f_maj['precision']} R={f_maj['recall']} F1={f_maj['f1']}")
    report["majority_common_overlap"] = {"rrf": r_maj, "filter": f_maj}

    # -----------------------------------------------------------------------
    # Head-to-head experimental candidate vs the (frozen) baseline, scored on
    # the IDENTICAL eval pairs so the McNemar/bootstrap test is genuinely paired.
    # Purely additive: absent --candidate this block is skipped and the report
    # above is unchanged.
    if cand_maps is not None:
        print("\n(5) CANDIDATE membership (experimental), overlap-aware [paired vs baseline]")
        c_vid_to_clusters, c_vid_first_cluster = cand_maps
        r_cand, rec_cand = eval_rrf(eval_pairs, text_to_vids, c_vid_to_clusters, c_vid_first_cluster,
                                    mode="overlap", decompose=True)
        ci_cand = bootstrap_ci(rec_cand)
        print(f"    CANDIDATE: P={r_cand['precision']} R={r_cand['recall']} F1={r_cand['f1']} "
              f"(tp={r_cand['tp']},fp={r_cand['fp']},fn={r_cand['fn']},tn={r_cand['tn']})")
        print(f"               P95CI={ci_cand['precision_95ci']} R95CI={ci_cand['recall_95ci']} "
              f"F1_95CI={ci_cand['f1_95ci']}")
        cb = r_cand["fn_buckets"]
        print(f"    CANDIDATE FN buckets: {cb}")
        report["proper_12k_candidate"] = {
            "rrf": {**r_cand, **ci_cand},
            "n_pairs": len(eval_pairs),
            "n_positives": sum(1 for _, _, e in eval_pairs if e),
            "baseline_membership": memb_path.name,
            "candidate_membership": Path(args.candidate).name,
        }
        paired_bc = relabel_paired_baseline_candidate(paired_significance(rec_r, rec_cand))
        print(f"    Paired (candidate - baseline) F1 diff 95% CI: "
              f"{paired_bc['f1_diff_candidate_minus_baseline_95ci']}  "
              f"McNemar chi2={paired_bc['mcnemar_chi2_cc']} p~={paired_bc['mcnemar_p_approx']:.2e}")
        print(f"    Discordant catches: candidate-only={paired_bc['candidate_only_catches']}, "
              f"baseline-only={paired_bc['baseline_only_catches']}")
        report["paired_baseline_vs_candidate"] = paired_bc

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
