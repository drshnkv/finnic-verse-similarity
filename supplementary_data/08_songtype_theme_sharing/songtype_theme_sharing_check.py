#!/usr/bin/env python
"""Robustness check behind the JEFUL §4.6 song-type/theme wording (2026-07-03).

Question: is the Phase E headline "45.5% of both-classified clusters fall under
the same thematic category" evidence that the corpora share the same kinds of
song? Decompose it and compare against a random-pairing baseline.

Inputs (both must exist; run from its floor2_rederive home):
  - cl_cluster_song_type_analysis.json   (Phase E, this folder — per_cluster
    ET/FI type lists + summary block)
  - deployment/drafts/song_type_theme_lookup.json  (type name -> 1 of 11
    themes; extracted by build_song_type_theme_lookup.py from
    deployment/song_type_index.json, whose `theme` field is assigned by
    classify_theme() in build_song_type_index.py — bilingual keyword rules on
    the type NAME, first match wins, default 'other')

Computed over the clusters typed on BOTH sides (n = 9,436):
  - exact shared type name (the two national catalogues are essentially
    disjoint inventories)
  - shared type name after matching through the LLM-generated natural-phrase
    translations (translation_batches/ at the repo root — the 2026-03-20..21
    claude-headless run of build_type_translations.py plus the 2026-07
    top-up batches, 290 batches in all — plus the 289 reviewed overrides in
    similarity/song_type_en_overrides.json). Matching is lowercased and
    punctuation-insensitive. This is the number the draft cites. Fairer
    than exact matching: cognate types normally carry different
    original-language names (Suur tamm vs Iso tammi), which exact matching
    can never link; coverage over the in-scope names is near-complete
    (1 lacks a translation).
  - shared >= 1 theme (the Phase E 45.5%), split into: only the catch-all
    'other' shared vs a substantive theme shared (anything but other/unknown)
  - chance baseline: shuffle the FI-side theme sets across clusters
    (seed 42, 200 shuffles) and recompute both sharing rates

Findings (2026-07-03 run; reproduced identically 2026-07-04 on the deployed
curated translation set): exact type name shared in 15/9,436 (0.2%);
matched via the
LLM translations 87/9,436 (0.9%) — composition of the 87: 72 clusters are
translation-linked clear cognates (Ussisõnad/Käärmeen sanat 'snake charm'
x16, Suur tamm/Iso tammi 'the great oak' x12, Suur härg/Iso härkä x7,
Mis on üks/Mikä yksi? x5, Tulesõnad/Tulen sanat x5, Müüdud neiu/Myyty
neito x5, Kurg kündmas/Kurki kyntämässä x4, sprain charms vs Niukahdus x4,
Sõrmus kadunud/Sormus kadonnut x3, Imemaa/Ihmemaa x3, Laevamäng/
Laivaleikki x3, Ilm udune/Ilma utuinen x2, new-moon greeting x2, Rikas ja
vaene/Rikas ja köyhä x1), 12 same-string cognates (Maie x6, Onnimanni x5,
Laula x1), and 3 artifacts where a misfiled Estonian poem (E, StK
signatures under Finnish-side IDs) brought its Estonian name to the FI
side (Hobune varastatud, Helise mets, Ori taevas). 1 in-scope type name
carries no LLM translation (15 at the 2026-07-03 run, before the top-up
batches landed; the 14 newly translated names created no new match).
Shared-any-theme 45.5% vs 41.2% chance; only-'other' 33.7%; substantive
11.8% vs 7.9% chance. So the 45.5% is mostly the catch-all sitting a few
points above chance — the draft dropped that sentence and leads with the
type-inventory disjointness (the LLM-translation-matched 87) + the
theme-enrichment chi-squared instead.

2026-07-04: a former secondary comparison through the mechanical
word-by-word glosses (song_type_index.json `en` field; 41/9,436) was
removed on Kaarel's decision — after the curated-translation merge the
index `en` field no longer carries the mechanical gloss, so the metric had
lost its referent. The article never cited it. All remaining counts and
the full match composition reproduce identically on the deployed set.

Output: songtype_theme_sharing_check.json + printed summary. Read-only inputs.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

RD = Path(__file__).resolve().parent
PROJECT_ROOT = RD.parents[2]
ANALYSIS = RD / "cl_cluster_song_type_analysis.json"
THEME_LOOKUP = PROJECT_ROOT / "deployment" / "drafts" / "song_type_theme_lookup.json"
TRANSLATION_BATCHES = PROJECT_ROOT / "translation_batches"
EN_OVERRIDES = PROJECT_ROOT / "similarity" / "song_type_en_overrides.json"
OUT = RD / "songtype_theme_sharing_check.json"

SEED = 42
N_SHUFFLES = 200
NON_SUBSTANTIVE = {"other", "unknown"}


def main():
    lookup = json.loads(THEME_LOOKUP.read_text())
    data = json.loads(ANALYSIS.read_text())
    summary = data["summary"]
    recs = [r for r in data["per_cluster"].values()
            if r.get("has_both_sides_typed")]

    def themes(types):
        return frozenset(lookup.get(t, "unknown") for t in types)

    et = [themes(r["et_types"]) for r in recs]
    fi = [themes(r["fi_types"]) for r in recs]
    n = len(recs)

    exact_type = sum(1 for r in recs if r.get("has_shared_type"))

    # LLM-translation matching: link cognate types whose original-language
    # names differ but whose natural-phrase claude-headless translations
    # coincide. Lowercased + punctuation-stripped
    # (recovers Mis on üks / Mikä yksi?).
    def norm_en(s):
        s = re.sub(r"[.,;:!?\"'()\[\]]", "", s.strip().lower())
        return re.sub(r"\s+", " ", s).strip()

    name2llm = {}
    for f in sorted(TRANSLATION_BATCHES.glob("batch_*.json")):
        for it in json.loads(f.read_text()):
            if isinstance(it, dict) and it.get("name") and it.get("en"):
                name2llm[it["name"]] = norm_en(it["en"])
    for k, v in json.loads(EN_OVERRIDES.read_text()).items():
        name2llm[k] = norm_en(v)

    llm_shared = 0
    llm_combos = Counter()
    llm_unresolved = set()
    for r in recs:
        et_llm = {name2llm[t]: t for t in r["et_types"] if t in name2llm}
        fi_llm = {name2llm[t]: t for t in r["fi_types"] if t in name2llm}
        llm_unresolved.update(t for t in r["et_types"] + r["fi_types"]
                              if t not in name2llm)
        inter = set(et_llm) & set(fi_llm)
        if inter:
            llm_shared += 1
            for g in inter:
                llm_combos[(g, et_llm[g], fi_llm[g])] += 1

    shared_any = sum(1 for a, b in zip(et, fi) if a & b)
    shared_subst = sum(1 for a, b in zip(et, fi) if (a & b) - NON_SUBSTANTIVE)
    only_catchall = shared_any - shared_subst
    unknown_in_shared = sum(1 for a, b in zip(et, fi) if "unknown" in (a & b))

    # cross-check against the Phase E summary block
    assert n == summary["both_sides_typed"], (n, summary["both_sides_typed"])
    assert shared_any == summary["shared_theme"], (shared_any,
                                                   summary["shared_theme"])
    assert exact_type == summary["shared_type"], (exact_type,
                                                  summary["shared_type"])

    random.seed(SEED)
    fi_s = fi[:]
    tot_any = tot_subst = 0
    for _ in range(N_SHUFFLES):
        random.shuffle(fi_s)
        tot_any += sum(1 for a, b in zip(et, fi_s) if a & b)
        tot_subst += sum(1 for a, b in zip(et, fi_s)
                         if (a & b) - NON_SUBSTANTIVE)

    result = {
        "generated": "2026-07-04",
        "n_both_sides_typed": n,
        "exact_shared_type_name": {"count": exact_type,
                                   "rate": round(exact_type / n, 4)},
        "shared_type_name_via_llm_translation": {
            "count": llm_shared,
            "rate": round(llm_shared / n, 4),
            "method": "claude-headless natural-phrase translations "
                      "(translation_batches/ 2026-03 run of "
                      "build_type_translations.py + 2026-07 top-up "
                      "batches + song_type_en_overrides.json), lowercased, "
                      "punctuation-stripped exact match",
            "type_names_without_translation": len(llm_unresolved),
            "match_composition": [
                {"translation": g, "et_name": et_n, "fi_name": fi_n,
                 "n_clusters": c}
                for (g, et_n, fi_n), c in llm_combos.most_common()],
        },
        "shared_any_theme": {"count": shared_any,
                             "rate": round(shared_any / n, 4)},
        "shared_only_catchall_other": {"count": only_catchall,
                                       "rate": round(only_catchall / n, 4)},
        "shared_substantive_theme": {"count": shared_subst,
                                     "rate": round(shared_subst / n, 4)},
        "unknown_theme_in_shared": unknown_in_shared,
        "chance_baseline": {
            "method": f"shuffle FI-side theme sets across clusters, "
                      f"{N_SHUFFLES} shuffles, seed {SEED}",
            "shared_any_theme_rate": round(tot_any / N_SHUFFLES / n, 4),
            "shared_substantive_theme_rate":
                round(tot_subst / N_SHUFFLES / n, 4),
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT.name}")


if __name__ == "__main__":
    sys.exit(main())
