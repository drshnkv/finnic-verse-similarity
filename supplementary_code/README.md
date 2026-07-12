# Reproducibility code

Source code for the cross-lingual Estonian–Finnish runosong verse-similarity
pipeline in:

> Veskis, Kaarel & Taive Särg. *Cross-Lingual Verse Similarity in Finnic
> Runosong: A Multi-Algorithm Approach Using LLM-Generated Translations.*
> Manuscript in preparation.

The scripts that produce the paper's results: the five verse-level similarity
algorithms (with the shared lemma resolver), their reciprocal-rank-fusion (RRF)
clustering, the cross-lingual cluster derivation, the two-member-floor
re-derivation, the leave-one-out ablation, the precision evaluation, and the
FILTER gold-standard comparison. Companion data is in `../supplementary_data/`.
Corpora: SKVR, KR, JR, ERAB.

## Scripts

`similarity/` — the pipeline:

- `verse_similarity_common.py` — shared helpers (normalisation, IO, sparse cosine)
- `runoverse_lemma_resolver.py` — lemma resolver used by the TF-IDF + translation legs
- `build_verse_jaccard.py`, `build_verse_tfidf.py`, `build_verse_charbigram.py`,
  `build_verse_translation.py`, `build_verse_sentence_embedding.py` — the five
  per-verse similarity algorithms
- `cluster_verses_rrf.py` — RRF fusion + verse clustering (threshold 0.033, two-member floor)
- `rrf_validation_stats.py` — re-runs the clustering to export the full cluster
  membership; `--out-tag` produces the leave-one-out and 3-algorithm variants
- `build_ws4_cross_lingual_5algo.py` — cross-lingual (ET–FI) cluster derivation
- `evaluate_sentence_similarity.py` — sentence-embedding precision evaluation
- `evaluate_gold_standard_proper.py` — FILTER co-membership P/R/F1 vs the SKVR gold standard (Table 2)
- `build_floor2_novel_sample.py` — samples novel clusters at the two-member floor
- `output/floor2_rederive/derive_ws4_floor2.py` — floor-2 cross-lingual aggregates + form/meaning classification
- `output/floor2_rederive/phase_c_table7.py` — floor-2 leave-one-out (Table 3)

`scripts/` — helpers that built the companion data deposit (Python stdlib only):

- `build_verse_pairs_csv.py` — data item 06 (flat ET–FI verse-pairs CSV)
- `build_cl_cluster_membership_export.py` — data item 05 (cross-corpus membership, IDs only)
- `correct_item5_deposit_counts.py` — count fix for data item 04

`valismaa_override.py` — language-reclassification overrides, imported across the
pipeline; it reads `data/valismaa_finnish_poems.json` (the 68-poem list,
bundled so the override leg runs without manual placement).

## Running

Python 3.10; `pip install -r requirements.txt` (EstNLTK and scikit-learn are not
required — see the note in `requirements.txt`). Each script has a module docstring
and, where relevant, `--help`. Broad order: the five `build_verse_*.py` →
`cluster_verses_rrf.py` (plus `rrf_validation_stats.py` to export membership) →
`build_ws4_cross_lingual_5algo.py` → the `floor2_rederive/` scripts → the
`evaluate_*` scripts. The RRF threshold (0.033) and the two-member cluster floor
are the script defaults throughout.

To regenerate the ablation memberships, after the RRF cache exists
(`cluster_verses_rrf.py --rebuild-cache`), from `similarity/`:

```bash
python -u rrf_validation_stats.py                                   # canonical full membership
python -u rrf_validation_stats.py --min-algos 3 --out-tag minalgos3 # 3-algorithm consensus
# five leave-one-out runs (drop one algorithm each) — Table 3:
python -u rrf_validation_stats.py --algos tfidf,translation,charbigram,sentence  --out-tag loo_minus_jaccard
python -u rrf_validation_stats.py --algos jaccard,translation,charbigram,sentence --out-tag loo_minus_tfidf
python -u rrf_validation_stats.py --algos jaccard,tfidf,charbigram,sentence       --out-tag loo_minus_translation
python -u rrf_validation_stats.py --algos jaccard,tfidf,translation,sentence      --out-tag loo_minus_charbigram
python -u rrf_validation_stats.py --algos jaccard,tfidf,translation,charbigram    --out-tag loo_minus_sentence
gzip -k -f output/rrf_cluster_membership*.jsonl
python -u evaluate_gold_standard_proper.py                          # Table 2 vs the gold standard
```

**External inputs not bundled.** The scripts read large pre-built indices derived
from the four corpora (the verse registry, the five per-algorithm neighbour
indices, `verse_metadata.jsonl.gz`, `gloss_index.json`) and, for the FILTER
comparison, the SKVR Verse Equivalence Gold Standard (Janicki, Kallio & Sarv 2023;
CC BY-NC-ND). These are multi-GB and/or third-party, so they are not redistributed
here; obtain them from the source archives. Every script resolves its inputs
relative to its own location — the deposit carries no absolute machine paths, and
every file compiles cleanly (`py_compile`).

## Licence

MIT (see `LICENSE`). The article itself is CC BY-NC-SA, but that non-commercial
term is a poor fit for code, so the code is released under MIT to maximise reuse.
