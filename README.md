# Cross-Lingual Verse Similarity in Finnic Runosong

Code and data underlying:

> Veskis, Kaarel & Taive Särg. *Cross-Lingual Verse Similarity in Finnic
> Runosong: A Multi-Algorithm Approach Using LLM-Generated Translations.*
> Manuscript in preparation.

The paper measures verse-level similarity across the Estonian and Finnish
runosong corpora (SKVR, KR, JR, ERAB) by fusing five algorithms — Jaccard,
TF-IDF, an LLM-translation pivot, character-bigram, and sentence-embedding —
with reciprocal-rank fusion, then clustering the fused neighbours.

- [`supplementary_code/`](supplementary_code/) — the build and analysis
  pipeline (the five similarity algorithms, RRF clustering, the leave-one-out
  ablation, the precision and gold-standard evaluations). Released under the
  **MIT** licence; see `supplementary_code/README.md`.
- [`supplementary_data/`](supplementary_data/) — the derived data (cluster
  memberships, cross-corpus verse pairs, form/meaning classification,
  ablation statistics, reclassification lists), referenced by verse ID.
  Released under **CC BY 4.0**; see `supplementary_data/README.md`.

The underlying corpus verse text belongs to the source archives (SKVR, KR, JR,
ERAB) and is **not** redistributed here; the data files reference verses by ID,
which can be browsed at <https://runoverse.org/>.
