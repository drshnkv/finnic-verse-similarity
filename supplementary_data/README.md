# Supplementary data — *Cross-Lingual Verse Similarity in Finnic Runosong* (Veskis & Särg, JEFUL)

Derived data for the cross-corpus (Estonian–Finnish) runosong verse-similarity
analysis. All values are at the **two-member clustering floor**
(`min_cluster_size = 2`) used throughout the paper. Corpora: SKVR, KR, JR, ERAB.
No running verse text is included beyond the short examples already quoted in the
article — verses are referenced by ID and can be browsed at
<https://runoverse.org/>. *ET*/*FI* mark the two source-corpus aggregates (ERAB
vs. SKVR+JR+KR), not a per-text language label (article §2.1).

## Contents

| Folder | What it is |
|--------|------------|
| `02_ablation/` | Leave-one-out statistics for the five algorithms (`phase_c_table7.json`) + the delta decomposition (`loo_delta_decomposition.json`, with the `.py` that produced it) |
| `03_reclassified_poems/` | 68 ERAB poems moved ET → Finnish/Ingrian (`valismaa_finnish_poems.json`) + 13 JR poems FI → Estonian (`jr_to_estonian_13.json`; 12 Seto, 1 Tori) |
| `04_form_meaning_classification/` | Form/meaning class + Jaccard scores for all 19,423 clusters (`rrf_ws4_cross_lingual_5algo.json`) |
| `05_cluster_membership/` | Complete membership of the 19,423 cross-corpus clusters, by verse ID |
| `06_verse_pairs/` | Flat ET–FI verse-pairs CSV — 120,590 cross-corpus pairs (105,784 distinct) |
| `07_example_tables/` | Example Tables S1–S4 (cluster + verse-pair examples with archival citations) |
| `08_songtype_theme_sharing/` | Song-type theme-sharing + theme-enrichment analysis (JSON) |

## Column codebook

**`05_cluster_membership/`** — three views of the same clusters, no verse text:

- `cl_cluster_membership.jsonl` (canonical, one cluster per line): `cluster_id`,
  `size`, `et_count`, `fi_count`, `unknown_count` (0 throughout), `category`
  (`form_based` / `meaning_based`), `char_unigram_jaccard`, `word_jaccard`,
  `kr_estonian_suspect`, `et_example_poem`, `fi_example_poem`, `members[]`
  (each `{vid, lang, place, poem}`).
- `cl_cluster_members.csv` (one row per cluster-member): `cluster_id`, `category`,
  `member_vid`, `member_lang` `{et, fi, unknown}`, `member_place` (`;`-joined),
  `member_poem_id`.
- `cl_clusters_index.csv` (one row per cluster): the per-cluster aggregates above.

**`06_verse_pairs/cl_verse_pairs.csv`** — one ET–FI co-membership pair per row:
`cluster_id`, `size`, `category`, `et_vid`, `et_place`, `fi_vid`, `fi_place`.

**`04_form_meaning_classification/rrf_ws4_cross_lingual_5algo.json`** — per cluster:
`cluster_id`, `size`, et/fi/pair counts, `char_{uni,bi,tri}gram_jaccard`,
`word_jaccard`, `category`, `shared_translations` (empty in this floor-2 export;
the label uses `char_unigram_jaccard` alone — form-based above 0.40, meaning-based
at or below).

**`07_example_tables/`** — Tables S1–S4; `uni_J` = the cluster's character-unigram
Jaccard (§4.3). Verses are cited by archival source (an ERAB reference for
Estonian; an SKVR volume-part-poem number or a JR/KR id for Finnish); the number
after the colon is the verse-line index.

## Usage notes

- **Soft / overlapping membership — `member_vid` is not a key.** A verse can belong
  to several clusters. There are 110,458 (cluster, member) rows over **84,265
  distinct verses** (47,065 ET + 37,200 FI). Summing `size` over-counts distinct
  verses by ≈ 31 %; the headline **120,590** counts each ET–FI verse pair once for
  every cluster it appears in, so a pair shared by more than one cluster is
  counted more than once; the number of **distinct** ET–FI pairs is **105,784**.
  Deduplicate on `(et_vid, fi_vid)` for the distinct count; count rows as-is to
  reproduce 120,590.
- **Verse-ID format** is `"<poem_id>:<verse_index>"`, and a poem ID can itself
  contain `:` `,` spaces `.` — use the explicit `member_poem_id` / `members[].poem`,
  or split on the **last** colon only.

## Rights and citation

The derived layer here (cluster assignments, classification, statistics, verdicts,
reclassification lists, counts) is released under **CC BY 4.0**. The underlying
corpus verse text (SKVR, KR, JR, ERAB) belongs to the source archives and is not
redistributed. Cite the JEFUL article as the primary reference.
