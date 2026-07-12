# Example tables S1–S4 — cross-corpus cluster and verse-pair examples

Tables S1–S4 accompany Section 4.2 of the article. Each verse is cited by
archival source (see the deposit's top-level `README.md` for the citation
format and the `uni_J` definition); the citation points to the single example
member shown, not the whole cluster.

**Table S1.** Form-based cross-corpus clusters. `uni_J` = the cluster's
character-unigram Jaccard similarity; the ET and FI verses shown are one
representative pair from each cluster.

| ID | ET example | FI example | uni_J |
|----|------------|------------|-------|
| 10137 | *vesi ala vesi päällä* (H II 40, 688 (28):6) | *vesi alla vesi päällä* (JR 47454:69) | 0.917 |
| 15862 | *ilmaseppa ilmarine* (H II 39, 523/44 (577):9) | *se on seppä ilmarine* (JR 29386) | 0.778 |
| 200 | *kullata suud ei kuluta* (H II 28, 450/1 (4):1) | *suutain kullata kuluta* (SKVR IV1 894:5) | 0.643 |

**Table S2.** Meaning-based cross-corpus clusters (columns as in Table S1).

| ID | ET example | FI example | uni_J |
|----|------------|------------|-------|
| 316098 | *kus on sie vanamies* (EÜS VI 938/9 (53):7) | *missä äijä* (SKVR X1 1341:8) | 0.267 |
| 92712 | *nägi õde kõndima* (E 61631/5:38) | *näki sisons kävelevän* (SKVR V1 498:38) | 0.375 |

**Table S3.** Cross-corpus verse pairs discovered by sentence embedding. Score =
sentence-embedding cosine similarity.

| ET verse | FI verse | Score | Cognates |
|----------|----------|-------|----------|
| *sündimasta kasvamasta* (E A 357 (3):20) | *syntymätä kasvamata* (JR 82768:1) | 0.881 | sündi-/synty-, kasva-/kasva- |
| *liha lihaga* (EKmS 4° 5, 84 (94):8) | *lihan liha* (SKVR XI 1511:6) | 0.930 | liha/liha |
| *neli nisa lehma all* (ERA II 9, 102 (12):3) | *nelj tissi lehmäl* (JR 55100:3) | 0.937 | neli/nelj, lehma/lehmä |
| *hakkan mina laulemaie* (EKS 31, 3 (4)) | *ruvennenko laulamahan* (SKVR I3 1294) | 0.885 | laulema/laulama |
| *ära sina nuta memmeke* (ERA I 4, 550 (2):19) | *elkää itkeä emmoin* (SKVR XV 940:58) | 0.874 | — (semantic: 'don't weep, mother') |

**Table S4.** Precision comparison of high-scoring cross-corpus verse pairs
(author-verified verdicts; Sections 4.2 and 5.2 of the article narrate the review).
The translation-pivot sample is exclusivity-filtered (absent from the sentence
embedding's top-50 neighbors); the sentence-embedding sample was drawn without
such a filter — post-hoc, 46 of its 50 pairs fall outside the translation-pivot's
top-50.

| Metric | Sentence embedding | Translation-pivot exclusive |
|--------|--------------------|-----------------------------|
| True parallels (TP) | 66% | 29% |
| Partial matches (P) | 24% | 52% |
| False positives (F) | 10% | 19% |
| Lenient precision (TP + P) | 90% | 81% |
| Pairs reviewed | 50 | 48 |
