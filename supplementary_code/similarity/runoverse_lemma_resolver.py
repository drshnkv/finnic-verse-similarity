"""
Strategy F' lemma resolver for the RunoVerse runosong corpus.

("Strategy F'" = this corpus-pipeline-counts + DeepSeek-LLM-counts +
phonological-gate lemma resolver; "DeepSeek" = the LLM whose per-wordform
lemma counts feed the resolver.)

Resolves the best lemma for ambiguous Estonian/Finnish wordforms (those that
appear under multiple lexicon entries) by combining two independent signals:
corpus-pipeline lemma counts (high volume, systematic errors) and DeepSeek
LLM lemma counts (language-aware, lower coverage). Unambiguous wordforms return
their single entry's lemma; unknown wordforms return the wordform itself.

Used by the TF-IDF and translation-pivot verse builders to map wordforms to
lemmas before similarity scoring.
"""

import json
import gzip
import unicodedata
from pathlib import Path
from typing import Dict, List


class RunoVerseLemmaResolver:
    """
    Strategy F' lemma resolver (corpus counts + DeepSeek LLM counts).

    The algorithm resolves ambiguous wordforms (those appearing under multiple
    lexicon entries) by combining two independent signals:
    - Corpus pipeline counts: high volume but with systematic errors
    - DeepSeek AI counts: language-aware but lower coverage

    For unambiguous wordforms (1 lexicon entry), returns that entry's lemma directly.
    For unknown wordforms (not in lexicon), returns the wordform itself.
    """

    DS_MIN_THRESHOLD = 1
    DS_SPLIT_THRESHOLD = 2

    def __init__(self, lexicon_data_path: Path, distribution_path: Path):
        """
        Load lexicon data and build lookup indices.

        Args:
            lexicon_data_path: Path to lexicon_data.json.gz (305K lemma entries)
            distribution_path: Path to wordform_lemma_distribution.json (400K ambiguous wordforms)
        """
        print(f"  Loading RunoVerse lexicon from {lexicon_data_path.name}...")
        with gzip.open(lexicon_data_path, 'rt', encoding='utf-8') as f:
            entries = json.load(f)

        # Build wf_to_entries: {wordform_lower: [{l, c, e, s, j}, ...]}
        # Only keep the 5 fields needed for the F' algorithm
        self.wf_to_entries: Dict[str, List[dict]] = {}
        n_mappings = 0
        for entry in entries:
            slim = {
                'l': entry.get('l', ''),
                'c': entry.get('c', 0),
                'e': entry.get('e', 0),
                's': entry.get('s', 0),
                'j': entry.get('j', 0),
            }
            for wf in entry.get('w', []):
                wf_lower = wf.lower()
                if wf_lower not in self.wf_to_entries:
                    self.wf_to_entries[wf_lower] = []
                self.wf_to_entries[wf_lower].append(slim)
                n_mappings += 1

        print(f"    {len(entries):,} lemma entries -> {len(self.wf_to_entries):,} unique wordforms "
              f"({n_mappings:,} total mappings)")

        print(f"  Loading distribution from {distribution_path.name}...")
        with open(distribution_path, 'r', encoding='utf-8') as f:
            self.wf_lemma_dist: Dict[str, list] = json.load(f)
        print(f"    {len(self.wf_lemma_dist):,} ambiguous wordforms loaded")

        self._cache: Dict[str, str] = {}

    def resolve(self, wordform: str) -> str:
        """Return the best lemma for a wordform using Strategy F'.

        Results are memoized — repeat lookups are O(1) dict reads.

        Returns:
            The resolved lemma string. For unknown wordforms, returns the
            wordform itself (lowercased).
        """
        wf = wordform.lower()
        if wf in self._cache:
            return self._cache[wf]

        entries = self.wf_to_entries.get(wf)
        if not entries:
            # Unknown wordform — use as-is
            self._cache[wf] = wf
            return wf
        if len(entries) == 1:
            # Unambiguous — single lexicon entry
            result = entries[0]['l']
            self._cache[wf] = result
            return result

        # Ambiguous — run full F' cascade
        result = self._pick_best(wf, entries)
        self._cache[wf] = result
        return result

    def _pick_best(self, wf: str, entries: List[dict]) -> str:
        """Full Strategy F' cascade for ambiguous wordforms.

        Args:
            wf: Lowercased wordform
            entries: List of lexicon entries containing this wordform

        Returns:
            The best lemma string
        """
        # Step 2: Load distribution data
        dists = self.wf_lemma_dist.get(wf)
        if not dists:
            # No distribution data — fall back to highest entry.c
            return max(entries, key=lambda e: e['c'])['l']

        # Step 3: Build corpus/DS score maps (NFC-normalize lemma keys)
        corpus_scores: Dict[str, int] = {}
        ds_scores: Dict[str, int] = {}
        for triple in dists:
            key = unicodedata.normalize('NFC', triple[0]).lower()
            corpus_scores[key] = corpus_scores.get(key, 0) + triple[1]
            ds_scores[key] = ds_scores.get(key, 0) + (triple[2] if len(triple) >= 3 else 0)

        # Step 4: Find best+second for each track (entries only)
        best_corpus = best_ds = None
        best_cs = second_cs = -1
        best_dss = second_dss = -1

        for entry in entries:
            lk = entry['l'].lower()
            cs = corpus_scores.get(lk, 0)
            ds = ds_scores.get(lk, 0)

            if cs > best_cs:
                second_cs = best_cs
                best_cs = cs
                best_corpus = entry
            elif cs > second_cs:
                second_cs = cs

            if ds > best_dss:
                second_dss = best_dss
                best_dss = ds
                best_ds = entry
            elif ds > second_dss:
                second_dss = ds

        if not best_corpus:
            return entries[0]['l']

        # Step 5: Agreement — corpus and DS pick the same lemma
        if best_corpus and best_ds and best_corpus['l'].lower() == best_ds['l'].lower():
            return best_corpus['l']

        # Step 6: No DS signal — corpus wins
        if not best_ds or best_dss == 0:
            return best_corpus['l']

        # Step 7: True-max DS guard — check if the DS champion is even in the entries
        true_max_ds = max(ds_scores.values()) if ds_scores else 0
        if best_dss < true_max_ds:
            return best_corpus['l']

        # Step 8: DS tie — top two DS scores equal, not informative
        if best_dss == second_dss:
            return best_corpus['l']

        # Helper: unanimity and language detection
        corpus_lemma_count = sum(1 for v in corpus_scores.values() if v > 0)
        corpus_is_unanimous = corpus_lemma_count <= 1
        ds_lemma_count = sum(1 for v in ds_scores.values() if v > 0)
        ds_is_unambiguous = ds_lemma_count == 1

        # Step 9: Cross-language exemption
        is_cross_lang = False
        if best_corpus and best_ds:
            c_e = best_corpus.get('e', 0)
            c_fi = best_corpus.get('s', 0) + best_corpus.get('j', 0)
            d_e = best_ds.get('e', 0)
            d_fi = best_ds.get('s', 0) + best_ds.get('j', 0)
            c_lang = 'et' if c_e > c_fi else ('fi' if c_fi > c_e else None)
            d_lang = 'et' if d_e > d_fi else ('fi' if d_fi > d_e else None)
            is_cross_lang = bool(c_lang and d_lang and c_lang != d_lang)

        if is_cross_lang and best_dss >= self.DS_MIN_THRESHOLD:
            if ds_is_unambiguous or best_dss >= self.DS_SPLIT_THRESHOLD:
                return best_ds['l']

        # Step 10: Unanimity gate + phonological gate
        if corpus_is_unanimous and best_dss >= self.DS_MIN_THRESHOLD:
            if ds_is_unambiguous or best_dss >= self.DS_SPLIT_THRESHOLD:
                ds_sim = self.levenshtein_similarity(wf, best_ds['l'])
                corpus_sim = self.levenshtein_similarity(wf, best_corpus['l'])
                if ds_sim >= corpus_sim:
                    return best_ds['l']

        # Step 11: Corpus tie — DS breaks it
        if best_cs == second_cs:
            if best_ds and best_dss > 0 and best_dss > second_dss:
                return best_ds['l']
            return max(entries, key=lambda e: e['c'])['l']

        # Step 12: Default — corpus winner
        return best_corpus['l']

    @staticmethod
    def levenshtein_similarity(a: str, b: str) -> float:
        """Normalized Levenshtein similarity: 1 - (editDist / max(len(a), len(b))).

        Returns 0.0 for empty strings, 1.0 for identical strings.
        """
        if not a or not b:
            return 0.0
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i]
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr.append(min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost  # substitution
                ))
            prev = curr
        max_len = max(m, n)
        return 1 - prev[n] / max_len if max_len > 0 else 0.0
