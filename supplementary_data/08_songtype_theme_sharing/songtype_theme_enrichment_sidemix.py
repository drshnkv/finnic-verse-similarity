#!/usr/bin/env python
"""Side-mix-corrected theme enrichment for the floor2 CL clusters (2026-07-03).

Question (JEFUL §4.6): the Phase E theme-enrichment table compares the theme
mix of typed poems inside cross-lingual (CL) clusters against the POOLED
corpus-wide theme distribution. That baseline is confounded twice:

  1. The ET and FI typed pools have very different genre make-ups
     (e.g. incantations ~22% of typed FI poems vs ~2% of typed ET poems;
     lyric ~26% of ET vs ~5% of FI).
  2. The ET:FI mix of typed poem-incidences INSIDE CL clusters differs from
     the corpus-wide mix, so pooled enrichment partly measures which side
     contributes more, not which themes are selected.

Correction: stratify by side. Expected count for a theme =
    share_ET(theme) x N_ET_in_CL  +  share_FI(theme) x N_FI_in_CL
i.e. each side's own corpus-wide theme distribution, weighted by how many
typed poem-incidences that side actually contributes to CL clusters. Also
reports fully per-side enrichment tables and a distinct-poem robustness
variant (each typed poem counted once if it appears in ANY CL cluster,
instead of once per CL cluster it appears in — Phase E uses the latter).

Method notes:
  - Reuses build_cl_cluster_geography_songtype.py UNCHANGED (imported, two
    I/O constants repointed to floor2, same as phase_e_geo_songtype.py):
    same membership, same detect_language(), same theme lookup.
  - First REPRODUCES the Phase E pooled table exactly (counts, enrichment
    ratios, chi2 = 4135.2) and asserts on it before computing anything new.
  - Also reports a VERSE-WEIGHTED baseline (P-13): each corpus poem is
    weighted by its verse count (verse_metadata.jsonl.gz), because clusters
    are built from verses and a longer poem has more chances to place one in
    a cluster; observed counts are unchanged. This length-aware baseline is
    the one the article's §4.6 reports (pooled chi2 = 1267, vs 2678 side-mix
    without length weighting).
  - Chi-squared uses the same >=5 expected/observed filter as Phase E;
    df = (#themes kept) - 1. Expected totals equal observed totals by
    construction in every variant.
  - Known classifier quirk, kept for consistency with Phase E:
    detect_language() sides 'Kanteletar ...' pids (KR corpus) as ET; the
    script reports the size of that group (~0.3% of typed incidences).

Inputs (floor2): rrf_ws4_cross_lingual_5algo.json (this folder),
rrf_cluster_membership_floor2_experimental.jsonl.gz (similarity/output/),
_pid_to_song_type_cache.json + song_type_theme_lookup.json (deployment).
Output: songtype_theme_enrichment_sidemix.json + printed tables. Read-only.
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

RD = Path(__file__).resolve().parent
SIM = RD.parents[1]  # similarity/
sys.path.insert(0, str(SIM))

import build_cl_cluster_geography_songtype as geo

geo.CL_CLUSTERS_FILE = RD / "rrf_ws4_cross_lingual_5algo.json"
geo.MEMBERSHIP_FILE = SIM / "output" / "rrf_cluster_membership_floor2_experimental.jsonl.gz"

ANALYSIS_JSON = RD / "cl_cluster_song_type_analysis.json"  # Phase E output (assert target)
OUT = RD / "songtype_theme_enrichment_sidemix.json"

MIN_COUNT = 5  # same filter as Phase E chi-squared


def load_verse_counts():
    """pid -> corpus verse count, from the full per-verse index that the
    clusters are built from (similarity/output/verse_metadata.jsonl.gz). This is
    each poem's *opportunity* to contribute a verse to a verse-built CL cluster:
    a long epic has many verses and so many chances to be pulled in, a short
    lullaby few. Used for the verse-weighted baseline (P-13)."""
    counts = Counter()
    with gzip.open(geo.VERSE_META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line).get("p", "")] += 1
    return counts


def chi2_table(obs_by_theme, exp_by_theme):
    """Chi-squared over themes with the Phase E >=5 filter. Returns (chi2, df, kept)."""
    chi2 = 0.0
    kept = []
    for theme in sorted(obs_by_theme.keys() | exp_by_theme.keys()):
        o = obs_by_theme.get(theme, 0)
        e = exp_by_theme.get(theme, 0.0)
        if o >= MIN_COUNT and e >= MIN_COUNT:
            chi2 += (o - e) ** 2 / e
            kept.append(theme)
    return chi2, len(kept) - 1, kept


def enrich_rows(obs, exp, corpus_share, n_obs_total):
    rows = []
    for theme in sorted(obs.keys() | exp.keys()):
        o = obs.get(theme, 0)
        e = exp.get(theme, 0.0)
        rows.append({
            "theme": theme,
            "obs": o,
            "expected": round(e, 1),
            "obs_rate": round(o / n_obs_total, 4) if n_obs_total else 0,
            "baseline_rate": round(corpus_share.get(theme, 0.0), 4),
            "enrichment": round(o / e, 3) if e > 0 else None,
        })
    rows.sort(key=lambda r: -(r["enrichment"] or 0))
    return rows


def main():
    theme_lookup = geo.load_theme_lookup()
    type_map = geo.load_song_type_cache()

    # --- corpus-wide per-side theme distributions (each typed poem once) ---
    side_of = {}
    corpus_theme = {"ET": Counter(), "FI": Counter()}
    kanteletar_et = 0
    for pid, stype in type_map.items():
        side = geo.detect_language(pid)
        side_of[pid] = side
        theme = theme_lookup.get(stype, "unknown")
        corpus_theme[side][theme] += 1
        if pid.startswith("Kanteletar") and side == "ET":
            kanteletar_et += 1
    n_corpus = {s: sum(c.values()) for s, c in corpus_theme.items()}
    corpus_share = {s: {t: c / n_corpus[s] for t, c in corpus_theme[s].items()}
                    for s in ("ET", "FI")}
    pooled_total = sum(n_corpus.values())
    pooled_share = {}
    for s in ("ET", "FI"):
        for t, c in corpus_theme[s].items():
            pooled_share[t] = pooled_share.get(t, 0) + c / pooled_total

    # --- recount typed poem-incidences inside CL clusters (Phase E convention:
    #     per (cluster, distinct pid)) + distinct-poem variant ---
    cl_ids = geo.load_cl_cluster_ids()
    membership = geo.load_cluster_membership(cl_ids)

    obs_inc = {"ET": Counter(), "FI": Counter()}
    distinct_pids = {"ET": set(), "FI": set()}
    for cid, members in membership.items():
        pids = set(geo.verse_id_to_pid(vid) for vid in members)
        for pid in pids:
            stype = type_map.get(pid)
            if not stype:
                continue
            side = side_of[pid]
            theme = theme_lookup.get(stype, "unknown")
            obs_inc[side][theme] += 1
            distinct_pids[side].add(pid)

    n_inc = {s: sum(c.values()) for s, c in obs_inc.items()}
    obs_pooled = obs_inc["ET"] + obs_inc["FI"]
    n_inc_total = sum(n_inc.values())

    # --- 1. reproduce the Phase E pooled table, assert before anything new ---
    phase_e = json.loads(ANALYSIS_JSON.read_text())["summary"]["theme_enrichment"]
    exp_pooled = {t: pooled_share.get(t, 0.0) * n_inc_total for t in obs_pooled}
    for row in phase_e:
        t = row["theme"]
        assert obs_pooled.get(t, 0) == row["cl_count"], (t, obs_pooled.get(t), row["cl_count"])
        mine = round(obs_pooled[t] / exp_pooled[t], 3)
        assert abs(mine - row["enrichment"]) <= 0.001, (t, mine, row["enrichment"])
    chi2_p, df_p, _ = chi2_table(obs_pooled, exp_pooled)
    assert abs(chi2_p - 4135.2) < 1.0, chi2_p
    print(f"Reproduced Phase E pooled table: {len(phase_e)} themes, "
          f"chi2={chi2_p:.1f} (df={df_p})\n")

    # --- 2. side-mix-corrected pooled expectation ---
    exp_smc = {}
    for t in obs_pooled.keys() | pooled_share.keys():
        exp_smc[t] = (corpus_share["ET"].get(t, 0.0) * n_inc["ET"]
                      + corpus_share["FI"].get(t, 0.0) * n_inc["FI"])
    smc_share = {t: e / n_inc_total for t, e in exp_smc.items()}
    chi2_c, df_c, _ = chi2_table(obs_pooled, exp_smc)

    # --- 2b. VERSE-WEIGHTED baseline (P-13): the side-mix baseline above counts
    #     each typed poem once, but the clusters are built from verses, so a
    #     longer poem has more chances for one of its verses to enter a CL
    #     cluster. Weight each corpus poem by its verse count, so the baseline
    #     absorbs the mechanical length effect; if epic/death stay enriched and
    #     lullabies depleted, the theme selectivity is real, not just length.
    #     Observed stays the poem-incidence count (§4.6 convention); only the
    #     baseline changes. Poems absent from the verse index get weight 0 —
    #     with no verses in the universe they cannot enter a cluster either. ---
    verse_counts = load_verse_counts()
    corpus_theme_vw = {"ET": Counter(), "FI": Counter()}
    vw_missing = {"ET": 0, "FI": 0}
    for pid, stype in type_map.items():
        side = side_of[pid]
        theme = theme_lookup.get(stype, "unknown")
        vc = verse_counts.get(pid, 0)
        if vc == 0:
            vw_missing[side] += 1
        corpus_theme_vw[side][theme] += vc
    n_corpus_vw = {s: sum(c.values()) for s, c in corpus_theme_vw.items()}
    corpus_share_vw = {s: {t: c / n_corpus_vw[s] for t, c in corpus_theme_vw[s].items()}
                       for s in ("ET", "FI")}
    exp_vw = {}
    for t in (obs_pooled.keys() | corpus_share_vw["ET"].keys()
              | corpus_share_vw["FI"].keys()):
        exp_vw[t] = (corpus_share_vw["ET"].get(t, 0.0) * n_inc["ET"]
                     + corpus_share_vw["FI"].get(t, 0.0) * n_inc["FI"])
    vw_share = {t: e / n_inc_total for t, e in exp_vw.items()}
    chi2_vw, df_vw, _ = chi2_table(obs_pooled, exp_vw)

    per_side_vw = {}
    for s in ("ET", "FI"):
        exp_s_vw = {t: corpus_share_vw[s].get(t, 0.0) * n_inc[s]
                    for t in obs_inc[s].keys() | corpus_share_vw[s].keys()}
        chi2_s_vw, df_s_vw, _ = chi2_table(obs_inc[s], exp_s_vw)
        per_side_vw[s] = {
            "n_incidences": n_inc[s],
            "chi2": round(chi2_s_vw, 1), "df": df_s_vw,
            "rows": enrich_rows(obs_inc[s], exp_s_vw, corpus_share_vw[s], n_inc[s]),
        }

    # --- 3. per-side enrichment ---
    per_side = {}
    for s in ("ET", "FI"):
        exp_s = {t: corpus_share[s].get(t, 0.0) * n_inc[s]
                 for t in obs_inc[s].keys() | corpus_share[s].keys()}
        chi2_s, df_s, _ = chi2_table(obs_inc[s], exp_s)
        per_side[s] = {
            "n_incidences": n_inc[s],
            "chi2": round(chi2_s, 1), "df": df_s,
            "rows": enrich_rows(obs_inc[s], exp_s, corpus_share[s], n_inc[s]),
        }

    # --- 4. distinct-poem robustness variant (poem counted once) ---
    obs_dist = {s: Counter(theme_lookup.get(type_map[p], "unknown")
                           for p in distinct_pids[s]) for s in ("ET", "FI")}
    n_dist = {s: sum(c.values()) for s, c in obs_dist.items()}
    obs_dist_pooled = obs_dist["ET"] + obs_dist["FI"]
    exp_dist_smc = {}
    for t in obs_dist_pooled.keys() | pooled_share.keys():
        exp_dist_smc[t] = (corpus_share["ET"].get(t, 0.0) * n_dist["ET"]
                           + corpus_share["FI"].get(t, 0.0) * n_dist["FI"])
    chi2_d, df_d, _ = chi2_table(obs_dist_pooled, exp_dist_smc)
    smc_dist_share = {t: e / sum(n_dist.values()) for t, e in exp_dist_smc.items()}

    result = {
        "generated": "2026-07-03",
        "inputs": {
            "cl_clusters": len(cl_ids),
            "typed_poems_corpus": {**n_corpus, "pooled": pooled_total},
            "typed_incidences_in_cl": {**n_inc, "pooled": n_inc_total},
            "distinct_typed_poems_in_cl": {**n_dist,
                                           "pooled": sum(n_dist.values())},
            "kanteletar_pids_sided_ET": kanteletar_et,
        },
        "corpus_theme_share_by_side": {
            s: {t: round(v, 4) for t, v in sorted(corpus_share[s].items())}
            for s in ("ET", "FI")},
        "phase_e_pooled_reproduced": {"chi2": round(chi2_p, 1), "df": df_p},
        "sidemix_corrected_pooled": {
            "chi2": round(chi2_c, 1), "df": df_c,
            "rows": enrich_rows(obs_pooled, exp_smc, smc_share, n_inc_total),
        },
        "verse_weighted_baseline_pooled": {
            "note": ("baseline weights each corpus poem by its verse count "
                     "(opportunity to enter a verse-built cluster); observed "
                     "unchanged. P-13."),
            "typed_verses_corpus": {**n_corpus_vw,
                                    "pooled": sum(n_corpus_vw.values())},
            "typed_poems_missing_from_verse_index": vw_missing,
            "chi2": round(chi2_vw, 1), "df": df_vw,
            "rows": enrich_rows(obs_pooled, exp_vw, vw_share, n_inc_total),
        },
        "verse_weighted_baseline_per_side": per_side_vw,
        "per_side": per_side,
        "distinct_poem_variant_sidemix_corrected": {
            "chi2": round(chi2_d, 1), "df": df_d,
            "rows": enrich_rows(obs_dist_pooled, exp_dist_smc,
                                smc_dist_share, sum(n_dist.values())),
        },
    }
    OUT.write_text(json.dumps(result, indent=2))

    def show(title, block):
        print(f"\n=== {title} (chi2={block['chi2']}, df={block['df']}) ===")
        print(f"{'theme':14} {'obs':>7} {'exp':>9} {'enrich':>7}")
        for r in block["rows"]:
            print(f"{r['theme']:14} {r['obs']:>7,} {r['expected']:>9,.1f} "
                  f"{r['enrichment']:>7}")

    show("Side-mix-corrected pooled", result["sidemix_corrected_pooled"])
    show("VERSE-WEIGHTED baseline, pooled (P-13)",
         result["verse_weighted_baseline_pooled"])
    show("VERSE-WEIGHTED baseline, ET side", per_side_vw["ET"])
    show("VERSE-WEIGHTED baseline, FI side", per_side_vw["FI"])
    show("ET side only", per_side["ET"])
    show("FI side only", per_side["FI"])
    show("Distinct-poem variant (side-mix-corrected)",
         result["distinct_poem_variant_sidemix_corrected"])
    print(f"\nWrote {OUT.name}")


if __name__ == "__main__":
    sys.exit(main())
