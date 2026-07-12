#!/usr/bin/env python
"""What is inside the 'death' theme? Read-only diagnostic (2026-07-03).

Source for the JEFUL §4.6 gloss "death-themed songs (the latter largely
ballads of killing)": lists the song types classified as 'death', per side,
with corpus poem counts and CL-cluster incidence counts (Phase E convention,
same loaders as songtype_theme_enrichment_sidemix.py), then sums the share
of death-theme CL incidences contributed by the killing/death ballads.

Theme provenance (build_song_type_index.py): themes come from bilingual
keyword rules over type names, first match wins, theme order incantations >
lullabies > epic > children > animal > calendar > wedding > work > lyric >
death; 16 of 170 death types instead inherit the theme from verse-sharing
neighbour types (themeSrc='propagated'). Epic is a closed list of
Kalevala-canon names checked before death, so hero-death types (e.g.
Lemminkäisen surma) go to epic; what lands in 'death' is death-themed
narrative outside that canon (ballads) plus death-lyric and grave songs.

Sensitivity (checked 2026-07-03): dropping the three questionable members
(Põdratapja - elk, not human death; propagated Ema õpetus, Ilma parandamine)
RAISES ET death enrichment 1.781 -> 1.819, so the §4.6 claim is robust.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

RD = Path(__file__).resolve().parent
SIM = RD.parents[1]  # similarity/
sys.path.insert(0, str(SIM))

import build_cl_cluster_geography_songtype as geo

geo.CL_CLUSTERS_FILE = RD / "rrf_ws4_cross_lingual_5algo.json"
geo.MEMBERSHIP_FILE = SIM / "output" / "rrf_cluster_membership_floor2_experimental.jsonl.gz"

DEATH_KW = ["kuolema", "tuonela", "manala", "hauta", "kalma",
            "surma", "kuolo", "haudan", "surm", "hauad", "tapja"]

# Ballads of (human) killing among the death-theme types: the Estonian
# daughter/wife/husband-killer ballads (Maie laul = Mehetapja variant,
# theme propagated) and the Finnish killing/legend ballads.
KILLING_BALLADS = {
    "ET": ["Tütarde tapja", "Tütrete tapja", "Tütre tapja", "Tütarde tapmine",
           "Tütred vette", "Naisetapja", "Mehetapja", "Maie laul", "Maielaul"],
    "FI": ["Laivassa surmattu veli", "Elinan surma",
           "Piispa Henrikin surmavirsi", "Velisurmaaja"],
}

theme_lookup = geo.load_theme_lookup()
type_map = geo.load_song_type_cache()

corpus_ct = defaultdict(Counter)   # type -> side -> n poems
for pid, stype in type_map.items():
    corpus_ct[stype][geo.detect_language(pid)] += 1

cl_ids = geo.load_cl_cluster_ids()
membership = geo.load_cluster_membership(cl_ids)
cl_ct = defaultdict(Counter)       # type -> side -> CL incidences
for cid, members in membership.items():
    for pid in set(geo.verse_id_to_pid(v) for v in members):
        st = type_map.get(pid)
        if st:
            cl_ct[st][geo.detect_language(pid)] += 1

for side in ("ET", "FI"):
    rows = []
    for st, th in theme_lookup.items():
        if th != "death":
            continue
        c = corpus_ct.get(st, Counter())[side]
        k = cl_ct.get(st, Counter())[side]
        if c or k:
            rows.append((k, c, st))
    rows.sort(reverse=True)
    tot_c = sum(r[1] for r in rows)
    tot_k = sum(r[0] for r in rows)
    print(f"\n=== DEATH types, {side} side: {len(rows)} types, "
          f"{tot_c:,} corpus poems, {tot_k:,} CL incidences ===")
    print(f"{'CL':>6} {'corpus':>7}  type name")
    for k, c, st in rows[:25]:
        print(f"{k:>6} {c:>7}  {st}")
    ballad_k = sum(cl_ct.get(st, Counter())[side] for st in KILLING_BALLADS[side])
    print(f"--> killing ballads {KILLING_BALLADS[side]}: "
          f"{ballad_k:,} of {tot_k:,} CL incidences = {ballad_k/tot_k:.1%}")

print("\n=== Types CONTAINING a death keyword but themed elsewhere (top by CL) ===")
rows = []
for st, th in theme_lookup.items():
    low = st.lower()
    if th != "death" and any(kw in low for kw in DEATH_KW):
        c = sum(corpus_ct.get(st, Counter()).values())
        k = sum(cl_ct.get(st, Counter()).values())
        if c or k:
            rows.append((k, c, th, st))
rows.sort(reverse=True)
print(f"{'CL':>6} {'corpus':>7} {'theme':<13} type name")
for k, c, th, st in rows[:15]:
    print(f"{k:>6} {c:>7} {th:<13} {st}")
print(f"... {len(rows)} such types total; "
      f"{sum(r[1] for r in rows):,} corpus poems, {sum(r[0] for r in rows):,} CL incidences")
