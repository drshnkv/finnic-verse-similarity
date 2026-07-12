#!/usr/bin/env python3
"""
build_cl_cluster_membership_export.py

Build the cross-corpus (cross-lingual) cluster-membership deposit
(IDs only, NO verse text):

    supplementary_data/05_cluster_membership/
        cl_cluster_membership.jsonl   (canonical archival form, one cluster/line)
        cl_cluster_members.csv        (one row per cluster-member)
        cl_clusters_index.csv         (one row per cluster: aggregates + examples)

  plus _build_report.txt in the deposit folder (provenance, input sha256s,
  corrected-vs-published aggregate deltas, any unresolvable verse IDs).

CRITICAL -- the entity-encoding join bug
----------------------------------------
The membership file stores some verse IDs with the HTML entity '&#xB0;' (a
degree sign); verse_metadata stores the DECODED character. EVERY metadata
lookup must html.unescape(vid) first, or 639 Estonian verses are silently
lost, and the Estonian-side counts come out too low.

Read-only; Python stdlib only. Safe to re-run (idempotent). This is a
provenance record of how folder 05 was built; the large membership and
metadata inputs are multi-GB and not bundled (see README).
"""
import csv
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from html import unescape

# --------------------------------------------------------------------------- #
# Paths (all relative to this script's location in the author's build tree)
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
# .../<deposit>/supplementary_code/scripts/ -> <deposit>/
DEPOSIT_ROOT = os.path.dirname(ROOT)
SIM_OUT = os.path.join(SCRIPT_DIR, "output")

SUMMARY = os.path.join(SIM_OUT, "floor2_rederive", "rrf_ws4_cross_lingual_5algo.json")
MEMB = os.path.join(SIM_OUT, "rrf_cluster_membership_floor2_experimental.jsonl.gz")
META = os.path.join(SIM_OUT, "verse_metadata.jsonl.gz")
KR_TSV = os.path.join(SIM_OUT, "floor2_rederive", "kr_estonian_misfiled_suspects.tsv")
ITEM04 = os.path.join(DEPOSIT_ROOT, "supplementary_data",
                      "04_form_meaning_classification", "rrf_ws4_cross_lingual_5algo.json")

DEPOSIT_DIR = os.path.join(DEPOSIT_ROOT, "supplementary_data", "05_cluster_membership")

# figures from the raw (non-entity-safe) join, reported for comparison
OLD = {"et_verses": 46423, "fi_verses": 37199, "poems": 57038, "pairs": 119388}

sys.path.insert(0, ROOT)
from valismaa_override import fi_override_pids, et_override_pids  # noqa: E402


def log(msg):
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path):
    st = os.stat(path)
    return {
        "path": os.path.relpath(path, ROOT),
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "sha256": sha256_file(path),
    }


# --------------------------------------------------------------------------- #
# 1. Summary
# --------------------------------------------------------------------------- #
log("[1/7] loading floor2 cross-lingual summary ...")
with open(SUMMARY, encoding="utf-8") as f:
    S = json.load(f)
clusters = S["clusters"]
assert S["total_cross_lingual_clusters"] == len(clusters) == 19423, (
    f"expected 19423 CL clusters, got total={S['total_cross_lingual_clusters']} "
    f"len={len(clusters)}")
summ = {c["cluster_id"]: c for c in clusters}
cl_ids = set(summ)
assert len(cl_ids) == 19423, "duplicate cluster_id in summary"
# shared_translations must be empty for all (verified) -> assert then drop
n_shared = sum(1 for c in clusters if c.get("shared_translations"))
assert n_shared == 0, f"shared_translations non-empty for {n_shared} clusters -- do not drop blindly"
cat_form = sum(1 for c in clusters if c["category"] == "form_based")
cat_mean = sum(1 for c in clusters if c["category"] == "meaning_based")
assert cat_form == 18376 and cat_mean == 1047, f"category split {cat_form}/{cat_mean} != 18376/1047"
log(f"      {len(clusters):,} clusters  (form {cat_form:,} / meaning {cat_mean:,})")

# --------------------------------------------------------------------------- #
# 2. Membership (pass 1): ordered member vids for the CL clusters
# --------------------------------------------------------------------------- #
log("[2/7] reading cluster membership ...")
memb = {}            # cluster_id -> [raw vid, ...] in file order
memb_lines = 0
with gzip.open(MEMB, "rt", encoding="utf-8") as f:
    for line in f:
        memb_lines += 1
        o = json.loads(line)
        cid = o["cluster_id"]
        if cid in cl_ids:
            assert len(o["members"]) == o["size"] == summ[cid]["size"], (
                f"size mismatch cluster {cid}: members={len(o['members'])} "
                f"size_field={o['size']} summary={summ[cid]['size']}")
            memb[cid] = o["members"]
assert set(memb) == cl_ids, "membership file is missing some CL clusters"
needed = {unescape(v) for vids in memb.values() for v in vids}
total_slots = sum(len(v) for v in memb.values())
log(f"      {memb_lines:,} membership lines; {total_slots:,} member-slots; "
    f"{len(needed):,} unique decoded vids")

# --------------------------------------------------------------------------- #
# 3. Verse metadata (single streaming sweep, capture only needed vids)
# --------------------------------------------------------------------------- #
log("[3/7] sweeping verse metadata (single pass) ...")
meta = {}            # decoded vid -> (t, l, pl, p)
poem_meta = {}       # poem id -> (l, pl)  -- poem-level language/place fallback
needed_poems = {v.rsplit(":", 1)[0] for v in needed}
meta_lines = 0
with gzip.open(META, "rt", encoding="utf-8") as f:
    for line in f:
        meta_lines += 1
        o = json.loads(line)
        v = o["v"]
        if v in needed:
            meta[v] = (o.get("t", ""), o.get("l", ""), o.get("pl", []), o.get("p", ""))
        p = o.get("p", "")
        if p in needed_poems and p not in poem_meta:
            poem_meta[p] = (o.get("l", ""), o.get("pl", []))
log(f"      {meta_lines:,} metadata lines scanned; {len(meta):,} of {len(needed):,} needed vids found; "
    f"{len(poem_meta):,} member poems captured for fallback")

# --------------------------------------------------------------------------- #
# 4. valismaa language override (mirror build_ws4_cross_lingual_5algo.py)
# --------------------------------------------------------------------------- #
FI_PIDS = set(fi_override_pids())
ET_PIDS = set(et_override_pids())


def lang_override(lang, pid):
    if lang == "et" and pid in FI_PIDS:
        return "fi"
    if lang == "fi" and pid in ET_PIDS:
        return "et"
    return lang


# --------------------------------------------------------------------------- #
# 5. KR-Estonian curated suspect poems (poem-level p values, runo KR 50 / KR 53)
# --------------------------------------------------------------------------- #
log("[4/7] loading curated KR-Estonian suspect poems ...")
KR_POEMS = set()     # exact metadata poem ids, e.g. "KR 50:18", "KR 53:7"
with open(KR_TSV, encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)  # header: poem_id, verse_text
    for row in r:
        if not row:
            continue
        pid = row[0]
        runo = pid.split(":", 1)[0]          # "KR 50:18" -> "KR 50"
        if runo in ("KR 50", "KR 53"):       # KR 34 = genuinely Finnish -> excluded
            KR_POEMS.add(pid)
log(f"      {len(KR_POEMS)} curated KR-Estonian poems (runo 50/53)")

# --------------------------------------------------------------------------- #
# 6. Resolve members; recompute per-cluster aggregates; pick example poems
# --------------------------------------------------------------------------- #
log("[5/7] resolving members and recomputing aggregates ...")
missing_vids = []                            # decoded vids absent from metadata
example_anomalies = []                       # clusters where an example text didn't resolve to a poem
kr_clusters = 0

deposit_jsonl_rows = []                      # one dict per cluster (IDs only)
deposit_member_rows = []                     # (cluster_id, category, vid, lang, place, poem)
deposit_index_rows = []                      # one row per cluster

# aggregate accumulators (corrected, entity-safe)
uniq_lang = {}                               # decoded vid -> lang  (for unique-verse + poem totals)
uniq_poem_lang = {}                          # poem -> set of langs seen (poems are corpus-specific)
sigma_et_fi = 0

for c in clusters:                           # summary order = stable output order
    cid = c["cluster_id"]
    category = c["category"]
    et_ex_text = c["et_example"]
    fi_ex_text = c["fi_example"]

    members_out = []                         # deposit member dicts {vid,lang,place,poem}
    et_n = fi_n = un_n = 0
    et_example_poem = None
    fi_example_poem = None
    cluster_is_kr = False

    for raw in memb[cid]:
        key = unescape(raw)
        rec = meta.get(key)
        if rec is None:
            # Exact verse line absent from the metadata: a single-distinct-word
            # verse dropped by the wordform leg that writes verse_metadata
            # (verse_similarity_common skips verses with < 2 distinct wordforms)
            # but kept in the RRF membership via the other legs. Language and
            # place are poem-level, so resolve them from the poem; text stays "".
            poem = key.rsplit(":", 1)[0]
            text = ""
            pm = poem_meta.get(poem)
            if pm is not None:
                lang = lang_override(pm[0], poem)
                place = ";".join(pm[1])
            else:
                lang, place = "unknown", ""
                missing_vids.append((cid, key))
        else:
            text, lang0, pl, poem = rec
            lang = lang_override(lang0, poem)
            place = ";".join(pl)
        if lang == "et":
            et_n += 1
        elif lang == "fi":
            fi_n += 1
        else:
            un_n += 1

        # example-poem selection: first member of the example's language whose
        # text matches the summary example (verified to resolve for all clusters).
        if et_example_poem is None and lang == "et" and text == et_ex_text:
            et_example_poem = poem
        if fi_example_poem is None and lang == "fi" and text == fi_ex_text:
            fi_example_poem = poem

        if poem in KR_POEMS:
            cluster_is_kr = True

        members_out.append({"vid": key, "lang": lang, "place": place, "poem": poem})
        deposit_member_rows.append((cid, category, key, lang, place, poem))

        # unique-verse / unique-poem accumulators
        if key not in uniq_lang:
            uniq_lang[key] = lang
        if poem:
            uniq_poem_lang.setdefault(poem, set()).add(lang)

    assert et_n + fi_n + un_n == c["size"], f"recomputed counts != size for cluster {cid}"
    sigma_et_fi += et_n * fi_n
    if cluster_is_kr:
        kr_clusters += 1
    if et_example_poem is None or fi_example_poem is None:
        example_anomalies.append((cid, et_example_poem is None, fi_example_poem is None))

    ju = c["char_unigram_jaccard"]
    jw = c["word_jaccard"]

    deposit_jsonl_rows.append({
        "cluster_id": cid, "size": c["size"],
        "et_count": et_n, "fi_count": fi_n, "unknown_count": un_n,
        "category": category, "char_unigram_jaccard": ju, "word_jaccard": jw,
        "kr_estonian_suspect": cluster_is_kr,
        "et_example_poem": et_example_poem, "fi_example_poem": fi_example_poem,
        "members": members_out,
    })
    deposit_index_rows.append((cid, c["size"], et_n, fi_n, un_n, category,
                               cluster_is_kr, ju, jw, et_example_poem, fi_example_poem))

log(f"      KR-Estonian-suspect clusters: {kr_clusters:,}")
log(f"      unresolvable vids (after unescape): {len(missing_vids)}")
log(f"      example-poem anomalies: {len(example_anomalies)}")

# --------------------------------------------------------------------------- #
# 7. Corrected aggregates (entity-safe)
# --------------------------------------------------------------------------- #
et_verses = sum(1 for l in uniq_lang.values() if l == "et")
fi_verses = sum(1 for l in uniq_lang.values() if l == "fi")
un_verses = sum(1 for l in uniq_lang.values() if l == "unknown")
et_poems = sum(1 for langs in uniq_poem_lang.values() if "et" in langs and "fi" not in langs)
fi_poems = sum(1 for langs in uniq_poem_lang.values() if "fi" in langs and "et" not in langs)
mixed_poems = sum(1 for langs in uniq_poem_lang.values() if "et" in langs and "fi" in langs)
total_poems = len(uniq_poem_lang)
# Article basis: poems containing at least one language-resolved (et/fi) verse.
# With the poem-level fallback every member resolves, so poems_unknown_only is 0
# and poems_resolved == total_poems. The split is kept for forward safety: if a
# future member's poem were itself wholly absent from the metadata, it would fall
# back to unknown and be excluded here.
poems_resolved = et_poems + fi_poems + mixed_poems
poems_unknown_only = total_poems - poems_resolved
NEW = {"et_verses": et_verses, "fi_verses": fi_verses, "poems": poems_resolved, "pairs": sigma_et_fi}

# --------------------------------------------------------------------------- #
# 8. Write outputs (IDs only, no text)
# --------------------------------------------------------------------------- #
os.makedirs(DEPOSIT_DIR, exist_ok=True)
log("[6/7] writing deposit files (IDs only, no text) ...")

p_jsonl = os.path.join(DEPOSIT_DIR, "cl_cluster_membership.jsonl")
with open(p_jsonl, "w", encoding="utf-8") as f:
    for row in deposit_jsonl_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

p_members = os.path.join(DEPOSIT_DIR, "cl_cluster_members.csv")
with open(p_members, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["cluster_id", "category", "member_vid", "member_lang",
                "member_place", "member_poem_id"])
    w.writerows(deposit_member_rows)

p_index = os.path.join(DEPOSIT_DIR, "cl_clusters_index.csv")
with open(p_index, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["cluster_id", "size", "et_count", "fi_count", "unknown_count",
                "category", "kr_estonian_suspect", "char_unigram_jaccard",
                "word_jaccard", "et_example_poem", "fi_example_poem"])
    w.writerows(deposit_index_rows)

# --------------------------------------------------------------------------- #
# 9. Verification gates
# --------------------------------------------------------------------------- #
log("[7/7] verification gates ...")
# Gate A
assert len(deposit_jsonl_rows) == 19423
assert cat_form == 18376 and cat_mean == 1047
assert len(missing_vids) == 0, f"unresolvable vids after poem-level fallback: {missing_vids}"
with open(ITEM04, encoding="utf-8") as f:
    item04_ids = {c["cluster_id"] for c in json.load(f)["clusters"]}
assert item04_ids == cl_ids, "export cluster_id set != item-04 summary cluster_id set"
# Gate B
csv_rows = len(deposit_member_rows)
assert csv_rows == total_slots == 110458, f"member rows {csv_rows} != slots {total_slots} != 110458"
# example anomalies must be zero (text-match-within-language resolves for all)
assert not example_anomalies, f"{len(example_anomalies)} clusters with unresolved example poem"
log("      all gates PASSED")

# --------------------------------------------------------------------------- #
# 10. Build report
# --------------------------------------------------------------------------- #
def delta(k):
    return NEW[k] - OLD[k]


report = os.path.join(DEPOSIT_DIR, "_build_report.txt")
with open(report, "w", encoding="utf-8") as f:
    f.write("Cross-corpus cluster-membership export -- build report\n")
    f.write("=" * 64 + "\n")
    f.write(f"generated_utc : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}\n")
    f.write(f"builder       : scripts/build_cl_cluster_membership_export.py\n\n")

    f.write("INPUTS (sha256 / size / mtime)\n")
    f.write("-" * 64 + "\n")
    for label, path, extra in [
        ("summary (floor2 CL)", SUMMARY, ""),
        ("membership", MEMB, "  [NB: *_experimental.jsonl.gz is the floor2 membership source]"),
        ("verse metadata", META, ""),
        ("KR suspects TSV", KR_TSV, ""),
    ]:
        st = file_stat(path)
        f.write(f"{label}\n  {st['path']}\n  sha256 {st['sha256']}\n"
                f"  {st['bytes']:,} bytes  mtime {st['mtime']}{extra}\n")
    f.write(f"membership lines scanned : {memb_lines:,}\n")
    f.write(f"metadata lines scanned   : {meta_lines:,}\n\n")

    f.write("CLUSTERS\n")
    f.write("-" * 64 + "\n")
    f.write(f"cross-lingual clusters   : {len(clusters):,}\n")
    f.write(f"  form_based             : {cat_form:,} (94.6%)\n")
    f.write(f"  meaning_based          : {cat_mean:,} (5.4%)\n")
    f.write(f"member-slots (sum size)  : {total_slots:,}\n")
    f.write(f"unique member verses     : {len(uniq_lang):,}\n")
    f.write(f"  in >=2 clusters        : overlapping membership -- sum(size) over-counts\n")
    f.write(f"member-language tally     : et={sum(1 for l in (m['lang'] for r in deposit_jsonl_rows for m in r['members']) if l=='et'):,}  "
            f"fi={sum(1 for l in (m['lang'] for r in deposit_jsonl_rows for m in r['members']) if l=='fi'):,}  "
            f"unknown={len(missing_vids)}\n")
    f.write(f"KR-Estonian-suspect clusters : {kr_clusters:,}  "
            f"(from {len(KR_POEMS)} curated runo-50/53 poems)\n\n")

    f.write("UNRESOLVABLE VERSE IDS (poem also absent from metadata)\n")
    f.write("-" * 64 + "\n")
    if missing_vids:
        for cid, vid in missing_vids:
            f.write(f"  cluster {cid}: {vid}\n")
    else:
        f.write("  none -- every member resolved; the single-distinct-word verses\n")
        f.write("  dropped by the wordform metadata leg are resolved at poem level\n")
    f.write("\n")

    f.write("CORRECTED AGGREGATES vs PUBLISHED (entity-decode + poem-level fallback)\n")
    f.write("-" * 64 + "\n")
    f.write(f"  Estonian verses : {OLD['et_verses']:,} -> {NEW['et_verses']:,}  (delta {delta('et_verses'):+,})\n")
    f.write(f"  Finnish verses  : {OLD['fi_verses']:,} -> {NEW['fi_verses']:,}  (delta {delta('fi_verses'):+,})\n")
    f.write(f"  unknown verses  : {un_verses}\n")
    f.write(f"  total unique    : {len(uniq_lang):,}  (et+fi+unknown)\n")
    f.write(f"  unique poems    : {OLD['poems']:,} -> {NEW['poems']:,}  (delta {delta('poems'):+,})"
            f"   [et {et_poems:,} / fi {fi_poems:,} / mixed {mixed_poems}]\n")
    if poems_unknown_only:
        f.write(f"    (article figure = poems with a language-resolved verse; the "
                f"{poems_unknown_only} poem(s) reachable only via the {un_verses} unresolved "
                f"verse(s) bring the distinct-poem total to {total_poems:,})\n")
    else:
        f.write(f"    (every member resolved, so this equals the {total_poems:,} "
                f"distinct poems total)\n")
    f.write(f"  cross-corpus pairs : {OLD['pairs']:,} -> {NEW['pairs']:,}  (delta {delta('pairs'):+,})"
            f"   [= sum(et_count*fi_count)]\n\n")

    f.write("OUTPUTS\n")
    f.write("-" * 64 + "\n")
    for path in [p_jsonl, p_members, p_index]:
        st = file_stat(path)
        f.write(f"  {st['path']}\n     sha256 {st['sha256']}  {st['bytes']:,} bytes\n")

log("")
log("Corrected aggregates:")
log(f"  ET verses {OLD['et_verses']:,} -> {NEW['et_verses']:,} ({delta('et_verses'):+,})")
log(f"  FI verses {OLD['fi_verses']:,} -> {NEW['fi_verses']:,} ({delta('fi_verses'):+,})")
log(f"  poems     {OLD['poems']:,} -> {NEW['poems']:,} ({delta('poems'):+,})")
log(f"  pairs     {OLD['pairs']:,} -> {NEW['pairs']:,} ({delta('pairs'):+,})")
log(f"\nDeposit : {os.path.relpath(DEPOSIT_DIR, ROOT)}/")
log(f"Report  : {os.path.relpath(report, ROOT)}")
log("DONE")
