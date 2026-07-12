"""Canonical helper for corpus language overrides.

Three override directions:

1. **välismaa → FI**: 68 ERAB poems whose text is actually Finnish/Ingrian,
   archived under ET-prefixed IDs with the catchall ``välismaa`` place tag.
   Override file: ``data/valismaa_finnish_poems.json``.

2. **JR → ET**: 13 JR (Finnish archive) poems whose text is Estonian:
   12 collected in Setumaa (Setu dialect), 1 in Tori (JR 79015).
   Hardcoded list (too small for a separate JSON file).

3. **KR Kalevipoeg → ET**: 22 poems in KR (Kirjalliset Runot) whose text is
   the Estonian national epic Kalevipoeg. These have ``l='et'`` in poem metadata
   but ``col='KR'``. Build scripts with ET_PREFIXES already include
   ``'Kalevipoeg'``; others default to ET correctly via prefix detection.

Build scripts that perform prefix-based language detection consult both
``is_fi_override()`` and ``is_et_override()`` before falling back to prefix
matching.
"""

from functools import lru_cache
import json
import os

OVERRIDE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data',
    'valismaa_finnish_poems.json',
)

_JR_ET_PIDS = frozenset([
    'JR 03705', 'JR 03706', 'JR 03707', 'JR 03708',
    'JR 72642', 'JR 72643', 'JR 72644', 'JR 72645',
    'JR 72646', 'JR 72647', 'JR 72648', 'JR 72649',
    'JR 79015',
])

_KR_ET_PIDS = frozenset([
    'Kalevipoeg 1', 'Kalevipoeg 2', 'Kalevipoeg 3', 'Kalevipoeg 4',
    'Kalevipoeg 5', 'Kalevipoeg 6', 'Kalevipoeg 7', 'Kalevipoeg 8',
    'Kalevipoeg 9', 'Kalevipoeg 10', 'Kalevipoeg 11', 'Kalevipoeg 12',
    'Kalevipoeg 13', 'Kalevipoeg 14', 'Kalevipoeg 15', 'Kalevipoeg 16',
    'Kalevipoeg 17', 'Kalevipoeg 18', 'Kalevipoeg 19', 'Kalevipoeg 20',
    'Kalevipoeg Sissejuhatuseks', 'Kalevipoeg Soovituseks',
])


@lru_cache(maxsize=1)
def fi_override_pids() -> frozenset:
    """Return the frozenset of all 68 välismaa-FI override pids."""
    with open(OVERRIDE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return frozenset(p['pid'] for p in data['finnish_pids'])


def is_fi_override(pid: str) -> bool:
    """Return True if ``pid`` is a välismaa Finnish-language override.

    For an override pid, downstream code should treat the *language* as
    Finnish while preserving the genuine archive *corpus* (typically ERAB).
    """
    return pid in fi_override_pids()


def et_override_pids() -> frozenset:
    """Return the frozenset of 35 ET-language override pids (13 JR + 22 KR Kalevipoeg)."""
    return _JR_ET_PIDS | _KR_ET_PIDS


def is_et_override(pid: str) -> bool:
    """Return True if ``pid`` is a JR or KR Estonian-language override.

    Covers two sets:
    * 13 JR poems whose text is Estonian (Setu dialect + Tori).
    * 22 KR Kalevipoeg poems (Estonian national epic).

    For an override pid, downstream code should treat the *language* as
    Estonian while preserving the genuine archive *corpus* (JR or KR).
    """
    return pid in _JR_ET_PIDS or pid in _KR_ET_PIDS
