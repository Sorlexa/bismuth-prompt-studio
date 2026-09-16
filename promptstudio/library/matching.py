#!/usr/bin/env python
"""
style_match.py -- cosine matching over the concept embeddings.

Powers (1) the artist-fit fix: pick artists whose measured fingerprint
matches the RESOLVED style, replacing the near-random floor pool; and
(2) describe-the-style: a freeform user description -> best styles +
best artists. Pure-python cosine (no numpy dependency for the studio's
runtime); the sets are small (~2k artists) so it is instant.
"""

import json
import math
import os
from promptstudio import paths as _paths

HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = None
_ARTIST = None


def _load(name):
    p = _paths.data(name)
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def _styles():
    global _STYLE
    if _STYLE is None:
        d = _load("style_embeddings.json") or {"names": [], "vecs": []}
        _STYLE = (d["names"], d["vecs"], {n: i for i, n in
                                          enumerate(d["names"])})
    return _STYLE


def _artists():
    global _ARTIST
    if _ARTIST is None:
        d = _load("artist_embeddings.json")
        _ARTIST = (d["names"], d["vecs"], d.get("posts", {})) if d \
            else ([], [], {})
    return _ARTIST


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos_unit(a_unit, b):
    n = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a_unit, b)) / n


def _rank(query_vec, names, vecs, top):
    qu = _norm(query_vec)
    scored = [(_cos_unit(qu, vecs[i]), names[i]) for i in range(len(names))]
    scored.sort(reverse=True)
    return scored[:top]


def artists_for_style(style_name, n=3, rng=None, pool=40):
    """the artist-fit fix: top artists whose fingerprint matches this
    style's precomputed vector. Variety: draw n from the top `pool` by
    cosine, weighted toward the closest."""
    snames, svecs, sidx = _styles()
    anames, avecs, _ = _artists()
    if style_name not in sidx or not anames:
        return []
    ranked = _rank(svecs[sidx[style_name]], anames, avecs, pool)
    if rng is None:
        return [a for _, a in ranked[:n]]
    picks, cand = [], list(ranked)
    for _ in range(min(n, len(cand))):
        ws = [max(0.01, s) ** 4 for s, _ in cand]   # sharply favour close
        j = rng.choices(range(len(cand)), weights=ws)[0]
        picks.append(cand.pop(j)[1])
    return picks


def match_description(desc, n_styles=5, n_artists=8):
    """describe-the-style: embed the freeform query live, return best
    styles and artists. Needs the embedding server (one call)."""
    from promptstudio.llm import server as llm_server
    qvec = llm_server.embed(["search_query: " + desc])[0]
    snames, svecs, _ = _styles()
    anames, avecs, _ = _artists()
    return {
        "styles": _rank(qvec, snames, svecs, n_styles),
        "artists": _rank(qvec, anames, avecs, n_artists),
    }


def ready():
    """the semantic matcher runs when everything it needs is present: the
    style and artist vectors (data/embeddings/), an embedding model named
    in llm_config.json ('embed') and llama.cpp to serve it. Otherwise the
    measured style-share draw and the lexical description fit stand in."""
    try:
        from promptstudio.llm import config as _cfg, server as _srv
        p = (_cfg.load() or {}).get("embed_path")
        if not p or not os.path.exists(p) or not (_srv._backend() or (None,))[0]:
            return False
    except Exception:
        return False
    return bool(_styles()[0]) and bool(_artists()[0])
