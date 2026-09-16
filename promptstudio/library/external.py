"""
external.py -- concepts the checkpoint knows that booru never tagged.

the author's: "the image generators can understand other concepts (not only
booru) ... but they still need to be structured and used in a meaningfull
way ... they need to be integrated into dependecies system and in the
concept slots they belong correctly".

Two rules make that work, and both are about NAMING:

1. THE KIND DECIDES THE SLOT. A concept here carries `kind`
   (artist/style/genre/medium), which is the axis it belongs to. It is not
   a loose word appended to the prompt.

2. THE KIND DECIDES HOW IT IS WRITTEN. the author's warning: an artist who is
   not a booru tag "are not named like '@incase' for anima and 'incase' for
   illustrious but rather look like 'incase style' or 'in the style of
   incase'". The '@' is anima's BOORU artist convention; putting it on a
   painter the model learnt from captions asks for a tag that does not
   exist. Other kinds are literal -- "Other concepts like new locations
   though look the same as tags ('market' is just a 'market' in any case)".

PRECEDENCE OVER CHARACTERS. 37 of these names are also booru characters,
and that collision is the bug this module exists to win: 'van gogh' is not
a danbooru artist, but 'van gogh (fate)' is a Fate servant, so asking for
the painter cast the servant. A typed external concept outranks a bare
character name; the qualified form ('van gogh (fate)') still wins for
anyone who means the servant.
"""

import json
import re

from promptstudio import paths as _paths

_CONCEPTS = None
_BY_KIND = None
_ALIAS = None


def load(reload=False):
    """-> {name: {kind, render, features, source}}"""
    global _CONCEPTS, _BY_KIND
    if _CONCEPTS is not None and not reload:
        return _CONCEPTS
    try:
        with open(_paths.data("external_concepts.json"),
                  encoding="utf-8-sig") as f:
            _CONCEPTS = json.load(f)["concepts"]
    except Exception:
        _CONCEPTS = {}
    # A LIGHTING CONCEPT IS LIGHTING (2026-09-14: 'cinematic lighting'
    # came from midlibrary's General Modifiers as a 'style' and sat in the
    # style band, capitalised): the name decides the kind, and a concept
    # that is not a person renders in lower case like every other tag
    for _n, _r in _CONCEPTS.items():
        if isinstance(_r, dict):
            if str(_n).lower().endswith((" lighting", " light", " lights")) and _r.get("kind") != "lighting":
                _r["kind"] = "lighting"
            if _r.get("kind") != "artist":
                _rd = _r.get("render") or {}
                for _ch, _v in list(_rd.items()):
                    if isinstance(_v, str):
                        _rd[_ch] = _v.lower()
    _BY_KIND = None
    global _ALIAS
    _ALIAS = {a: n for n, r in _CONCEPTS.items()
              for a in (r.get("aliases") or [])}
    return _CONCEPTS


_EQUIV = None


def booru_equivalents():
    """-> {typed phrase: booru tag} for concepts the tag path owns.

    the author's: "biker fashion = literally 'biker clothes' tag from booru and
    it describes and has right dependencies better". These are NOT external
    concepts -- booru already has them under another name, with measured
    co-occurrence behind it. The phrase still has to work when typed, so it
    resolves to the tag instead of being registered as a duplicate.
    """
    global _EQUIV
    if _EQUIV is None:
        try:
            with open(_paths.data("external_concepts.json"),
                      encoding="utf-8-sig") as f:
                _EQUIV = {k: v["tag"] for k, v in
                          (json.load(f).get("booru_equivalents") or {}).items()}
        except Exception:
            _EQUIV = {}
    return _EQUIV


def find_equivalents(base):
    """-> [(phrase, booru tag)] the user typed that booru already covers"""
    eq = booru_equivalents()
    if not eq or not base:
        return []
    low = " " + re.sub(r"[^a-z0-9 .'\-]+", " ", str(base).lower()) + " "
    low = re.sub(r"\s+", " ", low)
    return [(p2, t) for p2, t in eq.items() if (" " + p2 + " ") in low]


def by_kind(kind):
    """-> {name: rec} for one concept kind"""
    global _BY_KIND
    if _BY_KIND is None:
        _BY_KIND = {}
        for n, r in load().items():
            _BY_KIND.setdefault(r.get("kind") or "?", {})[n] = r
    return _BY_KIND.get(kind) or {}


def get(name):
    """the record for a name OR one of its aliases ('van gogh')."""
    con = load()
    n = str(name).strip().lower()
    if n in con:
        return con[n]
    real = (_ALIAS or {}).get(n)
    return con.get(real) if real else None


def canonical(name):
    """-> the registry key a name or alias refers to, or None"""
    con = load()
    n = str(name).strip().lower()
    if n in con:
        return n
    return (_ALIAS or {}).get(n)


# 'in the style of X' / 'X style' -- the user may write either, and both
# name the artist rather than a tag.
_STYLE_OF = re.compile(
    r"\b(?:in the style of|style of|art by|by)\s+([a-z0-9][a-z0-9 .'\-]{2,40})",
    re.I)


def _weak_surface(word):
    """-> True when a one-word alias/name is ordinary vocabulary."""
    w = str(word or "").lower().strip()
    if len(w) < 5:
        return True
    # an inflected English word is vocabulary, whatever surname it shares
    # (2026-09-14: 'wearing see-through shirt' drew Gillian Wearing)
    if w.endswith(("ing", "ed", "ly", "es", "er")) and len(w) >= 6:
        return True
    try:
        from promptstudio.engine import enhancer as _pe
        if w in _pe.FEMALE_NOUN_LIST or w in _pe.MALE_NOUN_LIST:
            return True
    except Exception:
        pass
    try:
        from promptstudio.engine import bridge as _lb
        return bool(_lb._common_word(w))
    except Exception:
        return False


def find_typed(base, kinds=None, include_pending=False):
    """-> [(name, rec)] for external concepts the user actually wrote.

    Longest match wins so 'alphonse mucha' is not read as 'mucha', and a
    name is only accepted on whole-word boundaries.

    PENDING CONCEPTS ARE NOT RETURNED BY DEFAULT. the author's: "we dont use the
    concept until its required fields are not filled". A concept whose
    fields are unfilled is registered and queued -- the DICE never roll
    it. `include_pending=True` is for what the user TYPED: a typed
    concept is honoured even before its fields are filled (its render
    forms exist), because "typed concepts are always honoured" outranks
    the queue.
    """
    con = load()
    if not con or not base:
        return []
    low = " " + re.sub(r"[^a-z0-9 .'\-]+", " ", str(base).lower()) + " "
    low = re.sub(r"\s+", " ", low)
    hits = []
    # an explicit 'in the style of NAME' is the strongest signal there is
    for m in _STYLE_OF.finditer(low):
        cand = m.group(1).strip()
        while cand:
            rec = con.get(cand)
            if rec and (not kinds or rec.get("kind") in kinds):
                hits.append((cand, rec))
                break
            cand = cand.rsplit(" ", 1)[0] if " " in cand else ""
    lookup = list(con.items()) + [(a2, con[n]) for a2, n in
                                  (_ALIAS or {}).items() if n in con]
    for name, rec in lookup:
        if kinds and rec.get("kind") not in kinds:
            continue
        # A BARE SURFACE MUST BE A NAME, NOT A WORD. The alias builder gave
        # 'Guerrilla Girls' the alias 'girls', and "two girls having a
        # picnic" then carried `Guerrilla Girls style`. A single-word
        # surface is accepted only when it is not ordinary vocabulary --
        # the same measured test the character matcher applies to bare
        # names (bridge._common_word) plus the gender nouns.
        if " " not in name and _weak_surface(name):
            continue
        if (" " + name + " ") in low and not any(name == h for h, _ in hits):
            hits.append((name, rec))
    # drop a name that is merely part of a longer name we already matched
    out = []
    for name, rec in hits:
        if any(name != o and name in o.split() for o, _ in hits):
            continue
        if not include_pending and rec.get("status") != "ready":
            continue
        out.append((name, rec))
    return out


def pending(kind=None):
    """-> {name: rec} still missing required fields, the work queue"""
    return {n: r for n, r in load().items()
            if r.get("status") != "ready" and (not kind or r.get("kind") == kind)}


def missing_report():
    """-> {kind: {field: how many concepts still lack it}}"""
    out = {}
    for r in load().values():
        if r.get("status") == "ready":
            continue
        d = out.setdefault(r.get("kind") or "?", {})
        for f in (r.get("missing") or []):
            d[f] = d.get(f, 0) + 1
    return out


def render(name, channel="tag", mode="anima"):
    """how this concept is written in the tag line or in the prose.

    `mode` is accepted for symmetry with the booru artist formatter; these
    concepts render the same for both models, because the difference
    between anima and illustrious is a BOORU convention ('@') and these are
    not booru tags.
    """
    rec = get(name) or {}
    r = rec.get("render") or {}
    return r.get(channel) or r.get("tag") or str(name)


def all_typed_names():
    """every name and alias, for matchers that must not steal them"""
    return set(load()) | set(_ALIAS or {})


def names_colliding_with_characters():
    """-> the names that a character matcher must not steal"""
    return {n for n, r in load().items() if r.get("char_collision")}
