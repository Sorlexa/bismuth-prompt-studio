"""
lexicon.py -- the English the engine needs to read a brief.

Built from Open English WordNet (CC BY 4.0) by tools/build/build_lexicon.py
into data/library/english_lexicon.json. Answers three questions:

    pos(word)        which parts of speech a word can be, inflected forms
                     included ('mixes' -> noun and verb, 'oily' -> adjective)
    ing_forms(word)  the -ing spellings a conjugated verb comes from
                     ('sat' -> sitting, 'lies' -> lying, 'hugged' -> hugging)
    someone(word)    whether a noun's main meaning is a person or an animal
                     ('worker' yes, 'engine' no); thing(word) the reverse

Every function returns None when the lexicon is missing or does not know the
word, so a caller keeps its own fallback for exactly those cases.

Morphology is WordNet's own: the forms the lexicon lists (irregular ones),
then morphy's detachment rules, each result checked against the lemmas.
"""

import json
import os

from promptstudio import paths as _paths

_LEX = {"data": None, "mtime": None}
_LIVE = {"person", "animal"}

# WordNet's morphy detachment rules (morph.c): suffix -> replacement
_DETACH = {
    "n": (("s", ""), ("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"),
          ("shes", "sh"), ("men", "man"), ("ies", "y")),
    "v": (("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"),
          ("ed", ""), ("ing", "e"), ("ing", "")),
    "a": (("er", ""), ("est", ""), ("er", "e"), ("est", "e")),
    "r": (),
}


def _load():
    p = _paths.data("english_lexicon.json", expect=False)
    try:
        mt = os.path.getmtime(p)
    except OSError:
        return None
    if _LEX["mtime"] != mt:
        try:
            with open(p, encoding="utf-8") as f:
                _LEX["data"] = json.load(f)
            _LEX["mtime"] = mt
        except Exception:
            _LEX["data"] = None
    return _LEX["data"]


def available():
    return bool(_load())


def _entry(lemma):
    """-> (pos letters, {class: senses}) for a lemma, or None"""
    lex = _load()
    if not lex:
        return None
    raw = (lex.get("words") or {}).get(lemma)
    if raw is None:
        return None
    pos, _, cls = raw.partition("|")
    counts = {}
    for part in cls.split(",") if cls else ():
        c, _, n = part.partition(":")
        counts[c] = int(n or 1)
    return pos, counts


def lemmas(word, pos):
    """-> the lemmas of `word` for one part of speech: the word itself, the
    listed forms, then morphy's detachments -- each a lemma of that part of
    speech in the lexicon"""
    lex = _load()
    if not lex:
        return []
    w = str(word or "").lower().strip()
    out = []

    def ok(lem):
        e = _entry(lem)
        return bool(e) and pos in e[0]
    if ok(w):
        out.append(w)
    for key in (lex.get("forms") or {}).get(w) or []:
        lem, _, p = key.rpartition(":")
        if p == pos and lem not in out:
            out.append(lem)
    for suf, rep in _DETACH.get(pos, ()):
        if w.endswith(suf) and len(w) > len(suf):
            lem = w[: len(w) - len(suf)] + rep
            if lem not in out and ok(lem):
                out.append(lem)
    return out


def pos(word):
    """-> the set of parts of speech the word can be ('n', 'v', 'a', 'r'),
    inflections included; None when the lexicon does not know it"""
    if not _load():
        return None
    got = {p for p in "nvar" if lemmas(word, p)}
    return got or None


def is_compound_noun(*words):
    """-> True when the words form a noun the lexicon lists ('coffee cup'),
    the last word taken in any of its noun forms ('coffee cups')"""
    if not _load() or len(words) < 2:
        return None
    head = [str(w).lower() for w in words[:-1]]
    for lem in lemmas(words[-1], "n") or [str(words[-1]).lower()]:
        e = _entry(" ".join(head + [lem]))
        if e and "n" in e[0]:
            return True
    return False


def _classes(word):
    """-> the classes holding the most senses of the word's noun lemmas
    (ties kept), or None when it is no noun the lexicon knows"""
    counts = {}
    for lem in lemmas(word, "n"):
        e = _entry(lem)
        for c, n in ((e[1] if e else None) or {}).items():
            counts[c] = counts.get(c, 0) + n
    if not counts:
        return None
    top = max(counts.values())
    return {c for c, n in counts.items() if n == top}


def someone(word):
    """-> True when the word's main noun meaning is a person or an animal,
    False when it is a noun whose main meaning is something else, None when
    the lexicon cannot say"""
    cls = _classes(word)
    if cls is None:
        return None
    return bool(cls & _LIVE)


def thing(word):
    s = someone(word)
    return None if s is None else not s


def ing_forms(word):
    """-> the -ing words a verb form comes from: for each verb lemma, the
    -ing form WordNet lists for it, else the regular spellings (e dropped,
    ie -> ying, the final consonant doubled). The caller checks them
    against the booru's vocabulary. None when the lexicon has no verb."""
    lex = _load()
    if not lex:
        return None
    lems = lemmas(word, "v")
    if not lems:
        return None
    out = []
    for lem in lems:
        for f in _listed_ing().get(lem, []):
            if f not in out:
                out.append(f)
        cands = [lem + "ing"]
        if lem.endswith("ie"):
            cands.insert(0, lem[:-2] + "ying")
        elif lem.endswith("e") and not lem.endswith(("ee", "ye", "oe")):
            cands.insert(0, lem[:-1] + "ing")
        if len(lem) >= 3 and lem[-1] in "bdgklmnprt" and lem[-2] in "aeiou" and lem[-3] not in "aeiou":
            cands.append(lem + lem[-1] + "ing")
        for f in cands:
            if f not in out:
                out.append(f)
    return out


_LISTED_ING = {"data": None, "mtime": None}


def _listed_ing():
    """-> {verb lemma: [listed -ing forms]} ('sit': ['sitting'])"""
    lex = _load()
    if _LISTED_ING["mtime"] != _LEX["mtime"]:
        out = {}
        for form, keys in ((lex or {}).get("forms") or {}).items():
            if not form.endswith("ing"):
                continue
            for key in keys:
                lem, _, p = key.rpartition(":")
                if p == "v":
                    out.setdefault(lem, []).append(form)
        _LISTED_ING["data"], _LISTED_ING["mtime"] = out, _LEX["mtime"]
    return _LISTED_ING["data"] or {}


# ---------------------------------------------------------------------------
# WHAT A WORD CAN BE SAID AS (2026-09-17): the relations the build kept --
# synonyms (depth 0), the direct broader term or an adjective's head (1), and
# terms further up (2-3) -- each target a word the booru uses.
_REL = {"mtime": None, "data": {}, "hub": {}, "cut1": 0, "cut2": 0}


def _relations():
    lex = _load()
    if _REL["mtime"] != _LEX["mtime"]:
        data, hub = {}, {}
        for w, raw in ((lex or {}).get("rel") or {}).items():
            items, seen = [], set()
            for part in raw.split(";"):
                t, c, d = part.rsplit(":", 2)
                items.append((t, c, int(d)))
                if int(d) >= 1 and t not in seen:
                    seen.add(t)
                    hub[t] = hub.get(t, 0) + 1
            data[w] = items
        vals = sorted(hub.values()) or [0]
        # HOW GENERIC A TARGET IS, MEASURED: how many words reach it. The
        # direct broader term may be anything but the top 2% ('device',
        # 'structure'); a term two or three levels up must be outside the
        # top 5% ('animal' is where 'kitten' ends, and says nothing).
        _REL.update(mtime=_LEX["mtime"], data=data, hub=hub,
                    cut1=vals[int(len(vals) * 0.98)], cut2=vals[int(len(vals) * 0.95)])
    return _REL


def related(word, classes=None):
    """-> [(target, depth)] a word can be said as, shallowest first, over the
    word and its noun and adjective lemmas; `classes` keeps only relations
    from senses in those categories. Generic targets are left out by depth
    (see _relations)."""
    rel = _relations()
    w = str(word or "").lower().strip()
    forms = [w] + [l for p in ("n", "a") for l in lemmas(w, p) if l != w]
    out = {}
    for f in forms:
        for t, c, d in rel["data"].get(f) or ():
            if classes is not None and c not in classes:
                continue
            if d == 1 and rel["hub"].get(t, 0) > rel["cut1"]:
                continue
            if d >= 2 and rel["hub"].get(t, 0) > rel["cut2"]:
                continue
            if t not in out or d < out[t]:
                out[t] = d
    return sorted(out.items(), key=lambda kv: kv[1])


def people(kind):
    """-> the words (plurals included) the lexicon names a person with:
    kind 'female' or 'male' (cast nouns), 'role' (a person of no stated
    gender: tailor, millionaire), 'youth' (their senses are a child's) or
    'sexual' (a sexualised child) -- see build_lexicon.py.
    Empty when the lexicon is missing."""
    lex = _load() or {}
    raw = (lex.get("people") or {}).get(kind) or ""
    return frozenset(w for w in raw.split(",") if w)


def style_words(word):
    """-> the words the artists' style descriptions use for `word` (its
    synonyms, similar and derived words, and its other spellings), looked up
    under the word and each of its lemmas; empty when unknown. See
    build_lexicon.py, 'style'."""
    lex = _load() or {}
    table = lex.get("style") or {}
    w = str(word or "").lower().strip()
    keys = {w}
    for p in "navr":
        keys.update(lemmas(w, p))
    out = set()
    for k in keys:
        out.update(x for x in (table.get(k) or "").split(",") if x)
    return frozenset(out)


_SEXUAL = {"mtime": None, "most": frozenset(), "any": frozenset()}


def sexual(word, how="most"):
    """-> True when the word's senses are in the dictionary's sexual domain:
    how='most' -- at least half of them ('erotic', 'sensual'); how='any' --
    one of them ('nude', 'romantic'). The word's lemmas count ('genitals').
    See build_lexicon.py."""
    lex = _load() or {}
    if _SEXUAL["mtime"] != _LEX["mtime"]:
        sx = lex.get("sexual") or {}
        _SEXUAL.update(mtime=_LEX["mtime"],
                       most=frozenset(x for x in (sx.get("most") or "").split(",") if x),
                       any=frozenset(x for x in (sx.get("any") or "").split(",") if x))
    words = _SEXUAL["most" if how == "most" else "any"]
    w = str(word or "").lower().strip()
    if w in words:
        return True
    return any(x in words for p in "navr" for x in lemmas(w, p))


_CLASSES = {"mtime": None, "data": {}}


def word_class(word, cls):
    """-> True when the word (or one of its lemmas) is in a class the
    terminology pass reads: 'body' (a sense that is a body part), 'clothing'
    (a sense under clothing), 'relational' (an adjective that only relates
    to a noun: 'facial'). See build_lexicon.py, word_classes."""
    lex = _load() or {}
    if _CLASSES["mtime"] != _LEX["mtime"]:
        _CLASSES.update(mtime=_LEX["mtime"], data={k: frozenset(v.split(",")) for k, v in
                                                   (lex.get("classes") or {}).items()})
    words = _CLASSES["data"].get(cls) or frozenset()
    w = str(word or "").lower().strip()
    if w in words:
        return True
    return any(x in words for p in ("n", "a") for x in lemmas(w, p))


def main_classes(word):
    """-> the noun categories holding most of the word's senses, or None"""
    return _classes(word)
