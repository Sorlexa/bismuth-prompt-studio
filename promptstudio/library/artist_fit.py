"""
artist_fit.py -- pick artists by how well they match EVERYTHING the user
said, and weight them by how well they match.

the author's:

    "if the user types 'a western comics style with detailed aesthetics,
     vivid colors and hypersexualized characters' our current generator
     will generalize the style to 'western comics style' and move spice to
     'nsfw' at best. So the artists will be picked just based on 2
     categories ... What about 'detailed aesthetics' and 'vivid colors'
     modifiers? We started vl jobs for artists EXACTLY to get those types
     of characteristics. A prompt ENHANCER should ENHANCE initial prompt,
     and generalizing to single style option is the opposite of
     enhancing/expanding."

So the whole description is the query, not the one style word it collapses
to. An artist that answers most of it is drawn more often AND written
stronger; one that answers only the general style is still reachable, just
quieter:

    fit 1.00  ->  @incase            (full strength, the default weight)
    fit 0.55  ->  (@belko:0.7)
    fit 0.20  ->  (@someone:0.5)

WHY LEXICAL AND NOT ONLY EMBEDDINGS. The embedding matcher is better and is
used when its server is up, but it needs the GPU and the fast path exists
precisely to avoid that. Overlap against the vision record's own words
costs nothing, works during a VL sweep, and degrades honestly: an artist
with no vision record scores neutral rather than zero, so nothing is
excluded for want of data.

THIS ONLY APPLIES WHERE CORRELATION IS MEASURABLE. the author's: the same
pickrate/strength rule suits styles, mediums, techniques and colour
schemes, but NOT "character/clothes/locations or other hard concepts" --
a location either is in the picture or is not, and half a castle is not a
weaker castle.
"""

import json
import re

from promptstudio import paths as _paths

_VISION = None
_STOP = set(
    "a an the of and or with in on at for to from by as is are be it its "
    "this that these those very more most some any all not no into over "
    "under about like style styles art image picture scene character "
    "characters girl boy woman man person people".split())


def vision():
    global _VISION
    if _VISION is None:
        try:
            with open(_paths.data("artist_styles_vision.json"),
                      encoding="utf-8-sig") as f:
                _VISION = json.load(f)["artists"]
        except Exception:
            _VISION = {}
    return _VISION


def _words(text):
    return {w for w in re.findall(r"[a-z]+", str(text).lower())
            if w not in _STOP and len(w) > 2}


_DOC_CACHE = {}


def _artist_words(name):
    """every word the vision record uses about this artist: its terms when the
    terminology pass has run (tools/vision/unify_looks.py -- one terminology,
    no noise), else the model's own text"""
    if name in _DOC_CACHE:
        return _DOC_CACHE[name]
    rec = vision().get(name) or {}
    terms = rec.get("terms") if isinstance(rec.get("terms"), dict) else None
    if terms is not None:
        bag = set()
        for t in (terms.get("agreed") or []) + (terms.get("once") or []):
            bag |= _words(t)
    else:
        bag = _words(rec.get("desc") or "")
        for t in (rec.get("traits") or []):
            bag |= _words(t)
    bag |= _words(rec.get("tradition") or "")
    _DOC_CACHE[name] = bag
    return bag


def described(name):
    """-> True when we know what this artist's work looks like"""
    return bool(_artist_words(name))


_SAID_AS = {}


def said_as(word):
    """-> the description words that answer a query word: the word itself,
    and what the dictionary says it is called in the descriptions (2026-09-17:
    the descriptions say 'dark' and 'moody', a user writes 'gloomy' -- exact
    matching found one artist). lexicon.style_words owns the table."""
    if word not in _SAID_AS:
        try:
            from promptstudio.engine import lexicon as _lx
            _SAID_AS[word] = frozenset({word}) | _lx.style_words(word)
        except Exception:
            _SAID_AS[word] = frozenset({word})
    return _SAID_AS[word]


def fit(name, query_words):
    """-> 0.0-1.0, how much of the user's description this artist answers.

    Scored as the share of the QUERY that the artist covers, not the share
    of the artist that the query covers: asking for two things and getting
    both is a full match even if the artist does ten other things too. A
    query word is answered by itself or by any word the dictionary says it
    is called in the descriptions (said_as).
    """
    if not query_words:
        return None                      # nothing was asked: no opinion
    bag = _artist_words(name)
    if not bag:
        return None                      # no vision record: no opinion
    hit = sum(1 for q in query_words if said_as(q) & bag)
    return hit / float(len(query_words))


def query_words(base, known_tags=()):
    """the descriptive part of the prompt.

    Words already recognised as booru tags are dropped: those are handled
    by the tag path and would only pull artists who draw that SUBJECT.
    What is left is the language the tag system had no home for -- exactly
    the 'detailed aesthetics, vivid colors' half that used to be discarded.
    """
    q = _words(base)
    for t in (known_tags or ()):
        q -= _words(t)
    return q


def embedding_fits(base, names, timeout_ok=True):
    """-> {name: fit} from the embedding matcher, or {} when unavailable.

    Better than word overlap because it understands synonymy: the user
    writes "hypersexualized" and the vision record says "sensual,
    exaggerated proportions", which lexical scoring cannot connect. Needs
    the embed server, so the fast path falls back to overlap rather than
    waking a GPU.

    Cosine over these vectors sits in a narrow band, so it is rescaled
    across the CANDIDATES rather than used raw -- the question is which of
    these artists answers the description best, not the absolute number.
    """
    if not base or not names:
        return {}
    try:
        from promptstudio.library import matching as _m
        if not _m.ready():
            return {}
        ranked = _m.match_description(base, n_styles=0,
                                      n_artists=len(names) or 8)
    except Exception:
        return {}
    # matching._rank returns (SCORE, NAME) -- unpacking it the other way
    # round silently produced a dict keyed by float, so every lookup missed
    # and the embedding path quietly fell back to word overlap.
    got = {n: sc for sc, n in (ranked.get("artists") or [])}
    got = {n: sc for n, sc in got.items() if n in set(names)}
    if len(got) < 2:
        return {}
    lo, hi = min(got.values()), max(got.values())
    if hi - lo < 1e-6:
        return {}
    return {n: (sc - lo) / (hi - lo) for n, sc in got.items()}


# fit -> the weight written into the prompt. Deliberately coarse: these are
# soft preferences and a long tail of 0.63s would be false precision.
def strength(f):
    if f is None:
        return 1.0
    if f >= 0.60:
        return 1.0
    if f >= 0.35:
        return 0.8
    if f >= 0.15:
        return 0.7
    return 0.5


def weight(f, k=4.0):
    """draw weight -- a better fit is picked more often, never exclusively"""
    if f is None:
        return 1.0
    return 1.0 + k * f


def render(formatted, f):
    """'@incase' + fit -> '@incase' or '(@incase:0.7)'.

    A weight of 1.0 is written as the bare name: '(@x:1)' and '@x' mean the
    same thing to the samplers, and the bare form keeps the prompt readable.
    """
    w = strength(f)
    if w >= 1.0:
        return formatted
    return "(%s:%s)" % (formatted, ("%.1f" % w).rstrip("0").rstrip("."))
