"""
affinity.py -- how the pools lean on each other, as a nudge.

the author's rework (2026-09-01) makes genre, style, medium and location
automatic: each is taken from the prompt when it is there, and invented
otherwise. When it is invented it should lean toward what already fits --
but he was explicit about how far that goes:

    "dependencies should not be strict (rather a nudge) and it should not
     exclude the possibility of selection of new genres/styles/mediums/
     locations that will be added"

and, on style specifically:

    "style dependency from genre should stay minimal (only strongly evident
     cases like the cyberpunk that you mentioned should really matter)"

So this module only ever returns WEIGHTS for a weighted draw. It never
filters. Three properties follow, and all three are deliberate:

  * an unmeasured candidate keeps weight 1.0, so a pool entry added
    tomorrow is selectable the day it lands;
  * a weak measured link also keeps weight 1.0 -- below the threshold the
    signal is noise, and cyberpunk->fantasy (0.014) must not steer
    anything the way cyberpunk->science fiction (0.379) should;
  * the strongest possible link multiplies a candidate's odds by K, it
    does not make the draw deterministic.

Measured by tools/build/build_pool_affinity.py into pool_affinity.json.
"""

import json

from promptstudio import paths as _paths

# Below this the measurement is noise and the candidate stays neutral.
# style<-genre is held much higher on purpose (the author's: minimal).
# genre -> occupation sits an order of magnitude lower than the style/genre
# numbers: 31 measured pairs, min 0.011, median 0.033, p90 0.100, and the one
# strong case (medieval -> knight) at 0.30. The band is pinned to that
# distribution rather than left on the default so the whole measured range
# counts instead of only its top fifth.
THRESHOLD = {
    "genre_occupation": 0.010,
    "style_genre": 0.15,
    "genre_location": 0.01,
    "style_medium": 0.02,
}
DEFAULT_THRESHOLD = 0.02

# The affinity at which a candidate reaches its maximum boost. Anything
# stronger is already overwhelming and does not need more help.
SATURATE = {
    "genre_occupation": 0.100,
    "style_genre": 0.35,
    "genre_location": 0.08,
    "style_medium": 0.06,
}
DEFAULT_SATURATE = 0.10

# Top of the boost range: a perfectly-fitting candidate is K times likelier
# than an unmeasured one, never certain.
K = 3.0

_AFF = None


def load(reload=False):
    global _AFF
    if _AFF is None or reload:
        try:
            with open(_paths.data("pool_affinity.json"), encoding="utf-8") as f:
                _AFF = json.load(f)
        except Exception:
            _AFF = {}
    return _AFF


def table(kind):
    """-> {source: {candidate: affinity}} for 'style_genre' etc."""
    return (load().get(kind) or {})


def score(kind, source, candidate):
    """-> the raw measured affinity, or 0.0 when unmeasured.

    `style_genre` is stored style-first (we ask the rare tag), so a caller
    weighting STYLES against a known genre passes source=style.
    """
    row = table(kind).get(source) or {}
    return float(row.get(str(candidate).lower(), 0.0))


def weight(kind, source, candidate):
    """-> a multiplier in [1.0, K]. 1.0 means 'no opinion'."""
    a = score(kind, source, candidate)
    lo = THRESHOLD.get(kind, DEFAULT_THRESHOLD)
    if a < lo:
        return 1.0
    hi = SATURATE.get(kind, DEFAULT_SATURATE)
    frac = min(1.0, (a - lo) / max(hi - lo, 1e-6))
    return 1.0 + (K - 1.0) * frac


def weights_for(kind, source, candidates, invert=False):
    """-> a weight per candidate, aligned with `candidates`.

    invert=True flips the lookup direction: the table is keyed by the RARE
    member (a style), so choosing a STYLE for a known genre means asking
    each candidate style what it thinks of that genre.
    """
    out = []
    for c in candidates:
        if invert:
            out.append(weight(kind, str(c), source))
        else:
            out.append(weight(kind, str(source), c))
    return out


def no_humans_share(genre):
    """-> measured share of this genre's posts tagged 'no humans'.

    Drives the author's "genre decides" rule for an empty prompt: a genre that
    is frequently peopleless may roll a subject-less scene, one that never
    is will not. Unknown genre -> 0.0, i.e. always peopled, which is the
    safe default for a prompt that asked for nothing.
    """
    return float((load().get("genre_no_humans") or {}).get(genre, 0.0) or 0.0)

# ---------------------------------------------------------------- seeds --
# How hard the PROMPT'S OWN WORDS may pull the genre roll.
#
# the author's, on seeing kimono -> new year measured at 0.475:
#
#     "kimono does not automatically mean new year theme nor new year theme
#      automatically means kimono though (it only narrows more plausible
#      options, so dont restrict 1 clothing = 1 genre, just narrow
#      possibilities)"
#
# So this deliberately does NOT reuse weight(), whose ceiling (K=3.0) a
# single strong pair reaches on its own -- one garment would then decide the
# genre outright. The pull is capped well below that and the evidence is
# damped, so a strong word makes a genre about two and a half times more
# likely than an unrelated one and nothing more. Every genre keeps a real
# share, and an unmeasured word changes nothing.
SEED_PULL_CAP = 1.5
SEED_PULL_K = 3.0


def genre_seed_weights(seeds, genres, net=None):
    """-> [weight] over `genres`, from what the prompt already says.

    Two sources, summed: the measured genre rows (a genre's own related
    tags) and the tag network. Both are evidence of the same kind -- this
    word tends to occur in that genre -- and both are only ever a nudge.
    """
    # COUNT TAGS CARRY NO GENRE EVIDENCE. '1boy' occurs in every genre, so
    # leaving it in saturated the cap for eight genres out of ten and the
    # words that DID mean something -- knight, castle -- were drowned. This
    # is the same contamination that made genre->occupation useless when it
    # was measured off the raw `person` flag; the fix belongs wherever a
    # measurement is read, not at one call site.
    from promptstudio.engine.scene import _is_count_tag
    terms = [str(t).lower() for t in (seeds or ())
             if not _is_count_tag(str(t).lower())]
    rows = table("genre_tags")
    out = []
    for g in genres:
        row = rows.get(g) or {}
        raw = 0.0
        for t in terms:
            raw += float(row.get(t) or 0.0)
            if net is not None:
                try:
                    raw += float(net.edge(t, g) or 0.0)
                except Exception:
                    pass
        # sqrt damping: a second piece of evidence still helps, but three
        # words about the same genre cannot run away with the roll
        pull = min(SEED_PULL_CAP, SEED_PULL_K * (raw ** 0.5)) if raw > 0 else 0.0
        out.append(1.0 + pull)
    return out
