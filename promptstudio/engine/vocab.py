"""
vocab.py -- the retrieval layer: what tags exist, what they mean, and which
of them a concept may legitimately map to.

Split out of bridge.py, which had grown to 4,100 lines. This group was the
only one that came away cleanly: it depends on nothing else in the pipeline,
while the resolve_* family and the two model bridges are entangled with a
dozen module constants each (docs/BRIDGE_DECOMPOSITION.md has the numbers).

What lives here:

  _vocab()          every tag the checkpoints have seen, normalised
  _glosses()        the concept library, hot-reloaded when it changes
  _gloss_of()       one tag's gloss and flags
  candidates_for()  a concept -> its ranked, gated shortlist
  confident_pick()  the engine's own answer when retrieval is unambiguous
  _slot_scope()     narrow a shortlist to one body slot
  _style_vocab()    the STYLE sphere, so style concepts cannot escape it
  _light_vocab()    the same discipline for lighting

The caches are module-level on purpose: these tables are large, read-only
and shared by every generation.
"""

import json
import os
import re

from promptstudio.engine import enhancer as pe
from promptstudio import paths as _paths


# ------------------------------------------------- candidate tag retrieval
_CAND_BLOCK = {"artist name", "artist self-insert", "tag", "tagme", "signature",
               "watermark", "text", "english text", "meme", "parody",
               "jpeg artifacts", "blurry", "lowres",
               "adult baby", "yukkuri shiteitte ne", "turn pale",
               "aged down", "aged up",
               # sweep round: meme/title/meta tags reached from innocent
               # descriptor and occupation concepts ('cute' -> an anime
               # title, 'artist' -> a commissioning meta tag)
               "can't be this cute", "cute & girly (idolmaster)",
               "artist connection", "student and teacher",
               "fighter jet", "variable fighter",  # 'a male fighter' is
               # not an aircraft -- either kind; the real fix is
               # occupation-sphere scoping (library phase)
               "holding hands is lewd",  # meme tag reached from an
               # innocent holding phrase
               "facial"}  # a CUM tag, not a face descriptor -- kept

               # reachable by typing and by the measured injectables
_VOCAB = None

# LIBRARY PHASE 1 (the author's description proposal, first layer): only
# GENERAL tags (danbooru category 0) are bridge-2 retrieval material.
# Artists, characters, copyrights and meta flow through their own engine
# channels -- and the whole hand-built meme/title blocklist above becomes
# a fossil the moment the category table exists ('can't be this cute' is
# a copyright BY DATA). Tags absent from the table (gelbooru-only
# spellings, curated pool names) default to general.
_TAG_CATS = None

# LIBRARY PHASE 2: per-tag glosses + machine flags (the author's description
# proposal in full). Glosses ride bridge-2's candidate lines so the
# model picks by MEANING, not name similarity; flags feed the verifiers
# ('meme'/'text'/'symbol' illegal everywhere in retrieval, 'view'
# illegal for body-description slots). Candidates without a gloss are
# logged to gloss_pending.json for the next build run (lazy growth).
_GLOSSES = None

_GLOSS_MTIME = 0

def _glosses():
    global _GLOSSES, _GLOSS_MTIME
    p = _paths.data("tag_glosses.json")
    try:
        mt = os.path.getmtime(p)
    except OSError:
        return {}
    if _GLOSSES is None or mt != _GLOSS_MTIME:
        try:
            with open(p, encoding="utf-8-sig") as f:
                _GLOSSES = json.load(f).get("glosses", {})
            _GLOSS_MTIME = mt
        except Exception:
            _GLOSSES = {}
    return _GLOSSES

def _gloss_of(t):
    e = _glosses().get(t) or {}
    return e.get("g"), e.get("f") or []

def _log_pending(tags_wanted):
    """candidates without glosses queue for the next build run"""
    if not tags_wanted:
        return
    p = _paths.data("gloss_pending.json")
    try:
        with open(p, encoding="utf-8-sig") as f:
            cur = set(json.load(f))
    except Exception:
        cur = set()
    new = cur | set(tags_wanted)
    if new != cur:
        try:
            from promptstudio.llm import config as _cfg
            if (_cfg.load() or {}).get("ledger", True):   # a release gathers nothing (2026-09-16)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(sorted(new), f)
        except Exception:
            pass

def _tag_cat(t):
    global _TAG_CATS
    if _TAG_CATS is None:
        try:
            with open(_paths.data("tag_categories.json"),
                      encoding="utf-8-sig") as f:
                _TAG_CATS = json.load(f).get("categories", {})
        except Exception:
            _TAG_CATS = {}
    return _TAG_CATS.get(t, 0)

def _vocab():
    """tag -> count over both boorus plus every curated pool; built once."""
    global _VOCAB
    if _VOCAB is not None:
        return _VOCAB
    import html as _html
    v = {}
    for name in ("danbooru_tags.json", "gelbooru_tags.json"):
        p = _paths.data(name)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                for t, n in json.load(f).items():
                    # the harvests store HTML entities ('cute &amp;
                    # girly', 'can&#039;t be this cute') -- every exact
                    # match against them silently failed until decoded
                    k = pe.normalize_tag(_html.unescape(t))
                    v[k] = max(v.get(k, 0), n)
    for pool, key in (("location_pool.json", None), ("style_pool.json", None)):
        p = _paths.data(pool)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                for t in json.load(f):
                    v.setdefault(t.lower(), 1)
    _VOCAB = v
    _load_aliases()
    return v

_ALIASES = None

def _load_aliases():
    """danbooru's alias table as redirects: 'hot spring' IS onsen, and
    retrieval must know it exactly instead of fuzzing to 'spring (object)'."""
    global _ALIASES
    if _ALIASES is not None:
        return _ALIASES
    _ALIASES = {}
    p = _paths.data("alias_map.json")
    try:
        with open(p, encoding="utf-8-sig") as f:
            raw = json.load(f)["aliases"]
        # the concept ledger's integrated aliases ride beside the booru's
        # (library/concepts.py integrate, 2026-09-11)
        try:
            with open(_paths.data("alias_map_local.json"), encoding="utf-8-sig") as f:
                raw.update((json.load(f) or {}).get("aliases") or {})
        except Exception:
            pass
        v = _VOCAB or {}
        _ALIASES = {a: c for a, c in raw.items()
                    if c in v and v[c] >= 100}
    except Exception:
        pass
    return _ALIASES

_FRANCHISE = None
# A copyright needs this many CHARACTER tags before its name is treated as
# franchise-bearing rather than an ordinary word. Measured, not guessed:
# `another`, `playboy`, `twitter` and `instagram` are all real copyright
# titles with 0 characters, and blocking on the bare name would have cost
# `looking at another` (433,889 posts), `playboy bunny` (147,467) and
# `dual persona` (62,764). `persona` itself has 11, which is why the line
# sits well above it. At 40 the rule covers pokemon (293), fate (376),
# blue archive (211), honkai: star rail (102) and touhou (44), and the
# 152 general tags it scopes are franchise-specific to a one.
_FRANCHISE_MIN_CHARS = 40


def _franchises():
    """-> {copyright name: character count} for names that carry a franchise.

    Also every copyright used as a PARENTHETICAL qualifier, whatever its
    character count: danbooru writes `ranger uniform (amphibia)` precisely
    to say the tag is scoped to that work, so the qualifier is the site's
    own statement that this is not the generic article.
    """
    global _FRANCHISE
    if _FRANCHISE is None:
        _FRANCHISE = {}
        try:
            with open(_paths.data("tag_categories.json"),
                      encoding="utf-8-sig") as f:
                cats = json.load(f)["categories"]
            counts = {}
            for t, c in cats.items():
                if c == 4 and "(" in t:
                    q = t.split("(", 1)[1].rstrip(")").strip()
                    counts[q] = counts.get(q, 0) + 1
            copy = {t for t, c in cats.items() if c == 3}
            _FRANCHISE = {w: counts.get(w, 0) for w in copy
                          if counts.get(w, 0) >= _FRANCHISE_MIN_CHARS}
            _FRANCHISE["_qualifiers"] = copy
        except Exception:
            _FRANCHISE = {}
    return _FRANCHISE


def _franchise_leak(tag, concept):
    """-> True when `tag` is scoped to a franchise `concept` never asked for.

    The LLM asked for a "ranger uniform" and got `pokemon ranger uniform`,
    because every tag in the vocabulary carrying the word `ranger` belongs
    to some franchise -- there is no generic one. Two of the three even
    say so in parentheses. Mapping onto one of them imports a whole art
    style and cast that nobody requested, and for a fantasy scene it is
    simply wrong.

    Emitting NOTHING is the right answer when the vocabulary has no
    generic tag for a concept: the prose still describes the uniform, and
    "no is better than genuinely wrong" is the standing rule here.
    """
    fr = _franchises()
    if not fr:
        return False
    quals = fr.get("_qualifiers") or set()
    have = " " + concept + " "
    if "(" in tag:
        q = tag.split("(", 1)[1].rstrip(")").strip()
        if q in quals and " " + q + " " not in have:
            return True
    for w in tag.replace("(", " ").replace(")", " ").split():
        if w == tag:
            continue
        if fr.get(w, 0) >= _FRANCHISE_MIN_CHARS and " " + w + " " not in have:
            return True
    return False


def _phrase_in(needle, hay):
    """-> True when `needle` occurs in `hay` as whole words ('tomb' is not
    in 'tomboy'; 'long hair' is in 'very long hair')."""
    return (" " + needle + " ") in (" " + hay + " ")


def candidates_for(concept, top=8, verb_first=False):
    """Booru tags that could express this concept: exact > substring > token
    overlap, popularity as the tie-break. Retrieval, never recall."""
    v = _vocab()
    c = pe.normalize_tag(concept)
    # alias redirect first: the concept may be danbooru's OWN name for a tag
    c = _load_aliases().get(c, c)
    if c in v and v[c] >= 100:
        return [(9, c)]
    # stopwords must not score: 'crying with eyes open' reached a serene
    # onsen because "with" counted as overlap
    _STOP = {"with", "and", "the", "a", "an", "in", "of", "on", "at", "her",
             "his", "their", "its", "is", "are", "over", "under", "to",
             # 'through' anchored 'self-soothing through touch' to 'ass
             # visible through thighs' -- function words never carry meaning
             "through", "by", "as", "while", "into", "from"}
    words = set(c.split()) - _STOP
    # light stemming: the model conjugates ('kisses', 'embracing') while
    # danbooru tags use base forms -- without this, 'kiss' has zero overlap
    # with 'kisses' and the anti-substring guard blocks the rest
    for w in list(words):
        if len(w) > 5 and w.endswith("ing"):
            words |= {w[:-3], w[:-3] + "e"}
        elif len(w) > 4 and w.endswith("es"):
            words.add(w[:-2])
        elif len(w) > 4 and w.endswith("ed"):
            words |= {w[:-2], w[:-1]}
        elif len(w) > 3 and w.endswith("s"):
            words.add(w[:-1])
    # The concept's HEAD NOUN anchors the sense: "chestnut brown hair" must
    # surface hair tags, not the nut -- a bare substring hit on "chestnut"
    # mapped exactly that in the first live run.
    # ACTION concepts lead with the verb ("kissing in the rain"): their
    # anchor is the FIRST word, or 'kiss' loses to rain tags on the head
    # filter. Object concepts keep last-word heads.
    head = ((c.split()[0] if verb_first else c.split()[-1])
            if c.split() else "")
    scored = []
    seen_emit = {}
    head_ok = set()
    # alias SURFACES score too: the fuzzy pass matches the alias's own words
    # ('bellybutton piercing') but emits the canonical tag it points at
    surfaces = [(t, n, t) for t, n in v.items()]
    surfaces += [(a, v.get(canon, 0), canon)
                 for a, canon in _load_aliases().items()]
    for t, n, emit in surfaces:
        # candidate hygiene: a tag the model could steer with needs real use,
        # a real name (no 'v'), and must not be negative-prompt vocabulary
        # ('artist name' reached a prompt through this hole)
        if n < 100 or len(t) < 3 or t in _CAND_BLOCK or emit in _CAND_BLOCK:
            continue
        if _tag_cat(emit) != 0 or _tag_cat(t) != 0:
            continue        # general tags only (library phase 1)
        # a general tag can still be SCOPED to a franchise
        if _franchise_leak(t, c) or _franchise_leak(emit, c):
            continue
        # banned-content vocabulary is NEVER offerable (the standing
        # content-filter policy, applied to retrieval itself), and
        # '(trend)' tags are meta-noise
        if re.search(r"\b(loli|shota|toddler)\b|\(trend\)$", t) or \
                re.search(r"\b(loli|shota|toddler)\b|\(trend\)$", emit):
            continue
        tw = set(t.split())
        ov = len(words & tw)
        # a short tag inside a longer word is noise, not meaning: 'axe' is a
        # substring of "relaxed" and reached a prompt that way -- and the
        # OTHER direction was still open: the concept 'tomb' sat inside
        # the tag 'tomboy' and a frog on a lily pad got a tomboy. Both
        # directions now match on WORD boundaries only.
        sub = _phrase_in(c, t) or (_phrase_in(t, c) and len(t) >= 5)
        if not sub and not ov:
            continue
        score = ov
        if sub:
            score = max(score, 2)
        if head and head in tw:
            score += 3                      # sense anchor on the head noun
            head_ok.add(emit)               # the SURFACE carried the head --
                                            # 'hot spring' anchors onsen even
                                            # though onsen contains no 'spring'
        if tw and tw <= words:
            score += 3                      # the tag appears whole in the
                                            # concept: 'yukata' in "pink
                                            # yukata with patterns" -- the
                                            # earlier lone-word PENALTY killed
                                            # exactly this and lost the
                                            # garment while keeping its print
        if score > 0:
            # BEST surface per emitted tag, not first: 'hot springs' (weak)
            # must not lock out 'hot spring' (whole-word, head-anchored)
            prev = seen_emit.get(emit)
            if prev is None or score > prev[0]:
                seen_emit[emit] = (score, n)
    scored = [(sc, n, emit) for emit, (sc, n) in seen_emit.items()]
    scored.sort(key=lambda x: (-x[0], -x[1]))
    out = [(sc, t) for sc, _, t in scored[:top * 2]]
    # When the head noun is represented at all, candidates missing it are cut:
    # for "long wavy chestnut hair" this keeps wavy hair / brown hair and
    # drops the nut, which the 0-2-picks rule had let back in.
    if head and any(head in t.split() or t in head_ok for _, t in out):
        out = [(sc, t) for sc, t in out
               if head in t.split() or t in head_ok or len(t.split()) > 1]
    return out[:top]

def confident_pick(concept, verb_first=False):
    """PICK DISCIPLINE: when retrieval is certain, the ENGINE maps the
    concept and the LLM never sees it -- 'collarbone' must not lose to
    'exposed bone' because a model liked the sound of it. Certain means: the
    concept IS a live tag, or the top candidate dominates (head-anchored,
    whole-word, clearly ahead of second place)."""
    v = _vocab()
    c = pe.normalize_tag(concept)
    if c in v and v[c] >= 100:
        return c
    ranked = candidates_for(concept, verb_first=verb_first)
    if not ranked:
        return None
    top_sc, top_t = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0
    if top_sc >= 5 and top_sc - second >= 2:
        return top_t
    return None

# ---------------------------------------------------------------- BRIDGE 2
_STYLE_VOCAB2 = None

def _style_vocab():
    """the STYLE sphere's tag universe: both style pools, their measured
    palettes and techniques, the colors group, and every '(style)' tag."""
    global _STYLE_VOCAB2
    if _STYLE_VOCAB2 is not None:
        return _STYLE_VOCAB2
    out = set()
    try:
        with open(_paths.data("style_pool.json"),
                  encoding="utf-8-sig") as f:
            sp = json.load(f)
        for k, v in sp.items():
            out.add(k.lower())
            out.update(t.lower() for t in (v.get("palette") or {}))
            out.update(t.lower() for t in (v.get("techniques") or {}))
    except Exception:
        pass
    try:
        with open(_paths.data("cultural_styles.json"),
                  encoding="utf-8-sig") as f:
            out.update(t.lower() for t in json.load(f)["pool"])
    except Exception:
        pass
    try:
        with open(_paths.data("danbooru_wiki.json"),
                  encoding="utf-8-sig") as f:
            grp = json.load(f)["groups"]
        for gname in ("colors",):
            for sec, tl in (grp.get(gname) or {}).items():
                out.update(t.lower() for t in tl)
        out.update(t.lower() for t in
                   (grp.get("image composition") or {}).get("techniques")
                   or [])
    except Exception:
        pass
    out.update(t for t in _vocab() if t.endswith("(style)"))
    # A MEDIUM IS NOT A STYLE. This sphere is the retrieval target for
    # every style-band concept, so anything in it is reachable from any
    # style phrase the pipeline cannot map directly: 'art deco' is not a
    # booru tag, so it retrieved its nearest neighbour here and emitted
    # 'lineart'. That put a medium on ~2% of images behind the medium
    # axis's back, which is deliberately rare (bridge.resolve_medium).
    # Medium belongs to one axis; the spheres must not overlap on it.
    try:
        from promptstudio.engine.bridge import is_medium_tag as _is_med
        out = {t for t in out if not _is_med(t)}
    except Exception:
        pass
    _STYLE_VOCAB2 = out
    return out

_HAIR_VOCAB = None

def _hair_vocab():
    """the HAIR sphere: the wiki's hair, hair-color and hair-styles groups
    -- 'ponytail', 'braid' and 'bangs' are hair tags that do not contain
    the word 'hair', and the substring scope was excluding exactly the
    hairstyle vocabulary the hair_style slot exists for."""
    global _HAIR_VOCAB
    if _HAIR_VOCAB is not None:
        return _HAIR_VOCAB
    out = set()
    try:
        with open(_paths.data("danbooru_wiki.json"),
                  encoding="utf-8-sig") as f:
            grp = json.load(f)["groups"]
        for gname in ("hair", "hair color", "hair styles"):
            for sec, tl in (grp.get(gname) or {}).items():
                out.update(t.lower() for t in tl)
    except Exception:
        pass
    # the wiki sections list variants ('high ponytail') but not the base
    # tags the sections are named after -- those are hair tags too
    out.update(("ponytail", "twintails", "braid", "twin braids", "bangs",
                "ahoge", "bob cut", "sidelocks", "updo", "topknot",
                "double bun", "single hair bun", "dreadlocks", "afro",
                "pompadour", "mohawk", "buzz cut", "undercut"))
    _HAIR_VOCAB = out
    return out

_LIGHT_VOCAB = None

def _light_vocab():
    """the LIGHTING sphere: the measured lighting pool plus the wiki's
    lighting group. A lighting concept that maps outside it maps wrong --
    'soft golden light from the left' had retrieved the CAMERA tag 'from
    side'; direction, when not drawn, belongs to the NL."""
    global _LIGHT_VOCAB
    if _LIGHT_VOCAB is not None:
        return _LIGHT_VOCAB
    out = set()
    try:
        with open(_paths.data("lighting_pool.json"),
                  encoding="utf-8-sig") as f:
            out.update(t.lower() for t in json.load(f))
    except Exception:
        pass
    try:
        with open(_paths.data("danbooru_wiki.json"),
                  encoding="utf-8-sig") as f:
            grp = json.load(f)["groups"]
        for sec, tl in (grp.get("lighting") or {}).items():
            out.update(t.lower() for t in tl)
    except Exception:
        pass
    out.update(("sunlight", "moonlight", "shadow", "silhouette",
                "light particles", "glowing", "sparkle", "lens flare"))
    _LIGHT_VOCAB = out
    return out

_FAMILY_LABEL = {"subject": "subject appearance", "act": "pose/action",
                 "inter": "interaction between subjects", "style": "art style",
                 "genre": "genre/setting", "scene": "location/scene",
                 "light": "lighting", "fx": "visual effect"}

# WRONG-BODY-HAIR HYGIENE: a head-hair concept must never map to the
# pubic/facial/armpit families the substring scope would let through
_NOT_HEAD_HAIR = ("pubic hair", "facial hair", "armpit hair",
                  "leg hair", "arm hair", "chest hair", "body hair")

# DESCRIPTION, NOT VIEW (the author's, body rule 13 extended): a body slot
# DESCRIBES the part; tags encoding visibility, exposure or handling are
# viewpoint/act facts with their own spice ('ass visible through thighs'
# -- front view, thigh gap, measured floor sensitive -- was mapped from
# 'softly rounded with a slight curve')
_NOT_DESCRIPTIVE = ("visible", "through", "peek", "grab", "slip",
                    "spill", "focus", "exposed", "flash", "out of")

def _slot_scope(cands_list, anchor, strict=False):
    """slot discipline for anchored concepts: an eye-slot concept may only
    map to eye tags, a hair-slot concept to hair tags -- 'gold flecks'
    inside an eye descriptor must not become the bare 'gold' tag. STRICT
    (hair/eyes, the one_of families) empties the list when nothing is
    in-slot: NL-only is the correct fate, the junk class is worse. Lenient
    (body parts) falls back to the unscoped list ('curvy' is a fair map
    for a body-shape phrase even without the anchor word in it). The bare
    anchor itself ('ears' from an ears descriptor) is dropped whenever a
    modified form is on the list."""
    aw = anchor.split()[0].lower()
    if strict:
        out = [t for t in cands_list if aw in t]
    else:
        # body parts: no anchor-word filter -- 'freckles' is a perfectly
        # good face-slot tag that does not contain the word 'face'
        out = list(cands_list)
    if aw in out:
        out = [t for t in out if t != aw]
    return out
