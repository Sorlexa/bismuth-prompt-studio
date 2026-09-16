"""
fastplan.py -- the no-LLM fast path.

WHY THIS EXISTS, AND WHY IT IS NOT THE v1 GENERATOR. v1 `enhance()` also
runs without an LLM, but it was abandoned mid-build and is frozen at the
knowledge base of that moment: it never reads the concept library
(tag_glosses, tag_categories), nor genre/medium/location/lighting pools,
nor the artist embeddings, and it emits its own construction order with no
presentation bands and -- measured -- no count tag at all in 60/60 runs.
Wiring that into the UI would ship a second, worse generator.

So the fast path is built the other way round: `bridge.generate()` already
resolves cast, scene type, genre, medium, camera, location, lighting, focus,
interactions and subject STRUCTURE mechanically, before the LLM is ever
called. The model only fills content words and writes prose. This module
supplies those two things without it:

  plan()      fills bridge1's JSON schema from the measured pools
  pick()      bridge2's choice: take the rank-1 candidate the engine
              already ranked (which is what the engine does anyway for
              hair/eyes, because the 8B ignores prefer-first ordering)

Everything downstream is unchanged, so the fast path inherits the whole
verifier chain: candidate verification, flag gates, arity, contradictions,
safety floors, the presentation band sort and the count tags.
"""

import json
import re
import os

from promptstudio import paths as _paths
from promptstudio.engine import slots as sm
from promptstudio.engine.tagnet import (SPECIES_RE as _SPECIES,
                                        NEVER_PROPOSE as _NEVER)

_TAGS = None        # tag -> danbooru post count
_FLAGS = None       # tag -> set(flags) from the concept library
_BY_FLAG = None     # flag -> [(tag, count)] sorted by popularity

# Slots the engine asks for, and how to find vocabulary for each. Hair and
# eyes are word-shaped families (danbooru names them '<colour> hair'), so a
# suffix match is exact and needs no taxonomy. Everything else comes from
# the concept library's own semantic flags -- the same source bridge 2
# gates on, which is the whole point: one knowledge base, not two.
_SUFFIX_SLOTS = {
    "eyes": (" eyes", ("closed eyes", "glowing eyes")),
}
_HAIR_COLOUR = re.compile(r"^[a-z\- ]+ hair$")
_HAIR_LEN = ("long hair", "short hair", "medium hair", "very long hair")
# every word that makes a tag ABOUT HAIR -- 'half updo' and 'hair ornament'
# were drawn as held objects; one vocabulary for "this is hair"
_HAIR_WORDS = re.compile(r"(?:^|\W)(hair|bangs|ponytail|twintails|braids?|bun|updo|"
                         r"drills?|sidelocks?|ahoge|intakes|fringe|pompadour|"
                         r"mohawk|dreadlocks|twin tails)(?:$|\W)")
_HAIR_STYLE = ("ponytail", "twintails", "braid", "bob cut", "hair bun",
               "side ponytail", "messy hair", "wavy hair", "straight hair",
               "curly hair", "hime cut", "twin braids", "low ponytail")

# clothing sub-families, matched inside the library's `clothing` flag
_WEAR = {
    "neckwear": ("scarf", "necklace", "choker", "necktie", "ribbon",
                 "collar", "bowtie", "pendant"),
    "headwear": ("hat", "cap", "beret", "hood", "helmet", "headband",
                 "crown", "tiara", "hairband", "veil"),
    "legwear": ("thighhighs", "pantyhose", "socks", "stockings", "kneehighs",
                "leggings", "tights", "boots"),
    "accessory": ("gloves", "glasses", "earrings", "bracelet", "belt",
                  "bag", "umbrella", "ring", "watch"),
}
_NOT_OUTFIT = ("hair", "eyes", "thighhighs", "pantyhose", "socks", "hat",
               "cap", "gloves", "scarf", "necklace", "choker", "boots")

# skin/pelt colouring reads as a species trait but is not in SPECIES_RE,
# and drawing-error tags ('fewer digits') are only half-covered by
# NEVER_PROPOSE. Neither belongs in a rolled prompt.
_NONHUMAN = re.compile(
    r"\b(fur|pelt|scales?|paws?|claws?|muzzle|snout|tentacles?)\b|"
    r"\b(skeleton|skull|patchwork skin|stitches|undead|corpse|gore)\b|"
    r"^(grey|gray|green|blue|purple|red|black|white) skin$|"
    r"\b(fewer|extra|missing) (digits|fingers|limbs|arms|legs)\b", re.I)

# EXTREMES ARE A REQUEST, NOT A DEFAULT. Nothing stops 'gigantic breasts'
# at the safe level -- it is not explicit -- but rolling it unprompted is
# the mean-regression the LLM path avoids by taste. The mechanical path
# needs the rule written down: body extremes only when the prompt asked.
_EXTREME = re.compile(
    r"\b(huge|gigantic|massive|enormous|giant|hyper)\b", re.I)

_MALE_TAG = re.compile(r"\b(male|boy|men|man|masculine|pectorals|penis|abs)\b", re.I)
_FEMALE_TAG = re.compile(r"\b(female|girl|women|woman|feminine|breasts|pussy|vagina)\b", re.I)

# THE MOODS ARE THE ARTISTS' OWN (the author, 2026-09-18: the prose writes
# "The mood is ..." from a hand list, while the descriptions name 42 moods the
# artists are actually described with). The look glossary owns them; a mood
# above the picture's spice level is not written (no 'erotic' at safe), and
# the hand list stays as the fallback when there is no glossary yet.
_MOODS = ("serene", "warm", "quiet", "lively", "wistful", "tense",
          "playful", "solemn", "dreamy", "melancholy", "intimate")
_MOOD_CACHE = {}


def moods(level="safe"):
    """-> the moods the prose may write at this spice level"""
    if level in _MOOD_CACHE:
        return _MOOD_CACHE[level]
    out = []
    try:
        from promptstudio.engine import slots as sm
        from promptstudio.library import looks as _looks
        rank = sm.SPICE_ORDER.get(str(level).lower(), 0)
        allowed = {lv for lv, i in sm.SPICE_ORDER.items() if i <= rank}
        for name, e in (_looks.vocabulary() or {}).items():
            if name.endswith(" mood") and (e.get("level") or "safe") in allowed:
                out.append(name[:-len(" mood")])
    except Exception:
        out = []
    _MOOD_CACHE[level] = tuple(sorted(out)) or _MOODS
    return _MOOD_CACHE[level]


def _load():
    global _TAGS, _FLAGS, _BY_FLAG
    if _BY_FLAG is not None:
        return
    _TAGS = {}
    try:
        with open(_paths.data("danbooru_tags.json"), encoding="utf-8-sig") as f:
            _TAGS = {k.replace("_", " ").lower(): v
                     for k, v in json.load(f).items()}
    except Exception:
        pass
    _FLAGS = {}
    try:
        with open(_paths.data("tag_glosses.json"), encoding="utf-8") as f:
            for t, e in (json.load(f).get("glosses") or {}).items():
                _FLAGS[t.lower()] = set(e.get("f") or [])
    except Exception:
        pass
    _BY_FLAG = {}
    for t, fl in _FLAGS.items():
        n = _TAGS.get(t, 0)
        if n < 2000:                 # the long tail is not worth rolling
            continue
        for f in fl:
            _BY_FLAG.setdefault(f, []).append((t, n))
    for f in _BY_FLAG:
        _BY_FLAG[f].sort(key=lambda kv: -kv[1])


def _gated(tag):
    """the concept library's hard gates -- never roll these unprompted.

    `theme` (added by tools/build/reflag_glosses.py) is a narrative,
    relationship or fetish CONCEPT -- 'netorare', 'cosplay', 'flashback',
    'age regression' -- typed by the user, never drawn for them."""
    return bool({"meme", "text", "format", "emote", "symbol", "theme"}
                & _FLAGS.get(tag, set()))


def _spice_never():
    """everything RULED OUT of the dice (the author's standing rules, stated
    once in the writers): the spice table's never-list (youth,
    mutilation, torture, scat) and its typed-only list (bdsm, rape,
    incest, animal play, exhibitionism...), and the body table's ruled
    builds (plump, fat, skinny). A tag here is never rolled by any draw,
    whatever its measured floor says; typed input is unaffected."""
    try:
        from promptstudio.engine import bridge as _lb
        return _lb.ruled_out()                 # one owner, in the bridge
    except Exception:
        return set()


def _bare_body(d, used):
    """the subject is undressed (2026-09-16): every garment on a body slot
    leaves the outfit, canon or rolled -- 'a girl taking a bath' kept her
    persona's jacket and shirt under 'nude'; accessories stay, as a naked
    subject may wear them"""
    from promptstudio.engine import bridge as _lbx
    before = list(d.get("outfit") or [])
    keep = _lbx.strip_body_garments(before + ["nude"])     # the state is being added: strip as if it were
    keep = [t for t in keep if t != "nude" or "nude" in before]
    for t in before:
        if t not in keep:
            used.discard(t)
    d["outfit"] = keep


def _walk_measured(rng, states, census, level, banks, used, taken=(), veto=None, normalise=True):
    """one draw walking MEASURED shares (the author's 2026-09-15: the face is
    drawn as the booru shows it, not as the co-occurrence network's hubs).
    `states` are (tag, share of the 1girl universe), largest first; every
    gate the other draws honour applies (floor, safety, the concept
    library's gates, the sex veto, what is already spent). normalise:
    P(t) = share / sum (the caller decided a draw happens); else the raw
    shares are walked and the remainder is 'none' (the mouth: .67 of
    pictures carry a state)."""
    cands = [(t, f) for t, f in (states or []) if t not in used and t not in taken
             and t in sm.SAFETY_FLOOR and not _gated(t) and not (veto and veto(t))
             and sm.content_allowed(t, census, level, banks)]
    if not cands:
        return None
    tot = sum(f for _, f in cands) if normalise else 1.0
    if tot <= 0:
        return None
    r = rng.random() * tot
    for t, f in cands:
        if r < f:
            return t
        r -= f
    return None


def _draw(rng, pool, census, level, banks, used, n=1, veto=None,
          net=None, seeds=None):
    """Draw n tags for a slot.

    CONTEXT FIRST. A popularity draw over a whole slot produces tags that
    are individually common and jointly absurd -- 'cat paws' and 'kabedon'
    at a picnic. So the co-occurrence network proposes against the scene's
    own seeds, and the slot's vocabulary is used only to TYPE the result.
    Popularity is the fallback for when the network has nothing to say.

    Every gate the LLM path honours still applies: safety, arity, the
    concept library's flags, and what the caller has already spent.
    """
    out = []
    if not pool:
        return out
    allowed = {t for t, _c in pool}
    seeded_species = any(_SPECIES.search(x) for x in (seeds or ()))
    _asked_extreme = any(_EXTREME.search(x) for x in (seeds or ()))

    def _ok(t):
        # ONE GATE FOR BOTH SOURCES. propose() enforces NEVER_PROPOSE, the
        # species rule and the popularity floor internally; the fallback
        # draw reaches the same pools WITHOUT those checks, so it has to
        # apply them explicitly or it re-admits exactly what the network
        # refuses -- measured: 'body fur' and 'fewer digits' at a picnic.
        if t in used or t in out or _gated(t):
            return False
        # UNMEASURED IS TYPED-ONLY, HERE TOO (the author's 2026-09-05): the bank
        # draws still trusted the regex fallback floor -- 'sweaty breasts'
        # and 'crossdressing' rolled into a safe prompt with no measured
        # floor at all. A tag the floor table has not measured is never
        # rolled; typed input is unaffected. The floors for every drawable
        # pool are measured by tools/build/build_safety_floor.py --tags-file.
        if t not in sm.SAFETY_FLOOR:
            return False
        if _NEVER.match(t) or t in _spice_never():
            return False
        if not seeded_species and (_SPECIES.search(t) or _NONHUMAN.search(t)):
            return False
        if _EXTREME.search(t) and not _asked_extreme:
            return False
        if veto and veto(t):
            return False
        try:
            return sm.content_allowed(t, census, level, banks)
        except Exception:
            return True

    # 1. what the scene actually implies -- SAMPLED, not taken in order.
    # The network's list is ranked by activation and, for a slot the scene
    # says little about, the same two or three of the pool's members sit
    # at its top for every seed: 'a girl in a park' drew bob cut / twin
    # braids / hime cut and nothing else in 60 seeds, while the popularity
    # roll below spreads over all thirteen styles. Everything the network
    # offers is still scene-implied; the pick among those offers is now a
    # popularity-weighted roll, the same weighting the fallback uses.
    if net is not None and seeds:
        try:
            _offered = []
            for t in net.propose(seeds, n * 12, exclude=tuple(used), rng=rng):
                if t in allowed and t not in _offered and _ok(t):
                    _offered.append(t)
            _cnt = dict(pool)
            while _offered and len(out) < n:
                _w = [max(1.0, _cnt.get(t, 0)) for t in _offered]
                t = rng.choices(_offered, weights=_w)[0]
                _offered.remove(t)
                out.append(t)
        except Exception:
            pass
    if len(out) >= n:
        return out

    # 2. fallback: weighted by popularity within the slot
    # THE COUNT IS THE WEIGHT (the author's 2026-09-13: 'flat ass' on .12 of the
    # lines, 'muscular' .17, 'no nose' .04 -- rare tags very overrated).
    # The .35 exponent flattened a 40:1 count ratio to 3.6:1; measured
    # beats reasoned, and the variety comes from the context (the
    # network's proposals, the slots), not from flattening the booru.
    top = pool[:400]
    weights = [max(1.0, c) for _t, c in top]
    names = [t for t, _c in top]
    for _ in range(n * 14):
        if len(out) >= n:
            break
        t = rng.choices(names, weights=weights)[0]
        if _ok(t):
            out.append(t)
    return out


def _lb_vocab():
    from promptstudio.engine import vocab as _vc
    return _vc._vocab()


def _flag_pool(flag, include=None, exclude=()):
    _load()
    pool = _BY_FLAG.get(flag) or []
    if include:
        pool = [(t, c) for t, c in pool if any(w in t for w in include)]
    if exclude:
        pool = [(t, c) for t, c in pool if not any(w in t for w in exclude)]
    return pool


# Anatomy that is never a held object and never a "pose".
_ANATOMY = re.compile(
    r"\b(breasts?|nipples?|areola|pussy|vagina|cervix|penis|testicles|"
    r"ass|butt|anus|crotch|groin|navel|thighs?|armpits?|skeleton|"
    r"skin|organs?|womb|uterus)\b", re.I)

# Body-part slots the engine asks for, and the words a tag must mention to
# be about that part. A tag drawn for 'face' has to be about a face --
# filling the slot with any body-flagged tag produced 'ass -> bright
# pupils' and 'markings -> cervix', which bridge 2 then had to rescue.
_PART_WORDS = {
    "face": ("face", "cheek", "chin", "jaw", "dimple",
             "eyebrow", "eyelash", "nose", "lip", "mouth", "teeth"),
    "eyes": ("eye", "pupil", "sclera", "iris", "eyelash"),
    "hair": ("hair", "bang", "ahoge", "sidelock"),
    "markings": ("tattoo", "scar", "mole", "birthmark", "freckle",
                 "marking", "bandage", "makeup", "lipstick"),
    "body shape": ("slender", "curvy", "muscular", "petite", "plump",
                   "toned", "athletic", "wide hips", "thin", "tall",
                   "short", "build"),
    "hands": ("hand", "finger", "nail", "palm", "wrist"),
    "legs": ("leg", "thigh", "knee", "calf", "ankle"),
    "skin": ("skin", "tan", "pale", "complexion", "freckle"),
}


_EXCLUSIVE = ("breast_size", "hair_length", "hair_color", "eye_color", "skin", "hair_top")
_EXCL_FAMS = {"data": None}


def _exclusive_family(tag):
    """the other members of the tag's exclusive body family (one breast
    size, one hair length...), from the body table; empty when none"""
    if _EXCL_FAMS["data"] is None:
        m = {}
        try:
            from promptstudio.engine import bridge as _lb
            fams = (_lb._body_table() or {}).get("families") or {}
            for k in _EXCLUSIVE:
                items = {str(t).lower() for t in (fams.get(k) or {}).get("items") or {}}
                for t in items:
                    m[t] = items
        except Exception:
            pass
        _EXCL_FAMS["data"] = m
    return set(_EXCL_FAMS["data"].get(str(tag or "").lower(), ()))


_APP_SHARE = {"data": None}


def _appearance_share():
    """P(any appearance tag | 1girl): the appearance pool's counts summed
    over the 1girl count (cached live count), capped at .5; .5 when the
    count is unknown (offline: the old always-on behaviour, halved)"""
    if _APP_SHARE["data"] is None:
        try:
            from promptstudio.engine import bridge as _lb
            tot = float(_lb._count_cached("1girl") or 0)
            pool = _person_pools().get("appearance") or []
            _APP_SHARE["data"] = min(0.5, sum(float(c or 0) for _t, c in pool) / tot) if tot else 0.5
        except Exception:
            _APP_SHARE["data"] = 0.5
    return _APP_SHARE["data"]


_GAZE_RE_FP = re.compile(r"\b(looking|gaze|glance|stare|staring|eye contact|sideways glance)\b")
_OPEN_EYES = frozenset(("wide-eyed", "open eyes", "half-closed eyes", "rolling eyes", "crazy eyes",
                        "empty eyes", "glowing eyes", "heart-shaped pupils", "slit pupils"))
_SHUT = {}


def _eyes_shut(act):
    """P(closed eyes | act) >= .5, live-cached (sleeping .63); False when
    unmeasured or no act"""
    if not act:
        return False
    a = str(act).lower()
    if a not in _SHUT:
        try:
            from promptstudio.engine import bridge as _lb
            sh = _lb._pair_share(a, "closed eyes")
            _SHUT[a] = bool(sh is not None and sh >= 0.5)
        except Exception:
            _SHUT[a] = False
    return _SHUT[a]


_REVEAL_RE = re.compile(r"(pull|lift|peek|open |unbutton|unzip|undress|dressing|strap slip|off shoulder|"
                        r"shot$|upskirt|slip$|lowered|removed|tug|hanging breasts|torn)")
_UNDER = frozenset(("panties", "bra", "underwear", "lingerie", "babydoll", "boxers", "briefs",
                    "male underwear", "sports bra", "camisole", "bloomers", "thong", "g-string"))
_OUTER_SLOTS = ("top", "bottom", "uniform", "swim", "traditional")


def _hide_covered_underwear(d, opts):
    """drop an underwear item beside an outer garment that hides it: the
    live pair lift P(inner | outer) / P(inner | 1girl) under 1, unless a
    reveal word is on the subject or in the level's injected content"""
    outfit = list(d.get("outfit") or [])
    inner = [t for t in outfit if t in _UNDER or t.split()[-1] in _UNDER]
    if not inner:
        return
    try:
        from promptstudio.engine import bridge as _lb
        slots = (_lb._clothes_table() or {}).get("slots") or {}
        outers = [t for t in outfit if t not in inner and any(
            t in (slots.get(sl) or {}) or t.split()[-1] in (slots.get(sl) or {}) for sl in _OUTER_SLOTS)]
        if not outers:
            return
        words = outfit + list((d.get("body") or {}).get("expression") or []) + list(d.get("self_actions") or []) \
            + list(opts.get("_required_content") or []) + list(opts.get("_user_tags") or [])
        if any(_REVEAL_RE.search(str(w).lower()) for w in words):
            return
        tot1 = float(_lb._count_cached("1girl") or 0)
        for u in inner:
            base_n = _lb._count_cached("1girl " + _lb._q(u))
            if not (tot1 and base_n):
                continue
            base = base_n / tot1
            for o in outers:
                on = float(_TAGS.get(o) or 0)
                pair = _lb._count_cached(_lb._q(o) + " " + _lb._q(u))
                if not on or pair is None:
                    continue
                if (pair / on) / base < 1.0:
                    d["outfit"] = [t for t in d["outfit"] if t != u]
                    break
    except Exception:
        return


def _lb_scene_details(place, rng):
    try:
        from promptstudio.engine import bridge as _lb
        return list(_lb.scene_details(place, rng))
    except Exception:
        return []


def _body_pool(part):
    """body-flagged tags that are actually about `part`"""
    words = _PART_WORDS.get(str(part).lower())
    if not words:
        words = (str(part).lower(),)
    if str(part).lower() == "hair":
        return _hair_style_pool()
    # MARKS HAVE ONE OWNER (2026-09-06): the body table rolls them at a
    # measured, detail-scaled share; 'scar on face' is not a face and
    # 'mole under eye' is not an eye
    if str(part).lower() != "markings":
        return _flag_pool("body", include=words, exclude=_PART_WORDS["markings"])
    return _flag_pool("body", include=words)


def _person_free(pool):
    """drop occupations from a GARMENT pool.

    the author's: "maid is not a clothes it is an occupation/profession". The
    concept library flags `maid` as BOTH person and clothing, so an outfit
    draw could dress Hatsune Miku as a maid -- naming a role, not a
    garment. The real garments are flagged clothing alone ('maid headdress',
    'maid apron', 'school uniform'), so the rule is simply: a tag that also
    names a PERSON is not an outfit item. Ten tags are dual-flagged --
    maid, pharaoh, playboy bunny and seven '(cosplay)' entries, none of
    which belongs in a clothing slot.

    Occupation is its own field in the plan schema; that is where a role
    belongs.
    """
    return [(t, c) for t, c in pool if "person" not in _FLAGS.get(t, set())]


_TAG_GENDER = {"mtime": 0, "data": {}}


def _tag_female_share(tag):
    """-> measured female share of a drawable tag (tag_gender.json), or
    None when unmeasured or under the post floor. Re-read on change, so a
    measurement still running in the background is picked up as it lands."""
    p = _paths.data("tag_gender.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _TAG_GENDER["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                blob = json.load(f)
            _TAG_GENDER["data"] = blob.get("tags") or {}
            _TAG_GENDER["floor"] = int(blob.get("min_posts") or 100)
            _TAG_GENDER["mtime"] = mt
    except (OSError, ValueError):
        return None
    rec = _TAG_GENDER["data"].get(str(tag or "").lower())
    if not rec or rec.get("female_share") is None:
        return None
    if int(rec.get("posts") or 0) < _TAG_GENDER.get("floor", 100):
        return None
    return float(rec["female_share"])


_FREE_ROLL_SLOTS = frozenset(("hair", "hair_style", "eyes"))

# hair tags that are a COLOUR modifier or a passing STATE, not a cut or a
# tie -- the colour slot and the contradiction table own those
_HAIR_NOT_STYLE = re.compile(
    r"\b(multicolored|streaked|gradient|two-tone|split-color|colored|inner|"
    r"wet|floating|blowing|facial|sideburns|stubble|beard|mustache|pubic|"
    r"armpit|chest|body|over|between|behind|in own|alternate|absurdly|"
    r"extremely|past|shiny|glowing|iridescent|blush|sex|severed|detached|"
    r"removed|no |visible|through|day|holder|roots|spread out)\b")
_HAIR_STYLE_POOL = None


def _hair_style_pool():
    """Every hairstyle the concept library flags `hair` (reflag_glosses.py)
    with enough posts to be worth rolling, minus colours, lengths and
    states. Replaces a hand-typed list of thirteen; 'wolf cut', 'drill
    hair' and 'hair intakes' were never rollable before."""
    global _HAIR_STYLE_POOL
    if _HAIR_STYLE_POOL is None:
        _load()
        skip = set(_HAIR_COLOURS) | set(_HAIR_LEN)
        _HAIR_STYLE_POOL = sorted(
            ((t, _TAGS.get(t, 0)) for t, fl in _FLAGS.items()
             if "hair" in fl and t not in skip
             and _TAGS.get(t, 0) >= 20000
             and not _HAIR_NOT_STYLE.search(t)
             and not _HAIR_COLOUR_WORD.search(t)),
            key=lambda kv: -kv[1])
    return _HAIR_STYLE_POOL


_HAIR_COLOUR_WORD = re.compile(
    r"\b(blonde|brown|black|blue|purple|pink|red|white|grey|gray|silver|"
    r"green|orange|aqua|lavender|platinum|yellow|maroon|rainbow)\b")


def _slot_pool(slot):
    _load()
    if slot == "hair":
        # THE COLOUR SLOT DRAWS COLOURS. '^... hair$' also matched 'very
        # long hair', 'wet hair', 'gradient hair', 'wavy hair' -- so the
        # length prefix doubled ("very long hair, very long hair, braid")
        # and a texture stood where the colour should be. _HAIR_COLOURS is
        # the one colour list (the typed-colour reader uses it too).
        return sorted(((t, _TAGS.get(t, 0)) for t in _HAIR_COLOURS
                       if t in _FLAGS), key=lambda kv: -kv[1])
    if slot == "hair_style":
        return _hair_style_pool()
    if slot == "eyes":
        suf, drop = _SUFFIX_SLOTS["eyes"]
        return sorted(((t, _TAGS.get(t, 0)) for t in _FLAGS
                       if t.endswith(suf) and t not in drop
                       and _TAGS.get(t, 0) > 5000), key=lambda kv: -kv[1])[:40]
    if slot == "outfit":
        return _person_free(_flag_pool("clothing", exclude=_NOT_OUTFIT))
    if slot in _WEAR:
        return _person_free(_flag_pool("clothing", include=_WEAR[slot]))
    return []


_PERSON_POOLS = None


def _person_pools():
    """the person-flag split: occupations / relationships / appearance.

    the author's: "both relationships and appearance tags are usefull". They are
    different slots, not one -- an appearance term describes ONE subject
    ('tomboy', 'dark-skinned female'), a relationship needs TWO ('sisters',
    'husband and wife'), and only an occupation answers "what do they do".
    """
    global _PERSON_POOLS
    if _PERSON_POOLS is None:
        _PERSON_POOLS = {"occupations": [], "relationships": [],
                         "appearance": []}
        try:
            with open(_paths.data("occupation_pool.json"),
                      encoding="utf-8") as f:
                b = json.load(f)
            for k in _PERSON_POOLS:
                _PERSON_POOLS[k] = sorted((b.get(k) or {}).items(),
                                          key=lambda kv: -kv[1])
        except Exception:
            pass
    return _PERSON_POOLS


def _occupations():
    """occupations only -- see _person_pools for why that is not the whole
    person flag."""
    _load()
    return _person_pools()["occupations"] or (_BY_FLAG.get("person") or [])


def _occupation_from_prompt(seeds):
    """an occupation the user actually named"""
    occ = {t for t, _c in _occupations()}
    for t in (seeds or ()):
        if t in occ:
            return t
    return None


def _occupation_from_clothes(seeds, net):
    """the profession the prompt's GARMENTS imply.

    'apron, maid headdress' is a maid without the word ever appearing.
    Only a clear winner counts -- a shirt implies nothing.
    """
    if not net:
        return None
    worn = [t for t in (seeds or ())
            if "clothing" in _FLAGS.get(t, set())]
    if not worn:
        return None
    best, score = None, 0.0
    for occ, _c in _occupations()[:400]:
        v = 0.0
        for w in worn:
            try:
                v += net.edge(w, occ)
            except Exception:
                pass
        if v > score:
            best, score = occ, v
    return best if score >= 0.12 else None


# how strong a location->occupation edge has to be to count as "this job
# belongs in this room". Measured: nurse/hospital .164, teacher/classroom
# .120, housewife/kitchen .102, and nothing else clears .02.
_LOC_OCC_EDGE = 0.02

# share of rolls taken from the fitting set. Same two-stage shape, and the
# same constant, as the location resolver in bridge.py: mostly the things
# that belong, but never ONLY them.
_OCC_FIT_SHARE = 0.65


def _occupation_fit(genre, loc_name, net, cands):
    """the occupations that actually belong to this genre or this place.

    A pure weighting pass cannot express this. The measured affinities span
    3x at most while the popularity term spans 10x across the pool, so a
    genre's own professions lose to whatever is merely popular -- a medieval
    scene came out full of train conductors, and a hospital produced
    everything except a nurse. So the fitting set is drawn as a SET, the way
    the location resolver already does it, rather than as a nudge that gets
    outvoted.
    """
    from promptstudio.engine import affinity as _af
    fit = {}
    for t, c in cands:
        w = 0.0
        if genre:
            a = _af.weight("genre_occupation", genre, t)
            if a > 1.0:
                w += a - 1.0
        if loc_name and net is not None:
            try:
                e = net.edge(loc_name, t)
            except Exception:
                e = 0.0
            if e >= _LOC_OCC_EDGE:
                w += 4.0 * e
        if w > 0.0:
            # popularity still whispers inside the fitting set, so a common
            # job wins ties -- it just no longer decides the question
            fit[t] = w * (max(1.0, float(c)) ** 0.12)
    return fit


def _roll_occupation(genre, loc_name, net, rng, census, level, banks, kind="female"):
    """genre and location choose the field; popularity only breaks ties.

    THE TABLE FIRST (the author's lookup tables, 2026-09-04): when the genre
    leaf has an entry in genre_occupations.json, the roll is the table's --
    its `none` share, its list, the measured gender filter -- and nothing
    below runs. A leaf without an entry keeps the roller below."""
    from promptstudio.engine import bridge as _lb
    try:
        got = _lb.occupation_for(
            genre, loc_name, kind, rng,
            allowed=lambda t: sm.content_allowed(t, census, level, banks))
        if got is not False:
            return got
    except Exception:
        pass
    from promptstudio.engine import affinity as _af
    cands = _occupations()[:300]
    if not cands:
        return None
    fit = _occupation_fit(genre, loc_name, net, cands)
    wide_names = [t for t, _c in cands]
    wide_w = []
    for t, c in cands:
        x = max(1.0, float(c)) ** 0.25          # a mild popularity floor
        x *= _af.weight("genre_occupation", genre, t) if genre else 1.0
        wide_w.append(x)
    fit_names = list(fit)
    fit_w = [fit[t] for t in fit_names]
    for _ in range(12):
        if fit_names and rng.random() < _OCC_FIT_SHARE:
            pick = rng.choices(fit_names, weights=fit_w)[0]
        else:
            pick = rng.choices(wide_names, weights=wide_w)[0]
        if _gated(pick):
            continue
        try:
            # the table's deny list holds for every roller ('booth babe'
            # rolled onto a maid from this path, golden 2026-09-04)
            if pick in set((_lb._occupation_table() or {}).get("deny") or ()):
                continue
            if not _lb.occupation_fits(pick, kind):   # the one gender rule
                continue
        except Exception:
            pass
        try:
            if not sm.content_allowed(pick, census, level, banks):
                continue
        except Exception:
            pass
        return pick
    return None


def _seeds(base, cast, opts, banks):
    """what this scene is about, as network seeds -- the user's own tags
    weigh most, then the resolved location/genre/medium."""
    seeds = {}
    try:
        from promptstudio.engine import enhancer as pe
        for t in (pe.parse_input(base or "", banks)[0] or []):
            seeds[str(t).lower()] = 1.0
    except Exception:
        pass
    loc = opts.get("_location") or (None, None, None)
    if loc[1]:
        seeds[str(loc[1]).lower()] = 1.0
    gen = opts.get("_genre") or (None, None)
    if gen[1]:
        seeds[str(gen[1]).lower()] = 0.8
    med = opts.get("_medium") or (None, None)
    if med[1]:
        seeds[str(med[1]).lower()] = 0.6
    lig = opts.get("_lighting") or {}
    for k in ("time", "weather"):
        if lig.get(k):
            seeds[str(lig[k]).lower()] = 0.5
    return seeds


# 'take your pick' was drawn as a POSE: a meme tag names a joke, not a
# body position. The gloss library flags memes, text and symbols (the
# retrieval layer already refuses them); the dice refuse them too.
_NOT_DRAWABLE = frozenset(("meme", "text", "symbol"))
_MEME_TEXT = {}


def _is_meme(tag):
    """'take your pick' carries only the `pose` flag, but its gloss says
    what it is: a meme. The gloss text is the data; read it."""
    t = str(tag or "").lower()
    if t in _MEME_TEXT:
        return _MEME_TEXT[t]
    try:
        from promptstudio.engine.vocab import _gloss_of
        g, _fl = _gloss_of(t)
        _MEME_TEXT[t] = bool(g) and ("meme" in str(g).lower() or "joke" in str(g).lower())
    except Exception:
        _MEME_TEXT[t] = False
    return _MEME_TEXT[t]

def _typed_action(base, cast, banks):
    """-> 'fellatio to the viewer' / 'sex' / '' from the user's own text"""
    try:
        from promptstudio.engine import enhancer as _pe
        tags = [str(t).lower() for t in (_pe.parse_input(base or "", banks)[0] or [])]
    except Exception:
        return ""
    acts = [t for t in tags if (sm.arity_of(t, banks) or 1) >= 2
            or sm.EXPLICIT_RE.search(t) or t in ("masturbation", "kissing", "hug")]
    # A VIEWPOINT IS THE CAMERA'S, NOT AN ACT: 'futanari pov' matches the
    # explicit stem and was picked as the act ("The girl is shown futanari
    # pov"); the pov family is owned by the camera resolver.
    # a futa-named tag says WHO the subject is, not what they do: "The
    # figure is shown futanari" was the typed act for "a futanari and a girl"
    acts = [t for t in acts if t not in sm.PAIRING_TAGS and not t.startswith("1")
            and t not in ("penis", "pussy", "nipples", "oral", "hetero")
            and not t.endswith("pov") and not sm.FUTA_NAMED.search(t)]
    if not acts:
        return ""
    act = acts[0]
    heads = sum(v for k, v in (cast or {}).items()
                if k in ("female", "male", "futa", "other") and isinstance(v, int))
    if "pov" in tags and heads <= 1 and (sm.arity_of(act, banks) or 1) >= 2:
        act += " to the viewer"
    return act


_HAIR_COLOURS = ("blonde hair", "brown hair", "black hair", "blue hair", "purple hair",
                 "pink hair", "red hair", "white hair", "grey hair", "silver hair",
                 "green hair", "orange hair", "aqua hair", "light brown hair",
                 "light blue hair", "dark blue hair", "lavender hair", "platinum blonde hair")
_EYE_COLOURS = ("blue eyes", "red eyes", "green eyes", "brown eyes", "purple eyes",
                "yellow eyes", "pink eyes", "aqua eyes", "orange eyes", "grey eyes",
                "black eyes", "golden eyes", "amber eyes", "silver eyes", "white eyes")


def _typed_colours_for(base, banks, si):
    """-> (hair colour, eye colour) the text types for subject `si`."""
    try:
        from promptstudio.engine import enhancer as _pe
        from promptstudio.engine import bridge as _lb
        tags = [str(t).lower() for t in (_pe.parse_input(base or "", banks)[0] or [])]
        spans = _lb._subject_spans(base or "")
    except Exception:
        return "", ""
    # TEXT ORDER, not tag order: the parser appends colours in the order
    # its rules fire, and 'red hair' (from a hyphen rule) came out ahead
    # of 'blonde hair' (from the scan) -- the elf turned red-haired
    low = str(base or "").lower()

    def _pos(t):
        w = t.split()[0]
        i = low.find(w)
        return i if i >= 0 else 10 ** 6
    hair = sorted((t for t in tags if t in _HAIR_COLOURS), key=_pos)
    eyes = sorted((t for t in tags if t in _EYE_COLOURS), key=_pos)
    n_beings = sum(c for _n, c in spans) or 1
    if n_beings <= 1:
        return (hair[0] if hair and si == 0 else "", eyes[0] if eyes and si == 0 else "")
    return (hair[si] if si < len(hair) else "", eyes[si] if si < len(eyes) else "")


def plan(base, cast, level, mode, opts, rng, subjects, tier_meta,
         interactions, banks):
    """bridge1's JSON schema, filled mechanically. Same shape, no model."""
    _load()
    net = (banks or {}).get("_net")
    seeds = _seeds(base, cast, opts, banks)
    # the cast's counts, plus the shape the content gate reads (see
    # slots.census_from_cast for what handing it the bare cast did)
    census = dict(cast)
    census.update(sm.census_from_cast(cast, list(seeds or ())))
    _typed_act = _typed_action(base, cast, banks)
    out_subjects = []
    used = set()
    for si, s in enumerate(subjects):
        d = {"who": "", "hair": "", "eyes": "", "outfit": [], "body": {},
             "pose": "", "self_actions": [], "held_object": None,
             "age": "", "alias": "", "race": "", "occupation": "", "fashion_style": "",
             "descriptors": []}
        kind = s.get("kind") or "female"
        d["kind"] = kind
        # A PERSONA IS NAMED BY ITS DISPLAY NAME, never its tag: the prose
        # template read the lowercase tag as a noun and wrote "The enoshima
        # junko; the enoshima junko is shown...". Same rule as
        # bridge._display_name (not imported: bridge imports this module).
        # TYPED APPEARANCE REACHES THE PROSE. "a blonde elf ranger and a
        # red-haired dwarf" put `blonde hair` and `red hair` in the tag
        # line and nothing in the paragraph. Colour words are handed to
        # the subject nouns in text order: the first colour describes
        # the first being.
        _typed_hair, _typed_eyes = _typed_colours_for(base, banks, si)
        _pn = str(s.get("persona") or "").split(" (")[0].strip()
        d["who"] = (" ".join(w[:1].upper() + w[1:] for w in _pn.split()) if _pn
                    else {"female": "girl", "male": "boy",
                          "futanari": "futanari"}.get(kind, "figure"))
        # A TAG CARRIES ITS OWN GENDER. 'toned male' landed on a female
        # subject because the body pool does not know whose body it is.
        # Word boundaries matter here: 'female' contains 'male'.
        _wrong = (_MALE_TAG if kind == "female" else
                  _FEMALE_TAG if kind == "male" else None)

        def _gender_veto(t, _w=_wrong, _k=kind):
            # the tag's own name first ('male ...' / 'female ...'), then
            # the MEASURED lean: 'lolita hairband' names no sex but is
            # worn by girls in 99% of its posts, and it landed on a boy
            if _w and _w.search(t):
                return True
            # a futa-named tag belongs to a futanari and nobody else:
            # `full-package futanari` was drawn as a girl's SKIN
            if _k != "futanari" and sm.FUTA_NAMED.search(t):
                return True
            share = _tag_female_share(t)
            if share is None:
                return False                  # unmeasured is neutral
            if _k == "male":
                return share >= 0.95
            if _k in ("female", "futanari"):
                return share <= 0.05
            return False
        # RACE -- the genre's table (the author's: genre only, never the place),
        # for an UNNAMED subject the user gave no race or species to. A typed
        # race word ('an elf in the forest', 'dragon girl') is law and is
        # already on the tag line; a persona has its canon.
        try:
            from promptstudio.engine import bridge as _lb
            _typed_race = ("+race:" in str((cast or {}).get("source") or "")
                           or any(sm.subject_kind(str(t).lower(), base) in ("race", "humanoid")
                                  for t in (seeds or ()))
                           or bool(_lb._species_phrase(str(base or "").lower())))
            if not _pn and not _typed_race:
                _rt = _lb.race_for(
                    (opts.get("_genre") or (None, None))[1], kind, rng,
                    allowed=lambda t: sm.content_allowed(t, census, level, banks))
                if _rt:
                    d["race"] = _rt
                    d["who"] = _lb.race_noun(_rt, d["who"])
        except Exception:
            pass
        # THE USER'S OWN WORDS NAME THE SUBJECT (the author's 2026-09-15: "sexy
        # girl should be the phrase 'sexy girl'"): a typed run of
        # subjective words with this subject's noun as its head ('sexy
        # girl', 'cute elegant woman') is the subject's name in the prose
        # ("At home, the sexy girl has ..."), once, for the first subject
        # of that noun. The same phrase sits on the tag line.
        try:
            from promptstudio.library import concepts as _clx
            _subjw = set(_clx.subjective_words())
            _noun = str(d.get("who") or "").lower()
            _taken = {str(x.get("who") or "").lower() for x in out_subjects}
            for _lo in (opts.get("_leftovers") or []):
                _ws = str(_lo).lower().split()
                if (len(_ws) >= 2 and _ws[-1] == _noun and all(w in _subjw for w in _ws[:-1])
                        and str(_lo).lower() not in _taken):
                    d["who"] = str(_lo).lower()
                    break
        except Exception:
            pass
        # OCCUPATION -- prompt first, then genre/location (see module docs)
        occ = None
        if "occupation" in (s.get("flavor") or []):
            occ = (_occupation_from_prompt(seeds)
                   or _occupation_from_clothes(seeds, net)
                   or _roll_occupation(
                       (opts.get("_genre") or (None, None))[1],
                       (opts.get("_location") or (None, None, None))[1],
                       net, rng, census, level, banks, kind=kind))
            if occ:
                d["occupation"] = occ
        # a profession steers the garments it comes with, but only for
        # slots the prompt has not already decided
        sub_seeds = dict(seeds)
        if occ:
            sub_seeds[occ] = 1.4
        # A TYPED FASHION EXPANDS INTO ITS GARMENTS (the author's: a fashion is
        # "not actual clothes pieces - but a set of specific clothes").
        #
        # PLACED, NOT SEEDED. Seeding was the wrong mechanism: a seed makes
        # tags CONNECTED TO it more likely, so seeding 'leather' produced
        # things that co-occur with leather while the slot still drew a
        # veil. A typed fashion states what IS worn, so its garments are
        # put into the slots they belong to. Slots the prompt or canon has
        # already decided are never overruled.
        _fashion = list(opts.get("_fashion_garments") or ())
        canon = s.get("locked_canon") or {}
        # THE CLOTHES TABLE (the author's last layer, 2026-09-05): the outfit
        # comes from the tree's attire by slot with its measured job / world /
        # place / season / race associations and the level's coverage; the
        # old clothing-bank draw stays where the table has nothing measured.
        # THE CANON GARMENTS ARE WORN (2026-09-14): a typed or persona
        # garment stands in the outfit the prose renders ("wears
        # see-through shirt"); the line carried it, the sentence did not
        for _cg in canon:
            try:
                from promptstudio.engine import bridge as _lb0
                _isg = _lb0._is_garment(str(_cg)) or any(
                    str(_cg).split()[-1] in items for items in
                    ((_lb0._clothes_table() or {}).get("slots") or {}).values())
            except Exception:
                _isg = False
            if _isg and _cg not in d["outfit"]:
                d["outfit"].append(_cg)
                used.add(_cg)
        _table_outfit = False
        if "outfit" in (s.get("slots") or []) and "outfit" not in canon \
                and s.get("tier") != "secondary" and not _fashion:
            try:
                from promptstudio.engine import bridge as _lb
                _loc = opts.get("_location") or (None, None, None)
                _lig = opts.get("_lighting") or {}
                # the activity is decided after the outfit for this subject;
                # the scene's act (rolled before the camera) is the one known
                _areg = ((_lb._activity_table() or {}).get("registry") or {}).get(
                    d.get("activity") or opts.get("_act") or "") or {}
                _ev = (opts.get("_event") or (None, None, {}))[2] or {}
                _ctx = {"kind": kind, "occupation": d.get("occupation") or _occupation_from_prompt(seeds),
                        "genre": (opts.get("_genre") or (None, None))[1],
                        "bucket": (_lb._genre_pool().get((opts.get("_genre") or (None, None))[1] or "") or {}).get("bucket"),
                        "place": _loc[1],
                        "place_class": (((_lb._activity_table() or {}).get("leaves") or {}).get("everyday") or {}).get("place_class", {}).get(_loc[1]),
                        "season": _lig.get("season"), "race": d.get("race") or _typed_race_word(base, seeds),
                        "level": level, "clothes_class": _areg.get("clothes") or _ev.get("clothes"),
                        "nude": bool(_areg.get("clothes") == "none"),
                        "canon": list(d["outfit"]),
                        # the scene's act (typed or rolled before the camera)
                        # leans the garments by its measured lifts (2026-09-11)
                        "act": opts.get("_act"),
                        "framing": (opts.get("_camera") or (None, None, None))[0]}
                _cl = _lb.clothes_for(_ctx, rng,
                                      allowed=lambda t: sm.content_allowed(t, census, level, banks)
                                      and not _gender_veto(t) and t not in used)
                if _ctx.get("nude") and not _cl.get("outfit"):
                    # THE ACTIVITY UNDRESSES (2026-09-15: 'a girl taking a bath'
                    # came out nude in a jacket, a shirt and thighhighs): the
                    # registry says clothes none, the clothes roll gave no
                    # outfit, and the fallback outfit draw dressed her anyway.
                    # The outfit is decided -- none -- and the whole state is
                    # walked by its measured shares.
                    _table_outfit = True
                    _loc8 = opts.get("_location") or (None, None, None)
                    try:
                        _ns8 = _lb.nudity_states(level, place=_loc8[1],
                                                 genre=(opts.get("_genre") or (None, None))[1],
                                                 occupation=d.get("occupation"), whole=True)
                    except Exception:
                        _ns8 = []
                    _bare_body(d, used)
                    _nv8 = _walk_measured(rng, _ns8, census, level, banks, used, taken=d["outfit"], veto=_gender_veto)
                    if _nv8:
                        used.add(_nv8)
                        d["outfit"].append(_nv8)
                    s["slots"] = [x for x in (s.get("slots") or []) if x not in ("legwear", "neckwear")]
                if _cl.get("outfit"):
                    for t in _cl["outfit"] + _cl.get("modifiers", []):
                        if t not in d["outfit"]:
                            d["outfit"].append(t)
                            used.add(t)
                    _table_outfit = True
                    # a place that undressed the subject (the onsen's 'nude')
                    # leaves the body bare: no legwear or neckwear slot after it
                    if any(_lb._is_whole_nudity(t) for t in _cl["outfit"]):
                        s["slots"] = [x for x in (s.get("slots") or []) if x not in ("legwear", "neckwear")]
            except Exception:
                pass
        for slot in (s.get("slots") or []):
            if slot in canon:                      # canon is never overruled
                continue
            # THE CLOTHES TABLE OWNS THE WEAR IT DECIDED (2026-09-17: 'pants,
            # socks' from the table, then 'thighhighs' from this draw; 'neck
            # ribbon' then 'necktie'). The table rolls legwear, neckwear,
            # headwear and accessories by their measured shares and gates
            # them against what is worn; a second draw here overruled a
            # decision already made -- including the table's "none". The
            # draw still runs and is discarded, so the dice after it fall
            # as they did.
            if slot == "outfit" and _table_outfit:
                continue
            _table_owns = _table_outfit and slot in _WEAR
            placed = None
            if _fashion and slot not in ("hair", "hair_style", "eyes"):
                # DRAW FROM THE GARMENT'S VARIANTS, don't place it bare.
                # the author's: "those are general clothes types and its ok I
                # just hope generator will add modifiers to them (like
                # 'frilled bikini' + colors and so on)". Placing 'bikini'
                # literally skipped the machinery that decorates it -- the
                # outfit pool holds 54 bikini variants. Restricting the
                # pool and drawing keeps colour, frills and cut, and keeps
                # the spice gate: the variants that imply a level ('bikini
                # bottom aside') are refused by content_allowed below the
                # level that permits them, which is how booru uses them.
                _pool = _slot_pool(slot)
                for _g in list(_fashion):
                    _vars = [(t, c) for t, c in _pool
                             if t == _g or _g in t.split()
                             or t.endswith(" " + _g)]
                    if not _vars:
                        continue
                    _got = _draw(rng, _vars, census, level, banks, used,
                                 net=net, seeds=sub_seeds)
                    if _got:
                        placed = _got[0]
                        _fashion.remove(_g)
                        break
            if placed:
                used.add(placed)
                d["outfit"].append(placed)
                continue
            # the same sex veto as every other draw: the slot draws (hair,
            # hairstyle, outfit...) had none, so 'lolita hairband' and 'hime
            # cut' kept landing on "a father and son" after the measured
            # lean was wired in everywhere else
            # COLOUR AND CUT ARE NOT IMPLIED BY A SCENE. The identity slots
            # (hair colour, hair style, eyes) roll free of the network: its
            # offers for them are the same two or three tags for every seed
            # ('a girl in a park': bob cut / twin braids / hime cut, 60/60).
            if slot == "nudity state" and any(_lb.implies_clothing(t) for t in (opts.get("_user_tags") or [])):
                continue                     # the request dressed this subject (canon yields to the undress)
            if slot == "nudity state":
                # THE UNDRESS IS MEASURED (2026-09-15): the nudity states
                # walked by their share of the level's rating universe,
                # lifted by the place, the genre and the occupation
                try:
                    _loc9 = opts.get("_location") or (None, None, None)
                    _ns = _lb.nudity_states(level, place=_loc9[1],
                                            genre=(opts.get("_genre") or (None, None))[1],
                                            occupation=d.get("occupation"), whole=True)
                except Exception:
                    _ns = []
                _nv = _walk_measured(rng, _ns, census, level, banks, used, taken=d["outfit"], veto=_gender_veto)
                if _nv:
                    _bare_body(d, used)
                    used.add(_nv)
                    d["outfit"].append(_nv)
                continue
            got = _draw(rng, _slot_pool(slot), census, level, banks, used,
                        net=(None if slot in _FREE_ROLL_SLOTS else net),
                        seeds=sub_seeds, veto=_gender_veto)
            if not got or _table_owns:
                continue
            v = got[0]
            used.add(v)
            if slot == "hair":
                d["hair"] = _typed_hair or v
            elif slot == "hair_style":
                d["hair"] = (d["hair"] + ", " + v).strip(", ")
            elif slot == "eyes":
                d["eyes"] = _typed_eyes or v
            else:
                d["outfit"].append(v)
        # AN OUTER GARMENT HIDES THE UNDERWEAR (2026-09-14)
        _hide_covered_underwear(d, opts)
        if _typed_hair and _typed_hair not in d["hair"]:
            d["hair"] = (_typed_hair + ", " + d["hair"]).strip(", ")
        if _typed_eyes and not d["eyes"]:
            d["eyes"] = _typed_eyes
        # hair length reads as part of the hair slot, not a separate one
        #
        # A LENGTH MUST NOT FIGHT THE CUT. This prepended a length drawn
        # blind to whatever the hair slot had already produced, so a
        # `bob cut` acquired "very long hair" in front of it -- and only
        # in the PROSE, because the length is never emitted as a tag.
        # That is how one line came to read "The girl has very long hair,
        # bob cut" while the tag line said `bob cut` alone: 34 of 300
        # sampled prompts carried the clash.
        #
        # Measured, conditioned on solo: bob cut + very long hair has
        # lift 0.023 and bob cut + long hair 0.034 -- both contradictions
        # -- while bob cut + short hair sits at 3.67, because a bob IS
        # short hair. So the length cannot simply be dropped or fixed to
        # one value; it has to agree with the cut. The contradiction
        # table is the one place that knows which pairs cannot share an
        # image, and it is consulted here rather than kept a second
        # opinion. Unmeasured pairs stay permitted, as everywhere else.
        # a hair style that names a length ('short hair with long locks',
        # 'long hair' inside a compound) fills the length slot itself
        # (2026-09-14 sweep: 'very long hair' beside 'short hair with long locks')
        _len_named = any(ln in d["hair"] for ln in _HAIR_LEN) or " short hair" in (" " + d["hair"])             or " long hair" in (" " + d["hair"])
        if d["hair"] and not _len_named and rng.random() < 0.75:
            _contra = banks.get("_contradictions") or {}
            _have = [h.strip() for h in d["hair"].split(",") if h.strip()]
            _fits = [ln for ln in _HAIR_LEN
                     if not any(ln in (_contra.get(h) or ())
                                or h in (_contra.get(ln) or ())
                                for h in _have)]
            if _fits:
                d["hair"] = rng.choice(_fits) + ", " + d["hair"]
        # THE FOCUSED PART GETS A SECOND DESCRIPTOR (the author's 2026-09-11):
        # what the camera emphasises is described more
        _focus_parts = set()
        try:
            from promptstudio.engine import bridge as _lb0
            _pf = {v: k for k, v in _lb0._PART_FOCUS.items()}
            _focus_parts = {_pf[f] for f in (opts.get("_focus") or []) if f in _pf}
        except Exception:
            _focus_parts = set()
        for part in (s.get("body_parts") or []):
            # skin and build belong to the body table below (it reads the
            # activity, so it runs after the pose block)
            # ...and so do the face, the nose, the eyebrows, the ears and
            # the body shape (2026-09-13): the menu part 'body shape' drew
            # 'muscular' on .17 of the lines and 'tall female' on .10
            # beside the table's measured build (.05 of 1girl posts carry
            # any build tag); 'face' at .60 drew 'no nose' (.002). The
            # table owns every trait it measures; the menu keeps the parts
            # it does not (hands, legs, breasts, ass, the anatomy).
            if str(part).lower() in ("skin", "build", "body", "figure", "marks", "markings",
                                     "body shape", "face", "nose", "eyebrows", "ears")                     and s.get("tier") != "secondary" and _body_table_present():
                continue
            # NOTHING BEATS NONSENSE. If no tag in the library is actually
            # about this part, the slot stays empty -- the LLM path can
            # invent a phrase, the mechanical path must not.
            got = _draw(rng, _body_pool(part), census, level, banks, used,
                        net=net, seeds=seeds, veto=_gender_veto)
            if got:
                d["body"][part] = [got[0]]
                used.add(got[0])
                if str(part).lower() in _focus_parts:
                    # the second descriptor of a focused part is never a
                    # second member of an exclusive family (2026-09-13:
                    # 'large breasts, small breasts' under a breast focus)
                    _excl = set(used) | _exclusive_family(got[0])
                    got2 = _draw(rng, _body_pool(part), census, level, banks, _excl,
                                 net=net, seeds=seeds, veto=_gender_veto)
                    if got2 and got2[0] != got[0]:
                        d["body"][part].append(got2[0])
                        used.add(got2[0])
        # THE EXPRESSION (the author's 2026-09-11): a measured draw for every main
        # subject (nearly every 1girl picture carries one); under a face
        # framing or a face focus the expression carries the picture: two
        if s.get("tier") != "secondary":
            try:
                from promptstudio.engine import bridge as _lb1
                _pe_share = float(_lb1.expression_share())
            except Exception:
                _pe_share = 0.6
            _face_on = bool(opts.get("_face_typed")) or any(
                f in ("portrait", "eye focus") for f in (opts.get("_focus") or []))
            _n_ex = 2 if _face_on else (1 if rng.random() < _pe_share else 0)
            _ex = []
            # A MOUTH STATE IS NOT AN EXPRESSION, AND THE FACE IS DRAWN BY
            # MEASURED SHARES (the author's 2026-09-15): the expression draw
            # walks the emotions' shares of the 1girl universe (looking at
            # viewer .51, smile .43, blush .42, closed eyes .11, :d .09 ...
            # -- the co-occurrence network had proposed its hubs, 'open
            # mouth' the whole expression of 18% of subjects); the mouth is
            # a second measured detail, the mouth states walked by their raw
            # shares (open mouth .28, closed mouth .15, parted lips .06 ...)
            # so .67 of subjects carry one, as the booru shows
            try:
                _estates = _lb.emotion_states()
                _mstates = _lb.mouth_states()
            except Exception:
                _estates, _mstates = [], []
            for _ in range(_n_ex):
                got = _walk_measured(rng, _estates, census, level, banks, used, taken=_ex, veto=_gender_veto)
                if got:
                    _ex.append(got)
                    used.add(got)
            if _ex:
                got = _walk_measured(rng, _mstates, census, level, banks, used, taken=_ex,
                                     veto=_gender_veto, normalise=False)
                if got:
                    _ex.append(got)
                    used.add(got)
            if _ex:
                d["body"]["expression"] = _ex
        # THE EYES ARE SHUT OR OPEN, ONE STATE (2026-09-14)
        _act_known = (d.get("activity") or opts.get("_act")
                      or ((opts.get("_pre_activity") or (None, {}))[1] or {}).get("activity") if si == 0
                      else d.get("activity") or opts.get("_act"))
        _typed_shut = any(t in ("closed eyes", "sleeping") for t in (opts.get("_user_tags") or []))
        if _eyes_shut(_act_known) or _typed_shut:
            if not _typed_eyes:
                d["eyes"] = ""
            _ex0 = list((d.get("body") or {}).get("expression") or [])
            # shut eyes take every other eye state with them (2026-09-16:
            # 'one eye closed' beside 'closed eyes' on a sleeper)
            _ex1 = [x for x in _ex0 if not _GAZE_RE_FP.search(x) and x not in _OPEN_EYES
                    and not (re.search(r"\b(?:eyes?|eyed|pupils)\b", x) and x != "closed eyes")]
            if "closed eyes" not in _ex1 and "closed eyes" not in used:
                _ex1.append("closed eyes")
                used.add("closed eyes")
            d["body"]["expression"] = _ex1
        # APPEARANCE descriptors (the author's: the appearance bucket is useful).
        # One per subject at most, gender-checked -- 'dark-skinned female'
        # must not land on a male subject.
        # ...at the slot's MEASURED share (2026-09-13): how often a 1girl
        # post carries any appearance tag at all (the pool's counts over
        # the 1girl count, capped) -- fired on every subject, the slot put
        # 'dark-skinned female' (.03 of posts) on .22 of the lines
        if "general" in (s.get("flavor") or []) and rng.random() < _appearance_share():
            got = _draw(rng, _person_pools()["appearance"], census, level,
                        banks, used, net=net, seeds=sub_seeds,
                        veto=_gender_veto)
            if got:
                d["descriptors"] = [got[0]]
                used.add(got[0])
        ap = s.get("action_plan") or {}
        # TYPED WINS HERE TOO. With the user's own act in the plan, a drawn
        # pose or self-action fights it ('broom riding' beside 'fellatio to
        # the viewer'); the dice add nothing to an act the user wrote.
        if _typed_act:
            ap = {}
        # THE POSE TABLE FIRST (the author's 2026-09-04): a main character ALWAYS
        # has a base stance, rolled by the place inside the genre leaf, with
        # optional leg / arm / hand / torso details under their constraints.
        # A typed act keeps its own postures (act_posture); a secondary
        # figure may have none.
        _table_acts = False
        if not _typed_act and s.get("tier") != "secondary" and not d.get("pose"):
            try:
                from promptstudio.engine import bridge as _lb
                _loc = opts.get("_location") or (None, None, None)
                _pair = (si == 0 and len(subjects) == 2
                         and all(x.get("tier") != "secondary" for x in subjects))
                # THE ACTIVITY FIRST (the author's 2026-09-04): what the subject is
                # doing, by place; its stance pins the pose below. A shared
                # activity (arity 2) is copied to the second main character.
                _pinned = None
                _shared = opts.get("_shared_activity") if si == 1 else None
                # the first main subject's activity was rolled BEFORE the
                # camera (bridge.generate, 2026-09-11): reused here, once
                _pre = opts.get("_pre_activity") if si == 0 else None
                _ac = _pre[1] if _pre else (_shared or _lb.activity_for(
                    (opts.get("_genre") or (None, None))[1], _loc[1], rng,
                    allowed=lambda t: sm.content_allowed(t, census, level, banks),
                    pair=_pair, level=level,
                    prefer=((opts.get("_event") or (None, None, {}))[2] or {}).get("activities"),
                    # the typed occupation leans the roll even when the flavour
                    # never asked for an occupation slot
                    occupation=d.get("occupation") or _occupation_from_prompt(seeds),
                    being=opts.get("_genre_anchor"),
                    stance=_lb.typed_stance(opts.get("_user_tags"))))
                if _ac is not False:
                    _table_acts = True
                if _ac:
                    d["self_actions"].append(_ac["activity"])
                    used.add(_ac["activity"])
                    _pinned = _ac.get("stance")
                    d["activity"] = _ac["activity"]
                    # the activity's object is the thing in the hand
                    _obj = _ac.get("object")
                    if _obj and not d.get("held_object") and _obj not in ("magic circle", "campfire") \
                            and _vocab_has(_obj):
                        d["held_object"] = _obj
                        used.add(_obj)
                    if _pair and int(_ac.get("arity") or 1) > 1:
                        opts["_shared_activity"] = _ac
                # A TYPED FACE FRAMING (portrait, close-up): a still stance,
                # no limb extras, no held object -- the picture is the face
                _face = bool(opts.get("_face_typed"))
                # A TYPED STANCE IS THE POSE (the author's 2026-09-15: 'sitting'
                # typed did not register; the pose rolled 'standing' and the
                # word sat in the subject band): the user's base stance pins
                # the pose as an activity's stance does
                _typed_st = _lb.typed_stance(opts.get("_user_tags"))
                if _typed_st:
                    _pinned = _typed_st
                _pz = _lb.pose_for(
                    (opts.get("_genre") or (None, None))[1], _loc[1], _loc[2], rng,
                    allowed=lambda t: sm.content_allowed(t, census, level, banks),
                    pair=_pair, base=_pinned,
                    hands_busy=bool(_ac and _ac.get("object")),
                    stance_nudge=({t: 0.2 for t in _MOVING_STANCES} if _face else None),
                    act=opts.get("_act"), level=level,
                    view=((opts.get("_camera") or (None, None, None))[1] or ""))
                if _face and rng.random() < 0.7:
                    _pz["extras"] = []          # mostly no limb extras under a face framing
                if _pz.get("base"):
                    d["pose"] = _pz["base"]
                    d["_from_table"] = True
                    # the typed act's implications are the typed act's: the
                    # verifier never prunes them for a rolled row (2026-09-11)
                    if _ac and _ac.get("typed"):
                        opts.setdefault("_act_derived", [])
                        opts["_act_derived"] += [x for x in (_pz["base"], d.get("held_object")) if x]
                    used.add(_pz["base"])
                    # 'running, running': the activity IS the stance
                    d["self_actions"] = [x for x in d["self_actions"] if x != _pz["base"]]
                    for _x in _pz.get("extras") or []:
                        if _x not in used:
                            d["self_actions"].append(_x)
                            used.add(_x)
                    if _pz.get("pair"):
                        d["self_actions"].append(_pz["pair"])
                        used.add(_pz["pair"])
                    ap = dict(ap)
                    ap["pose"] = False
            except Exception:
                pass
        # THE BODY TABLE (the author's 2026-09-04): race parts fixed, then build,
        # skin, marks and the state modifiers -- each only with its cause --
        # for a main character; the old per-part draw keeps the parts the
        # table did not cover.
        _covered = set()
        if s.get("tier") != "secondary":
            try:
                from promptstudio.engine import bridge as _lb
                _loc = opts.get("_location") or (None, None, None)
                _lig = opts.get("_lighting") or {}
                _pcls = None
                try:
                    _pcls = (((_lb._activity_table() or {}).get("leaves") or {}).get("everyday") or {}).get("place_class", {}).get(_loc[1])
                except Exception:
                    _pcls = None
                _areg = ((_lb._activity_table() or {}).get("registry") or {}).get(d.get("activity") or "") or {}
                _ctx = {"kind": kind, "race": d.get("race") or _typed_race_word(base, seeds),
                        "occupation": d.get("occupation") or _occupation_from_prompt(seeds),
                        "activity": d.get("activity"),
                        "activity_state": _areg.get("state"), "activity_clothes": _areg.get("clothes"),
                        "event": (opts.get("_event") or (None, None, None))[1],
                        "weather": _lig.get("weather"), "season": _lig.get("season"),
                        "place": _loc[1], "place_class": _pcls,
                        "genre": (opts.get("_genre") or (None, None))[1], "level": level,
                        "detail": _lb.DETAIL_SCALE.get(str(opts.get("detail") or "standard"), 1.0),
                        "act": opts.get("_act"),
                        "view": ((opts.get("_camera") or (None, None, None))[1] or ""),
                        # the parts this picture carries, for the part's states
                        # (2026-09-14): the focused parts, an open mouth's teeth
                        "parts": sorted(set(_focus_parts) | ({"teeth"} if "open mouth" in
                                        ((d.get("body") or {}).get("expression") or []) else set()))}
                _bd = _lb.body_for(_ctx, rng,
                                   allowed=lambda t: sm.content_allowed(t, census, level, banks)
                                   and not _gender_veto(t) and t not in used)
                for key, vals in (("race", _bd["parts"]), ("build", [_bd["build"]] if _bd["build"] else []),
                                  ("skin", [_bd["skin"]] if _bd["skin"] else []), ("marks", _bd["marks"]),
                                  ("state", _bd["states"] + ([_bd["state"]] if _bd.get("state") else [])),
                                  ("desc", _bd.get("desc") or [])):
                    vals = [v for v in vals if v]
                    if vals:
                        d["body"][key] = vals
                        used.update(vals)
                        _covered.add(key)
                opts.setdefault("_body_leans", {})[si] = {"hair": _bd["hair_lean"], "eyes": _bd["eyes_lean"],
                                                          "breasts": _bd["breast_lean"]}
            except Exception:
                pass
        if ap.get("pose"):
            # SOLO ARITY HERE TOO: 'missionary', 'girl on top', 'sitting on
            # person' sat in the pose pool with no partner check, the one
            # the act draw below has always applied.
            got = _draw(rng, _flag_pool("pose"), census, level, banks, used,
                        net=net, seeds=seeds,
                        veto=lambda t: (sm.arity_of(t, banks) or 1) > 1
                        or bool(_ANATOMY.search(t))
                        or "clothing" in _FLAGS.get(t, set())
                        or _NOT_DRAWABLE & _FLAGS.get(t, set())
                        or _is_meme(t)
                        or _gender_veto(t))
            if got:
                d["pose"] = got[0]
                used.add(got[0])
        n_act = int(ap.get("n_part_actions") or 0)
        if _table_acts:
            n_act = 0          # the activity table answered for this place
        if n_act:
            # SOLO ARITY: an individual action must not need a partner.
            acts = _draw(rng, _flag_pool("act"), census, level, banks, used,
                         n=n_act, net=net, seeds=seeds,
                         # bare 'holding' is the umbrella parent of every
                         # 'holding X'; the held-object slot owns it and
                         # "the girl is holding." is not a sentence
                         veto=lambda t: t == "holding"
                         or (sm.arity_of(t, banks) or 1) > 1
                         or bool(_ANATOMY.search(t))
                         or "clothing" in _FLAGS.get(t, set())
                         or _NOT_DRAWABLE & _FLAGS.get(t, set())
                         or _is_meme(t)
                         or _gender_veto(t))
            d["self_actions"] = acts
            used.update(acts)
        # A TYPED OBJECT IS THE HELD OBJECT (2026-09-12): 'sword' in the
        # text is 'holding sword', no roll -- for the first main subject
        if si == 0 and not d.get("held_object"):
            try:
                _v = _lb_vocab()
                for _t in (opts.get("_user_tags") or []):
                    _fl = _FLAGS.get(str(_t).lower(), set())
                    if (_fl & {"object", "weapon", "food", "tool", "instrument"}) and not (_fl & {"scenery", "location", "clothing", "race", "creature", "vehicle"}) \
                            and not _ANATOMY.search(str(_t)) and not _is_place(str(_t)):
                        _h = "holding " + str(_t).lower()
                        if not _v.get(_h):
                            continue                # not a thing the booru shows held
                        d["held_object"] = _h
                        used.add(_h)
                        break
            except Exception:
                pass
        if ap.get("object_ok") and not d.get("held_object") \
                and rng.random() < (0.3 if opts.get("_face_typed") else 1.0):
            # THE TREE'S HOLDING OBJECTS FIRST (2026-09-06): the activity
            # table's registry, weighted by the object's measured share at
            # this place; the bank draw below stays the fallback
            try:
                from promptstudio.engine import bridge as _lb
                _loc_o = opts.get("_location") or (None, None, None)
                _ob = _lb.object_for(_loc_o[1], rng,
                                     allowed=lambda t: sm.content_allowed(t, census, level, banks)
                                     and t not in used and t not in _spice_never())
                if _ob:
                    d["held_object"] = _ob
                    used.add(_ob)
            except Exception:
                pass
        if ap.get("object_ok") and rng.random() < (0.3 if opts.get("_face_typed") else 1.0):
            # a held object is a THING. The object flag also covers body
            # parts, and 'holding large breasts' is what that produces.
            got = _draw(rng, _flag_pool("object"), census, level, banks, used,
                        net=net, seeds=seeds,
                        veto=lambda t: bool(_ANATOMY.search(t))
                        or "body" in _FLAGS.get(t, set())
                        # 'holding long hair': hair and eyes are the
                        # subject's own, never the thing in the hand
                        or bool(_HAIR_WORDS.search(t))
                        or "eyes" in t.split() or t.endswith("eyes")
                        # 'holding long sleeves' again, by word: the object
                        # flag sits on garments the clothing flag missed
                        or t.endswith("sleeves")
                        # 'holding long sleeves': the object flag covers
                        # garments too, and a garment being worn is not
                        # something in the hand
                        or "clothing" in _FLAGS.get(t, set())
                        or _NOT_DRAWABLE & _FLAGS.get(t, set())
                        or _is_meme(t)
                        # 'holding mountain', 'weapon on back' (2026-09-04):
                        # a place, scenery or a worn position is nothing
                        # in the hand
                        or bool({"scenery", "location", "race", "creature"} & _FLAGS.get(t, set()))
                        or _is_place(t)
                        or bool(_WORN_POS.search(t)))
            if got and not d.get("held_object"):   # the activity's object stays
                d["held_object"] = got[0]
        out_subjects.append(d)

    # STYLE: the embedding match, not a random draw -- this is precisely the
    # artist-fit defect v1 still has.
    style = {"name": None, "decomposition": []}
    palette = []
    st = opts.get("_style")
    if isinstance(st, (list, tuple)) and st and st[0]:
        style["name"] = st[0]
        # DECOMPOSITION WITHOUT A MODEL. The section rules say a named
        # style must also ship 2-4 simpler descriptors, so a checkpoint
        # that does not know the name still gets something to work with.
        # The LLM writes those; the fast path had none at all. But
        # style_pool.json already carries them, MEASURED: palette and
        # techniques are real booru tags ('greyscale', 'film grain',
        # 'halftone'), so they map to themselves through bridge 2.
        try:
            from promptstudio.engine import bridge as _lb
            rec = (_lb._style_pool2() or {}).get(style["name"]) or {}
            # TECHNIQUES ONLY -- NOT MEDIUMS. A medium ('3d') is not a
            # descriptor of the style, it is a different rendering
            # entirely, and the pipeline already has a MEDIUM channel that
            # decides it. Emitting one here bypassed that channel and put
            # '3d' into every cyberpunk prompt.
            # ...and a medium hiding INSIDE `techniques` is still a
            # medium. 'lineart' is listed as a technique, so excluding the
            # `mediums` field alone let it onto ~2% of images anyway,
            # bypassing resolve_medium exactly as '3d' once did. Same rule,
            # applied to the tags rather than to the field name.
            # ...EMITTED AT THEIR MEASURED SHARE (the author's 2026-09-13:
            # greyscale on .31 of the lines, monochrome .25, colorful .22,
            # distortion .18 -- against .04, .05, .0006 and .0001 of 1girl
            # posts). The first two palette and technique tags of the
            # rolled style went on every line whatever their share under
            # the style (cubism: flat color .022, monochrome .022). Each
            # tag now rolls at P(tag | style), the pool's own measure; at
            # most two of each.
            # the ruled words (typed-only palettes among them, 2026-09-16)
            # are not drawn here unless the request typed them
            _typed_fp = {str(x).lower() for x in (opts.get("_user_tags") or [])}
            _ruled_fp = _lb.ruled_out()

            def _measured_pick(field):
                _c = [(t, float(f)) for t, f in (rec.get(field) or {}).items()
                      if t and float(f) > 0 and not _lb.is_medium_tag(t)]
                # every candidate still takes its roll, so the seed's stream
                # is unchanged; a ruled word that wins its roll is simply not
                # written, and each other word keeps its own measured chance
                return [t for t, f in _c if rng.random() < f
                        and (t not in _ruled_fp or t in _typed_fp)][:2]
            style["decomposition"] = _measured_pick("techniques")
            palette = _measured_pick("palette")
        except Exception:
            pass

    loc = opts.get("_location") or ("free", None, None)
    lig = opts.get("_lighting") or {}
    lighting = [v for k, v in lig.items()
                if k in ("time", "weather") and v] + list(lig.get("lighting") or [])
    # a typed prose-only lighting term ('soft lighting', 'stage lighting')
    # is said with the light (lighting rulings, 2026-09-15)
    try:
        from promptstudio.library import external as _ext0
        for _n, _rec in _ext0.find_typed(base or "", kinds=("lighting",), include_pending=True):
            _nl = ((_rec or {}).get("render") or {}).get("nl") or _n    # the concept's own words, not the alias typed
            if (_rec or {}).get("prose_only") and _nl not in lighting:
                lighting.append(_nl)
    except Exception:
        pass

    n = len(out_subjects)
    who = [s["who"] for s in out_subjects]
    count_sentence = (
        "%d %s" % (n, "figures" if n != 1 else who[0]) if n != 1
        else "%s %s" % ("an" if who[0][:1].lower() in "aeiou" else "a", who[0]))

    p = {
        "count_sentence": count_sentence,
        "subjects": out_subjects,
        # a creature is in the picture, not in the cast (see
        # slots.subject_kind): named in the prose, given no slot
        "creatures": list((cast or {}).get("creatures") or []),
        "secondaries_look": [],
        "collective": "",
        "background": loc[1] or "",
        # the place's own details, measured (bridge.scene_details, 2026-09-14)
        "scene_details": (_lb_scene_details(loc[1], rng) if loc[1] else []),
        "lighting": lighting,
        "event": ((opts.get("_event") or (None, None, None))[1] or ""),
        "effects": [],
        "palette": palette,
        # THE PAIR EDGE CARRIES THE ACT (2026-09-12): two mains and an act
        # of arity two make the edge's act, so the prose says it
        "interactions": _edges_with_act(interactions, opts, banks),
        # THE TYPED ACT IS THE STORY. A POV act has no second subject, so
        # it sits in no interaction edge -- and the template then wrote a
        # paragraph about a collar and a hospital with the fellatio the
        # user asked for nowhere in it. The acts the parser read from the
        # user's text are the action; 'to the viewer' when the camera is
        # the partner.
        "action": _typed_act,
        # the resolved camera reaches the prose like the setting does: a
        # typed `pov` had no sentence, so the paragraph never said whose
        # eyes the picture is seen through
        "camera": list(opts.get("_camera") or (None, None, None))[:2],
        "setting": loc[1] or "",
        "style": style,
        "looks": list(opts.get("_look_phrases") or []),
        "mood": rng.choice(moods(level)),
        "nl": "",
    }
    p["nl"] = render_nl(p, mode)
    return p


_VERBISH = re.compile(r"^(sit|stand|ly|kneel|walk|run|danc|smil|laugh|eat|drink|"
                      r"read|sleep|lean|hold|reach|look|jump|swim|fly|rest|"
                      r"pos|stretch|bend|kiss|hug|point|wav|cross|touch)")


def _edges_with_act(interactions, opts, banks):
    out = []
    act = opts.get("_act")
    for it in (interactions or []):
        it = dict(it)
        if not it.get("act") and act and (sm.arity_of(act, banks) or 1) >= 2 and len(it.get("participants") or []) >= 2:
            it["act"] = act
        out.append(it)
    return out


def _pair_act_phrase(act):
    """'are having sex in the cowgirl position', 'are kissing', 'are
    giving each other...': the pair form of the act's verb"""
    t = str(act or "").strip()
    try:
        from promptstudio.engine import bridge as _lb
        pos = set(((_lb._spice_table() or {}).get("slots") or {}).get("position") or {})
    except Exception:
        pos = set()
    if t in pos:
        return "having sex in the %s" % t if not t.endswith("position") else "having sex in the %s" % t
    p = _phrase(t)
    return p[3:] if p.startswith("is ") else p


def _phrase(tag):
    return _phrase0(tag).replace(" at viewer", " at the viewer").replace(" to viewer", " to the viewer")


def _phrase0(tag):
    """tags are noun phrases as often as verb phrases -- 'knees up' and
    'hood up' are states, not actions, and neither 'the girl is hood up'
    nor 'the girl with hood up' is a sentence. Verb-shaped tags become a
    present participle; the rest are stated as what is shown."""
    t = str(tag or "").strip()
    # a booru qualifier is bookkeeping, not English: 'genderswap (mtf)'
    # reads 'genderswap' in a sentence
    if t.endswith(")") and " (" in t:
        t = t[:t.rindex(" (")].strip()
    if not t:
        return ""
    words = t.lower().split()
    head = words[0]
    # AN ACT IS DONE, NOT SHOWN (the author's 2026-09-06): a position is what
    # the figure is in; a spice act takes the verb of its kind -- given
    # (the oral and hand family), had (the sex family); a registry
    # activity is done ('is doing yoga'); an -ing act is itself
    if words[-1] == "position":
        return "is in the %s" % t
    if _is_position(t):
        return "is in the %s position" % t      # doggystyle, missionary, standing sex, seiza
    if _registry_activity(t) and not head.endswith("ing"):
        return "is doing %s" % t                # yoga, gymnastics, kendo
    if _is_act(t) or _ACT_HAVE.search(t) or _ACT_GIVE.search(t):
        if head.endswith("ing"):
            return "is %s" % t                 # 'kissing', 'fingering': the act is its own verb
        if _ACT_GIVE.search(t):
            return "is giving %s" % t
        if _ACT_HAVE.search(t):
            return "is having %s" % t
        return "is engaged in %s" % t
    # A POSE OR A GESTURE READS BY ITS SHAPE (the author's 2026-09-13: 'is
    # shown arms up' on three lines in four). A wiki-named pose is the
    # pose; a stance or a style is what the figure is in; a lying detail
    # is how it lies; a verb head ('squatting', 'bent over') is done; a
    # preposition head ('on one knee') is a state; a body-part phrase
    # ('arms up', 'hands on own face', 'heart tail') is carried; a lone
    # noun of the feet slot is stood on ('tiptoes'), of the torso or legs
    # slot done ('handstand').
    if words[-1] == "pose" or words[-1].endswith("-pose"):   # 'pigeon pose', 'a-pose'
        return "is in %s%s" % ("" if t.startswith("the ") else "the ", t)   # 'the pose' takes no second article
    if words[-1] == "stance":
        return "is in a %s" % t
    if words[-1] == "style":
        return "is sitting %s" % t
    if _pose_slot(t) == "lying_detail":
        return "is lying %s" % t
    if head.endswith("ing") or head in _PARTICIPLE_HEADS:
        return "is %s" % t
    if words[-1].endswith("ing") and len(words) == 2:
        return "is %s %s-style" % (words[-1], words[0])
    if head == "all":
        return "is on %s" % t
    if head in ("on", "in", "at", "under", "against", "between", "behind", "over"):
        return "is %s" % t
    if any(w in _LIMB_WORDS for w in words):
        return "with %s" % t
    if head.endswith("ed") and len(words) <= 2:
        return "is %s" % t
    if len(words) == 1 and _pose_slot(t) == "feet":
        return "is on %s" % t
    if len(words) == 1 and _pose_slot(t) in ("torso", "legs"):
        return "is doing a %s" % t
    if len(words) == 1 and "pose" in _gloss_kinds(t):
        return "is %s" % t                 # a lone state word: midair, underwater, airborne
    return "with %s" % t


_LIMB_WORDS = frozenset("""arm arms leg legs hand hands knee knees foot feet head heads chest back
hip hips finger fingers thigh thighs tail eye eyes mouth tongue elbow elbows shoulder shoulders
wrist wrists ankle ankles toe toes palm palms fist fists lap waist neck chin cheek cheeks forehead
breast breasts ass butt heel heels""".split())
_POSE_SLOTS = {"data": None}
_PARTICIPLE_HEADS = frozenset(("bent", "bound", "hunched", "knelt", "stooped", "slumped", "sprawled"))
_REG_ACTS = {"data": None}
_POSITIONS = {"data": None}
_POSITION_GLOSS_RE = re.compile(r"\b(position|pose|posture)\b")


def _is_position(tag):
    """a named position: the spice table's position slot, or a tag whose
    booru gloss calls it a position or a pose ('wariza: a sitting
    position...'); never one that already ends in 'position' / 'pose'"""
    t = str(tag or "").lower().strip()
    if not t or t.endswith(" position") or t.endswith("pose") or t.endswith("ing"):
        return False
    if _POSITIONS["data"] is None:
        try:
            from promptstudio.engine import bridge as _lb
            sl = (_lb._spice_table() or {}).get("slots") or {}
            _POSITIONS["data"] = {str(k).lower() for k in (sl.get("position") or {})}
        except Exception:
            _POSITIONS["data"] = set()
    if t in _POSITIONS["data"]:
        return True
    try:
        from promptstudio.engine import bridge as _lb
        g = str((_lb._gloss_of(t) or ("", ()))[0] or "").lower()
        return bool(_POSITION_GLOSS_RE.search(g))
    except Exception:
        return False


def _registry_activity(tag):
    """the tag is an activity of the registry (genre_activities.json)"""
    if _REG_ACTS["data"] is None:
        try:
            from promptstudio.engine import bridge as _lb
            _REG_ACTS["data"] = {str(k).lower() for k in
                                 ((_lb._activity_table() or {}).get("registry") or {})}
        except Exception:
            _REG_ACTS["data"] = set()
    return str(tag or "").lower().strip() in _REG_ACTS["data"]


def _join_acts(phrases):
    """'is sitting, reading and holding book' -- one verb for the list: the
    second and later 'is ...' phrases drop their 'is', the last joins with
    'and' (2026-09-13: 'is sitting, is reading, is holding book')"""
    ph = [str(x).strip() for x in phrases if str(x).strip()]
    if not ph:
        return ""
    verbs = [ph[0]] + [x[3:] for x in ph[1:] if x.startswith("is ")]
    carried = [x for x in ph[1:] if not x.startswith("is ")]     # 'with arms up', 'in the ...'
    head = verbs[0] if len(verbs) == 1 else ", ".join(verbs[:-1]) + " and " + verbs[-1]
    return ", ".join([head] + carried)


def _pose_slot(tag):
    """the pose table's slot of a tag (legs, arms, hands, hands_self, torso,
    feet, named, lying_detail, pair) or None"""
    if _POSE_SLOTS["data"] is None:
        m = {}
        try:
            from promptstudio.engine import bridge as _lb
            tb = _lb._pose_table() or {}
            for slot, ent in (tb.get("slots") or {}).items():
                for t in (ent.get("items") or {}):
                    m[str(t).lower()] = slot
            for t in (tb.get("lying_detail") or []):
                m[str(t).lower()] = "lying_detail"
            pi = (tb.get("pairs") or {}).get("items") or []
            for t in (pi if isinstance(pi, list) else list(pi)):
                m[str(t).lower()] = "pair"
        except Exception:
            pass
        _POSE_SLOTS["data"] = m
    return _POSE_SLOTS["data"].get(str(tag or "").lower().strip())


def _gloss_kinds(tag):
    try:
        from promptstudio.engine import bridge as _lb
        return set(_lb._gloss_flags(tag) or ())
    except Exception:
        return set()


_ACT_GIVE = re.compile(r"\b(fellatio|cunnilingus|irrumatio|anilingus|paizuri|[a-z]*job|massage|"
                       r"licking|sucking|grab|kiss[a-z]*|spanking|facesitting)\b")
_ACT_HAVE = re.compile(r"\b(sex|penetration|insertion|threesome|foursome|fivesome|orgy|gangbang|"
                       r"spitroast|intercourse|position|doggystyle|missionary|prone bone|nelson|"
                       r"press|cowgirl|on top|straddle|congress|bukkake)\b")
_MOVING_STANCES = {"walking", "running", "jumping", "floating", "wading", "crawling", "swimming",
                   "flying", "dancing", "stretching", "fighting stance"}
_ACT_NAMES = {"data": None}


def _is_act(t):
    """the phrase names a spice act or position (whole words: 'fellatio to
    the viewer' does, 'kneeling' does not)"""
    if _ACT_NAMES["data"] is None:
        try:
            from promptstudio.engine import bridge as _lb
            sl = (_lb._spice_table() or {}).get("slots") or {}
            names = set(sl.get("act") or {}) | set(sl.get("position") or {})
            _ACT_NAMES["data"] = re.compile(r"\b(%s)\b" % "|".join(
                re.escape(n) for n in sorted(names, key=len, reverse=True))) if names else re.compile(r"(?!x)x")
        except Exception:
            _ACT_NAMES["data"] = re.compile(r"(?!x)x")
    if _ACT_NAMES["data"].search(str(t).lower()):
        return True
    # ...or any tag the gloss flags an act ('sex', 'kiss' are acts too)
    try:
        from promptstudio.engine import bridge as _lb
        _t0 = str(t).lower()
        for _sep in (" to the ", " with ", " on "):
            _t0 = _t0.split(_sep)[0]
        return "act" in _lb._gloss_flags(_t0.strip())
    except Exception:
        return False


def _names(subs):
    """every subject needs a referent of its own: two subjects that share a
    noun become 'the first girl' / 'the second girl', because the section
    rules forbid pronouns and an ambiguous 'the girl' is no better."""
    counts = {}
    for s in subs:
        counts[s.get("who")] = counts.get(s.get("who"), 0) + 1
    ORD = ("first", "second", "third", "fourth", "fifth")
    seen, out = {}, []
    for s in subs:
        w = s.get("who") or "figure"
        if counts.get(w, 0) > 1:
            i = seen.get(w, 0)
            seen[w] = i + 1
            out.append("%s %s" % (ORD[i] if i < len(ORD) else "other", w))
        else:
            out.append(w)
    return out


_GENDER_SUFFIX = re.compile(r"\s+(female|male|females|males)$")


def _desc(tag):
    """a body descriptor in a sentence about a named subject: 'muscular
    female' -> 'muscular', 'dark-skinned female' -> 'dark-skinned'. The
    noun already says the gender; the suffix is booru bookkeeping."""
    return _GENDER_SUFFIX.sub("", _plain(tag))


_NUM_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine")
_KIND_NOUN = {"female": ("girl", "girls"), "male": ("boy", "boys"),
              "futa": ("futanari", "futanari"), "other": ("figure", "figures")}


_NUDE_STATES = frozenset(("nude", "naked", "topless", "bottomless", "completely nude",
                          "partially nude", "underwear only", "barefoot"))


def _body_table_present():
    try:
        from promptstudio.engine import bridge as _lb
        return bool(_lb._body_table())
    except Exception:
        return False


_PLACES = None
_WORN_POS = re.compile(r"\b(on back|on shoulder|behind back|on head|in mouth|"
                       r"between breasts|on lap)\b")


def _is_place(tag):
    """is this a name in the location pool?"""
    global _PLACES
    if _PLACES is None:
        try:
            with open(_paths.data("location_pool.json"), encoding="utf-8-sig") as f:
                _PLACES = {k for k in json.load(f) if not str(k).startswith("_")}
        except Exception:
            _PLACES = set()
    return str(tag).lower() in _PLACES


def _vocab_has(tag):
    try:
        from promptstudio.engine import bridge as _lb
        return _lb._vocab().get(tag, 0) >= 100
    except Exception:
        return False


def _typed_race_word(base, seeds):
    """the race or humanoid word the user typed, if any ('cat girl', 'elf')"""
    try:
        for t in (seeds or ()):
            if sm.subject_kind(str(t).lower(), base) in ("race", "humanoid"):
                return str(t).lower()
    except Exception:
        pass
    return None


def _cast_opener(subs):
    """'Two girls.' / 'A girl and a boy.' / 'Two girls and a futanari.' --
    the count sentence for two or more subjects. The prose never stated
    how many were in frame; the anima model reads the paragraph, and a
    paragraph that names 'the first girl' and 'the second girl' without
    ever saying 'two girls' is a worse instruction than the tag line."""
    counts = {}
    for sub in subs:
        k = sub.get("kind") or {"girl": "female", "boy": "male",
                                "futanari": "futa"}.get(sub.get("who"), "other")
        k = {"futanari": "futa"}.get(k, k)
        counts[k] = counts.get(k, 0) + 1
    bits = []
    for k in ("female", "male", "futa", "other"):
        n = counts.get(k, 0)
        if not n:
            continue
        sing, plur = _KIND_NOUN[k]
        if n == 1:
            bits.append(("an " if sing[:1] in "aeiou" else "a ") + sing)
        else:
            bits.append("%s %s" % (_NUM_WORDS[n] if n < len(_NUM_WORDS) else str(n), plur))
    if not bits:
        return ""
    text = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]
    return text[:1].upper() + text[1:] + "."


def _plain(tag):
    """strip a trailing booru qualifier for prose: 'pom pom (clothes)' ->
    'pom pom', 'western comics (style)' -> 'western comics'"""
    t = str(tag or "").strip()
    if t.endswith(")") and " (" in t:
        t = t[:t.rindex(" (")].strip()
    return t


def _style_display(name):
    """'western comics (style)' -> 'western comics': the qualifier is booru
    bookkeeping and has no place in a sentence."""
    n = str(name or "")
    for q in (" (style)", " (medium)", " (theme)"):
        if n.lower().endswith(q):
            n = n[:-len(q)]
    return n


_ADVERB_PLACES = {"underwater", "outdoors", "indoors", "outside", "inside",
                  "underground", "aboard", "offshore", "overseas", "downtown",
                  "backstage", "onstage", "upstairs", "downstairs"}


def _is_adverb_place(st):
    return st.lower() in _ADVERB_PLACES


# A PLACE TAKES ITS OWN PREPOSITION. "set in train interior", "set in
# campfire", "set in summer festival" were the fault; one small table by
# the place's last word, 'in the ...' otherwise, and an interior is
# 'inside the ...'.
_PLACE_AT = frozenset(("festival", "party", "beach", "concert", "market",
                       "station", "platform", "stop", "counter", "altar",
                       "shrine", "pool", "lake", "table", "desk", "gate",
                       "entrance", "crossing", "intersection", "harbor",
                       "harbour", "pier", "dock", "port", "campsite", "camp",
                       "picnic", "aquarium", "zoo", "cafe", "bar", "school"))
_PLACE_ON = frozenset(("stage", "rooftop", "roof", "balcony", "bridge",
                       "boat", "ship", "deck", "bed", "cliff", "hill",
                       "mountain", "road", "street", "path", "field",
                       "farm", "island", "raft", "porch", "veranda",
                       "terrace", "staircase", "stairs", "train", "bus",
                       "bench", "swing", "playground", "court", "pitch",
                       "rink", "track", "runway"))
_PLACE_BY = frozenset(("campfire", "fireplace", "window", "river", "riverside",
                       "lakeside", "seaside", "shore", "waterfall", "fountain",
                       "pond", "bonfire", "hearth"))


def _place_phrase(st):
    low = st.lower().strip()
    if _is_adverb_place(low):
        return low
    if low.endswith(" interior"):
        return "inside the " + low[:-len(" interior")]
    # 'in space', 'in heaven': places that take no article
    if low in ("home",):
        return "at home"                  # a generic place word (2026-09-15)
    if low in ("space", "outer space", "heaven", "hell", "limbo",
               "cyberspace", "orbit", "nature", "town", "bed"):
        return "in " + low
    last = low.split()[-1] if low.split() else low
    if last in _PLACE_BY:
        prep = "by"
    elif last in _PLACE_ON:
        prep = "on"
    elif last in _PLACE_AT:
        prep = "at"
    else:
        prep = "in"
    return "%s the %s" % (prep, low)


def _view_sentence(view):
    """One sentence per viewpoint family: 'from X' views are seen from
    somewhere, angle views are shot at an angle, the rest are seen as-is.
    'Seen dutch angle.' was the fault."""
    v = view.lower()
    if v.startswith("from "):
        rest = v[5:]
        if rest in ("side", "front", "back"):
            rest = "the " + rest
        return "Seen from %s." % rest
    if v.endswith(" angle"):
        return "Shot at a %s." % v.replace("dutch", "Dutch")
    return "Seen %s." % v


def render_nl(p, mode):
    """The NL part, assembled from the plan.

    NAMING DISCIPLINE (the rule the LLM path is held to as well): every
    subject is named by noun or alias in every sentence that mentions it.
    A template cannot lose track of a referent, so the fast path satisfies
    this by construction instead of by instruction.
    """
    subs = p.get("subjects") or []
    names = _names(subs)
    if mode != "anima":
        # ILLUSTRIOUS TAKES PHRASES, NOT PROSE. The assembler comma-joins
        # this straight onto the tag line, so sentences arrive as
        # ', Rendered in the style of cyberpunk, The mood is playful'.
        bits = []
        if p.get("setting"):
            bits.append(str(p["setting"]))
        if (p.get("style") or {}).get("name"):
            # DO NOT DOUBLE THE WORD. Style names in the pool already end
            # in 'style'/'art'/'painting' often enough that appending it
            # blindly produced 'psychedelic art style style' -- and the
            # mangled copy no longer matched the tag line, so it survived
            # the de-duplication and shipped twice. Same test the anima
            # branch already applies.
            _sn = _style_display(p["style"]["name"])
            bits.append(_sn if _sn.lower().endswith(
                ("style", "art", "painting", "render", "aesthetic"))
                else _sn + " style")
        for lt in (p.get("lighting") or []):
            bits.append(str(lt))
        if p.get("mood"):
            bits.append("%s mood" % p["mood"])
        return ", ".join(b for b in bits if b)
    out = []
    if len(subs) >= 2:
        _op = _cast_opener(subs)
        if _op:
            out.append(_op)
    for s, who in zip(subs, names):
        # EVERY TAG THAT ENTERS A SENTENCE LOSES ITS BOORU QUALIFIER:
        # "wears pom pom (clothes)" is bookkeeping, not English
        bits = [_plain(b) for b in (s.get("hair"), s.get("eyes")) if b]
        # drawn BODY descriptors were never rendered ('stitched face',
        # 'medium breasts' reached the tag line and not the paragraph);
        # they belong in the same "has" list as hair and eyes
        # A GAZE IS DONE, NOT HAD (2026-09-15: "has looking at viewer,
        # closed mouth"): the gaze tags of the expression draw join the
        # subject's actions -- "is lying, looking at the viewer"
        _gazes = []
        for _part, _vals in (s.get("body") or {}).items():
            for v in (_vals or []):
                if not v:
                    continue
                if _part == "expression" and _GAZE_RE_FP.search(str(v)):
                    _gazes.append(v)
                else:
                    bits.append(_plain(v))
        # NUDITY IS A STATE, NOT A GARMENT: "wears nude" was the sentence
        _outfit = [_plain(x) for x in (s.get("outfit") or [])]
        _states = [x for x in _outfit if x.lower() in _NUDE_STATES]
        wear = ", ".join(x for x in _outfit if x.lower() not in _NUDE_STATES)
        # A NAME TAKES NO ARTICLE. The LLM plan hands a persona over as
        # its display name ('Enoshima Junko'), and "The Enoshima Junko;
        # the Enoshima Junko is shown..." is what the template made of it
        # when this paragraph stood in for a failed model render.
        named = bool(who) and who[:1].isupper()
        ref, ref_cap = (who, who) if named else ("the " + who, "The " + who)
        line = ref_cap
        if bits:
            line += " has %s" % ", ".join(_desc(x) for x in bits)
        if wear:
            line += "%s wears %s" % (" and" if bits else "", wear)
        if _states:
            line += "%s is %s" % (" and" if (bits or wear) else "", ", ".join(_states))
        acts = [a for a in ([s.get("pose")] + (s.get("self_actions") or []) + _gazes)
                if a]
        if acts:
            # nothing described yet -> "Firefly is holding..." rather than
            # "Firefly; Firefly is holding..."
            if line == ref_cap:
                line += " %s" % _join_acts([_phrase(a) for a in acts])
            else:
                line += "; %s %s" % (ref, _join_acts([_phrase(a) for a in acts]))
        if s.get("held_object"):
            # many object tags are already phrased as the act ('holding
            # fan'), so a bare prefix produces 'holding holding fan'
            obj = _plain(s["held_object"])
            line += (", %s" % obj if obj.lower().startswith("holding")
                     else ", holding %s" % obj)
        # a subject with nothing to say gets no sentence: "The second girl."
        # on its own is not a description
        if line.strip() != ref_cap.strip():
            out.append(line + ".")
    if p.get("action") and names:
        def _ref(nm):
            return nm if nm[:1].isupper() else "the " + nm
        _act = _plain(p["action"])
        # a route tag names the route, not the whole act: "are having
        # anal" is clipped; the sentence says anal sex / vaginal sex
        _act = {"anal": "anal sex", "vaginal": "vaginal sex",
                "anal to the viewer": "anal sex with the viewer",
                "vaginal to the viewer": "vaginal sex with the viewer",
                "sex to the viewer": "sex with the viewer"}.get(_act.lower(), _act)
        _head = _act.split()[0].lower() if _act else ""
        act_phrase = _act if _VERBISH.match(_head) else _pair_act_phrase(_act)
        _edge_acts = {str(it.get("act") or "").lower() for it in (p.get("interactions") or [])}
        if len(names) >= 2 and (sm.arity_of(str(p["action"]).split(" to ")[0]) or 1) >= 2:
            # a two-person act is told of the pair: "The girl and the boy
            # are having vaginal sex." / "... are kissing." -- once: the
            # pair edge tells it when it carries the same act
            line = "%s and %s are %s." % (_ref(names[0]), _ref(names[1]), act_phrase)
        else:
            line = "%s %s." % (_ref(names[0]), _phrase(p["action"]))
        if line:
            out.append(line[:1].upper() + line[1:])
    _told = str(p.get("action") or "").lower().split(" to ")[0]
    for it in (p.get("interactions") or []):
        act = it.get("act")
        if not act or (_told and str(act).lower() == _told):
            continue                    # the typed act sentence told it already
        parts = [names[i - 1] for i in (it.get("participants") or [])
                 if 0 < i <= len(names)]
        if len(parts) >= 2:
            out.append("The %s and the %s are %s." % (parts[0], parts[1], _pair_act_phrase(act)))
    _crs = [str(c) for c in (p.get("creatures") or []) if c]
    if _crs:
        _named = ["%s %s" % ("an" if c[:1] in "aeiou" else "a", c) for c in _crs]
        _list = (_named[0] if len(_named) == 1
                 else ", ".join(_named[:-1]) + " and " + _named[-1])
        out.append("%s %s also in the scene." % (
            _list[:1].upper() + _list[1:], "is" if len(_named) == 1 else "are"))
    _cam = p.get("camera") or [None, None]
    _view = str(_cam[1] or "").lower() if len(_cam) > 1 else ""
    if _view in ("pov", "male pov", "female pov", "futanari pov"):
        out.append("Seen from the viewer's point of view.")
    elif _view:
        out.append(_view_sentence(_view))
    if p.get("setting"):
        # the model's setting phrase may arrive capitalised and punctuated
        # ("Night park bench.") -- "set in Night park bench.." is not a
        # sentence
        _st = str(p["setting"]).strip().rstrip(".!, ")
        if _st[:1].isupper() and not (len(_st) > 1 and _st[1:2].isupper()):
            _st = _st[:1].lower() + _st[1:]
        # 'underwater', 'outdoors' are adverbs: "set underwater", not "set
        # in underwater"; a place noun keeps its "in"
        # an artificial backdrop is not a place: "against a white background"
        # THE PLACE OPENS THE DESCRIPTION (the author's 2026-09-13: "the
        # subject is set..." is no sentence a person writes): "In the
        # library, the girl has..." / "Against a white background, the
        # girl..."; a picture with nobody in it is a view of the place
        _ph = ("against a %s" % _st) if _st.endswith(" background") else _place_phrase(_st)
        _sd = [_plain(x) for x in (p.get("scene_details") or []) if x]
        if _sd:
            _ph += ", with " + (_sd[0] if len(_sd) == 1 else " and ".join(_sd[:2]))
        _open = _ph[:1].upper() + _ph[1:]
        if out:
            _first = out[0]
            if _first[:1].isupper() and not (len(_first) > 1 and _first[1:2].isupper())                     and _first.split()[0] in ("The", "A", "An", "Two", "Three", "Four", "Five", "Several"):
                _first = _first[:1].lower() + _first[1:]
            out[0] = "%s, %s" % (_open, _first)
        else:
            out.append("%s." % (_open if _st.endswith(" background") or _is_adverb_place(_st.lower())
                                else "A view of the %s" % _st.lower()))
    if (p.get("style") or {}).get("name"):
        out.append("Rendered in the style of %s."
                   % _style_display(p["style"]["name"]))
    if p.get("looks"):
        # the style box's phrases, in the user's words (2026-09-17)
        out.append("Drawn with %s." % (", ".join(p["looks"][:-1]) + " and " + p["looks"][-1]
                                       if len(p["looks"]) > 1 else p["looks"][0]))
    if p.get("event"):
        out.append("The occasion is %s." % _plain(p["event"]))
    if p.get("lighting"):
        out.append("Lit by %s." % ", ".join(p["lighting"]))
    if p.get("mood"):
        out.append("The mood is %s." % p["mood"])
    return " ".join(out)


def pick(ambiguous, cands):
    """bridge 2 without the model: the engine already ranked the candidates
    best-first, so rank 1 wins. This is not a downgrade for ranked slots --
    generate() already does exactly this for hair and eyes, on the grounds
    that the 8B ignores prefer-first ordering."""
    return {c: [cands[c][0]] for c in ambiguous if cands.get(c)}
