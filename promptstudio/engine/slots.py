#!/usr/bin/env python
"""
slot_model.py -- what the user actually addressed, slot by slot.

The generator used to know only what it managed to PARSE. A slot the user
plainly wrote about but that the parser fumbled looked exactly like a slot the
user never mentioned, so it got free-filled -- which is how an explicit
"disgusted expression" came back as a generated expression instead.

Three states per slot, and the middle one is the whole point:

    filled    a tag for this slot came out of the user input
              -> exclusive slots take nothing more; additive slots may extend
                 only through the ordinary conflict checks
    claimed   the text clearly addresses this slot but NO tag was extracted
              -> generate NOTHING here. We cannot check a generated tag for
                 agreement with words we failed to parse, so silence is the
                 only safe answer. An empty slot beats a contradicted one.
    free      the user never went near it -> fill as normal

Probes are deliberately conservative: a probe that fails to fire merely
restores the old behaviour, while a probe that fires wrongly costs richness.
"""

import json as _json
import re

from promptstudio.engine import tagnet as _tn

from promptstudio import paths as _paths

# Slots that hold exactly one value. Anything else may legitimately stack.
EXCLUSIVE = {"focus", "framing", "viewpoint", "gaze", "act"}

FOCUS_KEEP = {"solo focus"}      # 'solo focus' is about WHO, not about WHAT

FRAMING_TAGS = {"full body", "upper body", "lower body", "portrait", "close-up",
                "cowboy shot", "wide shot", "profile", "bust", "feet out of frame"}

VIEWPOINT_TAGS = {"pov", "male pov", "female pov", "from behind", "from above",
                  "from below", "from side", "from front", "dutch angle",
                  "straight-on", "high angle", "low angle", "overhead view"}

GAZE_TAGS = {"looking at viewer", "looking away", "looking back", "looking up",
             "looking down", "looking to the side", "eye contact",
             "looking at another", "looking afar"}

# Sex acts. The bank vocabulary carries most of them; the pattern catches the
# rest so the set does not silently trail the vocabulary the way a hand list does.
ACT_RE = re.compile(
    r"^(.*\b)?(handjob|footjob|blowjob|rimjob|titjob|fellatio|irrumatio|paizuri|"
    r"cunnilingus|masturbation|fingering|penetration|insertion|grinding|"
    r"frottage|tribadism|creampie|ejaculation|deepthroat|facesitting|"
    r"cowgirl position|reverse cowgirl position|missionary|doggystyle|"
    r"mating press|prone bone|piledriver|suspended congress|full nelson|"
    r"anal|vaginal sex|group sex|gangbang|threesome|orgy|sex)$")

# Gaze phrasings that must NOT be read as viewpoint. prompt_enhancer strips
# these before matching VIEWPOINT_MAP; the probes here have to make the same
# distinction or the analysis tool reports a missing viewpoint over "looking
# at the viewer", which is a gaze and was answered correctly.
GAZE_PHRASE_RE = re.compile(
    r"\b(look(ing|s|ed)?|star(e|es|ing)|gaz(e|es|ing)|glanc(e|es|ing)|"
    r"smil(e|es|ing)|wink(ing|s)?|peer(ing|s)?)\s+((back|down|up|over|straight)\s+)?"
    r"(at|to|toward|towards)\s+(the\s+)?(viewer|camera|you)\b")

SLOT_PROBES = {
    "expression": re.compile(
        r"\b(expressions?|facial|smil\w+|frown\w*|grin\w*|laugh\w*|cry\w*|tears?|"
        r"angry|anger|furious|mad|sad(ly)?|happy|joyful|disgust\w*|revolt\w+|"
        r"annoy\w*|irritat\w*|bored|blush\w*|pout\w*|smirk\w*|scowl\w*|glar\w+|"
        r"surpris\w+|shock\w+|scared|afraid|fear\w*|terrified|embarrass\w+|"
        r"ahegao|nervous|worried|anxious|confus\w+|serious|stoic|deadpan|smug|"
        r"ecstat\w+|blissful|sneer\w*|grimac\w+|"
        r"reluctan\w+|unwilling|unhappy|miserable|delight\w+|excited)\b"),
    "gaze": re.compile(
        r"\b(looking (at|away|back|up|down|aside|to)|eye contact|gazes?|gazing|"
        r"star(es|ing)|glanc\w+|avert\w+|meets? (his|her|their|the) eyes?)\b"),
    "act": re.compile(
        r"\b(handjob|blowjob|fellatio|paizuri|titfuck|footjob|rimjob|cunnilingus|"
        r"sex|fuck\w*|penetrat\w+|riding|rides|masturbat\w+|finger(s|ing)|"
        r"jerk\w+|strok\w+|suck\w+|lick\w+|kiss\w+|grop\w+|hump\w+|grind\w+|"
        r"creampie|anal|vaginal|missionary|doggystyle|cowgirl|mating press|"
        r"giving (him|her|them|the viewer)|performs?)\b"),
    "focus": re.compile(r"\b\w+ focus\b|\bfocus(ing|ed)? on\b|\bemphasis on\b"),
    "framing": re.compile(
        r"\b(full[- ]body|upper[- ]body|lower[- ]body|portrait|close[- ]?up|"
        r"cowboy shot|wide shot|head ?shot|bust shot|full[- ]length|"
        r"framed?|framing|from the (waist|knees|chest) (up|down))\b"),
    "viewpoint": re.compile(
        r"\b(pov|point of view|from behind|from above|from below|"
        r"from the (side|back|front|rear)|bird.?s[- ]eye|worm.?s[- ]eye|overhead|"
        r"(low|high) angle|to the (viewer|camera)|at the (viewer|camera)|"
        r"towards? (you|the viewer|the camera)|facing (the )?(viewer|camera)|"
        r"first[- ]person)\b"),
    "hair": re.compile(
        r"\b(hair|braids?|ponytail|twintails?|bun|bangs|blonde|blond|brunette|"
        r"redhead|balding|bald)\b"),
    "clothing": re.compile(
        r"\b(dress|shirt|blouse|skirt|bikini|swimsuit|lingerie|uniform|jacket|"
        r"coat|sweater|hoodie|pants|jeans|shorts|kimono|yukata|robe|armou?r|"
        r"nude|naked|topless|bottomless|wear\w*|dressed|outfit|clothes|clothing|"
        r"bra|panties|underwear|stockings|thighhighs|pantyhose|socks|shoes|"
        r"boots|heels|gloves|apron|leotard|corset|cloak|cape)\b"),
}


def _strip_qualifier(tag, vocab):
    """'red dress' -> 'dress', 'straddling paizuri' -> 'paizuri'.

    Group membership is written for the base tag, so a qualified form escapes
    every one-of rule: OUTFITS holds 'dress' and therefore never saw the user
    'red dress', and a second garment layered on top. Testing the base form as
    well closes that hole for every qualifier at once instead of enumerating
    colours.
    """
    parts = tag.split()
    for cut in range(1, len(parts)):
        base = " ".join(parts[cut:])
        if base in vocab:
            return base
    return None


def slot_of(tag, banks=None):
    """which slot a tag occupies, or None."""
    t = tag.lower().strip()
    vocab = (banks or {}).get("_scanvocab") or {}

    # The strongest signal there is: the user wrote 'disgusted EXPRESSION', so
    # whatever tag that resolved to belongs to the expression slot no matter what
    # any classifier thinks of it in isolation.
    hint = (((banks or {}).get("_slot_hints") or {}).get(t)
            or ((banks or {}).get("_slot_learned") or {}).get(t))
    if hint:
        return hint

    for probe in (t, _strip_qualifier(t, vocab) or ""):
        if not probe:
            continue
        if probe.endswith(" focus") and probe not in FOCUS_KEEP:
            return "focus"
        if probe in FRAMING_TAGS:
            return "framing"
        if probe in VIEWPOINT_TAGS:
            return "viewpoint"
        if probe in GAZE_TAGS:
            return "gaze"
        if ACT_RE.match(probe):
            return "act"

    role = _tn.role_of(t)
    if role in ("expression", "hair", "clothing", "lighting", "scene", "pose",
                "body"):
        # role_of files gaze tags under 'expression'; keep them apart, the two
        # are different slots and conflating them is how 'to the viewer' became
        # 'looking at viewer' instead of 'pov'.
        if role == "expression" and t in GAZE_TAGS:
            return "gaze"
        return role

    # Last resort: ask the graph. An unknown tag sits among its neighbours, and
    # if they overwhelmingly belong to one slot then so does it. This is what
    # keeps the slot map from trailing the vocabulary the way a hand list does.
    return _slot_by_neighbours(t, banks)


_NEIGHBOUR_CACHE = {}


def _slot_by_neighbours(tag, banks, top=40, margin=0.40):
    """Which slot do this tag's neighbours belong to?

    WEIGHT THE VOTE BY SPECIFICITY, not by edge strength. Every tag's strongest
    neighbours are hubs -- '1girl', 'long hair', 'solo', 'breasts' -- which sit
    beside everything and therefore say nothing about what this tag IS. Voting
    on raw edge weight gave 'disgust' 18% expression and left it unclassified,
    which kept it out of the contradiction table entirely: the one tag the
    original cafe fault turned on. Discounting hubs puts 'open mouth', 'blush'
    and 'frown' in charge of the answer, which is where it actually lives.
    """
    if tag in _NEIGHBOUR_CACHE:
        return _NEIGHBOUR_CACHE[tag]
    net = (banks or {}).get("_net")
    verdict = None
    row = getattr(net, "fwd", {}).get(tag) if net else None
    spec = getattr(net, "spec", {}) if net else {}
    if row:
        ranked = sorted(row.items(), key=lambda kv: -kv[1] * spec.get(kv[0], 1.0))
        ranked = [(nb, w * spec.get(nb, 1.0)) for nb, w in ranked[:top]]
        votes, total = {}, 0.0
        for nb, w in ranked:
            r = _tn.role_of(nb)
            if r == "expression" and nb in GAZE_TAGS:
                r = "gaze"
            elif nb.endswith(" focus") and nb not in FOCUS_KEEP:
                r = "focus"
            elif nb in FRAMING_TAGS:
                r = "framing"
            elif nb in VIEWPOINT_TAGS:
                r = "viewpoint"
            if r:
                votes[r] = votes.get(r, 0.0) + w
                total += w
        if total > 0 and votes:
            best, score = max(votes.items(), key=lambda kv: kv[1])
            if score / total >= margin:
                verdict = best
    _NEIGHBOUR_CACHE[tag] = verdict
    return verdict


def slot_states(text, user_tags, banks=None):
    """-> {slot: 'filled' | 'claimed' | 'free'}"""
    low = (text or "").lower()
    filled = set()
    for t in user_tags or ():
        s = slot_of(t, banks)
        if s:
            filled.add(s)

    # 'looking at the viewer' says where the EYES point, not where the CAMERA is.
    # Both contain 'at the viewer', so the gaze phrasings come out before the
    # viewpoint probe reads the line -- the same order prompt_enhancer uses.
    vp_low = GAZE_PHRASE_RE.sub(" ", low)

    states = {}
    for slot, probe in SLOT_PROBES.items():
        target = vp_low if slot == "viewpoint" else low
        if slot in filled:
            states[slot] = "filled"
        elif probe.search(target):
            states[slot] = "claimed"
        else:
            states[slot] = "free"
    for slot in filled:
        states.setdefault(slot, "filled")
    return states


def budget(states, slot, want):
    """how many tags the generator may add to `slot`."""
    st = states.get(slot, "free")
    if st == "claimed":
        return 0                       # never guess at words we could not parse
    if st == "filled" and slot in EXCLUSIVE:
        return 0                       # the user already settled it
    return want

# ---------------------------------------------------------------------------
# Who is in frame, and what may therefore be said about them.
#
# One definition, used by the generator AND the analysis tool. They had separate
# copies of the focus rule once and promptly disagreed with each other, so the
# subject model lives here and both import it.
# ---------------------------------------------------------------------------

import re as _re

_COUNT_RE = _re.compile(r"^(\d+)\+?(girls?|boys?|others?)$")

# TWO AXES, NOT ONE. ANATOMY is about what a body HAS: a penis-bearer
# (male or futa) satisfies `penis`; a female body (female or futa)
# satisfies `breasts`. A tag NAMED for a sex -- `male chest`, `male
# focus`, `mature female` -- is about what a body IS, and a futanari is
# not male: androgynous, closer to female, with a penis. Folding the
# word into the anatomy gate would have let `male chest` onto a futa
# (the author's: "male chest on a futa cast would be horrible"). `pov`
# qualifiers name the camera holder, not a body in the cast.
FEMALE_ANATOMY = _re.compile(
    r"\b(breasts?|cleavage|sideboob|underboob|boob|areolae?|nipples?|pussy|"
    r"vagina|vulva|clitoris|womb|ovaries)\b")
MALE_ANATOMY = _re.compile(
    r"\b(penis|testicles|scrotum|erection|foreskin|cock|balls)\b")
# needs an actual male: the word, or a male-crossdresser kind ('josou seme'
# is a crossdressing MAN topping; it was emitted for a futa on a woman)
MALE_NAMED = _re.compile(r"\bmale(?! pov)\b|\bjosou\b|\botoko no ko\b|"
                         r"\btraps?\b|\bfemboys?\b|\bshota\b")
FUTA_NAMED = _re.compile(r"\bfuta\w*(?! pov)\b|\bnewhalf\b")  # needs a futanari
FEMALE_NAMED = _re.compile(r"\bfemale(?! pov)\b")  # female or futa

# Acts that need somebody on the other end. With one subject in frame these are
# only coherent when the camera is the partner (pov).
PARTNERED_ACTS = {
    "handjob", "fellatio", "paizuri", "footjob", "cunnilingus", "irrumatio",
    "vaginal sex", "anal sex", "sex", "sex from behind", "missionary",
    "cowgirl position", "reverse cowgirl position", "doggystyle", "standing sex",
    "mating press", "prone bone", "deep penetration", "kissing", "hug",
    "standing missionary", "straddling paizuri", "sitting on face", "spooning",
    "squatting cowgirl position", "group sex", "gangbang", "double penetration",
    "suspended congress", "full nelson", "piledriver", "reach-around",
    "grabbing another's breast", "clothed sex", "happy sex", "imminent penetration",
}

# Content that is explicit however it arrives. The pose BANKS are gated by spice
# already, but the network is not: 'standing missionary' is a graph neighbour of
# 'standing' and walked straight into a spice=sfw prompt because netfill asks the
# network directly. Gating the tag rather than the bank closes every such route
# at once.
EXPLICIT_RE = _re.compile(
    r"\b(sex|fellatio|irrumatio|paizuri|handjob|footjob|rimjob|cunnilingus|"
    r"penetration|penetrated|creampie|ejaculation|cum|semen|masturbation|"
    r"fingering|anal|vaginal|penis|pussy|vulva|clitoris|testicles|nipples?|"
    r"areolae?|anus|deepthroat|bukkake|gangbang|orgy|dildo|vibrator|"
    r"missionary|doggystyle|cowgirl position|mating press|prone bone|"
    r"piledriver|full nelson|suspended congress|ahegao|orgasm|"
    # a futanari tag is never a safe tag: `full-package futanari` was
    # unmeasured, matched nothing here, and read as SAFE
    r"futa\w*|newhalf|dickgirl|shemale)\b")

# Bare nudity and heavy exposure: fine at suggestive, wrong at sfw.
# the FALLBACK for tags the floor table has not measured: it speaks in
# stems, because `nudist` and `nudity` fell through an exact-word list
# and read as safe
SUGGESTIVE_RE = _re.compile(
    r"\b(nud(?:e|es|ist|ists|ism|ity)|naked|topless|bottomless|no panties|nipple slip|areola slip|"
    r"spread legs|wide spread legs|presenting|upskirt|underboob|"
    r"bottomless|exposed)\b")


def subject_census(tags):
    """-> dict(females, males, others, total, solo, solo_focus, pov)

    The number every count-sensitive rule consults. Read from the count tags
    rather than guessed: '1girl, 1boy' is two subjects and licenses two of
    everything per-subject, while '1girl' alone licenses one.
    """
    low = {t.lower() for t in tags}
    females = males = others = futa = 0
    for t in low:
        m = _COUNT_RE.match(t)
        if m:
            n, kind = int(m.group(1)), m.group(2)
            if kind.startswith("girl"):
                females = max(females, n)
            elif kind.startswith("boy"):
                males = max(males, n)
            else:
                others = max(others, n)
        elif t == "multiple girls":
            females = max(females, 2)
        elif t == "multiple boys":
            males = max(males, 2)
        elif t == "1other":
            others = max(others, 1)
        elif t in ("1futa", "futanari"):
            futa = max(futa, 1)
    # A FUTA HAS A FEMALE BODY; A '1other' NEED NOT. Counting them together let
    # '1other' license 'large breasts' on a humanoid dachshund.
    return {"females": females, "males": males, "others": others + futa,
            "futa": futa,
            "total": females + males + others + futa,
            "solo": "solo" in low, "solo_focus": "solo focus" in low,
            "no_humans": "no humans" in low,
            # Is any living thing depicted? Poses, gazes and expressions are fine
            # on an owl and wrong on a landscape, and PERSON_TAG deliberately
            # excludes them so animals keep them -- so the census has to say
            # whether there IS an animal.
            "creature": bool(NONHUMAN_SUBJECT.search(" ".join(low))),
            "pov": bool(low & {"pov", "male pov", "female pov"})}


_SAFE_ONLY = None

# Vocabulary that may appear ONLY in a safe image. Youth-coded terms and
# family relationships: a woman with a child, or a brother and sister, is
# ordinary wholesome imagery -- the same words inside a sexual scene are
# not. Matched on whole words so 'childhood friend' is unaffected.
_SAFE_ONLY_RE = _re.compile(
    r"\b(child|children|kid|kids|baby|babies|infant|toddler|preteen|"
    r"teenage|sibling|siblings|brother|brothers|sister|sisters|"
    r"onee-?san|onii-?san|nephew|niece|son|daughter)\b", _re.I)


def census_from_cast(cast, typed=()):
    """The cast reader's counts ({female, male, futa, other}) in the shape
    content_allowed() reads ({females, males, others, total, solo, pov...}).

    ONE SHAPE FOR THE GATE. The fast planner handed the cast dict straight
    to content_allowed: 'total' was missing, so the gate read the frame as
    EMPTY and refused every person-describing tag ('a girl in a park' could
    draw bob cut, twin braids and hime cut -- the three hairstyles
    PERSON_TAG does not name -- and nothing else, 60 seeds out of 60), and
    the anatomy/arity checks died on a KeyError that the draw swallowed as
    'allowed'.
    """
    c = cast or {}
    f, m, fu, o = (int(c.get(k) or 0) for k in ("female", "male", "futa", "other"))
    low = {str(t).lower() for t in (typed or ())}
    total = f + m + o + fu
    return {"females": f, "males": m, "others": o + fu, "futa": fu,
            "total": total,
            # the cast reader has already decided how many are in frame;
            # 'solo' would only repeat total == 1 and, unlike a typed
            # `solo`, must not refuse an act with the viewer (heads_available
            # counts the camera when pov is set)
            "solo": False, "solo_focus": False,
            "no_humans": total == 0,
            "creature": bool(NONHUMAN_SUBJECT.search(" ".join(low))),
            "pov": bool(low & {"pov", "male pov", "female pov", "futanari pov"})}


def safe_only(tag):
    """-> True when this tag is allowed only at the safe level."""
    return bool(_SAFE_ONLY_RE.search(str(tag).lower()))


def content_allowed(tag, census, spice, banks=None):
    """May this GENERATED tag stand, given who is in frame and the spice?

    The user's own tags never come through here -- what they typed is the
    request, contradictions included.
    """
    t = tag.lower().strip()

    # The level is enforced on the TAG, because the network route around the
    # gated banks is wide open. Measured floor first, regex fallback -- see
    # safety_floor(). Legacy level names normalise inside above_ceiling.
    if above_ceiling(t, spice):
        return False

    # SAFE-ONLY VOCABULARY (the author's): family and youth words are fine in a
    # wholesome image and must never appear in anything above it. This is
    # the mirror of above_ceiling -- a tag with a MAXIMUM level rather than
    # a minimum one.
    if safe_only(t) and normalize_level(spice) != "safe":
        return False

    # Anatomy has to belong to somebody who is present.
    if FEMALE_ANATOMY.search(t)             and census["females"] + census.get("futa", 0) == 0             and census["total"] > 0:
        return False
    if MALE_ANATOMY.search(t) and census["males"] + census["others"] == 0             and not census["pov"]:
        return False

    # NOBODY IN THE CAST MEANS NOBODY TO DESCRIBE. Two ways to have nobody: the
    # prompt says `no humans` outright, or the census is simply empty. The
    # second used to slip through -- 'scenery, outdoors, cloud, ruins, building'
    # has no count tag and no `no humans` either, so body and clothing tags were
    # generated freely and a person was then inferred back out of them.
    empty = census.get("no_humans") or census.get("total", 0) == 0
    if empty and PERSON_TAG.search(t):
        return False
    # With nothing alive in frame, a pose or a gaze has no one to belong to:
    # 'no humans, scenery, standing, looking at viewer' describes a landscape
    # that is standing up.
    if empty and not census.get("creature"):
        if slot_of(t, banks) in ("pose", "gaze", "expression"):
            return False

    # An action needs enough people to perform it. The count comes from the
    # measured arity table, so 'gangbang' asks for three where 'kissing' asks
    # for two, instead of both being 'partnered' and treated alike.
    need = arity_of(t, banks)
    if need >= 2:
        if census["solo"]:
            return False
        if heads_available(census) < need:
            return False
    return True


# ---------------------------------------------------------------------------
# Pictures without people in them.
#
# The generator forced '1girl' onto every prompt, so "a frog emerging from dark
# calm water", "a zebra walking", "the owl is soft and fluffy" and "a little
# gnome in a blue hat" all came back as girls with breasts and long hair. The
# booru answer is the tag `no humans` (233,116 posts) -- a real, heavily used
# tag that says exactly this.
# ---------------------------------------------------------------------------

NONHUMAN_SUBJECT = _re.compile(
    r"\b(zebra|frog|whale|owl|bunny|rabbit|snake|lizard|turtle|dachsund|"
    r"dachshund|xenomorph|gnome|alien|dragon|cat|dog|horse|deer|fox|wolf|bear|"
    r"fish|bird|crow|raven|insect|spider|butterfly|goblin|orc|slime|golem|"
    r"robot|mecha|dinosaur|shark|octopus|squid|monster|creature|beast)\b",
    _re.I)

# Any of these means a person IS being described, whatever else the line says.
HUMAN_SUBJECT = _re.compile(
    r"\b(girls?|womb?[ae]n|wom[ae]n|lady|ladies|she|her|hers|boys?|m[ae]n|he|"
    r"his|him|person|people|child|children|character|nun|maid|knight|princess|"
    r"prince|queen|king|witch|nurse|teacher|idol|samurai|elf|chef|scientist|"
    r"barmaid|ballerina|dancer|athlete|worker|vendor|archer|sister|brother|"
    r"twins?|mother|father|wife|husband|human|guy|dude|waitress|priestess|"
    r"miko|succubus|goddess|milf|schoolgirl|bride|officer|soldier|pilot)\b",
    _re.I)

# An anthropomorphic animal IS a character -- it just is not a plain human.
# Words that name a human being outright, as opposed to a role anyone could
# fill. Only these veto the anthropomorphic reading.
HUMAN_CORE = _re.compile(
    r"\b(girls?|wom[ae]n|lady|ladies|she|her|hers|boys?|m[ae]n|he|his|him|"
    r"person|people|human|child|children|guy|dude|mother|father|wife|"
    r"husband|sister|brother|daughter|son)\b", _re.I)


ANTHRO = _re.compile(
    r"\b(humanoid|anthro|anthropomorphic|personified|furry|kemonomimi|"
    r"[a-z]+-?girl|[a-z]+-?boy|catgirl|foxgirl|bunnygirl|wearing|holds?|"
    r"holding|dressed)\b", _re.I)


# ---------------------------------------------------------------------------
# WHAT KIND OF BEING A NOUN NAMES (the author's race concept, 2026-09-03).
# "a knight and a dragon" counted the dragon as `1other` and dressed it in
# a shrug. A creature is in the picture, not in the cast: danbooru tags
# 'a wizard and his owl' as `1boy, owl` -- no `1other`. But the user may
# type "dragon girl", "bunny girl", "elf milf", "goblin futanari": a RACE
# word next to a person word is a MODIFIER of that person -- gender from
# the person word, clothes and features kept. We never roll such kinds;
# typed, they are honoured. Alone, a race word is somebody whose gender the
# booru knows by measure (race_gender.json: 'elf' 95% girl, 'dwarf' 85%
# man, 'orc' a coin toss; `1other` under 3% for every race). Four kinds,
# one classifier, read by the cast reader and by the no-people rule:
#
#   person    -- a human (gender noun, occupation, person-flagged word) or
#                any race + person-word compound ("dragon girl", "cat boy")
#   race      -- a race word standing alone (`race` flag in the concept
#                library, reflag_glosses.RACE) -> rolled by measured share
#   humanoid  -- a creature the text calls anthro/humanoid/furry -> `1other`
#   creature  -- an animal or monster: stays a tag, no slot, no clothes
# ---------------------------------------------------------------------------
_GENDER_HEAD = _re.compile(
    r"\b(girls?|boys?|wom[ae]n|m[ae]n|lady|ladies|guys?|maids?|princess|prince|"
    r"queen|king|mother|father|daughter|son|wife|husband|sister|brother|"
    r"milfs?|futanari|futas?|shemale|trap)$")
_ANTHRO_WORD = _re.compile(
    r"\b(humanoid|anthro|anthropomorphic|personified|furry|kemono)\b")


def _flags_of(noun):
    try:
        from promptstudio.engine.vocab import _gloss_of
        return tuple(_gloss_of(noun)[1] or ())
    except Exception:
        return ()


def is_race(noun):
    """-> True when the concept library flags this word `race`."""
    return (_flags_of((noun or "").lower().strip()) or ("",))[0] == "race"


def subject_kind(noun, text=""):
    """-> 'person' | 'race' | 'humanoid' | 'creature' | None (not a being).

    Person nouns and occupations are decided by the caller
    (bridge._is_humanlike) when this returns None."""
    n = (noun or "").lower().strip()
    if not n:
        return None
    if _GENDER_HEAD.search(n):
        return "person"
    fl = _flags_of(n)
    if fl[:1] == ("race",):
        return "race"
    # a role anyone can fill is a person, whatever else the library says
    # ('knight' is flagged person AND creature)
    if "person" in fl:
        return "person"
    if NONHUMAN_SUBJECT.search(n) or "creature" in fl:
        if _ANTHRO_WORD.search((text or "").lower()):
            return "humanoid"
        return "creature"
    return None


def _race_named(text):
    """-> True when the text names a race word on its own."""
    low = _re.sub(r"[^a-z0-9' ]", " ", (text or "").lower())
    words = low.split()
    for i, w in enumerate(words):
        for span in (2, 1):
            if i + span <= len(words) and is_race(" ".join(words[i:i + span])):
                return True
    return False


def nonhuman_character(text):
    """An anthropomorphic creature IS a character, but not a girl.

    'A humanoid miniature dachshund is a chef' and 'an anthro fox in a
    hoodie' both describe somebody doing something -- so suppressing the
    subject would be wrong -- but calling either of them '1girl' is just as
    wrong. Danbooru's answer is `1other` (127,133 posts). A race word alone
    is NOT this: its gender is measured (see subject_kind).
    """
    t = (text or "").lower()
    if not NONHUMAN_SUBJECT.search(t):
        return False
    # Only a word that names a HUMAN BEING rules this out. An occupation does
    # not: 'a humanoid miniature dachshund is a chef' says chef, and a chef is
    # a job, not a species.
    if HUMAN_CORE.search(t):
        return False
    return bool(ANTHRO.search(t))


def scene_has_no_people(text):
    """Does this request describe a picture with nobody in it?

    Deliberately conservative in both directions: it needs a non-human subject
    named outright, no human word anywhere, and nothing suggesting the creature
    is anthropomorphic. "a humanoid miniature dachshund is a chef" keeps its
    character; "a zebra walking across a surface" does not get given one.
    A race word (elf, goblin, gnome) names somebody, not nobody.
    """
    t = (text or "").lower()
    if not NONHUMAN_SUBJECT.search(t):
        return False
    if HUMAN_SUBJECT.search(t) or ANTHRO.search(t) or _race_named(t):
        return False
    return True


# Tags that describe a PERSON. With `no humans` in the prompt these are
# contradictions, and the generator would otherwise cheerfully produce
# "no humans, large breasts, long hair, looking at viewer".
PERSON_TAG = _re.compile(
    r"\b(breasts?|cleavage|sideboob|underboob|nipples?|navel|thighs?|"
    r"collarbone|midriff|hips?|ass|butt|buttocks|armpits?|foot|feet|toes|"
    r"soles|barefoot|legs?|"
    r"hair|bangs|ponytail|twintails|braid|blush|smile|frown|sweat|mole|"
    r"freckles|makeup|lipstick|nails|skindentation|"
    r"shirt|dress|skirt|bikini|panties|bra|gloves|thighhighs|uniform|"
    r"jacket|coat|cape|cloak|hat|jewelry|earrings|necklace|boots|shoes|"
    r"heels|socks|apron|nude|naked|barefoot|soaking feet|"
    r"hand on|arms? (up|behind|crossed)|1girl|1boy|1futa|solo)\b", _re.I)
# Deliberately NOT here: eyes, face, mouth, looking at viewer, sitting,
# standing, lying, full body. Animals do all of those, and danbooru tags them
# that way -- vetoing them would strip an owl of everything but the word owl.


# What a stated attribute drags along with it. 'futanari Tifa Lockhart fucks
# Aerith' binds 'futanari' to Tifa, but the parser then derives 'penis' and
# '1futa' from it -- and those were left for the profile scorer to place, which
# handed the penis to Aerith. A derived tag belongs to whoever the tag it came
# from belongs to.
ATTR_IMPLIES = {
    "futanari": _re.compile(
        r"^(1futa|dickgirl|penis|huge penis|large penis|small penis|erection|"
        r"testicles|penis on |penis under |futa )", _re.I),
    "pregnant": _re.compile(r"^(pregnant|large belly|navel bulge)$", _re.I),
    "milf": _re.compile(r"^(mature female|milf)$", _re.I),
    "muscular female": _re.compile(r"^(muscular|abs|biceps|toned)", _re.I),
}


def implied_by(attr, tag):
    """does `tag` come along with the stated attribute `attr`?"""
    pat = ATTR_IMPLIES.get(attr.lower().strip())
    return bool(pat and pat.match(tag.lower().strip()))


# ---------------------------------------------------------------------------
# How many people an action needs, measured rather than listed.
#
# PARTNERED_ACTS above is a hand set and covers what somebody thought of. This
# reads build_arity.py's measurement instead: solo share decides whether an
# action is something a person does alone, and the group share separates a pair
# from a crowd. The hand set stays as the fallback for anything unmeasured, and
# an action nothing knows about is treated as arity 1 -- the permissive answer,
# which is why build_arity.py prints its undecided list rather than hiding it.
# ---------------------------------------------------------------------------

_ARITY = None


def _arity_table():
    global _ARITY
    if _ARITY is None:
        _ARITY = {}
        import json as _json
        import os as _os
        p = _paths.data("arity.json")
        if _os.path.exists(p):
            try:
                with open(p, encoding="utf-8-sig") as f:
                    _ARITY = {k: v.get("arity", 1)
                              for k, v in _json.load(f).items()}
            except Exception:
                _ARITY = {}
    return _ARITY


# WHO IS PAIRED WITH WHOM. One definition: bridge._PAIRING maps cast kinds
# to these tags, and every arity check reads them here. `hetero` was being
# drawn as a SELF-ACTION on a lone dwarf at safe level, because the arity
# table did not list it and a tag the table does not know reads as solo.
# A pairing needs two people by definition.
PAIRING_TAGS = frozenset(("hetero", "yuri", "yaoi", "futa with female",
                          "futa with male", "futa with futa"))


def arity_of(tag, banks=None):
    """least number of people this action needs in frame"""
    t = tag.lower().strip()
    if t in PAIRING_TAGS:
        return 2
    table = _arity_table()
    if t in table:
        return table[t]
    base = _strip_qualifier(t, (banks or {}).get("_scanvocab") or {})
    if base and base in table:
        return table[base]
    if t in PARTNERED_ACTS or (base and base in PARTNERED_ACTS):
        return 2
    return 1


def heads_available(census):
    """subjects who can take part -- the camera counts when it is a participant"""
    return census.get("total", 0) + (1 if census.get("pov") else 0)


# ---------------------------------------------------------------------------
# A greyscale picture has no hue in it.
#
# Structural rather than measured pair by pair, because the cross product is
# three styles against every colour tag in the vocabulary. The measurement that
# justifies it, conditioned on `solo`:
#     greyscale + red dress 0.016 | monochrome + red dress 0.025
#     monochrome + purple cape 0.033 | monochrome + brown hair 0.033
#     monochrome + blue eyes 0.088 | sepia + red dress 0.089
# and the exception that makes it a rule rather than a blanket ban:
#     monochrome + spot color 19.2 -- one colour on a grey ground is a
#     technique, not a contradiction.
#
# Black, white and grey are not hues and stay legal: 'monochrome, black dress'
# is an ordinary picture.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# How spicy did the user actually ask for?
#
# Until now the dropdowns could never overrule the prompt, which made every
# "auto" option decorative -- the prompt won regardless, so choosing auto and
# choosing anything else came to the same thing whenever the prompt disagreed.
# Splitting the two makes each a real choice:
#
#     auto            read the level out of the prompt (what used to happen)
#     sfw/sug/expl    that level is the answer, and content above it is removed
#                     even when the user typed it
#
# Removing something the user typed is a strong move, so the caller is expected
# to SAY it did -- see Scene.reading(), which reports what the ceiling dropped.
# ---------------------------------------------------------------------------

# FOUR NAMED LEVELS, and the name IS the tag Anima emits. Danbooru rates on
# four (general/sensitive/questionable/explicit) and so does Anima's safety
# vocabulary; the old three-level scale collapsed sensitive+questionable into
# one "suggestive" and could not tell a bikini from full nudity -- exactly the
# distinction that matters most. The legacy names stay accepted at every entry
# point: "sfw" was the safe band, "suggestive" admitted the questionable band
# (SUGGESTIVE_RE covers nudity), so it maps to nsfw.
LEVELS = ("safe", "sensitive", "nsfw", "explicit")
SPICE_ORDER = {"safe": 0, "sensitive": 1, "nsfw": 2, "explicit": 3,
               "sfw": 0, "suggestive": 2}

def normalize_level(name):
    n = str(name or "").lower().strip()
    if n in ("sfw",):
        return "safe"
    if n in ("suggestive",):
        return "nsfw"
    return n if n in LEVELS else n


def _load_floors():
    p = _paths.data("safety_floor.json")
    try:
        with open(p, encoding="utf-8-sig") as f:
            raw = (_json.load(f) or {}).get("floors") or {}
        return {t: v.get("floor") for t, v in raw.items() if v.get("floor")}
    except Exception:
        return {}


SAFETY_FLOOR = _load_floors()


# Clothing worn wrongly, removed, or displaced. Matched as whole words so
# 'pullover' and 'opening' are untouched.
_DISPLACED_RE = _re.compile(
    r"\b(aside|untied|unzipped|undone|unbuttoned|unworn|pull|pulled|"
    r"lifted|lowered|removed|open|opened|slip|slingshot|see-through|"
    r"partially|displaced|half-removed)\b", _re.I)


def safety_floor(tag):
    """The mildest rating danbooru actually applies to this tag.

    MEASURED FIRST, REGEX AS FALLBACK. safety_floor.json samples 200 random
    posts per emittable tag; the keyword regex mis-called 62 of 485 tags, and
    always milder in the dangerous direction -- `bikini` is rated general on 0
    of 695,566 posts yet the regex read it as sfw. The regex only speaks for
    tags too small or too new to measure. No tag IS a rating (`kissing` runs
    25% general to 23% explicit); the floor is the mildest rating holding a
    real share, and a prompt's level is the max over its tags' floors.
    """
    t = str(tag).lower().strip()
    lv = SAFETY_FLOOR.get(t)
    if lv:
        return lv
    if EXPLICIT_RE.search(t):
        return "explicit"
    if SUGGESTIVE_RE.search(t):
        return "sensitive"     # the mildest reading of the old nudity band
    if _DISPLACED_RE.search(t):
        # CLOTHING IN A DISPLACED STATE CARRIES A LEVEL. the author's: "unworn
        # sandals, untied bikini top - some of those can be used (and booru
        # usually uses them that way) to imply spice level". They are too
        # small to have been measured, so they fell through to `safe` and
        # 'bikini bottom aside' could land on a safe image. The garment is
        # innocent; the state is the signal.
        return "sensitive"
    return "safe"


def implied_spice(tags):
    """the level a prompt is asking for: the max of its tags' measured floors"""
    worst = "safe"
    for t in tags or ():
        lv = safety_floor(t)
        if SPICE_ORDER[lv] > SPICE_ORDER[worst]:
            worst = lv
            if worst == "explicit":
                break
    return worst


def above_ceiling(tag, spice):
    """is this tag more explicit than the selected level allows?"""
    ceiling = SPICE_ORDER.get(normalize_level(spice), 2)
    return SPICE_ORDER[safety_floor(tag)] > ceiling


def resolve_spice(chosen, tags):
    """-> (level, source). 'auto' asks the prompt; anything else is the answer."""
    if not chosen or chosen == "auto":
        return implied_spice(tags), "prompt"
    return normalize_level(chosen), "selected"
