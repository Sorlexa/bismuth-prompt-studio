#!/usr/bin/env python
"""
prompt_enhancer.py — turn simple tags OR a short sentence into a rich, coherent prompt
for Illustrious (booru tags + weights) or Anima (tags + natural-language line).

Examples
--------
  python prompt_enhancer.py "1girl, solo, large breasts, sitting"
  python prompt_enhancer.py "a girl with big breasts sunbathing on a beach" --mode anima --spice suggestive
  python prompt_enhancer.py "1girl, solo" --spice explicit --location onsen -n 3
  python prompt_enhancer.py "2girls, yuri" --randomize --remix

Key behaviors
-------------
- Natural-language input is parsed into tags (phrase dictionary + bank vocabulary scan);
  in anima mode the original sentence also rides along as prose.
- A conflict engine prevents contradictions: breast size, hair/eye color, skin tone,
  time of day, base pose, camera framing, and per-garment color are one-of groups.
  The user's own tags always win and are never contradicted.
- One concise negative prompt is always emitted (NEG_LIGHT), filtered so it never
  negates anything the positive prompt asked for. --no-negative omits it entirely.
- Default is coherent/strict; --randomize loosens generated picks (user tags still win).
"""

import json
import math
import os
import re
import sys
from collections import Counter
from promptstudio.engine import slots as _sm
from promptstudio import paths as _paths


# ---------------------------------------------------------------------------- vocab
NORMALIZE = {
    # a species compound is one tag: 'bunny girl' read as 'bunny' gave the
    # girl a RABBIT beside her (the animal) instead of rabbit ears
    # the route tags in their canonical form (the concept library knows
    # `anal` and `vaginal`; the two-word forms are not tags)
    "anal sex": "anal", "vaginal sex": "vaginal", "anal fucking": "anal",
    "bunny girl": "rabbit girl", "bunnygirl": "rabbit girl",
    "bunny boy": "rabbit boy", "bunny ears": "rabbit ears",
    "big breasts": "large breasts", "huge tits": "huge breasts", "big tits": "large breasts",
    "big boobs": "large breasts", "boobs": "breasts", "tits": "breasts",
    "girl": "1girl", "a girl": "1girl", "boy": "1boy", "blond hair": "blonde hair",
    "dick": "penis", "cock": "penis",
}

_SINGLE_ALIASES = {}   # single ruled words, merged into WORD_TAGS / WORD_EXPAND below them

# THE RULED ALIASES ARE NORMALISATIONS TOO (2026-09-15): a multi-word phrase
# the rulings map to a booru tag ('portrait shot with bottom lighting' ->
# underlighting, 'watercolor painting style' -> watercolor (medium), 'top
# lighting' -> overlighting) is claimed by the parser like 'big breasts' ->
# large breasts. One owner: alias_map_local.json.
try:
    with open(_paths.data("alias_map_local.json"), encoding="utf-8-sig") as _f:
        for _a, _t in ((json.load(_f).get("aliases") or {}).items()):
            _a, _t = str(_a).lower().strip(), str(_t).lower().strip()
            if " " in _a and _a != _t and _a not in NORMALIZE:
                NORMALIZE[_a] = _t
            elif " " not in _a and _a != _t:
                # a single ruled word ('focused' -> concentrating, 'tomb' ->
                # graveyard) is read like 'blonde' -> blonde hair
                _SINGLE_ALIASES[_a] = _t
except Exception:
    pass

# A ROLE IS A ROLE (the author's 2026-09-04): a typed 'cultist' or 'witch hunter'
# is not mapped onto a neighbouring tag -- the word stays in the prose, and
# the role's appearance is the clothes table's business. No alias map here.

# Viewpoint is a RELATION, not a tag the user ever types, and it is a different
# thing from gaze. 'a handjob to the viewer' places the camera at the receiving
# end (pov); 'looking at the viewer' only says where the eyes point. Collapsing
# the two is how 'to the viewer' came back as 'looking at viewer' with no pov at
# all. Handled ahead of PHRASE_MAP because these are phrasings, not vocabulary.
GAZE_PHRASE_RE = _sm.GAZE_PHRASE_RE   # one definition, shared

VIEWPOINT_MAP = [
    (r"\b(to|at|toward|towards|into) (the )?(viewer|camera|you)\b", ["pov"]),
    # the POSSESSIVE viewer: "sucking a viewer's dick" / "your cock" puts
    # the camera at the receiving end just as "to the viewer" does
    (r"\b(?:the )?(?:viewer|camera)'s\b|\byour (?:dick|cock|penis|erection|hand|lap|face)\b", ["pov"]),
    (r"\bfacing (the )?(viewer|camera)\b", ["pov"]),
    (r"\b(pov|point of view|first[- ]person)\b", ["pov"]),
    (r"\bfrom (the )?(viewer|camera)(?:'s)? (perspective|point of view)\b", ["pov"]),
    (r"\bover[- ]the[- ]shoulder\b", ["over the shoulder"]),
    (r"\bfrom behind\b|\bfrom the (back|rear)\b", ["from behind"]),
    (r"\bfrom above\b|\bbird.?s[- ]eye|\boverhead\b|\bhigh angle\b", ["from above"]),
    (r"\bfrom below\b|\bworm.?s[- ]eye|\blow angle\b", ["from below"]),
    (r"\bfrom the side\b|\bside view\b", ["from side"]),
]

# NOTE: these mappings deliberately do NOT name a partner. They used to
# carry "1boy" each, which meant six copies of one decision -- and none of
# them could know that `pov` makes the VIEWER the partner, so "a blowjob to
# the viewer" invented a second man who then had to be placed in the prose.
# The single partner rule further down owns that call now.
# THE ONE GENDER VOCABULARY. NOUNS name a being and can be counted ("a
# mother and her daughter" is two women); pronouns prove presence only.
# `female`/`male` are adjectives ("female characters") and never count.
# Occupations are NOT here: whether a knight or a nurse is a man or a
# woman is a measured roll in bridge.resolve_cast, not a word list.
# bridge imports these; nothing else defines gender words.
FEMALE_NOUN_LIST = [
    "girl", "girls", "woman", "women", "lady", "ladies", "milf", "milfs",
    "mother", "mothers", "mom", "moms", "wife", "wives", "girlfriend",
    "girlfriends", "mistress", "maiden", "maidens", "queen", "queens",
    "princess", "princesses", "empress", "nun", "nuns", "witch", "witches",
    "goddess", "bride", "priestess", "miko", "succubus", "knightess",
    "widow", "schoolgirl", "schoolgirls", "gyaru", "daughter", "daughters",
    "sister", "sisters", "aunt", "aunts", "niece", "nieces", "grandmother",
    "grandmothers", "stepmother", "stepmothers"]
MALE_NOUN_LIST = [
    "boy", "boys", "man", "men", "guy", "guys", "dude", "dudes", "lad",
    "lads", "father", "fathers", "dad", "dads", "husband", "husbands",
    "boyfriend", "boyfriends", "gentleman", "gentlemen", "dilf", "dilfs",
    "king", "kings", "prince", "princes", "son", "sons", "brother",
    "brothers", "uncle", "uncles", "nephew", "nephews", "grandfather",
    "grandfathers", "stepfather", "stepfathers"]
# THE DICTIONARY'S PEOPLE (2026-09-17): 'a lass reading' cast no one. The
# lexicon lists the words that name a woman or a man by what they are
# (damsel, bloke, lass), read from WordNet's senses; the lists above stay
# the author's. Words whose senses are a child's are cast like 'schoolgirl'
# and refused like 'young girl' (bridge.youth_tags); sexualised ones are
# never cast. Occupations are not among them (see the note above).
_HAND_GENDER_NOUNS = frozenset(FEMALE_NOUN_LIST) | frozenset(MALE_NOUN_LIST)
try:
    from promptstudio.engine import lexicon as _lexicon
    FEMALE_NOUN_LIST = FEMALE_NOUN_LIST + sorted(_lexicon.people("female") - set(FEMALE_NOUN_LIST))
    MALE_NOUN_LIST = MALE_NOUN_LIST + sorted(_lexicon.people("male") - set(MALE_NOUN_LIST))
except Exception:
    pass
FEMALE_WORDS = "|".join(FEMALE_NOUN_LIST) + "|females?|she|her|hers"
MALE_WORDS = "|".join(MALE_NOUN_LIST) + "|males?|he|him|his"


def _plurals_of(nouns):
    s = set(nouns)
    return {w for w in s if (w.endswith("men") and w[:-3] + "man" in s)
            or (w.endswith("s") and (w[:-1] in s or w[:-2] in s or (w.endswith("ies") and w[:-3] + "y" in s)))}


# a word that is also a verb names somebody only after one of these, with
# at most one word between ('a skilled tailor', 'her widow', 'two wenches')
ARTICLE_RE = (r"(?:a|an|the|one|two|three|four|five|\d+|my|your|his|her|its|our|their|"
              r"this|that|these|those|another|some|every|each)")
_GENDER_MOD = {"re": None, "pl": None, "det": None}


def _article_people(text):
    """the dictionary's gendered words that are also verbs ('wench',
    'chairman', lexicon people female_det / male_det) read as 'woman' /
    'man' where an article stands before them"""
    if _GENDER_MOD["det"] is None:
        try:
            fd, md = _lexicon.people("female_det"), _lexicon.people("male_det")
        except Exception:
            fd, md = frozenset(), frozenset()
        words = sorted((fd | md) - _HAND_GENDER_NOUNS, key=len, reverse=True)
        _GENDER_MOD["det"] = (re.compile(r"\b(" + ARTICLE_RE + r"\s+(?:[a-z-]+\s+)?)(" + "|".join(map(re.escape, words))
                                         + r")\b", re.I) if words else None, fd, _plurals_of(fd) | _plurals_of(md))
    rx, fd, pl = _GENDER_MOD["det"]
    if rx is None:
        return str(text or "")

    def _sub(m):
        noun = m.group(2).lower()
        many = noun in pl
        return m.group(1) + (("women" if many else "woman") if noun in fd else ("men" if many else "man"))
    return rx.sub(_sub, str(text or ""))


def claim_gender_modifiers(text):
    """A GENDER WORD WRITTEN BEFORE A NOUN DECIDES THE NOUN (the author,
    2026-09-17: the dictionary's gender leans are right, and a knight is
    ambiguous until 'male', 'female' or 'futanari' is written before it).
    'a female mariner' was one woman AND one man: 'female' counted as a
    woman and 'mariner' -- a man by its definition -- as a man. The noun a
    gender word modifies (one word may sit between: 'female sea captain')
    takes that gender: a noun of the other gender is read as 'woman' /
    'man' (plural kept), and a dictionary noun after 'futa'/'futanari' as
    'futanari' -- the dictionary does not know futanari, so the typed word
    wins over it ('futanari girl' is the author's own idiom and stays).
    Used by the parser and the cast reader on their counting text only;
    the prompt's tags still see the noun."""
    text = _article_people(text)
    if _GENDER_MOD["re"] is None:
        nouns = sorted(set(FEMALE_NOUN_LIST) | set(MALE_NOUN_LIST), key=len, reverse=True)
        _GENDER_MOD["re"] = re.compile(
            r"\b(female|male|futanari|futa)(\s+(?:[a-z-]+\s+)?)(" + "|".join(map(re.escape, nouns)) + r")\b",
            re.I)
        _GENDER_MOD["pl"] = _plurals_of(FEMALE_NOUN_LIST) | _plurals_of(MALE_NOUN_LIST)
    fem, mal = set(FEMALE_NOUN_LIST), set(MALE_NOUN_LIST)

    def _sub(m):
        mod, gap, noun = m.group(1).lower(), m.group(2), m.group(3).lower()
        many = noun in _GENDER_MOD["pl"]
        if mod in ("futa", "futanari"):
            # the author's own words ('futanari girl') keep their reading;
            # the dictionary's genders yield
            if (noun in mal or noun in fem) and noun not in _HAND_GENDER_NOUNS:
                return mod + gap + ("futanari" if not many else "futas")
            return m.group(0)
        if mod == "female" and noun in mal and noun not in fem:
            return mod + gap + ("women" if many else "woman")
        if mod == "male" and noun in fem and noun not in mal:
            return mod + gap + ("men" if many else "man")
        return m.group(0)
    return _GENDER_MOD["re"].sub(_sub, str(text or ""))

PHRASE_MAP = [
    # PAIRING AND COUNTS ARE ENGINE FACTS, NOT PHRASE SIDE-EFFECTS.
    # These mappings used to add `hetero` to every partnered act and
    # `2boys, multiple boys` to gangbang/threesome -- so "two girls
    # having anal sex" came out `2girls, anal sex, hetero`, and "a
    # woman in a threesome" invented two men the user never wrote.
    # bridge._PAIRING already derives the pairing from the resolved
    # cast (measured: hetero / yuri / yaoi / futa with ...), and the
    # cast reader owns counts (threesome/gangbang now carry a total
    # in _PAIR_EXTRA; the gender split is rolled from measured
    # weights). One owner each; the maps keep only the act.

    # any determiner or possessive may sit between the verb and the noun:
    # "sucking a viewer's dick" matched nothing and lost the act
    (r"suck(ing|s|ed)? (?:on )?(?:(?:a |the |his |your )?(?:\w+'s )?)?(dick|cock|penis)",
     ["fellatio", "penis", "oral"]),
    (r"blow ?job", ["fellatio", "penis"]),
    (r"giv(ing|es) (a )?titjob|paizuri|tit ?fuck", ["paizuri", "penis"]),
    (r"rid(ing|es|den) (a |his )?(dick|cock|penis)|cowgirl", ["cowgirl position", "vaginal", "sex", "girl on top"]),
    # THE OBJECT NAMES THE ROUTE. "fucking her in the ass" mapped to
    # `vaginal sex` because the verb alone decided; 'in the ass' only
    # added the body part. A sex verb followed by an anal object is anal;
    # by a vaginal object, vaginal; with no route stated, `sex` alone --
    # the booru's general tag -- and nothing is assumed. Canonical tags
    # (`anal`, `vaginal`), which the concept library knows.
    # two shapes: "fucking her IN the ass" (any few words, then a
    # preposition) and "fucks her ass" / "pounding a girl's tight pussy"
    # (the object right after the verb, with a determiner, a possessive
    # and one adjective allowed). "with a big ass" never matches: 'with'
    # is not a determiner.
    (r"(?:fuck(?:ing|s|ed)?|pound(?:ing|s|ed)?|rail(?:ing|s|ed)?|penetrat(?:ing|es|ed|e)|"
     r"ramm(?:ing|ed)|thrust(?:ing|s)?|slamm(?:ing|ed)|takes?|taking|took)"
     r"(?:(?:\s+\w+){0,4}?\s+(?:in|up|into)"
     r"|(?:\s+(?:the|her|his|their|my|your|a|an|that|this))?(?:\s+\w+'s)?"
     r"(?:\s+(?:tight|fat|big|round|little|bare|wet|gaping|plump)))"
     r"\s+(?:the |her |his |their |my |your )?"
     r"(?:ass|asshole|anus|butt|rear|backdoor|behind)\b"
     r"|(?:fuck(?:ing|s|ed)?|pound(?:ing|s|ed)?)\s+(?:her|his|their|my|your)\s+"
     r"(?:ass|asshole|anus|butt)\b"
     r"|(?:ass|butt)[- ]?fuck\w*|anal(?:ly)? (?:sex|fucked|penetrat\w+)|\banal\b",
     ["anal", "sex"]),
    (r"(?:fuck(?:ing|s|ed)?|pound(?:ing|s|ed)?|rail(?:ing|s|ed)?|penetrat(?:ing|es|ed|e)|"
     r"ramm(?:ing|ed)|thrust(?:ing|s)?|slamm(?:ing|ed))"
     r"(?:(?:\s+\w+){0,4}?\s+(?:in|into|inside)"
     r"|(?:\s+(?:the|her|their|my|your|a|an|that|this))?(?:\s+\w+'s)?"
     r"(?:\s+(?:tight|wet|dripping|bare|little|pink|shaved|hairy))?)"
     r"\s+(?:the |her |their |my |your )?"
     r"(?:pussy|vagina|cunt|womb|slit)\b"
     r"|\bvaginal(?: sex)?\b", ["vaginal", "sex"]),
    # passive voice matters: 'being fucked' / 'gets railed' imply a partner too
    (r"(having|has) sex|fuck(ing|s|ed)|railed|pounded|penetrated|"
     # NOT "taken from behind": a photograph taken from behind reinforced
     # glass is not a sex act, and this pattern turned one into "vaginal
     # sex, hetero, sex from behind". "from behind" alone is already the
     # camera tag, and a real request says "fucked from behind", which the
     # verb catches.
     r"mating press|doggystyle|missionary|balls deep|"
     r"impaled on|bouncing on", ["sex"]),
    (r"gangbang|group sex|threesome|double penetration",
     ["group sex"]),
    (r"eating (her )?out|cunnilingus|going down on", ["cunnilingus", "oral"]),
    (r"creampie|cum(ming)? inside|filled with cum", ["cum in pussy", "creampie"]),
    (r"handjob|jerking (him )?off|stroking (his )?(dick|cock|penis)", ["handjob", "penis"]),
    (r"masturbat(ing|es)|touching herself|fingering herself", ["masturbation", "fingering"]),
    (r"two girls|2 girls", ["2girls"]),
    (r"kissing", ["kissing"]),
    (r"sunbath(ing|es)", ["sunbathing", "lying"]),
    (r"tak(ing|es) a (shower|bath)", ["shower", "bathing", "wet"]),
    (r"on (a|the) beach", ["beach"]),
    (r"in (a|the) pool", ["pool"]),
    (r"in (an|the) onsen|hot spring", ["onsen"]),
    (r"in (a|the) (bed|bedroom)", ["bedroom", "on bed"]),
    (r"at night", ["night"]),
    (r"in (a|the) (office|classroom|gym|library|forest|city)", None),  # scene word extracted below
    (r"look(ing|s) (back )?at (the )?(viewer|camera)", ["looking at viewer"]),
    (r"from behind", ["from behind"]),
    (r"undress(ing|es)|taking off (her )?clothes", ["undressing"]),
    (r"spread(ing|s) (her )?legs", ["spread legs"]),
    (r"bend(ing|s|t) over", ["bent over"]),
    (r"lying (down|on (her )?back)", ["on back", "lying"]),
    (r"wear(ing|s) (a )?bikini", ["bikini"]),
    (r"wear(ing|s) (a )?(school uniform|seifuku)", ["school uniform"]),
    (r"futanari|\bfuta\b|dickgirl|newhalf", ["futanari", "penis"]),
    (r"\byuri\b|lesbian|two girls (kissing|together)", ["yuri", "2girls"]),
    (r"tentacle", ["tentacles", "tentacle sex", "restrained"]),
    (r"bondage|shibari|tied up|\bropes?\b", ["bondage", "rope", "restrained"]),
    (r"monster girl|demon girl|slime girl", ["monster girl", "horns", "tail"]),
    (r"pregnant", ["pregnant", "large belly"]),
    (r"muscular|muscle|abs\b", ["muscular female", "abs", "toned"]),
    (r"nurse", ["nurse"]), (r"\bmaid\b", ["maid"]), (r"\belf\b", ["elf", "pointy ears"]),
    (r"succubus", ["succubus", "demon horns", "demon tail"]),
    (r"\bnaked\b|\bnude\b", ["nude"]),
    (r"smil(ing|es)", ["smile"]),
]

WORD_TAGS = {  # single words safely mappable when found in NL text
    "beach", "pool", "onsen", "bedroom", "office", "classroom", "gym", "library",
    "forest", "city", "night", "sunset", "rain", "snow", "bikini", "lingerie",
    "nude", "topless", "blonde", "redhead", "milf", "elf", "maid", "nurse",
    "shower", "bath", "couch", "kitchen", "car", "train", "wet",
}
WORD_EXPAND = {"blonde": "blonde hair", "redhead": "red hair", "milf": "mature female",
               "bath": "bathing", "car": "car interior"}

# ---------------------------------------------------------------------------- conflicts
BASE_POSES = {"sitting", "standing", "lying", "walking", "kneeling", "squatting",
              "all fours", "on back", "on side", "lying on side", "seiza", "running",
              "crouching", "bent over", "straddling", "top-down bottom-up", "on stomach"}
FRAMINGS = {"close-up", "full body", "upper body", "lower body", "cowboy shot",
            "wide shot", "face closeup", "portrait",
            # the framing scale is one-of end to end; without these the generator
            # happily emitted 'upper body' and 'very wide shot' in one prompt
            "very wide shot", "medium shot", "extreme close-up", "bust shot",
            "establishing shot", "full-length portrait", "close up"}
# The base garment is one-of per subject — layers (jacket, gloves, thighhighs) are
# not, so only the primary outfit belongs here. 'bikini' + 'dress' is not a look.
OUTFITS = {"bikini", "swimsuit", "one-piece swimsuit", "dress", "kimono", "yukata",
           "underwear", "bra", "panties",
           "leotard", "school uniform", "nurse uniform", "maid", "lingerie",
           "nude", "cheerleader uniform", "wedding dress", "gym uniform",
           "business suit", "sundress", "bodysuit", "overalls"}
COLORS = ("black", "white", "red", "blue", "green", "pink", "purple", "yellow",
          "orange", "brown", "grey", "gray", "silver", "blonde", "aqua", "gold")
GARMENTS = ("bikini", "swimsuit", "dress", "bra", "panties", "skirt", "shirt",
            "thighhighs", "stockings", "leotard", "jacket", "hoodie", "gloves",
            "socks", "boots", "heels", "choker", "sweater", "coat", "one-piece")
GAZE = {"looking at viewer", "looking away", "looking back at viewer", "looking back",
        "looking to the side", "looking down", "looking up", "eye contact",
        "closed eyes", "eyes closed", "looking at another", "looking ahead",
        "looking afar", "eyes out of frame", "one eye closed", "no eyes",
        "empty eyes", "rolling eyes"}
GROUPS = {
    "breast_size": {"flat chest", "small breasts", "medium breasts", "large breasts",
                    "huge breasts", "gigantic breasts"},
    "skin": {"pale skin", "fair skin", "tan", "tanned skin", "dark skin", "brown skin", "dark-skinned female"},
    "time": {"day", "night", "sunset", "sunrise", "morning", "dusk", "evening", "noon", "golden hour", "midnight"},
    "hair_length": {"very short hair", "short hair", "medium hair", "long hair",
                    "very long hair", "absurdly long hair", "bald"},
    # One position at a time. 'standing doggystyle' next to 'standing missionary'
    # asks for two incompatible acts in one frame.
    "sex_position": {"missionary", "cowgirl position", "reverse cowgirl position",
                     "doggystyle", "standing doggystyle", "standing missionary",
                     "standing sex", "mating press", "prone bone", "spooning",
                     "suspended congress", "piledriver", "full nelson",
                     "against wall", "lotus position", "amazon position"},
    "base_pose": BASE_POSES,
    "framing": FRAMINGS,
    "outfit": OUTFITS,
    "gaze": GAZE,
    # counts are one-of per gender ('1girl' + '1boy' is fine, '1girl' + '2girls' is not)
    "girl_count": {"1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
                   "multiple girls"},
    "boy_count": {"1boy", "2boys", "3boys", "multiple boys"},
}

# the single-word ruled aliases join the single-word tables (2026-09-15)
try:
    for _a, _t in _SINGLE_ALIASES.items():
        WORD_TAGS.add(_a) if isinstance(WORD_TAGS, set) else WORD_TAGS.__setitem__(_a, True)
        WORD_EXPAND.setdefault(_a, _t)
except Exception:
    pass

# Semantic incompatibility: tags matching one side may not coexist with the other
# (checked both directions). Catches meaning-level clashes that category groups miss.
CONFLICT_PAIRS = [
    # eyes shut vs. visible eye colour / detail / gaze
    # eyelashes/eyeliner stay valid with shut eyes — only visible-iris tags conflict
    (r"^(closed eyes|eyes closed)$",
     r"(?<!half-)(?<!half )\b(red|blue|green|yellow|purple|pink|brown|black|grey|gray|golden|orange|aqua|silver|amber|violet) eyes$|eye contact|looking at viewer|looking back at viewer|detailed eyes|glowing eyes|heterochromia|pupils"),
    # daylight/sun activities vs. night lighting
    (r"sunbath|sunlight|sunny|daylight|^day$|blue sky|noon|beach|poolside",
     r"moonlight|^night$|midnight|starry|nighttime|candlelight|neon lights|dark location|dimly lit"),
    (r"moonlight|^night$|midnight|starry|nighttime",
     r"sunbath|sunlight|sunny|daylight|^day$|blue sky|noon|god rays|dappled sunlight"),
    # indoor scenes vs. outdoor ground/vegetation/scenery.
    # Ambiguous tags (window, curtains, cityscape, rain, snow, sunlight, pool, onsen,
    # balcony, rooftop, flowers, bench) are in NEITHER list on purpose — they occur
    # legitimately on both sides and must not be blocked.
    (r"^(indoors|bedroom|classroom|office|library|locker room|changing room|bathroom|"
     r"kitchen|living room|ceiling|wooden floor|tatami|carpet|rug|couch|sofa|on bed|"
     r"bed sheet|dresser|chalkboard|refrigerator|bathtub|cubicle|hallway|fireplace|"
     r"chandelier|lockers|bookshelf|indoor pool|elevator|closet)$",
     r"^(outdoors|sky|blue sky|cloud|clouds|grass|lawn|tree|trees|forest|woods|ocean|"
     r"sea|beach|sand|sand dune|mountain|mountains|hill|hills|meadow|field|river|lake|"
     r"pond|waterfall|park|street|road|alley|sidewalk|pavement|horizon|waves|cliff|"
     r"desert|jungle|shore|seashore|garden|hedge|bush|forest path|dirt road)$"),
    # water immersion vs. dry-only descriptors
    (r"partially submerged|underwater|in water|onsen|hot spring|bathing",
     r"^dry$|dust|dusty"),
    # clothing state contradictions
    (r"^(nude|completely nude)$",
     r"^(school uniform|dress|sweater|coat|jacket|kimono|yukata|maid outfit|bikini|swimsuit|lingerie|leotard)$"),
    (r"^(bikini|swimsuit)$", r"^(school uniform|kimono|yukata|maid outfit|coat|business suit)$"),
    # cropped-away features vs. tags describing those features
    (r"eyes out of frame|head out of frame",
     r"eyes$|looking at viewer|eye contact|detailed eyes|face|expression|smile|blush"),
    # lying-implied activities vs. upright poses
    (r"sunbath|lying|on back|on stomach|all fours|sleeping|reclining",
     r"^(standing|walking|running|jumping|standing on one leg|contrapposto)$"),
    # camera sanity. NOTE: 'pov' = whose eyes the camera occupies; 'from behind' /
    # 'from below' = the subject's orientation to that camera — these compose freely
    # (pov + from behind is a standard framing) and must NOT be treated as exclusive.
    # Only genuine impossibilities go here: a face close-up can't show a subject
    # facing away unless they turn back (hence the looking-back escape hatch).
    (r"^(face closeup|portrait)$", r"^from behind$", r"looking back|turning head|over.the.shoulder"),
    (r"^close-?up$", r"^(wide shot|scenery|establishing shot|panorama)$"),
    (r"^(pov|male pov)$", r"^(wide shot|scenery|establishing shot)$"),
    # VISIBILITY, measured rather than assumed. Conditioned on 'solo' (multi-subject
    # images hide the effect — one character faces away while another faces the
    # camera), gelbooru gives these lifts against 'from behind':
    #     navel .088  collarbone .079  cleavage .107  stomach .082  abs .100
    #     breast focus .126  navel focus .000   <- all ~10x under-represented
    #     underboob .94  nipples .82  areolae 2.2  pubic hair .91  midriff .45
    #     groin .36  stomach bulge .52                <- perfectly normal, allowed
    # A bent-over or looking-back pose keeps the lower front visible, which is why
    # the second group survives; the navel and collarbone genuinely do not.
    (r"^(from behind|back focus|facing away)$",
     r"^(navel|collarbone|cleavage|stomach|abs|navel focus|breast focus)$"),
    (r"^(from behind|facing away)$", r"^(front view|facing viewer|straight-on)$"),
    # FRAMING vs POSE, measured on gelbooru conditioned on solo. A crop ending above
    # the waist cannot show what the legs are doing, so the pose tag is at best
    # wasted conditioning and at worst fights the framing. Lifts:
    #   portrait    x all fours .005  squatting .010  crossed legs .010  kneeling .011
    #                 spread legs .013  sitting .024  standing .038    (lying .095 ok)
    #   upper body  x kneeling .020  all fours .022  spread legs .022  squatting .025
    #                 crossed legs .046        (sitting .102, bent over .062 ok)
    #   cowboy shot x squatting .047            (everything else >= .058 ok)
    # 'full body' pairs with every pose at >= .85, so it is deliberately absent here.
    (r"^portrait$",
     r"^(sitting|kneeling|squatting|all fours|spread legs|crossed legs|standing)$"),
    (r"^upper body$",
     r"^(kneeling|squatting|all fours|spread legs|crossed legs)$"),
    (r"^cowboy shot$", r"^squatting$"),
    # 'scenery' means the LANDSCAPE is the subject. Measured on gelbooru,
    # conditioned on solo: scenery+from behind lift 4.96, +full body 1.04,
    # +1girl 0.95 - all normal - but +large breasts 0.098, +portrait 0.076 and
    # +face 0.000. So it does not fight having a character in frame, it fights
    # foregrounding their body or their face. The generator was emitting it
    # beside body detail 158 times per 1000 prompts.
    (r"^(scenery|landscape)$",
     r"^(large breasts|huge breasts|gigantic breasts|cleavage|navel|nipples|areolae|face|portrait|face closeup|close-?up|bust shot)$"),
    # A FOCUS tag names what fills the frame, so it has to be inside the crop.
    # "upper body" + "ass focus" asks for a close crop of something the crop cuts off.
    (r"^(upper body|portrait|face closeup|bust shot)$",
     r"^(ass|thigh|feet|foot|leg|groin|lower body) focus$|^lower body$"),
    (r"^(close-?up|face closeup|portrait)$", r"^(full body|very wide shot)$"),
    # KISSING is mouth-to-mouth and face-to-face: it cannot happen while looking
    # back over a shoulder, and an open mouth or a broad smile fights it too.
    (r"^kissing$", r"^(looking back|looking back at viewer|looking away|looking to the side)$"),
    (r"^kissing$", r"^(open mouth|shouting|laughing)$"),
    # near-duplicates: two spellings of one idea waste tokens and can fight
    (r"^blurry background$", r"^blurred background$"),
    (r"^depth of field$", r"^bokeh$", r"blurry"),
]

def _compile_pair(entry):
    """(a, b) or (a, b, escape) -> compiled triple; escape waives the conflict."""
    a, b = entry[0], entry[1]
    esc = entry[2] if len(entry) > 2 else None
    return (re.compile(a, re.I), re.compile(b, re.I),
            re.compile(esc, re.I) if esc else None)


COMPILED_PAIRS = [_compile_pair(e) for e in CONFLICT_PAIRS]
HAIR_COLOR_RE = re.compile(rf"^({'|'.join(COLORS)}) hair$")
EYE_COLOR_RE = re.compile(rf"^({'|'.join(COLORS)}) eyes$")
GARMENT_COLOR_RE = re.compile(rf"^({'|'.join(COLORS)}) ({'|'.join(GARMENTS)})$")
MULTICOLOR_OK = {"multicolored hair", "two-tone hair", "streaked hair", "gradient hair",
                 "colored inner hair", "split-color hair"}
LIGHT_RE = re.compile(r"(light|lit\b|glow|neon|backlig|sunset|sunrise|golden hour|moonl|chiaroscuro|contrast)")
class Conflicts:
    """One-of groups; user tags register first and always win."""

    # traits that belong to a PERSON — with several subjects in frame, several
    # different values are legitimate (two girls, two hair colours). Scene-level
    # facts (time of day, framing, base pose) stay one-of no matter how many people.
    PER_SUBJECT = {"breast_size", "skin", "hair_color", "eye_color",
                   "hair_length", "outfit"}

    # Traits that belong only to a female subject. Allowing one extra value
    # per SUBJECT let a '1boy' in frame license a second breast size — which
    # is not a second person's description, just a contradiction on the girl.
    FEMALE_ONLY = {'breast_size'}

    def __init__(self, strict=True, subjects=1, female_subjects=None):
        self.strict = strict
        self.subjects = max(1, subjects)
        self.female_subjects = max(1, subjects if female_subjects is None
                                   else female_subjects)
        self.taken = {}      # group key -> (tag, was_user)
        self.counts = Counter()   # group key -> how many values accepted
        self.multicolor = False
        self.tags = []       # every accepted tag, for pairwise semantic checks

    def _keys(self, tag):
        t = tag.lower().strip("() ").split(":")[0].strip()
        keys = []
        for g, members in GROUPS.items():
            if t in members:
                keys.append(g)
        if HAIR_COLOR_RE.match(t):
            keys.append("hair_color")
        if EYE_COLOR_RE.match(t):
            keys.append("eye_color")
        # any "<x> uniform" is a primary outfit, however franchise-specific,
        # so two of them are two different outfits on one subject
        if t.endswith(" uniform") or t == "uniform":
            keys.append("outfit")
        m = GARMENT_COLOR_RE.match(t)
        if m:
            keys.append("garment:" + m.group(2))
        # A qualified form must answer to its base tag's groups. OUTFITS lists
        # 'dress' and nothing else, so 'red dress' walked straight past the
        # one-outfit rule on the strength of a colour word. Testing the base
        # closes that for every qualifier at once rather than enumerating them.
        parts = t.split()
        for cut in range(1, len(parts)):
            base = " ".join(parts[cut:])
            for g, members in GROUPS.items():
                if base in members and g not in keys:
                    keys.append(g)
            if (base.endswith(" uniform") or base == "uniform")                     and "outfit" not in keys:
                keys.append("outfit")
        return keys

    def would_accept(self, tag):
        """Dry run of register() — lets the picker skip candidates that are already
        ruled out instead of spending a slot on them and having it filtered away."""
        t = tag.lower().strip("() ").split(":")[0].strip()
        for k in self._keys(tag):
            if k in self.taken:
                if k == "hair_color" and self.multicolor:
                    continue
                if k in self.PER_SUBJECT:
                    allowed = (self.female_subjects
                               if k in self.FEMALE_ONLY else self.subjects)
                    if self.counts[k] < allowed:
                        continue
                return False
        for a, b, esc in COMPILED_PAIRS:
            hit = (a.search(t) and any(b.search(x) for x in self.tags)) or \
                  (b.search(t) and any(a.search(x) for x in self.tags))
            if hit and not (esc and (esc.search(t) or any(esc.search(x) for x in self.tags))):
                return False
        return True

    def register(self, tag, user=False):
        """True if the tag may be added; records group ownership.

        The one-of sanity groups are enforced in EVERY mode, --randomize included:
        randomness means unexpected tag combinations, never a single subject with
        two breast sizes or a scene that is both day and night. What --randomize
        actually switches off is the danbooru coherence bias and profile borrowing
        (see enhance), not the sanity engine."""
        t = tag.lower().strip("() ").split(":")[0].strip()
        keys = self._keys(tag)
        if t in MULTICOLOR_OK:
            self.multicolor = True
        for k in keys:
            if k in self.taken:
                if user:
                    continue  # user contradicting themselves is their call
                if k == "hair_color" and self.multicolor:
                    continue
                # one description per subject: 2girls may have two hair colours,
                # but a lone subject may not have two breast sizes
                if k in self.PER_SUBJECT:
                    allowed = (self.female_subjects
                               if k in self.FEMALE_ONLY else self.subjects)
                    if self.counts[k] < allowed:
                        continue
                return False
        if not user:  # semantic pairwise check against everything accepted so far
            for a, b, esc in COMPILED_PAIRS:
                hit = (a.search(t) and any(b.search(x) for x in self.tags)) or \
                      (b.search(t) and any(a.search(x) for x in self.tags))
                if hit:
                    if esc and (esc.search(t) or any(esc.search(x) for x in self.tags)):
                        continue  # a reconciling tag makes the combination valid
                    return False
        for k in keys:
            self.taken.setdefault(k, (tag, user))
            self.counts[k] += 1
        self.tags.append(t)
        return True


# ---------------------------------------------------------------------------- helpers
# Generic halves of place names — matching on these alone would be meaningless
# ('room' appears everywhere), so only the distinctive half identifies the place.
GENERIC_PLACE_WORDS = {"room", "interior", "indoors", "outdoors", "view", "area",
                       "background", "wall", "floor", "panels", "seat", "hall",
                       "space", "table", "stool", "counter", "grounds", "park",
                       # Time, weather and light words modify a place, they
                       # never identify one. "night festival" matched any
                       # prompt containing "night", so "two boys fighting in
                       # an alley at night" acquired a festival.
                       "night", "nighttime", "day", "daytime", "morning",
                       "evening", "dusk", "dawn", "sunset", "sunrise",
                       "midnight", "noon", "summer", "winter", "spring",
                       "autumn", "rain", "rainy", "snow", "snowy", "storm",
                       "stormy", "cloudy", "sunny", "dark", "bright",
                       "light", "lights", "festival", "party", "scene"}


def scene_matches(scene_tag, text):
    """Does the text name this place? 'on the throne' must match 'throne room',
    and 'in a car' must match 'car interior' — phrase-only matching missed both."""
    st = scene_tag.lower()
    if len(st) >= 3 and re.search(r"\b" + re.escape(st) + r"\b", text):
        return True
    words = st.split()
    if len(words) > 1:
        for w in words:
            if len(w) >= 4 and w not in GENERIC_PLACE_WORDS and \
                    re.search(r"\b" + re.escape(w) + r"\w{0,2}\b", text):
                return True
    return False


def is_natural_language(text):
    """Prose vs. a tag list. Booru tags such as 'looking at viewer' or 'hand on hip'
    contain prepositions, so a stopword alone is not enough — but a leading article
    ('a girl getting fucked') is a reliable prose signal even in short input."""
    t = text.strip()
    commas = t.count(",")
    words = len(t.split())
    # determiner / quantifier opening: 'a girl ...', 'two girls ...', 'her ...'
    if commas == 0 and words >= 3 and re.match(
            r"^(a|an|the|one|two|three|four|several|some|her|his|my)\s", t, re.I):
        return True
    # 4+ words carrying an action verb reads as a sentence, not a tag list
    if commas == 0 and words >= 4 and re.search(r"\b\w{3,}ing\b", t, re.I):
        return True
    if words >= 5 and commas == 0 and re.search(
            r"\b(is|are|was|being|with|while|as|her|his|their|of|from the|into)\b", t, re.I):
        return True
    # A comma-free line of 4+ words containing a preposition or article is prose,
    # even when it opens with a tag-like token: '1girl in a dark alley' was being
    # swallowed whole as a single tag, so the alley was never seen and a selected
    # location could override a place the user had actually named. Reading it as prose
    # is safe now that the n-gram scanner pulls real tags back out of the sentence.
    if commas == 0 and words >= 4 and re.search(
            r"\b(in|on|at|by|near|under|over|inside|outside|beside|behind|a|an|the)\b",
            t, re.I):
        return True
    return words > 7 and commas < words / 5


# Words that are real danbooru tags but which, standing alone in a sentence, are
# almost always ordinary English rather than a tag request.
SCAN_STOP = {"with", "and", "the", "a", "an", "of", "in", "on", "at", "is", "are",
             "her", "his", "their", "him", "she", "he", "they", "it", "to", "for",
             "by", "as", "from", "that", "this", "very", "more", "most", "some",
             "who", "has", "have", "been", "into", "over", "under", "up", "down",
             "out", "off", "one", "two", "get", "gets", "make", "makes", "like"}


# LIGHT-VERB PHRASES. 'giving a blowjob', 'having sex', 'performing
# fellatio' -- the verb is the empty half of the construction and the ACT
# noun after it carries the whole meaning. Scanning it as vocabulary put a
# stray `giving` in the tag line beside `fellatio`, which is not a thing
# anyone asked to see drawn.
#
# It is only grammar when the object IS an act, and the gloss library
# already knows which tags those are: 'giving a gift' keeps the danbooru
# tag `giving`, because a gift is an object. So the class is closed and the
# test is measured -- no per-word blacklist.
LIGHT_VERBS = {"give", "gives", "giving", "have", "has", "having",
               "do", "does", "doing", "perform", "performs", "performing",
               "take", "takes", "taking", "receive", "receives", "receiving",
               "get", "gets", "getting", "make", "makes", "making",
               "engage", "engages", "engaging"}
# determiners between the verb and its object
_LV_SKIP = {"a", "an", "the", "her", "his", "their", "its", "him", "them",
            "some", "another", "one"}

_ACT_TAGS = None


def _act_tags():
    """every tag the gloss library flags as an act, lowercased"""
    global _ACT_TAGS
    if _ACT_TAGS is None:
        _ACT_TAGS = set()
        try:
            with open(_paths.data("tag_glosses.json"), encoding="utf-8-sig") as f:
                for k, v in (json.load(f).get("glosses") or {}).items():
                    if "act" in (v.get("f") or []):
                        _ACT_TAGS.add(str(k).lower())
        except Exception:
            pass
    return _ACT_TAGS


def _is_light_verb(words, i, resolve):
    """-> True when words[i] is the empty half of a light-verb phrase."""
    j = i + 1
    while j < len(words) and words[j] in _LV_SKIP:
        j += 1
    acts = _act_tags()
    if not acts:
        return False
    for size in (3, 2, 1):
        if j + size <= len(words):
            hit = resolve(" ".join(words[j:j + size]))
            if hit and str(hit).lower() in acts:
                return True
    return False


# Nouns that name a slot rather than describe anything: in 'disgusted expression'
# the tag is 'disgust' and 'expression' is scaffolding. Stripping these lets an
# ordinary English phrasing reach the tag it obviously means.
# English the boorus have no tag for at all. 'silver hair' returns zero posts on
# both danbooru and gelbooru -- the tag is 'grey hair' -- so a perfectly ordinary
# request lost its hair colour entirely. Kept deliberately small: everything that
# danbooru's own alias file covers is handled there, and this is only for words
# it has no opinion about.
SYNONYMS = {
    "silver hair": "grey hair", "silver-haired": "grey hair",
    "platinum hair": "grey hair", "ginger hair": "orange hair",
    "auburn hair": "brown hair", "raven hair": "black hair",
    "furious": "angry", "enraged": "angry", "livid": "angry",
    "terrified": "scared", "petrified": "scared",
    "delighted": "happy", "joyful": "happy", "cheerful": "smile",
    "miserable": "sad", "sorrowful": "sad", "melancholy": "sad",
    "revolted": "disgust", "repulsed": "disgust", "disgusted": "disgust",
    "irritated": "annoyed", "exasperated": "annoyed",
    "stares": "staring", "gazes": "staring", "glares": "glaring",
}


SCAN_ROLE_NOUNS = {
    "expression": "expression", "expressions": "expression", "face": "expression",
    "look": "expression", "looks": "expression", "vibe": "expression",
    "mood": "expression",
    "pose": "pose", "posing": "pose", "position": "pose",
    "style": "style", "hairstyle": "hair", "haircut": "hair",
    "colour": None, "color": None, "type": None, "kind": None,
    "outfit": "clothing", "clothing": "clothing", "clothes": "clothing",
    "attire": "clothing",
    "lighting": "lighting", "light": "lighting",
    "shot": "framing", "angle": "viewpoint", "view": "viewpoint",
    "background": "scene", "setting": "scene", "scene": "scene"}


# Danbooru aliases that are correct ON DANBOORU but ambiguous in English, so
# they mistranslate ordinary prose. "a photograph taken from behind
# reinforced glass" is not a sex act, and letting the alias stand turned it
# into one. Kept to phrases where the everyday reading is clearly the more
# likely one; anybody who means the tag can type the tag.
ALIAS_DENY = {"taken from behind", "on all fours", "from the back",
              "going down", "blowing", "eating out", "riding",
              "mounting", "taking off", "coming inside"}


LEARNED_SLOTS = "slot_hints.json"


def _save_learned_slots(banks):
    """persist what the scanner worked out, so it is known next time too"""
    try:
        path = _paths.data(LEARNED_SLOTS)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(banks.get("_slot_learned") or {}, f,
                      ensure_ascii=False, indent=1, sort_keys=True)
    except Exception:
        pass          # a cache that cannot be written is not worth an exception


def scan_vocab_tags(text, banks):
    """Longest-match n-gram scan of the text against the real booru vocabulary.

    The old scan only knew tags that happened to be in a curated bank, so a prompt
    saying 'small breasts and huge penis' produced neither: 'small breasts' was in
    no bank, and 'huge penis' in none either. Both are ordinary danbooru tags. This
    matches 4-word phrases first and marks the words consumed, so 'small breasts'
    does not also yield a bare 'breasts', and the user's own words survive intact.
    """
    vocab = banks.get("_scanvocab") or {}
    if not vocab:
        return []
    aliases = banks.get("_aliases") or {}
    canon_map = banks.get("_canon") or {}
    slot_hints = banks.setdefault("_slot_hints", {})
    learned = banks.setdefault("_slot_learned", {})
    words = re.findall(r"[a-z0-9'\-]+", text.lower())
    used = [False] * len(words)
    out = []

    def resolve(cand):
        """literal tag, else danbooru alias, else the phrase minus a role noun.

        THE QUALIFIED FORM IS TRIED FIRST for a trailing role noun. Booru
        writes these tags with a qualifier -- `western comics (style)`
        (2,308 posts), `watercolor (medium)` -- while people type "western
        comics style". Without this the phrase decomposed into `western` +
        `comic`, and `western` aliases to `cowboy western`, so asking for
        western COMICS produced a cowboy setting and drew artists for the
        wrong genre. 146 style tags and the whole medium family are
        reachable this way.
        """
        parts0 = cand.split()
        if len(parts0) > 1 and parts0[-1] in ("style", "medium"):
            qual = " ".join(parts0[:-1]) + " (" + parts0[-1] + ")"
            if qual in vocab:
                return qual
            qa = aliases.get(qual)
            if qa:
                return qa
        if cand in vocab:
            # CANONICALISE EVEN WHEN THE LITERAL EXISTS. `blowjob` is a real
            # gelbooru tag AND danbooru aliases it to `fellatio`, so keeping
            # the literal emitted both spellings of one act.
            c2 = canon_map.get(cand)
            if c2 and c2 in vocab:
                return c2
            return cand
        a = aliases.get(cand)
        if a:
            return a
        # 'disgusted expression', 'smiling face', 'kneeling pose' — the trailing
        # noun names the SLOT and carries no meaning of its own. Strip it and try
        # the modifier alone, which is where the real tag lives.
        parts = cand.split()
        if len(parts) > 1 and parts[-1] in SCAN_ROLE_NOUNS:
            head = " ".join(parts[:-1])
            hit = head if head in vocab else aliases.get(head)
            if hit:
                # The scaffolding word we just threw away NAMES THE SLOT. Keep
                # that: 'disgust' is not in any expression list and role_of does
                # not know it, but the user wrote 'disgusted EXPRESSION', so the
                # slot is settled beyond doubt. Recording it is how an unknown
                # tag still counts as filling the slot it was written for.
                _slot = SCAN_ROLE_NOUNS[parts[-1]]
                slot_hints[hit] = _slot
                # LEARN IT PERMANENTLY. A hint discovered mid-parse used to live
                # only for that call, so 'disgust' was an expression while the
                # prompt that taught us was being built and an unknown tag five
                # minutes later. A classifier whose answer depends on call
                # history is not a classifier.
                if _slot and learned.get(hit) != _slot:
                    learned[hit] = _slot
                    _save_learned_slots(banks)
                return hit
        return None

    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            if any(used[i:i + size]):
                continue
            cand = " ".join(words[i:i + size])
            if size == 1 and (cand in SCAN_STOP or len(cand) < 3):
                continue
            if size == 1 and cand in LIGHT_VERBS                     and _is_light_verb(words, i, resolve):
                continue
            hit = resolve(cand)
            if hit is None:
                continue
            for j in range(i, i + size):
                used[j] = True
            out.append(hit)
    return out


_ARTICLE_RE = re.compile(r"^(?:an?|the)\s+", re.I)


def segment_is_tag(seg, banks):
    """Is this comma-separated piece a TAG the user typed, or a piece of prose?

    Vocabulary wins outright. 'hand on own hip' is four words with a preposition
    and reads exactly like prose to the sentence detector, but it is a real
    danbooru tag and must survive verbatim. Only a segment the vocabulary does
    not know gets judged on its shape.
    """
    s = normalize_tag(seg)
    if s in (banks.get("_scanvocab") or {}) or s in NORMALIZE:
        return True
    if (banks.get("_aliases") or {}).get(s):
        return True
    # The vocabulary does not know the whole segment. If it knows the PARTS,
    # then this is a phrase to mine, not a tag to emit -- 'girl eating ramen'
    # is three words with no preposition or article, so no sentence-shape rule
    # catches it, but the scanner reads 'ramen' and 'eating' straight out of it.
    # Emitting a phrase whole when its pieces are real tags is never right.
    # Three or more words that the booru vocabulary does not know is prose, and
    # emitting it whole is never right. Either the scanner can mine real tags
    # out of it ('girl eating ramen' -> eating, ramen) or it carries nothing a
    # model can use ('surrounds this classical structure.') -- and in both cases
    # the words still reach the model through the prose line, where they belong.
    if len(seg.split()) >= 3:
        return False
    return not is_natural_language(seg)


def parse_input(base, banks):
    """-> (tags, nl_sentence or None). NL input is mined for tags."""
    # INPUT IS ROUTINELY A MIXTURE, and used not to be treated as one.
    #
    # Every prose rule in is_natural_language() requires commas == 0, so
    # 'ballerina stretching in an empty studio, mirrors' failed the sentence
    # test, went down the tag-list path, and the whole first clause was emitted
    # as though it were a booru tag. Writing a sentence and then adding a couple
    # of tags after it is one of the most ordinary ways to type a prompt, and it
    # produced 16 junk tags in a 100-prompt sample.
    #
    # So the decision is made per segment. Prose segments are mined for tags the
    # usual way; tag segments are kept exactly as written, because under the
    # precedence rule what the user typed is the request.
    # Strip a leading English article from each segment before anything looks at
    # it. No booru tag begins with 'a ' or 'the ', so 'A photorealistic, ultra
    # detailed, ...' was emitting the tag 'a photorealistic' -- the article was
    # the only thing standing between it and the real tag 'photorealistic'.
    segments = [_ARTICLE_RE.sub("", x.strip()) for x in base.split(",")]
    segments = [x for x in segments if x]
    tag_segments = [x for x in segments if segment_is_tag(x, banks)]
    prose_segments = [x for x in segments if x not in tag_segments]

    if not prose_segments and not is_natural_language(base):
        out = [NORMALIZE.get(t.lower(), t) for t in segments]
        # A COMMA-SEPARATED LIST STILL NEEDS TO SAY WHO IS IN IT. This path
        # returned early, so the subject fallback further down never ran and
        # 'medium breasts, multicolored hair, white hoodie' produced a prompt
        # describing a person in detail without ever saying 1girl. 'solo' does
        # not count -- it says how many, not who.
        low_out = {t.lower() for t in out}
        if not any(re.match(r"^\d+\+?(girls?|boys?|others?)$|^multiple |^1futa$",
                            t) for t in low_out) and "no humans" not in low_out:
            # A gaze, an expression or a pose implies a person just as surely
            # as a body part does: 'looking at viewer, indoors, curtains' has
            # somebody doing the looking.
            if any(_sm.slot_of(t, banks) in ("hair", "body", "clothing",
                                             "gaze", "expression", "pose")
                   for t in low_out):
                out.insert(0, "1girl")
            elif any(_sm.slot_of(t, banks) == "scene" or t in INDOOR_PLACES
                     or t in OUTDOOR_PLACES or t == "scenery"
                     for t in low_out):
                # A place with nobody in it is a picture of the place. Saying so
                # outright beats leaving the subject slot empty, which reads to
                # everything downstream as "not decided yet" rather than "none".
                out.append("no humans")
        return out, None

    text = base.lower()
    tags = []
    # 'princess', 'witch', 'nurse' etc. name a female subject just as clearly as
    # 'girl' does — the original list missed them and produced no character tag.
    # ONE GENDER VOCABULARY (module level, shared with bridge.resolve_cast).
    # This used to be a third private copy, and it gendered OCCUPATIONS:
    # nurse/teacher/maid female, knight/butler/prince male -- so "a
    # female knight" produced 1girl AND 1boy, while the cast reader
    # treats an occupation as a measured genderless roll. An occupation
    # carries no gender here; a noun that IS a gender (princess, king,
    # nun, butler is not) stays.
    female = r"\b(?:" + FEMALE_WORDS + r")\b"
    male = r"\b(?:" + MALE_WORDS + r")\b"
    # PLURALS COUNT. The male pattern had 'boy' but not 'boys', so "two boys
    # fighting in an alley" matched nothing at all and fell through to the
    # '1girl' default -- two boys became one girl.
    # A PICTURE WITH NOBODY IN IT. The subject detection below always produced
    # somebody, so "a frog emerging from dark calm water", "a zebra walking"
    # and "the owl is soft and fluffy" all came back as girls with breasts and
    # long hair. The booru answer is `no humans` (233,116 posts), and saying it
    # outright is far better than leaving the subject slot empty and hoping.
    _no_people = _sm.scene_has_no_people(claim_gender_modifiers(text))   # 'a wench' is somebody
    if _no_people:
        tags.append("no humans")
    # An anthropomorphic creature is somebody, just not a girl. '1other' is the
    # booru tag for exactly this and it keeps the gnome a gnome.
    _other_char = (not _no_people) and _sm.nonhuman_character(text)
    if _other_char:
        tags.append("1other")

    _num = {"two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}
    # A POV QUALIFIER IS THE CAMERA HOLDER, NOT A SUBJECT (one rule, one
    # place): the subject count reads the text with "male pov" / "female
    # pov" / "futanari pov" removed. "male pov, nurse giving a handjob"
    # counted the 'male' and produced a man in a nurse costume every time;
    # the cast reader (bridge.resolve_cast) strips the same phrase.
    _text_nf = re.sub(r"\b(?:futanari|female|male) pov\b", " ", text, flags=re.I)
    _text_nf = claim_gender_modifiers(_text_nf)
    # one modifier may sit between the number and the noun: "two ELF
    # girls" counted as one girl and rolled a boy beside her
    _f2 = re.search(r"\b(two|2|three|3|four|4) (?:[a-z\-]+ )?(girls|women|ladies|sisters)\b", _text_nf)
    _m2 = re.search(r"\b(two|2|three|3|four|4) (?:[a-z\-]+ )?(boys|men|guys|brothers)\b", _text_nf)
    if not _no_people and not _other_char:
        if _f2:
            tags.append(f"{_num.get(_f2.group(1), 2)}girls")
        elif re.search(r"\b2girls\b", _text_nf):
            tags.append("2girls")
        elif re.search(female, _text_nf):
            tags.append("1girl")
        if _m2:
            tags.append(f"{_num.get(_m2.group(1), 2)}boys")
        elif re.search(male, _text_nf):
            tags.append("1boy")
    # 'looking at the viewer' is where the EYES point; 'a handjob to the viewer'
    # is where the CAMERA is. Both contain 'at/to the viewer', so the gaze
    # phrasings are removed before the viewpoint patterns get to look.
    vp_text = GAZE_PHRASE_RE.sub(" ", text)
    for pat, mapped in VIEWPOINT_MAP:
        if re.search(pat, vp_text):
            tags += mapped
            break                      # one viewpoint; the first match wins
    for pat, mapped in PHRASE_MAP:
        if mapped and re.search(pat, text):
            tags += mapped
    # "<colour>-haired" / "<colour>-eyed" ARE THE COLOUR TAGS. 'red-haired
    # dwarf' produced no `red hair`: the scanner reads whole words and the
    # hyphenated adjective is neither 'red hair' nor a tag. A shape rule,
    # so every colour the vocabulary knows is covered at once.
    for m in re.finditer(r"\b([a-z]+(?: [a-z]+)?)[- ](haired|eyed)\b",
                         str(base or "").lower()):
        _suffix = " hair" if m.group(2) == "haired" else " eyes"
        _words = m.group(1).split()
        # 'light brown-haired' is two words; 'a red-haired' is one plus an
        # article -- try the longer form, then the last word alone
        for cand in (" ".join(_words) + _suffix, _words[-1] + _suffix):
            if cand in (banks.get("_scanvocab") or {}) or cand in NORMALIZE:
                tags.append(NORMALIZE.get(cand, cand))
                break
    # THE LONGEST PHRASE CLAIMS ITS WORDS -- for every scanner, not just
    # one. scan_vocab_tags() has always resolved longest-first and claimed
    # the words it used ('office lady' is one tag, so it never emits
    # 'office'); the three scanners below read the RAW text, so the
    # single-word list still produced `office` and the location loop put
    # the scene in an office when the text said "office lady on the
    # train". They now read the text with every multi-word tag the
    # vocabulary scan found blanked out. A place named on its own
    # ("a nurse in an office") is untouched.
    # normalized multiword phrases FIRST ("big breasts" -> large breasts):
    # a normalised phrase claims its words before any scanner sees them,
    # so 'bunny girl' -> rabbit girl leaves no 'bunny' behind for the
    # vocabulary scan or the single-word list to turn into a rabbit (the
    # animal standing next to her)
    _claimed = text
    for phrase, norm in sorted(NORMALIZE.items(), key=lambda kv: -len(kv[0])):
        if " " in phrase:
            _pat = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
            if re.search(_pat, _claimed):
                tags.append(norm)
                _claimed = re.sub(_pat, " ", _claimed)
    _scanned = scan_vocab_tags(_claimed, banks)
    for _v in sorted((v for v in _scanned if " " in v), key=len, reverse=True):
        _claimed = re.sub(r"(?<![a-z0-9])" + re.escape(_v) + r"(?![a-z0-9])",
                          " ", _claimed)
    for w in _claimed.replace(",", " ").split():
        w = w.strip(".!?")
        if w in WORD_TAGS:
            tags.append(WORD_EXPAND.get(w, w))
    # everything the booru vocabulary recognises, longest phrase first
    tags += _scanned


    # FUTANARI IS NOT HETERO. The act phrases assume a male partner and add
    # '1boy' + 'hetero', which is wrong when the penetrating partner is the futa:
    # a futa with a woman is 'futa with female', and the pair is 1futa + 1girl,
    # not 2girls and certainly not hetero.
    # A POV QUALIFIER IS THE CAMERA HOLDER, NOT A SUBJECT. The scanner
    # read `futanari` out of "futanari pov" and this block then made the
    # girl a futa. If the word occurs nowhere else in the text, the futa
    # is behind the camera and the subject tags must not say otherwise.
    if not re.search(r"\bfuta\w*", _text_nf, re.I):
        tags = [t for t in tags if t.lower() not in ("futanari", "1futa")]
    low_tags = {t.lower() for t in tags}
    if "futanari" in low_tags:
        # The strip of hetero/1boy/2boys that used to sit here undid the
        # phrase maps' invented partner -- and also erased a REAL man: "a
        # futanari and a man having sex" came out with no man at all. The
        # maps no longer add partners or pairing (bridge._PAIRING owns
        # that), so there is nothing left to undo, and a man the user
        # wrote stays written.
        partnered_act = any(re.search(r"sex\b|fellatio|paizuri|handjob|cunnilingus|"
                                      r"penetrat|creampie|cowgirl|anal", t, re.I)
                            for t in tags)
        # Decide from the TAGS, not by re-matching the raw text: by this point the
        # female partner may only be present as '1girl', 'mature female' or a
        # character name, none of which the prose regex sees.
        has_female = any(t.lower() in ("1girl", "2girls", "3girls", "multiple girls",
                                       "mature female", "milf", "yuri")
                         for t in tags)
        # THE PAIRING TAG IS NOT EMITTED HERE ANY MORE. bridge._PAIRING
        # derives 'futa with female' / 'futa with male' / 'futa with futa'
        # from the resolved cast, and it is the only owner of that fact --
        # this branch was the second owner, the same duplication that put
        # `hetero` on a two-girl scene. The COUNT is still this parser's
        # job: a partnered act needs the woman counted, and `1futa` below
        # is what tells the partner rule the futa is not solo.
        if partnered_act and has_female:
            if not any(t.lower() in ("1girl", "2girls") for t in tags):
                tags.append("1girl")
        if not any(t.lower() == "1futa" for t in tags):
            tags.append("1futa")     # count tag; 'futanari' stays as the content tag
        low_tags = {t.lower() for t in tags}

    # THE VIEWER IS A PARTNER. slots.heads_available() has always counted
    # the camera as a participant ("the camera counts when it is a
    # participant"), but this rule did not -- so "a woman giving a blowjob
    # TO THE VIEWER" invented a second man, and the prose then had to place
    # him, producing a bystander standing beside a POV act. When `pov` is
    # present the partner IS the person holding the camera.
    # a partnered act cannot be solo — make sure a partner is present and drop 'solo'
    if any(re.search(r"sex\b|fellatio|paizuri|handjob|cunnilingus|penetrat|creampie|"
                     r"cowgirl|group sex|anal", t, re.I) for t in tags):
        # futas the TEXT names count too: "two futas having sex" has no
        # futa count tag (there is no multi-futa vocabulary), so this
        # rule invented a man for them
        if not any(t.lower() in ("1boy", "2boys", "multiple boys", "2girls",
                                 "futa with female", "1futa", "yuri", "pov")
                   for t in tags) and not re.search(
                r"\bfutas?\b|\bfutanari\b", _text_nf, re.I):
            tags.append("1boy")
        tags = [t for t in tags if t.lower() != "solo"]
    # a prose prompt with no character count at all still needs a subject --
    # unless it is a picture of a frog, in which case the subject is the frog
    # `1futa` IS a subject count: without it here a solo futa got a
    # phantom girl inserted beside her
    if not _no_people and not _other_char and not any(
            re.match(r"^\d(girl|boy|futa)s?$|^multiple", t) for t in tags):
        tags.insert(0, "1girl")

    seen, out = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    # Tag segments the user typed alongside the prose are the request too, and
    # the n-gram scanner only recovers the ones the booru vocabulary knows. Add
    # them back verbatim so an exact tag is never lost to a sentence sitting
    # next to it.
    # RESCUE PASS: a slot the user plainly wrote about that produced no tag.
    #
    # The main scan is deliberately conservative, so an ordinary English word for
    # something real -- 'furious', 'silver hair', 'stares' -- can leave a slot
    # empty. When a probe says the user addressed that slot and nothing came out,
    # try harder on exactly those words: synonyms first, then the low-floor
    # vocabulary, then a light plural strip into the alias table. Nothing here
    # runs unless a slot is otherwise going to be answered with silence.
    _states = _sm.slot_states(base, out, banks)
    _low_vocab = banks.get("_scanvocab_low") or {}
    _aliases = banks.get("_aliases") or {}
    for _slot, _st in _states.items():
        if _st != "claimed":
            continue
        _probe = _sm.SLOT_PROBES.get(_slot)
        if not _probe:
            continue
        for _m in _probe.finditer(text):
            _w = _m.group(0).strip()
            # The probe matches the head word, but the meaning often sits in the
            # modifier in front of it: the hair probe fires on 'hair' while the
            # tag we need is 'silver hair'. Try the phrase before the bare word.
            _before = text[:_m.start()].split()
            _cands = []
            if _before:
                _cands.append((_before[-1] + " " + _w).strip())
            if len(_before) > 1:
                _cands.append((_before[-2] + " " + _before[-1] + " " + _w).strip())
            _cands.append(_w)
            _hit = None
            for _c in _cands:
                _hit = (SYNONYMS.get(_c)
                        or (_c if _c in _low_vocab else None)
                        or _aliases.get(_c))
                if not _hit and _c.endswith("s") and len(_c) > 4:
                    _stem = _c[:-1]
                    _hit = (SYNONYMS.get(_stem) or _aliases.get(_stem)
                            or (_stem if _stem in _low_vocab else None))
                if _hit:
                    break
            if _hit and _hit.lower() not in {t.lower() for t in out}:
                out.append(_hit)
                banks.setdefault("_slot_hints", {})[_hit.lower()] = _slot
                break

    have = {t.lower() for t in out}
    for seg in tag_segments:
        norm = NORMALIZE.get(seg.lower(), seg)
        if norm.lower() in have:
            continue
        # Skip a segment the scanner has already accounted for. The prose pass
        # reads the whole line, so 'disgusted expression' has usually become
        # 'disgust' by now; re-adding the raw words would put the echo straight
        # back, which is the fault this whole change exists to remove.
        mined = scan_vocab_tags(seg, banks)
        if mined and all(m in have for m in mined):
            continue
        out.append(norm)
        have.add(norm.lower())

    # The last-resort '1girl' must not undo the no-humans decision: a picture of
    # a frog with nothing else recognisable in it is still a picture of a frog.
    if not out:
        out = ["no humans"] if _no_people else ["1girl"]
    return out, base.strip()


def _tn_role(tag):
    try:
        from promptstudio.engine import tagnet as _tn
        return _tn.role_of(tag)
    except Exception:
        return None


_BANKS_CACHE = None


def load_all_banks(reload=False):
    """curated banks + harvested tag overlay + learned locations/flavors/lighting.
    Curated entries always win; learned ones only add new keys.

    CACHED. Building this costs ~6.5 s -- most of it rebuilding the 29k-row
    tag network and normalising 1.1M tag strings -- and bridge.generate()
    called it on EVERY generation. The LLM path hid the cost behind a
    15-40 s model call; the no-LLM path could not, and it dominated the
    fast mode entirely (7.2 s per prompt, of which 6.5 s was this).

    The studio has always treated the result as shared read-only state
    (Handler.banks is loaded once and reused for every request), and
    generate() writes nothing back into it, so one instance is correct.
    Pass reload=True after a build tool rewrites a data file.
    """
    global _BANKS_CACHE
    if _BANKS_CACHE is not None and not reload:
        return _BANKS_CACHE
    _BANKS_CACHE = _load_all_banks_uncached()
    return _BANKS_CACHE


def _load_all_banks_uncached():
    # THE V1 TAG BANKS ARE GONE (2026-09-16): prompt_banks.json and its
    # harvested / learned overlays were hand weights ('seductive smile 4',
    # 'detailed face 3') the v2 engine never rolled from; their only effect
    # left was to turn non-booru phrases typed in a brief into tags. Every
    # vocabulary the studio reads is measured now (the booru counts, the
    # glosses, the pools). The keys stay as empty shapes for the readers.
    banks = {"locations": {}, "style_flavors": {}, "lighting_moods": {}, "lighting": [],
             "camera": [], "expression": [], "body_detail": [], "hair_bonus": [],
             "clothing": {}, "pose": {}, "_overlay": {}, "_extra_locations": {},
             "_extra_flavors": {}, "_extra_moods": {}}

    # danbooru's own co-occurrence data: {tag: [[related, frequency], ...]}
    banks["_graph"] = {}
    gpath = _paths.data("tag_graph.json")
    if os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8-sig") as f:
            banks["_graph"] = json.load(f)
    # danbooru -> gelbooru spelling, applied in anima mode only (its README asks
    # for the gelbooru version of any tag the two sites spell differently)
    banks["_bridge"] = {}
    bpath = _paths.data("tag_bridge.json")
    if os.path.exists(bpath):
        with open(bpath, "r", encoding="utf-8-sig") as f:
            banks["_bridge"] = json.load(f)

    # Category banks: curated entries plus danbooru-popularity entries, split into
    # the dropdown ('top', capped 50/30/30) and everything else ('tail'). The cap
    # bounds the DROPDOWN only — both halves load here and both stay reachable
    # through 'auto' and 'random'.
    # the medium family is its own axis (resolve_medium): the category banks
    # below keep their flavour entries free of medium members
    try:
        from promptstudio.engine import bridge as _lb
        _is_medium = _lb.is_medium_tag
    except Exception:
        def _is_medium(_t):
            return False

    def _drop_mediums(entries):
        """-> entries without the medium-family members"""
        out = []
        for e in (entries or []):
            t = str(e[0] if isinstance(e, (list, tuple)) else e)
            if _is_medium(t):
                continue
            out.append(e)
        return out

    banks["_categories"] = {}
    cpath = _paths.data("category_banks.json")
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8-sig") as f:
            cats = json.load(f)
        for kind, blob in cats.items():
            top, tail = blob.get("top") or {}, blob.get("tail") or {}
            # A whole FLAVOUR entry whose defining tag is a body or clothing tag
            # is not a style at all - 'gradient horns', 'gradient wings' rode in on
            # the word 'gradient'. Drop the entry rather than emit it as a style.
            if kind == "flavors":
                # "gradient" is a style word, but "gradient legwear" /
                # "gradient nails" are garments and body details that rode in
                # on it. Only the genuinely stylistic gradients survive.
                _GRAD_OK = ("gradient background", "gradient", "rainbow gradient",
                            "gradient sky")
                # MEDIUM IS ITS OWN AXIS, AND ITS OWN OWNER. 'lineart',
                # 'sketch' and 'photorealistic' sat in the flavour bank's
                # top 30, so they were drawn as flavours several percent of
                # the time -- straight past resolve_medium and the rarity
                # it deliberately applies. Two axes emitting one family
                # under two different policies is the bug; the medium
                # vocabulary is excluded here so the medium resolver is the
                # only thing that can introduce a medium. Typed input is
                # unaffected: it never comes from a pool.
                def _is_style(entries):
                    for e in (entries or []):
                        t = str(e[0] if isinstance(e, (list, tuple)) else e)
                        if _is_medium(t):
                            return False
                        if (t.lower().startswith("gradient ")
                                and t.lower() not in _GRAD_OK):
                            return False
                        return _tn_role(t) not in ("body", "clothing", "hair",
                                                   "pose", "expression", "scene")
                    return True
                top = {k2: _drop_mediums(v) for k2, v in top.items()
                       if _is_style(v)}
                tail = {k2: _drop_mediums(v) for k2, v in tail.items()
                        if _is_style(v)}
                top = {k2: v for k2, v in top.items() if v}
                tail = {k2: v for k2, v in tail.items() if v}
            banks["_categories"][kind] = {
                "top": top, "tail": tail, "all": dict(top, **tail)}

    # Vocabulary the free-text scanner may recognise: real booru tags only, with a
    # popularity floor so a stray word does not become a tag. Multi-word phrases are
    # far more specific than single words, so they need a much lower floor.
    scan = {}
    for src, mult in (("danbooru_tags.json", 1), ("gelbooru_tags.json", 1)):
        sp = _paths.data(src)
        if not os.path.exists(sp):
            continue
        with open(sp, "r", encoding="utf-8-sig") as f:
            for tag, n in json.load(f).items():
                t = normalize_tag(tag)
                w = len(t.split())
                if w > 4:
                    continue
                # Short tags are not noise — 'bed', 'bra', 'hat', 'ass', 'sky',
                # 'pov', 'cup' are among the most used tags on the site. Blanket
                # len < 4 made every one of them unreachable from free text. Only
                # 31 three-letter tags clear 50k posts, so the high floor keeps
                # the stray-word protection without the collateral damage.
                if len(t) < 3:
                    continue
                if len(t) == 3 and w == 1:
                    floor = 50000
                else:
                    floor = 2000 if w == 1 else 300
                if n >= floor:
                    scan[t] = max(scan.get(t, 0), n)
    # EITHER SPELLING MUST BE TYPEABLE. The two boorus disagree on ~60
    # tags, and the popularity floors above admit only the popular side --
    # so 'fingers to cheek' (133 danbooru posts) was simply not a word the
    # reader knew, while 'finger to cheek' (7,067) was. The user should not
    # have to know which booru a phrase came from: both sides of every
    # bridged pair are accepted here, and bridge.spell_for() then writes
    # whichever form the target model has actually seen more.
    try:
        _pairs = {}
        with open(_paths.data("tag_bridge.json"), encoding="utf-8-sig") as f:
            _pairs = json.load(f)
        for _a, _b in _pairs.items():
            _a, _b = normalize_tag(str(_a)), normalize_tag(str(_b))
            if _a in scan or _b in scan:
                _n = max(scan.get(_a, 0), scan.get(_b, 0))
                scan[_a] = max(scan.get(_a, 0), _n)
                scan[_b] = max(scan.get(_b, 0), _n)
    except Exception:
        pass

    # THE SCANNER MUST RECOGNISE EVERYTHING THE STUDIO ITSELF SHIPS.
    #
    # The floors above are there to stop a stray word becoming a tag, but they
    # were also silently excluding 45 of our own curated places -- 'cathedral'
    # (715 posts), 'attic' (96), 'throne room' (245), 'basement' (118). Typing
    # "a nun praying inside a cathedral" lost the cathedral entirely, and then
    # the selected location filled the empty place slot with a library. A tag we
    # chose to ship is a tag we mean; popularity has no vote on it.
    curated = set(INDOOR_PLACES) | set(OUTDOOR_PLACES)
    # Creature names, for the same reason. 'owl' is 5,711 posts and three
    # letters, so both floors excluded it -- and a prompt about an owl that
    # cannot say 'owl' is not much of a prompt.
    curated |= {"owl", "zebra", "frog", "snake", "bunny", "rabbit", "fox",
                "wolf", "bear", "deer", "horse", "whale", "shark", "octopus",
                "squid", "lizard", "turtle", "crow", "raven", "spider",
                "butterfly", "dragon", "gnome", "goblin", "orc", "slime",
                "golem", "alien", "dinosaur", "penguin", "panda", "tiger",
                "lion", "monkey", "sheep", "goat", "cow", "pig", "duck",
                "swan", "eagle", "hawk", "parrot", "dolphin", "jellyfish"}
    # The occupation pool, for the same reason: 'hunter' (247 posts on
    # gelbooru, absent from the local count file) typed as "a hunter in the
    # forest" produced no role at all (2026-09-04). A role the tables can
    # roll is a role the scanner must read.
    try:
        with open(_paths.data("occupation_pool.json"), encoding="utf-8-sig") as _f:
            curated.update(str(_t).lower() for _t in
                           (json.load(_f).get("occupations") or {}))
    except Exception:
        pass
    # The ACTIVITY REGISTRY, for the same reason (the author's 2026-09-11): 50
    # of the 221 activities the studio rolls and prints were words the
    # scanner did not read -- 'yoga' (867 posts), 'gymnastics', 'tennis',
    # 'baking', 'sewing' sit under the 2,000-post single-word floor, so
    # "a woman doing yoga" rolled 'walking' instead. An activity the
    # tables can roll is an activity the scanner must read.
    try:
        with open(_paths.data("genre_activities.json"), encoding="utf-8-sig") as _f:
            # ...every activity, the prose-only ones too (2026-09-15: they
            # are measured through an anchor now; 'forging' typed is the act,
            # said in the prose and never emitted as a tag)
            curated.update(str(_t).lower() for _t, _d in
                           ((json.load(_f).get("registry") or {}).items()))
    except Exception:
        pass
    for _t in curated:
        _t = normalize_tag(_t)
        if len(_t) >= 3 and len(_t.split()) <= 4:
            scan.setdefault(_t, 1)

    banks["_scanvocab"] = scan

    # A SECOND, LOWER-FLOOR VOCABULARY used only when a slot probe has already
    # fired. 'furious' is a real danbooru tag with 986 posts, below the 2,000
    # floor that keeps stray words out of the general scan -- so 'a furious
    # expression' produced no expression at all. Dropping the floor everywhere
    # would admit 'seal', 'recorder' and 'heater' to ordinary prose; admitting
    # them only where the user has demonstrably written about that slot does not.
    scan_low = {}
    for src in ("danbooru_tags.json", "gelbooru_tags.json"):
        sp = _paths.data(src)
        if not os.path.exists(sp):
            continue
        with open(sp, "r", encoding="utf-8-sig") as f:
            for tag, n in json.load(f).items():
                t = normalize_tag(tag)
                w = len(t.split())
                if w > 4 or len(t) < 3:
                    continue
                if n >= (300 if w == 1 else 100):
                    scan_low[t] = max(scan_low.get(t, 0), n)
    banks["_scanvocab_low"] = scan_low

    # The Style pool (58 curated entries with measured setting-pulls and
    # derived artists) -- the artist subsection reads it for style->artist
    # derivation. Absent file just means no derivation channel.
    try:
        with open(_paths.data("style_pool.json"), encoding="utf-8-sig") as f:
            banks["_style_pool"] = json.load(f)
    except Exception:
        banks["_style_pool"] = {}

    # NL-channel cultural styles (anima only): studio looks, director hands,
    # technique phrases -- A/B-tested vocabulary with zero booru presence.
    try:
        with open(_paths.data("cultural_styles.json"),
                  encoding="utf-8-sig") as f:
            banks["_cultural"] = json.load(f)
    except Exception:
        banks["_cultural"] = {}

    # Measured incompatible pairs. Absent until build_contradictions.py has run,
    # in which case the generator simply does not use them -- an unmeasured pair
    # is not evidence of a fault.
    cpath = _paths.data("contradictions.json")
    banks["_contradictions"] = {}
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8-sig") as f:
            banks["_contradictions"] = {k: set(v) for k, v in json.load(f).items()}

    # Surface form -> canonical tag, straight from danbooru's own tag_aliases.
    # The people who maintain the tags already wrote down that 'disgusted' means
    # 'disgust', 'blowjob' means 'fellatio' and 'titfuck' means 'paizuri'. This
    # file was downloaded for tag_bridge.py and never given to the scanner, which
    # is why an explicit 'disgusted expression' produced no tag at all.
    aliases = {}
    from promptstudio.engine.aliases import booru_aliases
    _al = booru_aliases()
    if _al:
        if True:
            for surface, canon in _al.items():
                s = normalize_tag(surface)
                c = normalize_tag(canon)
                # only aliases that land on vocabulary the scanner would accept,
                # and never one that shadows a real tag with its own meaning
                if not (c in scan and s not in scan and len(s) >= 3):
                    continue
                # A surface form that is a strict PREFIX of the tag it maps to is
                # a truncation, not a synonym: 'sitting on' -> 'sitting on person'
                # fires on 'sitting on a bench' and invents a person who is not
                # there. Matching a fragment means guessing the missing words.
                if c.startswith(s + " "):
                    continue
                if s in ALIAS_DENY:
                    continue
                # a gendered noun never aliases to a FOCUS tag: 'man' ->
                # 'male focus' made every man a camera fact (2026-09-12)
                if c.endswith(" focus") and (s in FEMALE_NOUN_LIST or s in MALE_NOUN_LIST
                                             or s in ("man", "woman", "girl", "boy", "men", "women", "girls", "boys")):
                    continue
                aliases[s] = c
    banks["_aliases"] = aliases

    # CANONICAL FORM, for tags that are BOTH a tag and an alias of another.
    # The map above deliberately skips those (`s not in scan`) because it is
    # an EXPANSION map -- turning a non-tag surface form into a tag. But
    # `blowjob` is a real gelbooru tag (183,647 posts) AND danbooru says it
    # means `fellatio` (97,851), so a prompt that mentioned one came out
    # carrying both spellings of a single act. Collapsing is a different job
    # from expanding, so it gets its own map.
    canon = {}
    if _al:
        if True:
            for surface, target in _al.items():
                sfc, tgt = normalize_tag(surface), normalize_tag(target)
                if sfc == tgt or sfc in ALIAS_DENY:
                    continue
                # both sides must be real vocabulary, else this is expansion
                if sfc in scan and tgt in scan and not tgt.startswith(sfc + " "):
                    canon[sfc] = tgt
    banks["_canon"] = canon

    banks["_slot_learned"] = {}
    lp = _paths.data(LEARNED_SLOTS)
    if os.path.exists(lp):
        try:
            with open(lp, "r", encoding="utf-8-sig") as f:
                banks["_slot_learned"] = json.load(f)
        except Exception:
            pass

    banks["_profiles"] = {}
    ppath = _paths.data("character_profiles.json")
    if os.path.exists(ppath):
        with open(ppath, "r", encoding="utf-8-sig") as f:
            banks["_profiles"] = json.load(f)

    # The same data as a network: spreading activation, hubness correction and
    # mutual-coherence scoring, restricted to vocabulary the models actually know
    # (danbooru + the finetune vocabulary + everything already in the banks) so it
    # can never propose a tag no checkpoint has ever seen.
    try:
        from promptstudio.engine import tagnet as tag_net
        vocab = set(banks["_graph"])
        dpath = _paths.data("danbooru_tags.json")
        if os.path.exists(dpath):
            with open(dpath, "r", encoding="utf-8-sig") as f:
                vocab |= {normalize_tag(t) for t in json.load(f)}
        dcounts = {}
        if os.path.exists(dpath):
            with open(dpath, "r", encoding="utf-8-sig") as f:
                dcounts = json.load(f)
        gpath = _paths.data("gelbooru_tags.json")
        if os.path.exists(gpath):
            with open(gpath, "r", encoding="utf-8-sig") as f:
                for k, v in json.load(f).items():
                    kk = normalize_tag(k)
                    dcounts[kk] = max(dcounts.get(kk, 0), v)
        banks["_net"] = tag_net.TagNet(banks["_graph"], vocab, dcounts)
    except Exception as e:
        # a silent None here degrades every prompt with no sign of why
        sys.stderr.write(f"tag network unavailable: {e!r}\n")
        banks["_net"] = None

    # SANITISE EVERY BANK AT ONE CHOKE POINT. The banks are built by several
    # scripts from several sources, so junk that gets past one of them (booru
    # event tags like 'nue day', tags naming a drawing error) otherwise has to be
    # filtered at every point of use. Bracketed tags are NOT dropped: '(medium)'
    # is part of real danbooru tag names and is escaped at output instead.
    try:
        from promptstudio.engine import tagnet as _tn

        def _clean_list(entries):
            out = []
            for e in entries or []:
                t = e[0] if isinstance(e, (list, tuple)) and e else e
                if isinstance(t, str) and _tn.NEVER_PROPOSE.match(t):
                    continue
                out.append(e)
            return out

        def _scrub(node):
            if isinstance(node, dict):
                return {k: _scrub(v) for k, v in node.items()}
            if isinstance(node, list):
                return _clean_list(node)
            return node

        for _key in ("locations", "style_flavors", "lighting_moods", "lighting",
                     "camera", "expression", "body_detail", "hair_bonus",
                     "_overlay", "_extra_locations", "_extra_flavors", "_extra_moods",
                     "clothing", "pose", "_categories"):
            if _key in banks:
                banks[_key] = _scrub(banks[_key])
    except Exception as e:
        sys.stderr.write(f"bank sanitiser skipped: {e!r}\n")

    return banks



# A1111 / regional-prompter CONTROL KEYWORDS. These are parser directives, not
# descriptors: BREAK pads the prompt to the next 75-token chunk, AND composes
# separate conditionings, ADDBASE/ADDCOL/ADDROW/ADDCOMM split regions. They carry
# no visual meaning, so they must never be learned as tags or emitted by the
# generator — only a human placing them deliberately should introduce them.
CONTROL_RE = re.compile(r"^(break|and|addbase|addcol|addrow|addcomm|"
                        r"bre[ak]{2,}|\.{2,}|-{2,})$", re.I)


# Underscores: danbooru stores multi-word tags as long_hair, but in a prompt the
# space form is equivalent and is what people actually write. Keeping both creates
# duplicate vocabulary entries ('looking_at_viewer' AND 'looking at viewer'), so
# everything is normalised to spaces — EXCEPT tags where the underscore is the tag
# (emoticon faces), which lose their meaning if you split them.
EMOTICON_TAGS = {"^_^", "^_-", "-_-", ">_<", "@_@", ";_;", "o_o", "0_0", "x_x", "+_+",
                 "=_=", "._.", "\\m/_", ">_@", "^q^", "@_<", "u_u", "t_t", "v_v"}
SYMBOL_ONLY_RE = re.compile(r"^[^a-z0-9]*_[^a-z0-9]*$", re.I)


def normalize_tag(t):
    """canonical prompt form of a scraped tag"""
    t = (t or "").strip().lower()
    if not t:
        return ""
    if t in EMOTICON_TAGS or SYMBOL_ONLY_RE.match(t):
        return t                      # ^_^ , >_< , ._. keep their underscores
    # conditioning TOKENS, not multi-word tags: 'score_9' is one identifier and
    # becomes meaningless if split into 'score 9'
    if re.match(r"^(score|source|rating|quality)_\w+$", t):
        return t
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()



# --------------------------------------------------------------------------
# typo / junk defence for anything mined from real human prompts
# --------------------------------------------------------------------------
MALFORMED_RE = re.compile(r"^[^a-z0-9]|[^a-z0-9\s\-'()_.:!/\\]|^\W|\s{2,}|^.{0,2}$|^.{41,}$", re.I)


# danbooru expression tags built from punctuation: ^_^  >_<  :d  ;)  @_@  o_o  :3
# Must contain at least one emoticon symbol, so plain short words never qualify.
FACE_TAG_RE = re.compile(r"^(?=.*[\^>@;:=<>_\\/|+*~])"
                         r"[\^>@;:=<>_\\/|+*~()\-.,'\"a-z0-9]{1,8}$", re.I)


def is_malformed(tag):
    """obvious junk: control keywords, stray punctuation, unbalanced parens, length."""
    t = tag.strip()
    if not t:
        return True
    if FACE_TAG_RE.match(t):          # a face tag, not junk
        # emoticons legitimately carry unmatched brackets — ';)' , ':(' , '>:('
        return bool(re.search(r"(.)\1{3,}", t))
    if CONTROL_RE.match(t) or MALFORMED_RE.search(t):
        return True
    # booru tags never start with an article — 'the pose' is a prose fragment
    if re.match(r"^(the|a|an)\s", t, re.I):
        return True
    if t.count("(") != t.count(")"):
        return True
    if re.search(r"(.)\1{3,}", t):           # 'aaaa', '!!!!'
        return True
    if sum(ch.isdigit() for ch in t) > len(t) / 2:
        return True
    return False


def _transposed(a, b):
    """'gril' vs 'girl' — two adjacent characters swapped."""
    if len(a) != len(b):
        return False
    d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]


def _edit1(a, b):
    """True when a and b are within one insert/delete/substitute/transposition."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = sum(1 for x, y in zip(a, b) if x != y)
        if diff == 1:
            return True
        return _transposed(a, b)
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] != b[j]:
            if skipped:
                return False
            skipped = True
            j += 1
            continue
        i += 1
        j += 1
    return True


def typo_tags(counts, rare_max=25, ratio=6):
    """Rare tags one edit away from a much more common one are typos of it.
    ('beautifull' next to 3000x 'beautiful', 'chrothes' next to 'clothes')"""
    from collections import defaultdict as _dd
    common = [(t, c) for t, c in counts.items() if c >= rare_max]
    buckets = _dd(list)
    for t, c in common:
        buckets[(t[:1], len(t))].append((t, c))
    drop = set()
    for t, c in counts.items():
        if c >= rare_max or len(t) < 4:
            continue
        for L in (len(t) - 1, len(t), len(t) + 1):
            for t2, c2 in buckets.get((t[:1], L), ()):
                if c2 < c * ratio:
                    continue
                # transpositions are near-certain typos even in short words;
                # substitutions in short words are often real ('cat' vs 'car'),
                # so those need >= 5 characters.
                if _transposed(t, t2) or (len(t) >= 5 and _edit1(t, t2)):
                    drop.add(t)
                    break
            if t in drop:
                break
    # transposition of the FIRST two letters changes the bucket key, so check those
    for t, c in list(counts.items()):
        if c >= rare_max or len(t) < 4 or t in drop:
            continue
        swapped = t[1] + t[0] + t[2:]
        if counts.get(swapped, 0) >= c * ratio:
            drop.add(t)
    return drop


NEG_LIGHT = "worst quality, low quality, bad anatomy, bad hands, watermark, signature"

# A negative term is dropped when the positive prompt deliberately asks for that look.
# (negative pattern, positive pattern that protects it)
NEG_PROTECT = [
    (r"chromatic aberration", r"chromatic aberration|synthwave|vaporwave|glitch|retro"),
    (r"^(monochrome|greyscale|grayscale)$", r"monochrome|greyscale|grayscale|sketch|ink wash|sumi-e|pencil drawing|charcoal|limited palette"),
    (r"^sketch$", r"sketch|lineart|unfinished|rough|pencil drawing|charcoal"),
    (r"simple background", r"simple background|white background|studio|plain background|gradient background"),
    (r"^(realistic|photorealistic)$", r"realistic|photorealistic|hyperreal|film grain|analog photo|semi-realistic|film_photo"),
    (r"^3d$", r"\b3d\b|render|octane|cgi"),
    (r"flat colou?rs?", r"flat colou?r|cel shading|anime screencap|no lineart|flat vector"),
    (r"^blurry$", r"\bblurry (background|foreground)\b|motion blur|blur\b"),
    (r"jpeg artifacts", r"jpeg artifacts|film grain|glitch|analog"),
    (r"^(muscular|abs|toned)$", r"muscular|\babs\b|toned|fit body|athletic"),
    (r"^(fat|chubby|plump)$", r"chubby|plump|thick|curvy|bbw"),
    (r"^old$", r"mature female|milf|older|aged"),
    (r"^(text|english text|caption|speech bubble)$", r"\btext\b|speech bubble|dialogue|caption|sign\b"),
    (r"^censored", r"censored|bar censor|mosaic"),
    (r"depth of field|bokeh", r"depth of field|bokeh|blurry background"),
    (r"^(lips|nose)$", r"\blips\b|glossy lips|nose\b"),
    (r"^(dark|dark location|dimly lit)$", r"dark location|dimly lit|low key|night|moody|noir"),
    (r"^(bright|high contrast)$", r"high contrast|bright|vivid|saturated"),
]
def negative_for(banks=None, mode=None, strength=None):
    """One concise negative. Long negatives fight the positive prompt more than they
    help, and scraped donor negatives are worse still (dead embeddings,
    contradictions). Anima gets the negative its own README recommends — its quality
    scale is different, so negating 'worst quality, low quality' alone would leave
    the score axis unconditioned."""
    if mode == "anima" and banks:
        neg = (banks.get("anima_tokens") or {}).get("negative")
        if neg:
            return ", ".join(neg)
    return NEG_LIGHT


EXPLICIT_RE = re.compile(r"pussy|penis|sex\b|cum\b|nipples|fellatio|nude|naked|vaginal|anal|masturbat|paizuri|nsfw|breasts out|topless")
COUNT_RE = re.compile(r"^\d(girl|boy)s?$|^solo$|^multiple")
# Anima-3.8B guidance: state where each subject is. Positions are handed out in a
# fixed reading order so two subjects never claim the same spot.
POSITIONS = ["on the left", "on the right", "in the centre",
             "in the upper left", "in the lower right"]
# Positions and relationships the USER stated. Anything strictly specified in the
# initial prompt is the rule: the generator may rephrase it or move it to satisfy
# the model's prompt-format guidance, but it may never mean something else. Before
# this, positions were assigned by shuffling POSITIONS — so asking for "Tifa on the
# left" could produce "Tifa Lockhart is on the right".
_POS_WORD = (r"(?:(?:upper|lower|top|bottom)\s+)?"
             r"(?:left|right|centre|center|middle|background|foreground|front|back)")
USER_POS_RE = re.compile(
    r"(?P<who>[A-Za-z][A-Za-z'’.\- ]{1,40}?)\s+"
    r"(?:is\s+|are\s+|sits\s+|sitting\s+|stands\s+|standing\s+|placed\s+|positioned\s+)?"
    r"(?:on|at|in|to|towards?)\s+(?:the\s+)?(?P<pos>" + _POS_WORD + r")\b", re.I)
USER_REL_RE = re.compile(
    r"(?P<a>[A-Za-z][A-Za-z'’.\- ]{1,40}?)\s+(?:is\s+|are\s+)?"
    r"(?P<rel>behind|in front of|next to|beside|near|above|below|on top of|"
    r"underneath|under|leaning on|holding|hugging|embracing|facing)\s+"
    r"(?P<b>[A-Za-z][A-Za-z'’.\- ]{1,40}?)(?=[,.;]|$| and | while | with )", re.I)


def _article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _place_phrase(scene_tags):
    """A scene tag list is not a noun phrase. 'on bed' and 'indoors' are not places
    you can put after 'set in a', which is how 'set in a on bed indoors' happened."""
    for s in scene_tags:
        sl = s.lower().strip()
        if sl in ("indoors", "outdoors", "scenery", "background"):
            continue
        if sl.split()[0] in ("on", "in", "at", "under", "near", "behind", "beside"):
            continue
        return f"{_article(sl)} {sl}"
    for s in scene_tags:                       # nothing better — take a preposition
        if s.lower().strip().split()[0] in ("on", "in", "at"):
            return s.lower().strip()
    return None


VERB_SPLIT = re.compile(
    r"\b(fucks?|fucking|rides?|riding|sucks?|sucking|kisses|kissing|hugs?|hugging|"
    r"holds?|holding|licks?|licking|pounds?|pounding|takes?|taking|grabs?|grabbing|"
    r"is|are|was|were|and then|while|as)\b", re.I)


def character_attributes(text, names, banks):
    """What the USER said about each named character, from their own words.

    'young futanari tifa lockhart with small breasts and huge penis fucks milf
    aerith gainsborough' describes two different people. Describing both from
    their danbooru profiles throws all of that away and returns stock traits, so
    the prose ends up saying nothing the prompt asked for.

    English puts the modifiers before the name ('young futanari tifa lockhart')
    and any elaboration after it in a 'with ...' clause, so each character claims
    the words from the previous name up to its own, plus a following 'with'
    clause that runs until the next name or the next verb.
    """
    low = text.lower()
    spans = []
    for name in names:
        i = low.find(name)
        if i >= 0:
            spans.append((i, i + len(name), name))
    if not spans:
        return {}
    spans.sort()
    out = {}
    for idx, (start, end, name) in enumerate(spans):
        prev_end = spans[idx - 1][1] if idx else 0
        before = low[prev_end:start]
        # a verb between the previous name and this one ends the previous clause
        parts = VERB_SPLIT.split(before)
        before = parts[-1] if parts else before

        after = ""
        nxt = spans[idx + 1][0] if idx + 1 < len(spans) else len(low)
        tail = low[end:nxt]
        m = re.search(r"\bwith\b(.*)", tail, re.S)
        if m:
            after = VERB_SPLIT.split(m.group(1))[0]
            # a "with ..." clause ends at a locative preposition: "aerith with
            # huge breasts IN THE CHURCH" puts the church in the scene, not on her
            after = re.split(r"\b(?:in|at|on|inside|outside|near|behind|under|beside|by)\b", after)[0]

        attrs = []
        for chunk in (before, after):
            for t in scan_vocab_tags(chunk, banks):
                # A camera or scene word standing next to a name is not a trait
                # of that person: 'pov, futanari Tifa Lockhart' bound 'pov' to
                # Tifa, and pov describes where the camera is, not who she is.
                if _sm.slot_of(t, banks) in ("viewpoint", "framing", "focus",
                                             "scene", "lighting"):
                    continue
                if t not in attrs:
                    attrs.append(t)
        if attrs:
            out[name] = attrs
    return out


def _join(items, conj="and"):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def category_entries(banks, kind, fallback):
    """Every entry in a category — dropdown AND tail. The cap limits what the UI
    lists, never what the generator can reach."""
    cat = (banks.get("_categories") or {}).get(kind)
    return (cat or {}).get("all") or fallback


# Both guides want the same shape: macro first, micro last. Illustrious' own
# guidance spells it out as quality/aesthetic -> count -> character+artist ->
# appearance+clothing -> action+pose -> setting; Anima's README fixes the first
# sections and leaves the rest free, so the same macro-to-micro order is used
# inside its general block. Without this the tags came out in whatever sequence
# the generator happened to add them, which reads as noise.
ORDER_QUALITY, ORDER_STYLE, ORDER_COUNT, ORDER_CHAR, ORDER_ARTIST = 0, 1, 2, 3, 4
ORDER_APPEARANCE, ORDER_CLOTHING = 5, 6
ORDER_ACT, ORDER_POSE, ORDER_SCENE, ORDER_LIGHT, ORDER_CAMERA = 7, 8, 9, 10, 11


# The place sets come from location_pool.json now -- danbooru's own locations wiki,
# classified indoor / building / outdoor-natural / outdoor-manmade. They used to
# be two hand-written sets covering 80 places while the location dropdown offered
# 239, so 159 locations could not drive the air tags or the collision check at all.
#
# BUILDINGS COUNT AS INDOOR HERE, and only here. Danbooru files them as a third
# thing because you can be inside a church or looking at one, and that is the
# right answer for deciding what CONFLICTS. But for deciding what to ADD, a
# building named as the setting means you are in it -- which is why "church"
# has to keep its "indoors" and not sprout "outdoors, sky". The two uses want
# different answers from the same fact, so the collision rule will read the
# kinds directly rather than reuse these sets.
def _load_place_kinds():
    path = _paths.data("location_pool.json")
    kinds = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                kinds = {k: v.get("kind", "") for k, v in json.load(f).items()}
        except Exception:
            kinds = {}
    return kinds

PLACE_KINDS = _load_place_kinds()
INDOOR_PLACES = {t for t, k in PLACE_KINDS.items()
                 if k in ("indoor", "building")}
OUTDOOR_PLACES = {t for t, k in PLACE_KINDS.items() if k.startswith("outdoor")}
if not INDOOR_PLACES:                 # pool missing: keep the old floor
    INDOOR_PLACES = {"church", "classroom", "bedroom", "bathroom", "kitchen",
                     "office", "library", "cafe", "restaurant", "bar (place)"}
    OUTDOOR_PLACES = {"beach", "forest", "mountain", "park", "street", "city"}

_ARTISTS = None          # {name: post_count}, 576k entries, loaded once
_ARTIST_DRAW = None      # names >= ARTIST_FLOOR, the uniform random pool
ARTIST_FLOOR = 100       # capability grounding for GENERATED artists only --
                         # a typed artist bypasses it entirely. 100 (the author,
                         # 2026-09-17): the fingerprints and the safety levels
                         # are measured from 100 posts (max of danbooru and
                         # gelbooru), so every drawable artist is a measured one


_ARTIST_DENY = None


def artist_denylist():
    """artist tags that must never be drawn OR accepted as typed.

    Kept as data, not a regex over names: a keyword sweep flagged twelve
    drawable artists containing loli/shota/child/gore, and sampling their
    posts showed most are ordinary handles where the word is incidental
    ('child (isoliya)' 0%, 'gore (white gore)' 0%). 'lolita' also names a
    real fashion style. Entries go in on evidence.
    """
    global _ARTIST_DENY
    if _ARTIST_DENY is None:
        _ARTIST_DENY = set()
        try:
            with open(_paths.data("artist_denylist.json"),
                      encoding="utf-8") as f:
                _ARTIST_DENY = {k.lower()
                                for k in (json.load(f).get("deny") or {})}
        except Exception:
            pass
    return _ARTIST_DENY


def _artist_pool():
    global _ARTISTS, _ARTIST_DRAW
    if _ARTISTS is None:
        p = _paths.data("artist_pool.json")
        try:
            with open(p, encoding="utf-8-sig") as f:
                _ARTISTS = json.load(f)["artists"]
        except Exception:
            _ARTISTS = {}
        deny = artist_denylist()
        # removed from the POOL as well as the draw list: the pool is what
        # decides whether a typed token counts as an artist, so leaving a
        # denied name there would still let it through when typed
        for n in list(_ARTISTS):
            if n.lower() in deny:
                del _ARTISTS[n]
        # A NAME MUST HAVE LETTERS OR DIGITS IN IT. danbooru carries artists
        # literally named `@ (artist)`, `@ . @ (kjjw2272)`, `@@@ (eckzahn)`;
        # rendered with the anima prefix and the punctuation cleaned they
        # come out as "@ @" -- a token nothing can steer with. A shape
        # rule, not a list: the bare name (before any qualifier) must
        # contain an alphanumeric character.
        def _steerable(n):
            return bool(re.search(r"[a-z0-9]", n.split(" (")[0], re.I))
        _ARTIST_DRAW = sorted(n for n, c in _ARTISTS.items()
                              if c >= ARTIST_FLOOR and n.lower() not in deny
                              and _steerable(n))
    return _ARTISTS, _ARTIST_DRAW


_ARTIST_SAFETY = None


def artist_safety():
    """-> (safe_set, not_safe_set), measured. See build_artist_safety.py.

    the author's: "it should be simplified to safe and not safe artists (not
    safe pool is the sensitive/nsfw/explicit pool from tags - this will
    make pools roughly equal)". Measured 1655 / 1355, and the vision
    descriptions correct 59 of them -- every correction in the same
    direction, safe -> not_safe, because the tags under-report.
    """
    global _ARTIST_SAFETY
    if _ARTIST_SAFETY is None:
        safe, not_safe = set(), set()
        try:
            with open(_paths.data("artist_safety.json"),
                      encoding="utf-8-sig") as f:
                blob = json.load(f)
            safe = {k.lower() for k in (blob.get("safe") or {})}
            not_safe = {k.lower() for k in (blob.get("not_safe") or {})}
            _ARTIST_LEVELS.clear()
            _ARTIST_LEVELS.update({k.lower(): v for k, v in (blob.get("levels") or {}).items()})
        except Exception:
            pass
        _ARTIST_SAFETY = (safe, not_safe)
    return _ARTIST_SAFETY


_ARTIST_LEVELS = {}
# TWO POOLS (the author's 2026-09-15): the tame pool (artists measured safe or
# sensitive) serves safe and sensitive prompts; the hot pool (measured
# nsfw or explicit) serves nsfw and explicit prompts. "Sensitive artists
# pollute nsfw and explicit prompts -- they are too tame." The measure
# behind the pools is four levels (build_artist_safety.py: the higher of
# the fingerprint's top-tag floor and the vision sweep's eroticism); the
# gate is two.
_ARTIST_POOL_OF = {"safe": "tame", "sensitive": "tame", "nsfw": "hot", "explicit": "hot"}


def artist_level(name):
    """-> the artist's measured level (safe/sensitive/nsfw/explicit), or
    None when unmeasured"""
    artist_safety()
    return _ARTIST_LEVELS.get(str(name).lstrip("@").replace(chr(92), "").strip().lower())


def artist_pool_of(name):
    """-> 'tame', 'hot', or None (unmeasured)"""
    return _ARTIST_POOL_OF.get(artist_level(name) or "")


def artist_allowed(name, spice):
    """-> may GENERATION use this artist at this level?

    One predicate for every generation path. The first cut of this gated
    only the free random draw, which is the path that fires LEAST often:
    style-match runs first and supplies most artists, so not-safe artists
    still reached safe images. Any new artist source must come through
    here rather than gain its own copy of the rule.
    """
    safe, not_safe = artist_safety()
    if not safe and not not_safe:
        return True                      # table missing: old behaviour
    if not _ARTIST_LEVELS:               # an old two-list table: its two pools
        n = str(name).lstrip("@").strip().lower()
        return n in safe if _sm.normalize_level(spice) == "safe" else (n in safe or n in not_safe)
    want = _ARTIST_POOL_OF.get(_sm.normalize_level(spice) or "safe", "tame")
    return artist_pool_of(name) == want


def artist_draw_pool(spice):
    """the artists GENERATION may draw at this level.

    Two pools, per the author's design: a safe image draws only from artists
    whose own measured work is safe; above safe, both pools are open,
    because a not-safe artist is a capability rather than a requirement.

    UNMEASURED ARTISTS ARE NOT DRAWN. An artist who cannot be classified
    cannot be matched to the level, which is the entire point of the
    control. Since the floor and the fingerprints both stand at 100 posts
    (2026-09-17), all 24,880 drawable artists are measured; the rule
    remains for a pool refreshed ahead of its fingerprints.
    TYPED artists are unaffected -- they never come from this pool.
    """
    _pool, draw = _artist_pool()
    safe, not_safe = artist_safety()
    if not safe and not not_safe:
        return draw                      # table missing: old behaviour
    out = [a for a in draw if artist_allowed(a, spice)]
    return out or draw


def _is_artist_name(name):
    """-> True when the bare name is one the artist pool knows."""
    try:
        bare = str(name or "").lstrip("@").replace(chr(92), "").strip().lower()
        pool, _draw = _artist_pool()
        return bool(bare) and bare in pool
    except Exception:
        return False


def _artist_format(name, mode):
    """Anima artists MUST carry '@' (README); Illustrious never does.

    Parentheses are ESCAPED: half of danbooru's artist names carry a
    qualifier -- `atelier (arainydancer)` -- and every downstream consumer
    (our own cleanup included) reads bare parens as emphasis syntax, which is
    how `@atelier (arainydancer)` shipped as `@atelier`."""
    bare = name.lstrip("@").strip()
    bare = bare.replace("(", chr(92) + "(").replace(")", chr(92) + ")")
    return ("@" + bare) if mode == "anima" else bare


def unclaimed_text(text, banks):
    """The text with every span the parser claims as a tag blanked.

    A WORD THE PARSER READS AS A TAG IS NOT AN ARTIST WORD (the author's
    2026-09-15: 'a naked cat girl' drew the artist @naked cat, 52 posts --
    'naked' is the nudity state 'nude' and 'cat girl' the race, both claimed
    by parse_input before the artist scan ever saw the text). The artist scan
    reads the text the way it already reads it without character names and
    place phrases: without the parser's own claims. The claim channels are
    the parser's (PHRASE_MAP and VIEWPOINT_MAP regexes, the multi-word
    NORMALIZE aliases, the vocabulary scan, WORD_TAGS), so a new alias or a
    new tag claims its words here too with no second list. '@name' tokens
    are held out: an explicit artist is never a claim. Each blanked span
    becomes a '|' word so no n-gram can bridge it.
    """
    s = str(text or "").lower()
    held = {}

    def _hold(m):
        k = "heldartist%dx" % len(held)
        held[k] = m.group(0)
        return k
    s = re.sub(r"@[\w()'-]+", _hold, s)
    for _map in (PHRASE_MAP, VIEWPOINT_MAP):
        for pat, mapped in _map:
            if mapped:
                s = re.sub(pat, " | ", s)
    for phrase in sorted(NORMALIZE, key=len, reverse=True):
        if " " in phrase:
            s = re.sub(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", " | ", s)
    for v in sorted(set(scan_vocab_tags(s, banks)), key=len, reverse=True):
        s = re.sub(r"(?<![a-z0-9])" + re.escape(str(v).lower()) + r"(?![a-z0-9])", " | ", s)
    s = " ".join("|" if w.strip(".!?") in WORD_TAGS else w for w in s.split())
    for k, v in held.items():
        s = s.replace(k, v)
    return s


_FINGERPRINTS = None


def _fingerprints():
    """{artist: {"tags": [top co-occurring tags], "posts": n}} -- the
    measured subjects of every fingerprinted artist (build_artist_fingerprints.py)"""
    global _FINGERPRINTS
    if _FINGERPRINTS is None:
        try:
            with open(_paths.data("artist_fingerprints.json"), encoding="utf-8-sig") as f:
                _FINGERPRINTS = json.load(f)["fingerprints"]
        except Exception:
            _FINGERPRINTS = {}
    return _FINGERPRINTS


def _fingerprint_fits(tags, names, banks):
    """-> {artist: 0..1} the idf-weighted share of the picture's informative
    tags the artist's fingerprint answers. A tag's weight is log(top / its
    posts): '1girl' weighs nothing, 'anal' or 'plate armor' a lot, so the
    fit says whether the artist draws THIS kind of picture, not whether
    they draw girls. Artists with no fingerprint are absent (neutral)."""
    fps = _fingerprints()
    if not fps or not tags or not names:
        return {}
    vocab = dict(banks.get("_scanvocab_low") or {})
    vocab.update(banks.get("_scanvocab") or {})
    if not vocab:
        return {}
    top = float(max(vocab.values()))
    want = {}
    for t in tags:
        t = str(t).lstrip("@").replace(chr(92), "").strip().lower()
        if not t or _MEASURE_SKIP.match(t) or t in want:
            continue
        p = vocab.get(t)
        if p:
            v = math.log(top / float(p))
            if v > 0:
                want[t] = v
    if not want:
        return {}
    out = {}
    for a in names:
        hit = sum(want[t] for t in ((fps.get(a) or {}).get("tags") or []) if t in want)
        if hit:
            out[a] = hit
    # rescaled across the candidates, as the embedding fit is: the question
    # is which of these artists answers this picture best (a fingerprint is
    # sixteen tags, so the raw share of a thirty-tag picture is always small)
    top_hit = max(out.values()) if out else 0.0
    return {a: v / top_hit for a, v in out.items()} if top_hit > 0 else {}


def pick_artists(tags_in, banks, mode, rng, enabled, base_text="",
                 spice="safe", fast=False, scene_tags=(), look_text=""):
    """The artist subsection: prompt > resolved style > random (the author's).

    Returns (typed_normalised, generated, note). TYPED ARTISTS ARE ALWAYS
    HONOURED, checkbox state irrelevant -- the checkbox governs generation
    only. A typed token counts as an artist if it starts with '@', or if it
    is in the artist pool AND is not also a general tag (an artist named after
    a common word must not hijack the word).

    'Resolved style' means whatever style the prompt carries, auto included;
    until the Style dropdown is wired into enhance, the prompt is the only
    resolution channel visible here, so this reads style_pool entries out of
    tags_in and draws from that style's measured artists (weighted by their
    share of the style's posts). No style -> uniform draw over the >=100-post
    pool: NOT popularity-weighted (the author's overruled that), the floor is only
    capability grounding. Count: 1-3 at 50/30/20.
    """
    pool, draw = _artist_pool()
    styles = banks.get("_style_pool") or {}
    # 'general' is every booru tag the reader knows at any floor (2026-09-13:
    # 'cottage' -- 'in a cluttered cottage' -- became the artist @cottage
    # because the place tag sits under the 2,000-post scan floor)
    general = dict(banks.get("_scanvocab_low") or {})
    general.update(banks.get("_scanvocab") or {})

    typed, rest = [], []
    for t in tags_in:
        raw = str(t).strip()
        # one normal form: the pool keys carry spaces, so '@shirow_masamune'
        # and the text's 'shirow masamune' are the same artist, once
        bare = raw.lstrip("@").lower().replace("_", " ")
        if raw.startswith("@") or (bare in pool and bare not in general):
            typed.append(_artist_format(bare, mode))
        else:
            rest.append(t)
    # TYPED ARTISTS MUST BE FOUND IN THE RAW TEXT TOO. parse_input is built
    # for content words and mangles names -- '@shirow_masamune' and 'drawn by
    # wlop' both came out of it with no artist tag at all. Scan the text's own
    # 1-3 word n-grams against the pool ('@'-prefixed tokens count even off
    # the pool: the user was explicit). The general-vocab exclusion keeps an
    # artist named after a common word from hijacking the word.
    if base_text:
        # only the words the parser left unclaimed can be a bare artist name
        words = re.findall(r"@?[\w()'-]+|\|", unclaimed_text(base_text, banks))
        words = [w.replace("_", " ") for w in words]
        have = {t.lstrip("@").lower() for t in typed}
        i = 0
        while i < len(words):
            hit = None
            if words[i].startswith("@"):
                # AN '@' NAME IS AS LONG AS THE POOL SAYS (2026-09-16: typed
                # '@himajin noizu' gave '@himajin noizu, @himajin' -- the
                # '@' branch took one word, and the parser had the full
                # name already). The longest pool name starting here wins;
                # a name the pool does not know stays one token, which is
                # how an unknown artist is written ('@some_artist').
                hit = words[i].lstrip("@")
                for j in (3, 2):
                    g = " ".join([hit] + words[i + 1:i + j])
                    if len(words[i:i + j]) == j and g in pool:
                        hit = g
                        i += j - 1
                        break
            else:
                for j in (3, 2, 1):
                    g = " ".join(words[i:i + j])
                    # Bare matches must clear the floor: the pool holds
                    # artists literally named `in a` (7 posts) and `at night`
                    # (2), and a micro-artist whose name doubles as an English
                    # phrase IS the phrase. '@' names skip this -- explicit.
                    if (g and g in pool and g not in general
                            and pool[g] >= ARTIST_FLOOR):
                        hit = g
                        i += j - 1
                        break
            if hit and hit not in have:
                typed.append(_artist_format(hit, mode))
                have.add(hit)
            i += 1
    if typed or not enabled or not pool:
        note = {"source": "prompt" if typed else "off",
                "artists": typed, "style": None}
        banks["_artist_note"] = note
        return typed, [], rest

    n = rng.choices((1, 2, 3), weights=(50, 30, 20), k=1)[0]
    low = {str(t).lower() for t in tags_in}
    style = next((s for s in styles if s in low), None)
    picks = []
    src = "random"
    # STYLE-MATCHED ARTISTS (the author's: the old measured-share draw picked
    # artists that rarely fit the style). The embedding matcher picks
    # artists whose measured fingerprint matches the resolved style's
    # vector -- a real stylistic match instead of a weak co-occurrence.
    if style:
        try:
            from promptstudio.library import matching as _match
            if _match.ready():
                picks = _match.artists_for_style(style, n=n, rng=rng)
                if picks:
                    src = "style-match"
        except Exception:
            picks = []
    # fallback to the old measured shares, then to a random draw
    if style and not picks and (styles[style].get("artists") or {}):
        cand = list(styles[style]["artists"].items())
        src = "style-share"
        while cand and len(picks) < n:
            total = sum(w for _, w in cand)
            r = rng.random() * total
            for i, (a, w) in enumerate(cand):
                r -= w
                if r <= 0:
                    picks.append(a)
                    cand.pop(i)
                    break
    # THE LEVEL APPLIES TO EVERY PATH (the author's rework point 5). style-match
    # and style-share are evidence-driven, but evidence about STYLE says
    # nothing about content: an artist can be the closest stylistic match
    # and still be one whose work is explicit. Filtering here rather than
    # inside each branch is what stops the next artist source from missing
    # the rule.
    picks = [a for a in picks if artist_allowed(a, spice)]
    level_draw = artist_draw_pool(spice)

    # THE WHOLE DESCRIPTION IS THE QUERY (the author's). The prompt is not
    # collapsed to one style word plus a spice level: every descriptor the
    # user wrote -- "detailed aesthetics, vivid colors" -- scores the
    # candidates, and the score decides BOTH how often an artist is drawn
    # and how strongly they are written. An artist with no vision record
    # scores neutral, so the sweep improves this without ever excluding
    # anyone for missing data.
    fits = {}
    try:
        from promptstudio.library import artist_fit as _fit
        # the style box's phrases describe the look too (2026-09-17)
        _q_text = ((base_text or "") + ", " + (look_text or "")).strip(", ")
        q = _fit.query_words(_q_text, tags_in)
        if q and level_draw:
            # FAST MODE STAYS OFF THE GPU. The embedding matcher is the
            # better scorer, but it wakes the embed server -- and fast mode
            # exists precisely so a prompt costs no model call. Word overlap
            # against the vision records is free and still differentiates.
            if not fast:
                fits = _fit.embedding_fits(_q_text, level_draw)
            if not fits:
                fits = {a: _fit.fit(a, q) for a in level_draw}
                fits = {a: v for a, v in fits.items() if v}
    except Exception:
        fits = {}

    # THE ARTIST'S OWN SUBJECTS (the author's 2026-09-15: "they are still picked
    # mostly randomly"): the fingerprint is the top tags of the artist's
    # own posts, measured, and the picture's tags are known by now. An
    # artist whose top tags cover what this picture is of (its acts, its
    # garments, its world) is drawn more often -- by the idf-weighted share
    # of the picture's informative tags the fingerprint answers -- and the
    # description fit above multiplies in. An unfingerprinted artist is
    # neutral, never excluded; nobody is picked exclusively.
    fp_fits = _fingerprint_fits(scene_tags or tags_in, level_draw, banks) if level_draw else {}
    if len(picks) < n and level_draw:
        names2 = list(level_draw)

        def _wf(a):
            w9 = 1.0 + 8.0 * fp_fits.get(a, 0.0)
            f9 = fits.get(a)
            if f9 is not None:
                try:
                    w9 *= _fit.weight(f9)
                except Exception:
                    pass
            return w9
        w2 = [_wf(a) for a in names2]
        for _ in range(n * 12):
            if len(picks) >= n:
                break
            a = rng.choices(names2, weights=w2)[0]
            if a not in picks:
                picks.append(a)
    while len(picks) < n and level_draw:
        a = rng.choice(level_draw)
        if a not in picks:
            picks.append(a)

    out = []
    for a in picks:
        formatted = _artist_format(a, mode)
        f = fits.get(a)
        if f:
            try:
                formatted = _fit.render(formatted, f)
            except Exception:
                pass
        out.append(formatted)
    banks["_artist_note"] = {"source": src, "artists": out, "style": style,
                             "pool": _ARTIST_POOL_OF.get(_sm.normalize_level(spice) or "safe"),
                             "fits": {a: round(v, 2)
                                      for a, v in fits.items()
                                      if a in picks and v},
                             "subject_fits": {a: round(v, 2)
                                              for a, v in fp_fits.items()
                                              if a in picks and v}}
    return [], out, rest


                          # draws one NL-channel cultural style


_MEASURE_SKIP = re.compile(
    r"^(masterpiece|best quality|amazing quality|very aesthetic|good quality|"
    r"superb quality|absurdres|incredibly absurdres|highres|newest|recent|mid|"
    r"early|old|year \d{4}|high detail|8k|ultra detailed|intricate details|"
    r"refined details|delicate details|nuanced details|professional illustration|"
    r"masterwork|score_\d|safe|sensitive|nsfw|explicit)$")


def _injectable(target, census, banks, rng):
    """A tag whose measured floor IS the target level and that fits the cast.

    Drawn from safety_floor.json itself -- the floor table doubles as the
    vocabulary of level-appropriate content, so injection cannot invent a tag
    or reach for one whose floor overshoots."""
    pool = [t for t, lv in _sm.SAFETY_FLOOR.items() if lv == target]
    rng.shuffle(pool)
    for t in pool:
        if not _sm.content_allowed(t, census, target, banks):
            continue
        need = _sm.arity_of(t)
        if need and need > max(1, census.get("total", 1)):
            continue
        return t
    return None
