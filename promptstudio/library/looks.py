"""
looks.py -- the words for how a picture looks, as the artists' own
descriptions say them (2026-09-17).

The vision sweep describes every artist from their pictures ('loose
linework', 'visible brushstrokes', 'muted palette', 'rim light'). Those
words are a vocabulary of looks. They serve three things:

1. THE STYLE HINT (read_style_hint). Every comma-separated entry of the
   style box is a look, kept whole (the author: "the 'loose linework' part
   should be respected too, not only for artist matching but for the
   generated prompt"):
   - an entry the studio already knows (a booru tag or its alias, a style
     of the pool, a medium spelling, a checkpoint concept) is read as
     before;
   - an entry a booru tag means as a whole becomes that tag ('rim light'
     -> rim lighting), found through the dictionary's words for each of
     its words -- never cut down to one of its words;
   - anything else stays the user's phrase, on the tag line and in the
     prose; it never reaches the word scanners, so 'muted palette' cannot
     become a paint palette in the scene.
2. THE SUGGESTIONS (vocabulary): the phrases the descriptions use for at
   least MIN_ARTISTS artists, with how many, age and lettering screened;
   those that lean to nsfw artists are marked.
3. THE MEASURE (description_lift): how the descriptions pair two looks --
   a typed look against a rolled style or medium the booru cannot measure
   together.

It also owns the AGE SCREEN every description and suggestion passes
(age_words): the vision sweep imports it.

Source: data/library/artist_signatures_vision.json once the new sweep has
written it (fields, agreement marks), else artist_styles_vision.json (the
traits of the first sweep).
"""

import json
import os
import re

from promptstudio import paths as _paths

MIN_ARTISTS = 10          # a suggestion is a phrase the descriptions use for this many artists

# ---------------------------------------------------------------- the age screen
# THE AGE SCREEN (the author, 2026-09-17): a description never states age.
# The studio's youth vocabulary (tags, lexicon youth words, the prose floor)
# and the words that state an age at all; the words the author ruled
# typed-ok for adults ('schoolgirl', 2026-09-06) are exempt, and words like
# 'innocent' or 'small chest' are not age words (adults play and have them).
AGE_WORDS = re.compile(r"\b(young\w*|youth\w*|juvenile|adolescen\w*|pubescent|prepubescent|underage|"
                       r"minors?|childlike|child-like|kids?|teen\w*|\d+\s*(?:years?[- ]old|yo))\b", re.I)
LETTERING = re.compile(r"\b(texts?|lettering|watermarks?|signatures?|logos?|credits?|captions?|"
                       r"speech bubbles?|sound effects?|onomatopoeia)\b", re.I)


def age_words(obj):
    """-> the age words a text, a list or a record uses ([] when none)"""
    from promptstudio.engine import bridge as lb

    def flat(o):
        if isinstance(o, dict):
            return " ".join(flat(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return " ".join(flat(v) for v in o)
        return str(o or "")
    text = flat(obj)
    low = " " + text.lower() + " "
    hits = set(lb._invent_age_hit(text)) | {m.group(0).lower() for m in AGE_WORDS.finditer(text)}
    for w in _youth_words():
        if re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(w), low):
            hits.add(w)
    return sorted(h for h in hits if h not in lb._YOUTH_EXEMPT)


_YW = {"data": None}


def _youth_words():
    if _YW["data"] is None:
        from promptstudio.engine import bridge as lb
        _YW["data"] = sorted(w for w in (lb.youth_tags() | lb._SEXUAL_YOUTH) if len(w) > 3)
    return _YW["data"]


# ---------------------------------------------------------------- the descriptions
_SRC = {"key": None, "records": None, "bags": None, "levels": None}


def _norm(phrase):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 \-]+", " ", str(phrase).lower())).strip()


def _records():
    """-> {artist: [(phrase, field or None, agreed)]}, from the newest source"""
    new = _paths.data("artist_signatures_vision.json", expect=False)
    old = _paths.data("artist_styles_vision.json", expect=False)
    gl = _paths.data("look_glossary.json", expect=False)
    key = tuple(os.path.getmtime(p) if p and os.path.exists(p) else 0 for p in (new, old, gl))
    if _SRC["key"] == key:
        return _SRC["records"]
    out, raw = {}, {}
    # the first sweep for every artist it described, the new sweep over it
    # artist by artist as its night slices arrive (2026-09-17)
    try:
        with open(old, encoding="utf-8-sig") as f:
            for a, r in (json.load(f).get("artists") or {}).items():
                if (r or {}).get("traits"):
                    raw[a] = [(p, None, True) for p in r["traits"]]
                    out[a] = _terms_of(r) or raw[a]
    except Exception:
        pass
    try:
        with open(new, encoding="utf-8") as f:
            for a, r in (json.load(f).get("artists") or {}).items():
                if (r or {}).get("fields"):
                    raw[a] = [(p, fld, mark == "agreed") for fld, marks in r["fields"].items()
                              for mark in ("agreed", "once") for p in (marks.get(mark) or [])]
                    out[a] = _terms_of(r) or raw[a]
    except Exception:
        pass
    _SRC.update(key=key, records=out, raw=raw, bags=None)
    return out


def _terms_of(rec):
    """-> the terminology pass's terms of a record as [(term, None, agreed)],
    or None when the pass has not run on it"""
    t = (rec or {}).get("terms")
    if not isinstance(t, dict):
        return None
    return [(x, None, True) for x in (t.get("agreed") or [])] + [(x, None, False) for x in (t.get("once") or [])]


def raw_records():
    """-> {artist: [(phrase, field, agreed)]}: the descriptions as the model wrote
    them, for the terminology pass"""
    _records()
    return _SRC.get("raw") or {}


def _bags():
    """-> {artist: set of description words} (digits kept: '3d')"""
    recs = _records()
    if _SRC["bags"] is None:
        _SRC["bags"] = {a: {w for p, _f, _g in items for w in re.findall(r"[a-z0-9]+", str(p).lower())
                            if len(w) > 1}
                        for a, items in recs.items()}
    return _SRC["bags"]


def _nsfw_artists():
    """the artists the booru measures at nsfw or explicit (artist_safety)"""
    if _SRC["levels"] is None:
        try:
            with open(_paths.data("artist_safety.json"), encoding="utf-8-sig") as f:
                lv = json.load(f).get("levels") or {}
            _SRC["levels"] = {a.lower() for a, v in lv.items() if v in ("nsfw", "explicit")}
        except Exception:
            _SRC["levels"] = set()
    return _SRC["levels"]


# ---------------------------------------------------------------- one look, one phrase
# ONE LOOK, ONE PHRASE (the author, 2026-09-17: "comedic and romantic mood",
# "romantic and comedic mood" and "comedic and romantic" were three looks
# linking to different artists, and one artist's description could hold
# several forms of the same meaning). A description phrase is broken into
# its ATOMS -- one quality and what it qualifies: 'comedic and romantic mood'
# is 'comedic mood' and 'romantic mood'; 'large, expressive eyes' is 'large
# eyes' and 'expressive eyes'; 'highly detailed and glossy skin' is 'highly
# detailed skin' and 'glossy skin'. The parts of speech are the dictionary's
# (lexicon.py). Atoms are compared by their words, in any order and in the
# singular, so 'cel-shaded digital rendering' and 'digital cel-shaded
# rendering' agree. A pair the dictionary keeps whole ('black and white') is
# never split. The raw descriptions stay as the model wrote them; this reads
# them, so the rules can improve without describing anyone again.
# the words that join looks, prepositions included ('split panels in comics');
# 'on' and 'of' also join a relation word to its look ('focus on large
# breasts', 'heavy use of halftone') -- that relation is no look
_JOINERS = ("and", "or", "but", "yet", "while", "though", "although", "versus", "vs", "to", "with", "without",
            "plus", "on", "of", "in", "at", "for", "from", "by", "into", "over", "under", "within", "across", "as",
            "like", "via", "through", "using", "featuring", "including", "showing", "depicting", "against",
            "beside", "between", "among", "amid", "around", "than")
_CONNECT = re.compile(r"(\s*[,;/&()\[\]]\s*|\s+(?:%s)\s+)" % "|".join(_JOINERS))
_RELATION = re.compile(r"^\s+(?:on|of)\s+$")
# FILLER IS NO LOOK (the author, 2026-09-17: "yet whimsical mood", "and mood",
# "as subjects", "acts depicted", "one image"): articles, joining words left at
# the edge of a run, the words that say how often, and the verbs a description
# uses to point at its subject
_DROP = {"a", "an", "the", "some", "very", "often", "sometimes", "frequently", "occasionally", "usually",
         "typically", "mostly", "generally", "consistently", "predominantly", "primarily", "mainly",
         "commonly", "always", "various", "varied", "recurring", "overall", "both", "either", "such",
         "especially", "particularly", "notably", "also", "more", "most", "less", "certain", "one", "two",
         "depicted", "shown", "featured", "portrayed", "presented", "displayed", "included", "seen",
         "its", "their", "his", "her", "this", "that", "these", "those", "each", "every", "other", "many",
         "several", "multiple", "numerous", "frequent", "occasional", "typical", "common", "usual",
         "general", "consistent", "specific", "particular", "distinct", "notable"} | set(_JOINERS)
# a verb that opens a run points at what follows ('features cat ears')
_POINTING = {"features", "shows", "depicts", "uses", "includes", "contains", "displays", "emphasizes",
             "emphasises", "showcases", "portrays", "presents", "employs", "utilizes", "utilises"}
# A HEAD THAT COUNTS OR RELATES IS NO LOOK ('wide range', 'high level', 'heavy
# use'): with 'of' after it the whole run goes ('wide range of colors' is the
# colors)
_QUANTITY = {"range", "variety", "mix", "array", "assortment", "selection", "number", "amount", "lot", "lots",
             "plenty", "series", "collection", "set", "kinds", "kind", "types", "type", "sorts", "sort", "use",
             "usage", "level", "levels", "degree", "abundance", "inclusion", "application", "choice", "choices",
             "hint", "hints", "presence", "variations", "variation", "combination", "blend"}
# THE OLD SWEEP'S RATINGS ARE NO LOOKS (the author's ruling: the level is the
# booru's): its eroticism labels, and a look that is an absence ('no nudity',
# 'non-sexualised', 'not explicit')
# qualities of degree say how much, not what ('heavy', 'subtle')
_DEGREE = {"heavy", "strong", "subtle", "slight", "high", "low", "significant", "noticeable", "considerable",
           "great", "minor", "major", "moderate", "notable", "marked", "clear", "overt", "obvious", "certain",
           "extreme", "mild", "light", "deep", "full", "partial", "physical", "visual"}
_RATING = {"none", "chaste", "suggestive", "explicit", "nsfw", "sfw", "r18", "r-18", "questionable", "no", "not",
           "non", "nonsexual", "clothed"}
# WRAPPER NOUNS ADD NO LOOK (the author, 2026-09-17: "whimsical elements /
# aesthetics / themes / tone", "action scenes / themes / action-oriented
# composition"): a quality with one of these is the quality. They come in
# three families, because a wrapper still says WHAT is qualified: 'warm tones'
# (a palette) is not 'warm mood', and 'curvy figures' (a body) is not 'curvy
# lines'. Measuring which nouns share their qualities did not separate them
# (frequent qualities like 'digital' made 'linework' look like 'rendering'),
# so the families are listed -- correct the lists, not the looks.
_GENERIC = {"mood", "moods", "atmosphere", "atmospheres", "theme", "themes", "element", "elements", "aesthetic",
            "aesthetics", "feel", "feeling", "vibe", "vibes", "register", "tone", "sensibility", "sensibilities",
            "quality", "qualities", "touch", "energy", "spirit", "undertone", "undertones", "overtone",
            "overtones", "sense", "air", "flavor", "flavour", "nature", "style", "styles", "look", "looks",
            "scene", "scenes", "scenario", "scenarios", "situation", "situations", "setting", "settings",
            "context", "contexts", "moment", "moments", "content", "focus", "emphasis", "depiction",
            "depictions", "imagery", "image", "images", "art", "artwork", "artworks", "work", "works",
            "illustration", "illustrations", "piece", "pieces", "design", "designs", "tradition",
            "traditions", "influence", "influences", "format", "technique", "techniques", "presentation",
            "appearance", "detail", "details", "effect", "effects", "motif", "motifs", "pattern", "patterns",
            "composition", "compositions", "framing", "approach", "treatment", "storytelling", "subject matter",
            "portrayal", "portrayals", "rendition", "renditions", "vibe", "flair", "quality"}
_COLOR = {"palette", "palettes", "color", "colors", "colour", "colours", "coloring", "colouring", "colorings",
          "tones", "hue", "hues", "scheme", "schemes", "colorway", "colouration", "coloration"}
_FIGURE = {"anatomy", "anatomies", "body", "bodies", "figure", "figures", "form", "forms", "physique",
           "physiques", "build", "builds", "frame", "frames", "proportion", "proportions", "shape", "shapes",
           "silhouette", "silhouettes", "character", "characters", "girl", "girls", "woman", "women",
           "females", "males", "subject", "subjects", "protagonist", "protagonists", "people", "person",
           "persons", "individuals"}
_WRAPPERS = _GENERIC | _COLOR | _FIGURE
# 'watercolor-like rendering' is a watercolor rendering, 'action-oriented' is
# action, 'pastel-heavy palette' a pastel palette
_LIKE = re.compile(r"-(?:like|style|styled|esque|inspired|ish|influenced|oriented|heavy|driven|focused|centric|"
                   r"centered|centred|based|themed|leaning|rich|laden|infused|tinged|type|looking)$")
_POS_CACHE = {}


def _pos(w):
    if w not in _POS_CACHE:
        try:
            from promptstudio.engine import lexicon as _lx
            _POS_CACHE[w] = frozenset(_lx.pos(w) or ())
        except Exception:
            _POS_CACHE[w] = frozenset()
    return _POS_CACHE[w]


def _quality(w):
    """a word that can qualify a noun: an adjective ('romantic' is also a
    noun, 'comedic' unknown to the dictionary) -- not a noun-only word
    ('hearts', 'portraits')"""
    p = _pos(w)
    return not p or "a" in p or "n" not in p


_ATOMS = {}


def atoms(phrase):
    """-> the single looks a description phrase holds, in its own words"""
    key0 = str(phrase or "")
    if key0 in _ATOMS:
        return _ATOMS[key0]
    import unicodedata
    s = " " + unicodedata.normalize("NFKD", key0).encode("ascii", "ignore").decode().lower().strip() + " "
    # the dictionary's fixed pairs stay whole ('black and white')
    for m in re.finditer(r"\b([a-z-]+) (and|or) ([a-z-]+)\b", s):
        whole = m.group(0)
        if _pos(whole) & {"n", "a"}:
            s = s.replace(whole, whole.replace(" ", "-"))
    parts = _CONNECT.split(s)
    segs = []
    for i in range(0, len(parts), 2):
        ws = [_LIKE.sub("", w) for w in re.findall(r"[a-z0-9][a-z0-9-]*", parts[i])]
        ws = [w for w in ws if w and w not in _DROP]
        while ws and ws[0] in _POINTING and len(ws) > 1:
            ws = ws[1:]
        joiner = parts[i + 1] if i + 1 < len(parts) else ""
        if not ws:
            continue
        # a relation before 'on' / 'of' is no look ('focus on', 'heavy use of')
        if _RELATION.match(joiner) and ((len(ws) == 1 and not _quality(ws[0])) or ws[-1] in _QUANTITY):
            continue
        segs.append(ws)
    if not segs:
        _ATOMS[key0] = []
        return []
    # A LIST SHARES ITS NOUN AT THE END: a one-word run that can qualify
    # ('comedic', 'romantic') takes the noun of the next run of two or more
    # words ('romantic and comedic mood'); runs of single nouns stay apart
    # ('hearts and sparkles')
    shared = [None] * len(segs)
    for i, ws in enumerate(segs):
        # ...and so does a run made only of qualities ('highly detailed' of
        # 'highly detailed and glossy skin')
        only_q = all(_quality(w) or _pos(w) == {"r"} for w in ws) and "a" in _pos(ws[-1])
        if (len(ws) == 1 and _quality(ws[0])) or (len(ws) > 1 and only_q):
            nxt = next((segs[j] for j in range(i + 1, len(segs)) if len(segs[j]) > 1), None)
            if nxt is not None:
                shared[i] = nxt
    # ...and a lone noun after a run with qualities takes those qualities
    # ('detailed hair and clothing' is detailed clothing too; 'large breasts
    # and buttocks' large buttocks)
    lent = [None] * len(segs)
    for i, ws in enumerate(segs):
        if i and len(ws) == 1 and not _quality(ws[0]) and shared[i] is None and shared[i - 1] is None:
            prev_mods = _head(segs[i - 1])[0]
            if prev_mods and all(_quality(w) or _pos(w) == {"r"} for w in prev_mods):
                lent[i] = prev_mods
    out = []
    for ws, donor, borrowed in zip(segs, shared, lent):
        if donor is not None:
            head_words = _head(donor)[1]
            mods = ws
        elif borrowed is not None:
            mods, head_words = list(borrowed), list(ws)
        else:
            mods, head_words = _head(ws)
        units, carry, i = [], [], 0
        while i < len(mods):
            w = mods[i]
            # an adverb qualifies the word after it ('highly detailed')
            if _pos(w) == {"r"} or (not _pos(w) and w.endswith("ly")):
                carry.append(w)
                i += 1
                continue
            # two qualities the booru or the dictionary keep as one ('high
            # contrast' palette is not a high palette and a contrast palette)
            if i + 1 < len(mods) and _pair(w, mods[i + 1]):
                units.append(carry + [w, mods[i + 1]])
                carry = []
                i += 2
                continue
            units.append(carry + [w])
            carry = []
            i += 1
        if carry:
            units.append(carry)
        if not units:
            out.append(" ".join(head_words))
        else:
            out += [" ".join(u + head_words) for u in units]
    res = list(dict.fromkeys(out))
    _ATOMS[key0] = res
    return res


_PAIRS = {}


def _pair(a, b):
    """two words that name one quality: a booru tag or alias ('high contrast'),
    or a compound the dictionary knows"""
    k = a + " " + b
    if k not in _PAIRS:
        _PAIRS[k] = k in _tag_posts() or k in _aliases() or bool(_pos(k))
    return _PAIRS[k]


def _head(ws):
    """-> (qualities, head words): the last word, with the noun-only words
    before it ('close-up portraits' stays whole; 'skin texture' is a quality
    of texture)"""
    # a run that ends on a word the dictionary knows and that cannot be a
    # noun names no thing: its words are qualities ('attractive, sexualised')
    if _pos(ws[-1]) and "n" not in _pos(ws[-1]) and len(ws) > 1:
        return list(ws), []
    head = [ws[-1]]
    mods = list(ws[:-1])
    while mods and _pos(mods[-1]) == {"n"}:
        head.insert(0, mods.pop())
    return mods, head


_KEYS = {}
_BASE_CACHE = {}


def _base(w):
    """-> a word in the form that compares: American spelling, a noun in the
    singular, a verb form at its lemma ('cel-shaded' and 'cel shading' both
    shade; 'rendering' render; 'stylised' stylize)"""
    if w in _BASE_CACHE:
        return _BASE_CACHE[w]
    try:
        from promptstudio.engine import lexicon as _lx
    except Exception:
        _lx = None
    out = _american(w)
    if _lx is not None:
        if out.endswith(("ed", "ing")) and "v" in _pos(out):
            cands = [x for x in _lx.lemmas(out, "v") if x != out]
            if cands:
                out = min(cands, key=len)
        elif out.endswith("s") and "n" in _pos(out):
            cands = [x for x in _lx.lemmas(out, "n") if x != out]
            if cands:
                out = min(cands, key=len)
    _BASE_CACHE[w] = out
    return out


def atom_key(atom):
    """-> the words of an atom, compared as the same look: split at hyphens,
    in their base form, in no order, wrapper nouns folded into their family
    ('whimsical themes' = 'whimsical mood'; 'warm tones' = 'warm palette';
    'curvy figures' = 'curvy anatomy'); a rating or an absence is marked
    '#rating'"""
    a = str(atom).lower()
    if a in _KEYS:
        return _KEYS[a]
    raw = re.findall(r"[a-z0-9]+", a.replace("-", " "))
    key, marks, wrapped = [], set(), []
    for w in raw:
        w0 = _american(w)
        if w0 in _RATING:
            marks.add("#rating")
            continue
        if w0 in _WRAPPERS:
            wrapped.append(_base(w0))
            if w0 in _COLOR:
                marks.add("#color")
            elif w0 in _FIGURE:
                marks.add("#figure")
            continue
        key.append(_base(w0))
    if any(w.startswith("non") and len(w) > 5 and _pos(w[3:]) for w in raw):
        marks.add("#rating")
    if not key:
        key, marks = wrapped, marks & {"#rating"}
    _KEYS[a] = frozenset(key) | frozenset(marks)
    return _KEYS[a]


def _content_tag(name):
    """-> the booru tag a look names about the subject, or None: the look
    without its wrapper nouns, in the singular ('school uniforms' -> school
    uniform, 'blue hair colors' -> blue hair), whose gloss flags are all
    about the subject (hair, clothing, body, expression, an object...)"""
    try:
        from promptstudio.engine import bridge as _lb
    except Exception:
        return None
    posts, al = _tag_posts(), _aliases()
    words = [w for w in name.split() if w not in _WRAPPERS]
    cands = [name, " ".join(words)]
    if words:
        cands.append(" ".join(words[:-1] + [_base(words[-1])]))
    for c in cands:
        c = c.strip()
        t = c if c in posts else al.get(c)
        if t and t in posts:
            fl = set(_lb._gloss_flags(t) or ())
            if fl and not (fl & _LOOK_FLAGS):
                return t
    return None


_SPELL = ((r"our$", "or"), (r"ours$", "ors"), (r"our(ed|ing|ful|less)$", r"or\1"), (r"ise$", "ize"),
          (r"ised$", "ized"), (r"ising$", "izing"), (r"isation$", "ization"), (r"yse$", "yze"),
          (r"ysed$", "yzed"), (r"tre$", "ter"), (r"tres$", "ters"), (r"ogue$", "og"), (r"lled$", "led"),
          (r"lling$", "ling"), (r"aemia$", "emia"), (r"grey", "gray"))
_SPELL_CACHE = {}


def _american(w):
    """-> the American spelling when the dictionary knows it as the same word
    ('colours' -> colors, 'idealised' -> idealized, 'grey' -> gray): the
    variant is taken only when it is a word of the same parts of speech"""
    if w in _SPELL_CACHE:
        return _SPELL_CACHE[w]
    out = w
    for pat, sub in _SPELL:
        cand = re.sub(pat, sub, w)
        if cand != w and _pos(cand) and (_pos(cand) & _pos(w) or not _pos(w)):
            out = cand
            break
    _SPELL_CACHE[w] = out
    return out


_VOCAB = {"key": None, "data": None}


# ---------------------------------------------------------------- what is a look
_LOOKTAGS = {"data": None, "words": set()}


def _look_tags():
    """-> the booru tags of style, medium, lighting, effect, view and the like
    (their gloss flags), and the studio's own style vocabulary (the style pool,
    the medium spellings): the words a look may be or be built from"""
    if _LOOKTAGS["data"] is None:
        out = set()
        try:
            with open(_paths.data("tag_glosses.json"), encoding="utf-8-sig") as f:
                for t, e in (json.load(f).get("glosses") or {}).items():
                    if set((e or {}).get("f") or ()) & _LOOK_FLAGS:
                        out.add(str(t).lower().replace("_", " "))
        except Exception:
            pass
        try:
            from promptstudio.engine import bridge as _lb
            out |= {str(k).lower() for k in (_lb._style_pool2() or {}) if not str(k).startswith("_")}
            ts = (_lb._medium_pool() or {}).get("typed_surfaces") or {}
            out |= {str(k).lower() for k in ts if not str(k).startswith("_")}
            out |= {str(v).lower() for v in ts.values() if isinstance(v, str)}
        except Exception:
            pass
        # a medium tag is also its bare name ('ink (medium)' is ink)
        out |= {re.sub(r"\s*\([^)]*\)\s*$", "", t) for t in out}
        _LOOKTAGS["data"] = out
        # ...and the words that qualify in look tags describe as modifiers
        # ('anime' of 'anime coloring', 'cel' of 'cel shading'; not 'background'
        # of 'simple background', which is what those tags qualify) -- never a
        # body or a garment
        try:
            from promptstudio.engine import lexicon as _lx
        except Exception:
            _lx = None
        _LOOKTAGS["words"] = {w for t in out for w in re.findall(r"[a-z0-9-]+", t)[:-1]
                              if len(w) > 2 and not (_lx and (_lx.word_class(w, "body") or _lx.word_class(w, "clothing")))}
    return _LOOKTAGS["data"]


_GENRES = {"data": None}


def _genres():
    """the studio's genres (genre_pool.json) -- a genre is what a picture is
    about, not how it looks"""
    if _GENRES["data"] is None:
        try:
            with open(_paths.data("genre_pool.json"), encoding="utf-8-sig") as f:
                g = json.load(f)
            names = {str(k).lower() for k in (g.get("genres") or g) if not str(k).startswith("_")}
        except Exception:
            names = set()
        names |= {n.replace("science fiction", "sci-fi") for n in names}
        # and every booru tag its wiki calls a genre ('a science fiction genre')
        try:
            with open(_paths.data("tag_glosses.json"), encoding="utf-8-sig") as f:
                for t, e in (json.load(f).get("glosses") or {}).items():
                    if re.search(r"genre", str((e or {}).get("g") or "")[:120], re.I):
                        names.add(str(t).lower().replace("_", " "))
        except Exception:
            pass
        _GENRES["data"] = names
    return _GENRES["data"]


# a head a look tag may drop when the tag covers it: the tag's kind (its gloss
# flags) or its definition says so. 'dark mood' is not the booru's dark (a
# lighting), 'realistic anatomy' not realistic (a style), 'heavy shading' not
# heavy; 'backlit lighting' is backlighting, 'chibi proportions' chibi, 'manga
# style' comic
_HEAD_KINDS = (
    ({"mood", "moods", "atmosphere", "atmospheres", "feel", "feeling", "vibe", "vibes", "tone", "energy", "spirit",
      "undertones", "overtones", "sensibility", "air", "theme", "themes"}, {"theme"}, None),
    ({"style", "styles", "aesthetic", "aesthetics", "look", "looks", "art", "artwork", "technique", "techniques",
      "imagery", "rendering", "renderings", "render", "quality", "touch", "sensibilities", "elements", "element"},
     {"style", "medium", "format", "theme"}, None),
    (_COLOR, set(), r"colou?r|palette|hue|tone|monochrom|gr[ae]yscale|saturat|black and white"),
    (_FIGURE, set(), r"proportion|body|bodies|anatom|figure|head|limb|deformed"),
    ({"lighting", "light", "lights"}, {"lighting"}, None),
    ({"shading", "shade", "shadows"}, set(), r"shad"),
    ({"linework", "lines", "line", "strokes", "outlines"}, set(), r"\bline|stroke|outline|hatch"),
    ({"effect", "effects"}, {"effect"}, None),
    ({"framing", "composition", "compositions", "shot", "shots", "angle", "perspective"}, {"view", "camera"}, None),
)
_GLOSS = {"data": None}


def _gloss(tag):
    """-> (flags, definition) of a booru tag from tag_glosses.json"""
    if _GLOSS["data"] is None:
        out = {}
        try:
            with open(_paths.data("tag_glosses.json"), encoding="utf-8-sig") as f:
                for t, e in (json.load(f).get("glosses") or {}).items():
                    out[str(t).lower().replace("_", " ")] = (frozenset((e or {}).get("f") or ()), str((e or {}).get("g") or ""))
        except Exception:
            pass
        _GLOSS["data"] = out
    return _GLOSS["data"].get(tag) or _GLOSS["data"].get(re.sub(r"\s*\([^)]*\)\s*$", "", tag)) or (frozenset(), "")


def _covers(tag, head):
    """True when a look tag covers a head the description put after it"""
    flags, text = _gloss(tag)
    for heads, kinds, pattern in _HEAD_KINDS:
        if head in heads:
            return bool(flags & kinds) or bool(pattern and re.search(pattern, text, re.I))
    return False


def canonical_tag(atom, alias_ok=True):
    """-> the look tag a description phrase means as a whole, or None: the
    phrase itself, or the phrase with trailing heads the tag covers taken off
    (_covers: 'backlit lighting' -> backlighting, 'pixel art style' -> pixel
    art, 'chibi proportions' -> chibi); the last word also in the singular
    ('sparkles' -> sparkle); through the booru's aliases when alias_ok; only
    tags of the look (style, medium, lighting, effect...). A single word of
    degree ('heavy') is no tag of a look, nor a tag that also names a thing
    ('ink', 'military', 'red eyes')."""
    a = _norm(atom).replace("-and-", " and ")
    words = a.split()
    posts, al, looktags = _tag_posts(), _aliases(), _look_tags()

    def lookup(c):
        c = c.strip()
        if not c:
            return None
        for t in (c, al.get(c), al.get(c.replace(" ", "-")), al.get(c.replace("-", " "))):
            if t is None or (t != c and not alias_ok):
                continue
            if t in looktags and (t in posts or " " in t or t in al.values()):
                if (" " not in t and t in _DEGREE) or _thing_flags(t):
                    return None
                return t
        return None

    def forms(ws):
        yield " ".join(ws)
        if ws and ws[-1].endswith("s") and _base(ws[-1]) != ws[-1]:
            yield " ".join(ws[:-1] + [_base(ws[-1])])
    for c in forms(words):
        t = lookup(c)
        if t:
            return t
    trimmed, dropped = list(words), []
    while len(trimmed) > 1 and trimmed[-1] in (_WRAPPERS | {h for hs, _k, _p in _HEAD_KINDS for h in hs}):
        dropped.append(trimmed.pop())
        for c in forms(trimmed):
            t = lookup(c)
            if t and all(_covers(t, h) for h in dropped):
                return t
    return None


_THING_FLAGS = {"object", "symbol", "clothing", "body", "hair", "character", "location"}
# the subject's sex is no look ('female anatomy', 'male figures')
_SEX = {"female", "male", "futanari", "futa"}


def _thing_flags(tag):
    """-> the gloss flags of a tag that name a thing, not a look"""
    try:
        from promptstudio.engine import bridge as _lb
        return set(_lb._gloss_flags(tag) or ()) & _THING_FLAGS
    except Exception:
        return set()


def _subject_tag(atom):
    """-> (tag, flags) when the phrase (its trailing wrappers off, the last word
    singular, through aliases) is a booru tag whose flags are not the look's:
    a body, hair, clothing, an object -- else None"""
    a = _norm(atom)
    words = a.split()
    cands = [a]
    trimmed = list(words)
    while len(trimmed) > 1 and trimmed[-1] in _WRAPPERS:
        trimmed = trimmed[:-1]
        cands.append(" ".join(trimmed))
    for c in list(cands):
        ws = c.split()
        if ws and ws[-1].endswith("s"):
            cands.append(" ".join(ws[:-1] + [_base(ws[-1])]))
    posts, al, looktags = _tag_posts(), _aliases(), _look_tags()
    for c in cands:
        t = c if c in posts else al.get(c)
        if t and t in posts:
            # a medium is no thing ('ink work' is ink (medium)); a pose, an act
            # or a mood is no subject ('dynamic poses', 'affectionate mood')
            if t + " (medium)" in looktags:
                return None
            fl = _thing_flags(t)
            if fl:
                return t, fl
    return None


# EYES AND HAIR ARE NO LOOK (the author, 2026-09-18): 'expressive eyes', 'large
# anime eyes', 'detailed hair', 'flowing hair' -- qualified or not, and the
# signature sweep no longer asks for them
EYES_HAIR = re.compile(r"(?<![a-z])(?:eyes?|eyed|eyelash(?:es)?|lashes|eyelids?|irises|iris|pupils?|"
                       r"hairs?|haired|hairstyles?|haircuts?|hairdos?)(?![a-z])")


def scope(atom):
    """-> None when the phrase is a look, else why it is not (the author,
    2026-09-17): a look describes HOW something is drawn -- a quality and what
    it qualifies ('exaggerated facial expressions'), a quality alone
    ('whimsical'), or a look tag ('speed lines', 'pixel art'). Not a look:
    a bare noun that describes nothing ('facial expressions', 'hand gestures',
    'background rendering'), a bare body part ('buttocks'), a booru tag about
    the subject ('blue eyes', 'long hair', 'animal ears' -- while a body's
    size or shape stays: 'large breasts', 'wide hips', 'perky breasts'),
    clothing, a genre ('fantasy settings'), a word of degree alone ('heavy'),
    wrapper nouns alone."""
    a = _norm(atom)
    words = a.split()
    plain = [w for w in words if w not in _WRAPPERS]
    if not plain:
        return "wrapper nouns alone"
    if EYES_HAIR.search(a):
        return "eyes or hair"
    try:
        from promptstudio.engine import lexicon as _lx
    except Exception:
        _lx = None

    def cls(w, c):
        return bool(_lx and _lx.word_class(w, c))
    genres = _genres()

    def is_genre(t):
        return bool(t) and (t in genres or t.replace("science fiction", "sci-fi") in genres)
    if len(plain) == 1 and plain[0] in _DEGREE and not any(w in _COLOR or w in ("detail", "details", "effect", "effects") for w in words):
        return "degree alone"                  # 'heavy emphasis'; 'light palette', 'high detail' stay
    joined = " ".join(plain)
    if is_genre(joined) or all(is_genre(w) or is_genre(w.replace("-", " ")) for w in plain):
        return "genre"
    tag = canonical_tag(a)
    if is_genre(tag):
        return "genre"
    if len(plain) == 1 and plain[0] not in _DEGREE and any(d + " " + plain[0] in _look_tags() for d in ("high", "low")):
        return "degree missing"                # 'contrast' -- the booru's look is high or low contrast
    # the phrase itself a look tag is a look ('pixel art style', 'speed lines'); a
    # single word only when its tag names no thing ('military' is also uniforms)
    own = canonical_tag(a, alias_ok=False)
    if own:
        return None
    looktags, lookwords = _look_tags(), _LOOKTAGS["words"]
    head, mods = plain[-1], plain[:-1]

    def participle(m):
        return m.endswith(("ed", "ing")) and "v" in _pos(m) and not (cls(m, "clothing") or cls(m, "body"))

    def describing(m):
        if cls(m, "relational"):
            return False                       # 'facial' of 'facial expressions'
        if m in looktags or m.replace("-", " ") in looktags or m in lookwords:
            return True                        # 'watercolor shading', 'anime eyes'
        p = _pos(m)
        if not p:
            return not cls(m, "body")          # unknown words: 'cel-shaded', 'chibi'
        return "a" in p or p == {"r"} or participle(m)
    if plain and all(w in _SEX for w in plain[:-1] or plain):
        return "subject tag"
    if cls(head, "body") and mods and all(cls(m, "colour") for m in mods):
        return "subject tag"                   # 'red eyes', 'pink hair'
    subject = None if (len(plain) == 1 and plain[0] in lookwords and words[-1] in
                       ("style", "styles", "aesthetic", "aesthetics", "art", "look", "looks")) else _subject_tag(a)
    if subject:                                # ('cel style' is no cel sheet)
        t, fl = subject
        if len(plain) == 1 and fl & {"body", "clothing"} and "a" in _pos(plain[0]) and not cls(plain[0], "colour"):
            return None                        # a body type: 'curvy anatomy', 'petite figures', 'nude figures'
        if "body" in fl and mods:
            # a body's size or shape describes it ('large breasts', 'wide hips',
            # 'perky breasts'); a colour or a kind names what it has
            if any(cls(m, "size") or (_pos(m) and "a" in _pos(m) and "n" not in _pos(m) and not cls(m, "colour"))
                   for m in mods):
                return None
        return "subject tag"
    if len(plain) == 1:
        w = plain[0]
        if w in looktags or w in lookwords:
            return None                        # 'anime style', 'painterly'
        if cls(w, "relational"):
            return "describes nothing"         # 'genital depiction'
        if cls(w, "body") and "a" not in _pos(w):
            return "bare body part"
        if cls(w, "clothing") and "a" not in _pos(w):
            return "clothing"
        if _pos(w) and "n" in _pos(w) and "a" not in _pos(w) and not participle(w):
            if _lx and _lx.sexual(w, "most"):
                return None                    # 'nudity', 'eroticism'
            return "bare noun"
        return None
    if cls(head, "clothing") and "a" not in _pos(head) and not cls(head, "body"):
        return "clothing"
    if mods and all(is_genre(w) for w in mods):
        return "genre"
    if not any(describing(m) for m in mods):
        return "bare body part" if cls(head, "body") else "describes nothing"
    return None


_SHARES = {"data": None, "base": None}


def tag_level(tag):
    """-> the level a look tag needs, from the booru's rating shares
    (safety_floor.json): the highest level whose posts (that level and above)
    are at least half of the tag's AND at least twice the share among all
    measured tags -- 'bdsm' 88% nsfw or explicit is nsfw, while 'backlighting'
    (17% nsfw or explicit, like the booru at large) stays safe; without shares,
    the measured floor"""
    from promptstudio.engine import slots as sm
    if _SHARES["data"] is None:
        try:
            with open(_paths.data("safety_floor.json"), encoding="utf-8-sig") as f:
                fl = json.load(f).get("floors") or {}
        except Exception:
            fl = {}
        _SHARES["data"] = fl
        tot, n = {}, 0
        for e in fl.values():
            sh = (e or {}).get("share") or {}
            if sh:
                n += 1
                for k, v in sh.items():
                    tot[k] = tot.get(k, 0.0) + float(v)
        _SHARES["base"] = {k: v / n for k, v in tot.items()} if n else {}
    e = _SHARES["data"].get(tag) or {}
    sh, base = e.get("share") or {}, _SHARES["base"] or {}
    if not sh:
        return sm.safety_floor(tag) if tag in _tag_posts() else "safe"
    order = ("sensitive", "nsfw", "explicit")
    best = "safe"
    for i, lv in enumerate(order):
        cum = sum(float(sh.get(x, 0)) for x in order[i:])
        bcum = sum(float(base.get(x, 0)) for x in order[i:]) or 1e-9
        if cum >= 0.5 and cum / bcum >= 2:
            best = lv
    return best


def _aliases():
    try:
        from promptstudio.engine.aliases import booru_aliases
        return booru_aliases()
    except Exception:
        return {}
_LOOK_FLAGS = {"style", "medium", "lighting", "effect", "view", "camera", "format", "theme"}
_FIELD_NOUN = {"mood": "mood", "palette": "palette", "lighting": "lighting"}


_LEVELS = ("safe", "sensitive", "nsfw", "explicit")
_LEVEL_CACHE = {}


def look_level(look, field=None, lean=True):
    """-> 'safe' | 'sensitive' | 'nsfw' | 'explicit': how much a look SAYS
    (the author, 2026-09-17: "why are some looks nsfw, like '3d rendering'?"
    -- the old mark measured the artists a look describes, and 3D artists lean
    adult). A whole look that is a booru tag (itself, its alias, its lemma)
    takes the tag's measured floor. A word counts only when the dictionary
    gives it a sexual sense: then its booru tag's floor decides ('semen'
    explicit, 'nude' sensitive, 'breasts' safe), and a word that is no booru
    tag is nsfw when at least half of its senses are sexual ('erotic',
    'genitalia', 'sensual' -- not 'romantic' or 'chaste', which have one such
    sense among several; and 'facial' of 'facial expressions' has none, so the
    booru's 'facial' does not reach it). The new sweep's erotic_style field is
    at least nsfw. The highest wins.
    THE ARTISTS DECIDE HOW FAR (lean: the share of the look's artists the
    booru rates nsfw or explicit is at least twice the share among all
    described artists): a sexual word that is no booru tag, or a rating name,
    makes a look sensitive, and nsfw only when its artists lean adult -- the
    dictionary alone marked 'whimsical', the lean alone '3d rendering' and
    'warm tones'. A booru tag keeps its floor either way."""
    key = (str(look).lower(), field, bool(lean))
    if key in _LEVEL_CACHE:
        return _LEVEL_CACHE[key]
    from promptstudio.engine import slots as sm
    try:
        from promptstudio.engine import lexicon as _lx
        from promptstudio.engine.aliases import booru_aliases
        al = booru_aliases()
    except Exception:
        _lx, al = None, {}
    posts = _tag_posts()
    order = {lv: i for i, lv in enumerate(_LEVELS)}

    def tag_floor(t):
        cands = {t}
        if t in al:
            cands.add(al[t])
        if _lx:
            for p in ("n", "a"):
                cands |= set(_lx.lemmas(t, p))
            cands |= {al[c] for c in list(cands) if c in al}
        floors = [tag_level(c) for c in cands if c in posts]
        return max(floors, key=lambda f: order.get(f, 0)) if floors else None
    # THE FIELD IS WHERE THE MODEL WROTE IT, NOT WHAT THE LOOK SAYS (the author,
    # 2026-09-18: "why is 'detailed rendering' nsfw?" -- the sweep writes how
    # the sexual content is drawn under erotic_style, technique words and all,
    # and the field alone used to make them nsfw). It now only carries a look
    # that already says something sexual the rest of the way.
    best = "safe"
    low = _norm(look)
    whole = tag_floor(low)
    if whole and order.get(whole, 0) > order[best]:
        best = whole
    for w in re.findall(r"[a-z0-9][a-z0-9-]*", low.replace("-", " ")):
        # the boorus' own rating names state a level ('explicit content'; the
        # dictionary reads 'explicit' as 'precisely expressed'), when the
        # artists bear it out
        if w in sm.SPICE_ORDER and w not in ("safe", "auto"):
            fl = w if lean else "sensitive"
        elif _lx is None or not _lx.sexual(w, "any"):
            continue                    # a word with no sexual sense says nothing ('facial expressions')
        else:
            # a sexual word that is a booru tag takes the tag's floor ('nude'
            # sensitive, 'semen' explicit, 'breasts' safe); one that is not is
            # sensitive, and nsfw when the look's artists lean adult
            fl = tag_floor(w)
            if fl is None and _lx.sexual(w, "most"):
                fl = "nsfw" if lean else "sensitive"
        if fl and order.get(fl, 0) > order[best]:
            best = fl
    if field == "erotic_style" and best != "safe" and order[best] < order["nsfw"]:
        best = "nsfw"
    _LEVEL_CACHE[key] = best
    return best


def vocabulary():
    """-> {look: {"artists": n, "field": f or None, "level": lv, "nsfw": bool}}: the single
    looks (atoms) the descriptions use for at least MIN_ARTISTS artists, each
    written in its most common form. Left out: a look stating age or naming
    lettering; the old sweep's ratings and absences ('chaste', 'no nudity');
    a head that counts or relates ('wide range', 'heavy use'); a look made of
    wrapper nouns alone ('women subjects'); a lone noun that is no look tag
    and says nothing sexual ('works', 'detail'); and a booru tag about the
    subject ('white hair', 'school uniforms' -- the brief's). 'level' is what
    the look says and its artists bear out (look_level); 'nsfw' is a level of
    nsfw or explicit."""
    recs = _records()
    if _VOCAB["key"] == _SRC["key"] and _VOCAB["data"] is not None:
        return _VOCAB["data"]
    # THE GLOSSARY OWNS THE TERMS once the terminology pass has run
    # (tools/vision/unify_looks.py): its canonical terms, their artists and
    # levels; the rules below are the reader for descriptions it has not seen
    try:
        with open(_paths.data("look_glossary.json"), encoding="utf-8") as f:
            terms = json.load(f).get("terms") or {}
    except Exception:
        terms = {}
    if terms:
        out = {t: {"artists": e["artists"], "field": e.get("field"), "level": e.get("level") or "safe",
                   "nsfw": e.get("level") in ("nsfw", "explicit"), "tag": e.get("tag")}
               for t, e in terms.items() if e.get("artists", 0) >= MIN_ARTISTS}
        bad = set(age_words(list(out)))
        out = {k: v for k, v in out.items() if not any(re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(b), k)
                                                        for b in bad)}
        _VOCAB.update(key=_SRC["key"], data=out)
        return out
    who, fields, forms = {}, {}, {}
    for a, items in recs.items():
        for p, fld, _g in items:
            for at in atoms(p):
                if len(at) > 48:
                    continue
                # a bare quality in a field that names its noun is that noun's
                # ('dramatic' under mood is 'dramatic mood', not the noun
                # 'dramatic' takes most elsewhere)
                if fld in _FIELD_NOUN and " " not in at and _quality(at):
                    at = at + " " + _FIELD_NOUN[fld]
                k = atom_key(at)
                who.setdefault(k, set()).add(a.lower())
                forms.setdefault(k, {}).setdefault(at, set()).add(a.lower())
                if fld:
                    fields.setdefault(k, {}).setdefault(fld, 0)
                    fields[k][fld] += 1
    nsfw = _nsfw_artists()
    base = len(nsfw & {a.lower() for a in recs}) / float(len(recs) or 1)
    posts, al = _tag_posts(), _aliases()
    try:
        from promptstudio.engine import bridge as _lb
        from promptstudio.engine import lexicon as _lx
    except Exception:
        _lb = _lx = None
    out = {}
    for k, arts in who.items():
        if len(arts) < MIN_ARTISTS or "#rating" in k:
            continue
        name = max(forms[k], key=lambda f: (len(forms[k][f]), -len(f)))
        if LETTERING.search(name) or AGE_WORDS.search(name):
            continue
        words = name.split()
        if words[-1] in _QUANTITY or any(w in _QUANTITY for w in k):
            continue
        plain = [w for w in k if not w.startswith("#")]
        # a look made of wrapper nouns alone names nothing ('women subjects',
        # 'detail', 'images')
        if all(w in {_base(x) for x in _WRAPPERS} | _WRAPPERS for w in plain) and \
                not any((t in posts or t in al) and (set(_lb._gloss_flags(al.get(t, t)) or ()) & _LOOK_FLAGS)
                        for t in (name,) if _lb is not None):
            continue
        # a quality of degree alone names nothing ('heavy emphasis' is 'heavy')
        if len(plain) == 1 and plain[0] in _DEGREE:
            continue
        # a lone noun is a look only as a look tag ('stippling', 'sparkles') or a
        # sexual word ('nudity'); 'works', 'clothing', 'shading' alone say
        # nothing -- judged by the word as written ('clothing', not 'clothe')
        surf = [w for w in words if w not in _WRAPPERS] or words
        if len(plain) == 1 and len(surf) == 1 and "n" in _pos(surf[0]) and "a" not in _pos(surf[0]):
            t = name if name in posts else al.get(name, al.get(surf[0], surf[0]))
            is_look_tag = _lb is not None and t in posts and bool(set(_lb._gloss_flags(t) or ()) & _LOOK_FLAGS)
            if not is_look_tag and not (_lx and _lx.sexual(surf[0], "most")):
                continue
        # A CHARACTER'S TAG IS THE BRIEF'S (the author, 2026-09-17: 'white
        # hair'): a look that is a booru tag about the subject -- hair,
        # clothing, body, expression, an object -- belongs in the brief, which
        # suggests it; the style box keeps the tags of style, medium, lighting,
        # effect and view
        if _content_tag(name):
            continue
        fld = max(fields[k], key=fields[k].get) if k in fields else None
        prev = out.get(name)
        if prev and prev["artists"] >= len(arts):
            continue
        share = len(arts & nsfw) / float(len(arts))
        lv = look_level(name, fld, lean=base > 0 and share >= 2 * base)
        out[name] = {"artists": len(arts), "field": fld, "level": lv, "nsfw": lv in ("nsfw", "explicit")}
    bad = set(age_words(list(out)))
    if bad:
        out = {k: v for k, v in out.items() if not any(re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(b), k)
                                                        for b in bad)}
    _VOCAB.update(key=_SRC["key"], data=out)
    return out


def description_lift(a, b, min_expected=5.0):
    """-> P(both) / P(a)P(b) over the described artists, a and b each read
    as all of its words in an artist's description ('visible brushstrokes'
    against '3d'); None when the expected count of artists with both is
    under min_expected (unmeasured is neutral)"""
    bags = _bags()
    n = float(len(bags))
    if not n:
        return None
    wa = set(re.findall(r"[a-z0-9]+", str(a).lower())) - {"and", "of", "with"}
    wb = set(re.findall(r"[a-z0-9]+", str(b).lower())) - {"and", "of", "with"}
    if not wa or not wb:
        return None
    na = sum(1 for bag in bags.values() if wa <= bag)
    nb = sum(1 for bag in bags.values() if wb <= bag)
    if na * nb / n < min_expected:
        return None
    nab = sum(1 for bag in bags.values() if wa <= bag and wb <= bag)
    return (nab / n) / ((na / n) * (nb / n))


# ---------------------------------------------------------------- the style hint
_TAGS = {"data": None}


def _tag_posts():
    """-> {tag: posts}, the larger of danbooru's and gelbooru's count"""
    if _TAGS["data"] is None:
        d = {}
        for name in ("danbooru_tags.json", "gelbooru_tags.json"):
            try:
                with open(_paths.data(name), encoding="utf-8-sig") as f:
                    for t, n in json.load(f).items():
                        t = str(t).lower().replace("_", " ")
                        d[t] = max(d.get(t, 0), int(n or 0))
            except Exception:
                pass
        _TAGS["data"] = d
    return _TAGS["data"]


def _studio_knows(seg):
    """an entry the studio already reads: a booru tag or alias, a style of the
    pool, a medium spelling, a checkpoint concept"""
    if seg in _tag_posts():
        return True
    try:
        from promptstudio.engine.aliases import booru_aliases
        if seg in booru_aliases():
            return True
    except Exception:
        pass
    try:
        from promptstudio.engine import bridge as lb
        if seg in (lb._style_pool2() or {}):
            return True
        if seg in ((lb._medium_pool() or {}).get("typed_surfaces") or {}):
            return True
    except Exception:
        pass
    try:
        from promptstudio.library import external as _ext
        if _ext.get(seg):
            return True
    except Exception:
        pass
    return False


def whole_tag(phrase, max_combos=400):
    """-> the booru tag that means the whole phrase, or None: each word or
    one of the dictionary's words for it ('light' -> lighting), all of them
    together a booru tag ('rim light' -> rim lighting); the most used tag
    wins. One word of a phrase is never enough."""
    words = _norm(phrase).split()
    if not 1 <= len(words) <= 4:
        return None
    try:
        from promptstudio.engine import lexicon as _lx
        options = [[w] + sorted(x for x in _lx.style_words(w) if " " not in x) for w in words]
    except Exception:
        options = [[w] for w in words]
    posts = _tag_posts()
    best, combos = None, [[]]
    for opt in options:
        combos = [c + [o] for c in combos for o in opt][:max_combos]
    for c in combos:
        cand = " ".join(c)
        if cand in posts and (best is None or posts[cand] > posts[best]):
            best = cand
    return best


def read_style_hint(hint):
    """-> {"known": [...], "known_tags": [...], "tags": [...], "phrases": [...]}
    for the style box: what the studio reads as before (and which of those are
    booru tags, measured against a rolled style), the tags that mean a whole
    entry, and the entries kept as the user's phrases"""
    out = {"known": [], "known_tags": [], "tags": [], "phrases": [], "from": {}}
    for raw in re.split(r"[,;\n]+", str(hint or "")):
        seg = raw.strip()
        low = _norm(seg)
        if not low:
            continue
        if low.startswith("@") or "(" in seg or _studio_knows(low) or _studio_knows(seg.lower()):
            out["known"].append(seg)
            if low in _tag_posts():
                out["known_tags"].append(low)
            continue
        tag = whole_tag(low)
        if tag:
            out["tags"].append(tag)
            out["from"][tag] = low
        else:
            out["phrases"].append(low)
    return out
