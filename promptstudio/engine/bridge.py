from promptstudio import paths as _paths
from promptstudio.llm import config as _llm_config
#!/usr/bin/env python
"""
llm_bridge.py -- ARCHITECTURE v2: the two bridges between the engine and the
local LLM. See SECTION_SPEC.md "ARCHITECTURE v2".

V1 STATUS -- the pipeline works end to end (cast -> bridge 1 -> bridge 2 ->
verify -> assemble, ~10-15 s/prompt on the 5060) and the free-generation case
already produces coherent invented scenes. KNOWN DEFECTS, in priority order:

  1. FIXED 2026-08-29: pick discipline. Three layers -- confident_pick()
     lets the ENGINE map exact/dominant retrievals ('exposed collarbone' ->
     collarbone) so the model never sees them; ambiguous concepts reach the
     model with a top-5 RANKED list and a prefer-first instruction; stopwords
     no longer score ('crying with eyes open' had reached a serene onsen on
     the strength of the word "with").
  2. FIXED 2026-08-29: UP-injection. When the measured level of the final
     tag set sits below the target, the engine draws up to 2 injectable
     tags at the target level via pe._injectable (census-checked against
     the actual cast); reported as
     injected_for_level. Verified live: tame tea prompt + explicit target
     -> 'cum in mouth' injected, measured level reaches explicit.
  3. FIXED 2026-08-29: style-scoped decomposition retrieval. Ambiguous
     concepts whose origin band is "style" retrieve against _style_vocab()
     only (style_pool keys + palettes + techniques + cultural pool +
     colors group + composition techniques + '*(style)' tags), never the
     whole booru -- 'simplified forms' can no longer reach 'simplified
     chinese text'. Empty after scoping = NL-only, the correct fate for a
     style phrase without a tag.
  4. Illustrious phrase dedup is exact-match only, so near-duplicates ride
     along (crop top / white crop top).
  5. The concept library is not consulted yet -- every named style is
     re-derived per generation until concept_library.json exists.
  6. FIXED 2026-08-29: alias blindness. alias_map.json (40,893 active
     danbooru aliases via build_alias_map.py) backs three retrieval layers:
     exact redirect ('hot spring' -> onsen, engine-mapped), alias SURFACES
     in fuzzy scoring (the alias text matches, the canonical tag is emitted:
     'bellybutton piercing' -> navel piercing), best-surface-per-tag dedup,
     and head-anchor credit carried by the surface (onsen contains no
     'spring', but its surface did).

    initial prompt -> [engine: cast, spice, checkboxes] -> BRIDGE 1 (LLM
    builds the scene plan + NL part, section rules as its construction
    prompt) -> BRIDGE 2 (LLM translates its OWN concepts into booru tags,
    constrained to engine-retrieved candidates) -> [engine: VERIFY, assemble]

Division of labour, per the author's decisions:
  * CODE decides: cast (census; cast_weights roll when the prompt is silent),
    spice target (resolve_spice + warn-and-switch), section-1 conditioning,
    candidate retrieval, and ALL verification.
  * LLM decides: every concept (styles/locations/outfits are its free choice
    when the checkbox says generate), the decomposition of named concepts,
    the NL wording, and which candidate tag fits each concept.
  * The LLM is never trusted on tag strings: bridge-2 output is filtered
    against the retrieval shortlists -- the very first live test emitted four
    tags that were on no list. The filter is the architecture, not hygiene.

    python llm_bridge.py "1girl relaxing at an onsen" --mode anima
    python llm_bridge.py "incase style pinup in a loft" --mode illustrious \
                         --spice nsfw --gen-style --gen-location
"""

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from promptstudio.engine import enhancer as pe   # noqa: E402
from promptstudio.engine import slots as sm
from promptstudio.engine import affinity as _affinity
from promptstudio.engine import fastplan as _fastplan        # noqa: E402
from promptstudio.engine import lexicon as _lex              # noqa: E402
from promptstudio.engine.vocab import (  # noqa: E402
    _FAMILY_LABEL,
    _NOT_DESCRIPTIVE,
    _NOT_HEAD_HAIR,
    _gloss_of,
    _glosses,
    _hair_vocab,
    _light_vocab,
    _log_pending,
    _slot_scope,
    _style_vocab,
    _vocab,
    candidates_for,
    confident_pick)


# no API constant any more: the endpoint is whatever our own llama-server
# child reports. There is no second server to point at.
MODEL = _llm_config.load().get("model_id") or "qwen3-8b-heretic"
# Qwen3 needs ' /no_think' to skip its reasoning block and the reply split
# on '</think>'. Both are meaningless for other families, so they follow
# the configured model rather than being baked in.
THINKING = bool(_llm_config.load().get("thinking", True))


# ---------------------------------------------------------------- LLM client
# THE PLAN'S SHAPE AS A SCHEMA (2026-09-11): llama.cpp
# compiles it to a grammar, so the writer cannot return malformed JSON --
# the mechanical fallback was 20-40 percent of seeds on free-form JSON
_S = {"type": "string"}
_SN = {"type": ["string", "null"]}
_SL = {"type": "array", "items": _S}
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "count_sentence": _S,
        "subjects": {"type": "array", "items": {
            "type": "object",
            "properties": {"who": _S, "hair": _S, "eyes": _S, "outfit": _SL,
                           "body": {"type": "object", "additionalProperties": _SL},
                           "pose": _S, "self_actions": _SL, "held_object": _SN, "age": _S,
                           "alias": _S, "race": _S, "occupation": _S, "fashion_style": _S,
                           "descriptors": _SL},
            "required": ["who", "hair", "eyes", "outfit", "body", "pose", "self_actions", "held_object",
                         "age", "alias", "race", "occupation", "fashion_style", "descriptors"],
            "additionalProperties": False}},
        "secondaries_look": _SL, "collective": _S, "background": _S,
        "lighting": _SL, "effects": _SL, "palette": _SL,
        "interactions": {"type": "array", "items": {
            "type": "object",
            "properties": {"participants": {"type": "array", "items": {"type": "integer"}},
                           "act": _S, "direction": _S, "positions": _S, "secondary_contact": _SN},
            "required": ["participants", "act", "direction", "positions", "secondary_contact"],
            "additionalProperties": False}},
        "action": _S, "setting": _S,
        "style": {"type": "object", "properties": {"name": _SN, "decomposition": _SL},
                  "required": ["name", "decomposition"], "additionalProperties": False},
        "mood": _S, "nl": _S},
    "required": ["count_sentence", "subjects", "secondaries_look", "collective", "background",
                 "lighting", "effects", "palette", "interactions", "action", "setting", "style",
                 "mood", "nl"],
    "additionalProperties": False}
_MAP_SCHEMA = {"type": "object", "additionalProperties": _SL}
# one receipt per attempt (stage, seconds, tokens, finish reason, status);
# generate() clears the list and returns it as `llm_calls`
_CALLS = []
CANCEL = threading.Event()      # set by the studio's cancel button; checked before every model call


class Cancelled(RuntimeError):
    pass


def chat(system, user, temp=0.7, max_tokens=700, retries=1, schema=None, stage=None):
    if CANCEL.is_set():
        raise Cancelled("generation cancelled")
    STATUS["detail"] = "model call: %s" % (stage or ("json" if schema else "text"))
    STATUS["calls"] = STATUS.get("calls", 0) + 1
    """retries=1 and a LONG socket window: a timed-out client does NOT
    cancel the server's generation, so fast-fail retries stack zombie
    generations behind each other and the queue death-spirals -- one
    patient attempt beats three impatient ones (measured the hard way
    under VRAM contention with the image app).

    The endpoint comes from llm_server (the generator's OWN llama-server
    child, idle auto-unload). There is no fallback:
    if that engine cannot start, the error says so rather than quietly
    routing the work to a third-party server that may not be running the
    configured model."""
    _req = {
        "model": MODEL,
        "messages": [{"role": "system",
                      "content": system + (" /no_think" if THINKING else "")},
                     {"role": "user", "content": user}],
        "temperature": temp, "max_tokens": max_tokens,
        # REASONING OFF, THE WAY THE MODEL ACTUALLY LISTENS. ' /no_think'
        # (below) is the Qwen3 switch and does nothing on Qwen3.5, which
        # would otherwise spend the whole budget in reasoning_content and
        # return an empty answer. This template switch works on 3.5 and is
        # inert elsewhere, so both families get one request shape.
        "chat_template_kwargs": {"enable_thinking": False},
        # sampler pinned EXPLICITLY -- host defaults must never decide
        # quality
        "top_p": 0.95, "top_k": 40, "min_p": 0.05}
    if schema is not None:
        _req["response_format"] = {"type": "json_object", "schema": schema}
    body = json.dumps(_req).encode()
    for attempt in range(retries + 1):
        _t0 = time.time()
        _rc = {"stage": stage or ("json" if schema else "text"), "seconds": None,
               "prompt_tokens": None, "completion_tokens": None, "finish_reason": None,
               "status": "failed", "attempt": attempt}
        _CALLS.append(_rc)
        try:
            from promptstudio.llm import server as llm_server
            api = llm_server.ensure_up()
            req = urllib.request.Request(
                api, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            ch = (d.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            out = (msg.get("content") or "")
            _u = d.get("usage") or {}
            _rc.update(prompt_tokens=_u.get("prompt_tokens"), completion_tokens=_u.get("completion_tokens"),
                       finish_reason=ch.get("finish_reason"))
            # A REASONING MODEL CAN SPEND THE WHOLE BUDGET THINKING. Qwen3.5
            # returns its chain in a separate `reasoning_content` field and
            # leaves `content` EMPTY when it runs out of room -- so a budget
            # sized for a non-reasoning model reads as a blank answer rather
            # than as truncation, which is exactly how the first VL
            # benchmark "proved" the 4B produced nothing. ('/no_think'
            # suppresses this on Qwen3 but NOT on Qwen3.5.) One retry with
            # room to finish, and only when the server says it was cut off.
            # ...and a CUT answer of any kind gets one retry with room
            # (2026-09-11): a schema-constrained plan truncated mid-object
            # was a fallback before
            if ch.get("finish_reason") == "length" and attempt < retries + 1 and \
                    (not out.strip() and msg.get("reasoning_content") or schema is not None):
                _rc["truncated"] = True
                bigger = json.loads(body.decode())
                bigger["max_tokens"] = min(4096, int(max_tokens) * 3)
                req2 = urllib.request.Request(
                    api, data=json.dumps(bigger).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req2, timeout=600) as r2:
                    d2 = json.load(r2)
                _ch2 = (d2.get("choices") or [{}])[0]
                out = (_ch2.get("message", {}).get("content") or "")
                _u2 = d2.get("usage") or {}
                _rc.update(finish_reason=_ch2.get("finish_reason"), retried_budget=bigger["max_tokens"],
                           completion_tokens=(_rc.get("completion_tokens") or 0) + (_u2.get("completion_tokens") or 0))
            _rc.update(status="ok", seconds=round(time.time() - _t0, 2))
            return (out.split("</think>")[-1] if THINKING else out).strip()
        except Exception as _e:
            _rc.update(status="error", error=type(_e).__name__, seconds=round(time.time() - _t0, 2))
            if attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))


def parse_json_block(text):
    """The model wraps JSON in prose or fences often enough to always fish.

    RAW_DECODE, NOT A GREEDY MATCH. The old pattern spanned from the
    FIRST brace to the LAST one, so a model that emitted its object
    and then carried on talking -- or emitted two objects -- produced
    a capture that was not valid JSON, and the whole generation died
    with `JSONDecodeError: Extra data`. An unhandled crash is the
    worst possible response to a model being chatty, which is a thing
    models do.

    Decoding ONE value from a brace and ignoring whatever follows is
    what "fish the JSON out" was always meant to mean. Each brace is
    tried in turn, so prose before, prose after, fences, and a
    preamble object all resolve to the first thing that parses.
    """
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object in LLM output")


# ------------------------------------------------------------- engine facts
_CAST_CLASS = {
    "1girl solo": {"female": 1}, "2girls": {"female": 2},
    "1girl + viewer": {"female": 1, "viewer": 1},
    "1girl + 1boy": {"female": 1, "male": 1},
    "1boy solo": {"male": 1}, "2boys": {"male": 2},
    "3girls": {"female": 3}, "others/mixed": {"other": 1},
    "1boy + offscreen": {"male": 1, "viewer": 1},
    "2girls + 1boy": {"female": 2, "male": 1},
    "6+girls": {"female": 6}, "4girls": {"female": 4},
    "1girl + 2boys": {"female": 1, "male": 2}, "5girls": {"female": 5},
    "3girls + 1boy": {"female": 3, "male": 1},
    "1girl + 3boys": {"female": 1, "male": 3},
    "2girls + 2boys": {"female": 2, "male": 2},
    "futa": {"futa": 1, "female": 1},
    "large mixed group": {"female": 4, "male": 2},
}


# ONE VOCABULARY FOR GENDER EVIDENCE. resolve_cast had five separate
# regexes for "this word means a woman is here" and they had drifted:
# `milf` was in the phantom-girl guard but not in the gate that decides
# whether the census counts at all, so "blonde milf ... blowjob to the
# viewer" parsed to 1girl + mature female and then resolved to NO
# HUMANS. Every gender test below reads these two lists and nothing
# else, so a word added here is added everywhere at once.
# THE GENDER VOCABULARY LIVES IN THE PARSER (enhancer.FEMALE_NOUN_LIST
# and friends) and is imported here, so the cast reader and the parser
# can never disagree about which words mean a woman is present. They
# did: the parser gendered occupations (knight = man) and this file
# did not, and "a female knight" came out as a woman AND a man.
_FEMALE_NOUN_LIST = pe.FEMALE_NOUN_LIST
_MALE_NOUN_LIST = pe.MALE_NOUN_LIST
_FEMALE_WORDS = pe.FEMALE_WORDS
_MALE_WORDS = pe.MALE_WORDS

_GENDERLESS = re.compile(
    r"(\d+|one|two|three|four|five|a couple of|a group of)\s+"
    r"(people|persons?|figures?|subjects?|characters?)", re.I)
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "a couple of": 2, "a group of": 4}

# SOME PERSON NOUNS CARRY THEIR OWN COUNT. _GENDERLESS needs a number in
# front of it ("2 people"), so a word that is inherently plural fell
# through and the prompt came out peopleless: "lovers embracing in a
# candlelit bedroom" resolved to an empty bedroom, and the model dutifully
# wrote "there is no living subject".
#
# The vocabulary is DATA-DRIVEN, not a hand list: occupation_pool's
# `relationships` bucket is by construction "how subjects relate" -- every
# entry (siblings, husband and wife, cousins, brother and sister) denotes
# two or more people. A handful of ordinary English pair words that the
# booru vocabulary simply does not carry are added on top, the same way
# _WORDNUM spells out number words the tag system never sees.
_PAIR_EXTRA = {"lovers": 2, "couple": 2, "twins": 2, "partners": 2,
               "spouses": 2, "newlyweds": 2, "friends": 2,
               # a CROWD is scenery (the 'crowd' tag), never four subjects
               "group": 4, "family": 3,
               # group acts carry a TOTAL, never a gender split: the
               # parser used to hardcode `2boys` for these, inventing men
               # the user never wrote; the split is rolled from measured
               # weights like every other genderless count
               # a noun that IMPLIES a mix says so: (total, min female,
               # min male). "gangbang" rolled as four women and paired as
               # yuri, which is not what the word means.
               "threesome": (3, 0, 0), "gangbang": (4, 1, 2),
               "orgy": (4, 1, 1)}
_PAIR_NOUNS = None


def _pair_nouns():
    """-> {phrase: how many people it implies}"""
    global _PAIR_NOUNS
    if _PAIR_NOUNS is None:
        out = dict(_PAIR_EXTRA)
        try:
            with open(_paths.data("occupation_pool.json"),
                      encoding="utf-8-sig") as f:
                rel = json.load(f).get("relationships") or {}
            for t in rel:
                t = str(t).lower().strip()
                if t and t not in out:
                    # triplets/quintuplets/sextuplets say their own number
                    out[t] = (3 if "triplet" in t else
                              5 if "quintuplet" in t else
                              6 if "sextuplet" in t else 2)
        except Exception:
            pass
        _PAIR_NOUNS = out
    return _PAIR_NOUNS


def _pair_count_in(base):
    """-> the count a plural person noun implies, or 0.

    Longest phrase wins, so 'brother and sister' is not read as 'sister'.
    """
    return _pair_shape_in(base)[0]


def _pair_shape_in(base):
    """-> (total, min_female, min_male) the plural person noun implies."""
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", (base or "").lower()) + " "
    best, shape = "", (0, 0, 0)
    for phrase, cnt in _pair_nouns().items():
        if len(phrase) > len(best) and (" " + phrase + " ") in low:
            best = phrase
            shape = tuple(cnt) if isinstance(cnt, (tuple, list))                 else (int(cnt), 0, 0)
    return shape


_COUNT_TAG = {"female": "1girl", "male": "1boy", "futa": "1futa", "other": "1other"}


def _cast_weights(mode, level=None):
    """the measured cast classes, each kept only where its count tags are
    (2026-09-18: '1futa' measured explicit on 96% of its posts, and the roll
    put a futa into an nsfw couple -- the cast was the one die that read no
    floor). A typed cast is unaffected; without a level nothing is gated."""
    with open(_paths.data("cast_weights.json"), encoding="utf-8-sig") as f:
        cw = json.load(f)[mode if mode in ("anima", "illustrious")
                          else "anima"]
    w = dict(cw["weights"])
    if level:
        rank = sm.SPICE_ORDER.get(sm.normalize_level(level), 0)
        w = {n: x for n, x in w.items()
             if all(sm.SPICE_ORDER.get(sm.safety_floor(_COUNT_TAG[k]), 0) <= rank
                    for k in _CAST_CLASS.get(n, {}) if k in _COUNT_TAG)}
    return w


def _roll_cast_shape(total, min_f, min_m, mode, rng, level=None):
    """a measured cast class of this total honouring the minimums, or None"""
    cw = {"weights": _cast_weights(mode, level)}
    fits = []
    for n, w in cw["weights"].items():
        c = _CAST_CLASS.get(n, {})
        tot = sum(v for k, v in c.items()
                  if k in ("female", "male", "futa", "other"))
        if tot == total and c.get("female", 0) + c.get("futa", 0) >= min_f                 and c.get("male", 0) >= min_m:
            fits.append((n, w))
    if not fits:
        return None
    roll = rng.choices([n for n, _ in fits], weights=[w for _, w in fits])[0]
    return dict(_CAST_CLASS[roll]), roll


_PERSON_NOUNS = None


def _person_noun_in(base):
    """-> the person-flagged occupation noun found in the text, or None.
    Data-driven from the concept library's `person` flag (occupation
    tags: nurse, office lady, secretary, idol...). Whole-phrase match,
    longest wins. Excludes tags that are also clothing (a 'maid' is
    usually the outfit, not a declared subject)."""
    global _PERSON_NOUNS
    if _PERSON_NOUNS is None:
        _PERSON_NOUNS = set()
        try:
            gl = _glosses()
            for k, v in gl.items():
                fl = v.get("f") or []
                if "person" in fl and "clothing" not in fl \
                        and 3 <= len(k) <= 24 and "(" not in k:
                    _PERSON_NOUNS.add(k)
        except Exception:
            pass
    low = " " + re.sub(r"[^a-z0-9 ]", " ", (base or "").lower()) + " "
    hits = [n for n in _PERSON_NOUNS if " " + n + " " in low]
    return max(hits, key=len) if hits else None


_SUBJECT_NOUNS = None
_CONJ_RE = re.compile(r"\b(and|plus|alongside|versus|vs|beside|next to)\b")
_DET_TAIL_RE = re.compile(
    r"\b(a|an|the|another|one|two|three|his|her|their|some|\d+)\b[a-z ]*$")


def _subject_nouns(base):
    """-> one entry per DISTINCT being the text names by role or species.

    THE CAST READER COUNTED NOUNS, NOT PEOPLE. 'a blonde elf ranger and a
    red-haired dwarf blacksmith' is two, and it produced `1girl` -- the
    occupation path finds the first person-flagged noun and returns
    exactly one subject however many the prompt describes.

    Two corrections, both from what that sentence actually contains:

    A SPECIES NAMES A BEING. `ranger` is not a booru tag at all, so a
    person-flagged scan cannot see it; what it can see is `elf` and
    `dwarf`, which the gloss library flags `creature`. An elf is a
    subject in the same way a nurse is.

    A SUBJECT IS A NOUN PHRASE, NOT A NOUN. 'dwarf blacksmith' is two
    nouns and one man. What separates two beings is a CONJUNCTION and a
    DETERMINER of their own -- 'an elf AND A dwarf'. Without both, the
    nouns are describing one person, which is why 'a girl, elf, throne
    room' stays a single elf girl and 'a nurse in a maid outfit' stays a
    nurse.
    """
    return [n for n, _c in _subject_spans(base)][:6]


_NUM_BEFORE = re.compile(
    r"(?:^|\s)(\d+|one|two|three|four|five|six)(?:\s+(?:more|other))?\s+$")


_GARMENT_HEADS = frozenset((
    "outfit", "outfits", "dress", "dresses", "uniform", "uniforms", "costume",
    "costumes", "headdress", "apron", "aprons", "cap", "hat", "shoes",
    "bikini", "cosplay", "clothes", "clothing", "attire", "suit", "gloves",
    "stockings", "collar", "cafe"))
_WEARABLE_PERSON = None


def _wearable_person_nouns():
    """person nouns the gloss library ALSO flags as clothing (maid, nurse...)"""
    global _WEARABLE_PERSON
    if _WEARABLE_PERSON is None:
        _WEARABLE_PERSON = set()
        try:
            for k, v in _glosses().items():
                fl = v.get("f") or []
                if "person" in fl and "clothing" in fl                         and 3 <= len(k) <= 24 and "(" not in k:
                    _WEARABLE_PERSON.add(k)
        except Exception:
            pass
    return _WEARABLE_PERSON


_CAMERA_PHRASE_RE = re.compile(
    r"\b(from (?:behind|above|below|the side|the front|the back)(?: (?:the|a|an|her|his|their) [a-z]+)?|"
    r"cowboy shot|full body|upper body|lower body|wide shot|close-?up(?: (?:on|of) (?:her|his|their|the) [a-z]+)?|"
    r"portrait|profile|low angle|high angle|dutch angle|bird's-eye view|worm's-eye view|"
    r"(?:male |female |futanari )?pov|looking at viewer|eye contact|(?:\w+ )?focus)\b", re.I)


_PEOPLE_GLOSS_RE = re.compile(r"\b(group of \w+|crowd|people|persons|bystanders|spectators|audience|onlookers|passers-?by)\b", re.I)


def _names_people(tag):
    """a tag whose gloss is about people (a crowd, an audience) is a
    background of people, not a place to be in"""
    try:
        tb = _json_table(_GLOSSES, "tag_glosses.json") or {}
        e = (tb.get("glosses") or {}).get(str(tag).lower().strip()) or {}
        g = str((e.get("g") if isinstance(e, dict) else "") or _gloss_of(tag)[0] or "")
        return bool(_PEOPLE_GLOSS_RE.search(g))
    except Exception:
        return False


def _sans_camera(base):
    """the text without its camera phrases: 'cowboy shot' is a framing,
    not a cowboy; 'from behind the queen' names no second queen"""
    return re.sub(r"\s+", " ", _CAMERA_PHRASE_RE.sub(" ", str(base or "")))


def _text_person_words(base):
    """person nouns the noun set lacks: -ist practitioners, agent nouns,
    the extra person list -- read from this text"""
    low = " " + re.sub(r"[^a-z0-9 ]", " ", str(base or "").lower()) + " "
    out = set()
    try:
        from promptstudio.engine import lexicon as _lx
        _roles = _lx.people("role")          # the dictionary's roles: millionaire, beekeeper (2026-09-17)
        _roles_det = _lx.people("role_det")  # ...and those that are verbs too: tailor, usher
    except Exception:
        _roles, _roles_det = frozenset(), frozenset()
    for w in set(re.findall(r"[a-z]{4,}", low)) & _roles_det:
        if re.search(r"(?:^|\s)" + pe.ARTICLE_RE + r"\s+(?:[a-z-]+\s+)?" + w + r"\s", low):
            out.add(w)
    for w in re.findall(r"[a-z]{4,}", low):
        if w in _PERSON_EXTRA or w in _AGENT_NOUNS or w in _roles \
                or (w.endswith("s") and (w[:-1] in _AGENT_NOUNS or w[:-1] in _PERSON_EXTRA)):
            out.add(w)
        elif re.fullmatch(r"[a-z]{4,}ist", w) and _is_humanlike(w):
            out.add(w)
    return out


def _subject_spans(base):
    base = _sans_camera(base)
    return _subject_spans_text(base)


def _subject_spans_text(base):
    """-> [(noun, count)] per distinct being, in text order.

    A NUMBER IN FRONT OF A NOUN MULTIPLIES IT. "a woman and her two
    sisters" is three women; the count used to take the largest single
    piece of evidence (two, from 'sisters') instead of adding the beings
    up. Each kept noun now carries the number written before it, and the
    caller sums.
    """
    global _SUBJECT_NOUNS
    _person_noun_in(base)                      # builds _PERSON_NOUNS
    if _SUBJECT_NOUNS is None:
        # gendered nouns are subject nouns: 'daughter' is no booru tag and
        # carries no gloss flag, but it names a person as surely as 'nurse'
        _SUBJECT_NOUNS = (set(_PERSON_NOUNS or ()) | set(_FEMALE_NOUN_LIST)
                          | set(_MALE_NOUN_LIST))
        try:
            for k, v in _glosses().items():
                fl = v.get("f") or []
                if "creature" in fl and "clothing" not in fl                         and 3 <= len(k) <= 24 and "(" not in k:
                    _SUBJECT_NOUNS.add(k)
        except Exception:
            pass
    low = " " + re.sub(r"[^a-z0-9 ]", " ", (base or "").lower()) + " "
    spans = []
    for n in _SUBJECT_NOUNS | _wearable_person_nouns() | _text_person_words(base):
        pat = " " + n + " "
        start = 0
        while True:
            k = low.find(pat, start)
            if k < 0:
                break
            # A MAID IS A PERSON UNLESS SHE IS AN OUTFIT. Person nouns that
            # are also garments were excluded outright, so "a maid serving
            # tea" resolved to no humans. The garment reading is the one
            # with a garment word right after it ("maid outfit", "maid
            # dress"); anything else is somebody.
            if n in _wearable_person_nouns():
                nxt = low[k + 1 + len(n):].split()
                if nxt and nxt[0] in _GARMENT_HEADS:
                    start = k + 1
                    continue
            spans.append((k + 1, k + 1 + len(n), n))
            start = k + 1
    if not spans:
        # NOBODY THE LISTS KNOW, SOMEBODY THE DICTIONARY DOES (2026-10-03: 'a tired
        # office worker collapsed on the couch' came out `no humans` -- 'worker',
        # 'employee', 'clerk', 'farmer', 'cashier' are in no role list and are no
        # booru tags). Only when the text would otherwise hold nobody: a word the
        # lexicon files as a person and nothing else, standing after a determiner
        # ('a tired office worker'), is the being. Never the onlooker.
        try:
            from promptstudio.engine import lexicon as _lxp
            for m in re.finditer(r"(?<= )(?:" + pe.ARTICLE_RE + r") ((?:[a-z-]+ ){1,3})", low):
                _at = m.start(1)
                for w in m.group(1).split():
                    if len(w) >= 4 and w not in _NOT_A_BEING and (_lxp.main_classes(w) or set()) == {"person"}:
                        spans.append((_at, _at + len(w), w))
                        break
                    _at += len(w) + 1
        except Exception:
            pass
    if not spans:
        return []
    # longest match wins where two overlap ('office lady' beats 'lady')
    spans.sort(key=lambda sp: (sp[0], -(sp[1] - sp[0])))
    keep = []
    for sp in spans:
        if keep and sp[0] < keep[-1][1]:
            continue
        keep.append(sp)

    def _count(sp):
        m = _NUM_BEFORE.search(low[:sp[0]])
        if not m:
            return 1
        w = m.group(1)
        return int(w) if w.isdigit() else _WORDNUM.get(w, 1)

    out = [(keep[0][2], _count(keep[0]))]
    for idx in range(1, len(keep)):
        gap = low[keep[idx - 1][1]:keep[idx][0]]
        # A CONJUNCTION BETWEEN TWO PERSON NOUNS IS TWO PEOPLE. The
        # determiner test guarded against 'dwarf blacksmith' (one man,
        # two nouns, NO conjunction) -- but it also refused 'father and
        # son' and 'brother and sister', which nobody writes with a
        # second article. The conjunction is the evidence; the article
        # was never what separated them.
        #
        # SO IS A VERB. "goblin boy fucking elf milf" has no conjunction
        # and no second article, and the reader saw one being; the verb
        # between the two noun phrases is what separates actor from
        # receiver. A participle or a contact verb in the gap counts.
        #
        # AND A RACE WORD JOINS THE PERSON WORD RIGHT AFTER IT (the author's
        # race concept): 'goblin' + 'boy' with nothing between them is
        # one being called 'goblin boy', gendered by the head.
        if not gap.strip() and _GENDER_HEAD_RE.match(keep[idx][2]):
            prev_noun, prev_cnt = out[-1]
            if prev_noun == keep[idx - 1][2]:
                out[-1] = (prev_noun + " " + keep[idx][2], prev_cnt)
                continue
        if _CONJ_RE.search(gap) or _VERB_GAP_RE.search(gap):
            # A DEFINITE MENTION IS A BACK-REFERENCE (2026-09-14): 'the
            # girl', 'this girl', 'that girl' after a girl was introduced
            # names her again ("a sexy girl sleeping ... emphasizing the
            # girl's sexuality" is one girl); a number in front still
            # counts, and a noun not yet introduced is a new being
            _before = low[:keep[idx][0]].split()
            _det = _before[-1] if _before else ""
            _noun = keep[idx][2]
            _sing = _noun[:-1] if _noun.endswith("s") and not _noun.endswith("ss") else _noun
            if _det in _BACKREF_DETS and not _NUM_BEFORE.search(low[:keep[idx][0]]) \
                    and any(o[0] == _noun or o[0] == _sing or o[0].endswith(" " + _sing)
                            for o in out):
                continue
            out.append((keep[idx][2], _count(keep[idx])))
    return out[:6]


_BACKREF_DETS = frozenset(("the", "this", "that", "same", "said"))
# persons by the dictionary who are not in the picture, or are nobody in particular
_NOT_A_BEING = frozenset(("viewer", "viewers", "audience", "spectator", "spectators", "onlooker", "onlookers",
                          "artist", "author", "painter", "illustrator", "photographer", "user", "reader",
                          "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody",
                          "person", "people", "human", "humans", "character", "characters", "self", "other",
                          "others", "owner", "stranger", "creator", "designer"))


_GENDER_HEAD_RE = re.compile(
    r"^(girls?|boys?|wom[ae]n|m[ae]n|lady|ladies|guys?|milfs?|futanari|futas?|"
    r"princess|prince|queen|king|maids?|mother|father|daughter|son|wife|"
    r"husband|sister|sisters|brother|brothers)$")
# a verb between two person nouns: a participle, or one of the contact
# verbs in any form ('fucks', 'kissed', 'holds')
_VERB_GAP_RE = re.compile(
    r"\b(?:\w{3,}ing|fuck\w*|pound\w*|rail\w*|penetrat\w*|kiss\w*|hug\w*|"
    r"hold\w*|carr\w+|lift\w*|grop\w*|spank\w*|lick\w*|suck\w*|rid\w+|"
    r"push\w*|pull\w*|hit\w*|slap\w*|feed\w*|teach\w*|serv\w+|"
    r"watch\w*|chas\w+|fight\w*|meet\w*|greet\w*|help\w*|beside|"
    r"behind|next to|in front of|facing|with|versus|vs)\b")


# A CREATURE IS NOT AUTOMATICALLY A PERSON. 'an elf and a wolf' is one
# person and one animal, and counting both as people gendered the wolf.
# The gloss library already says which is which, in its own definitions:
# elf is "A humanoid fantasy race", orc "A monstrous humanoid creature",
# mermaid "A woman with the bottom half of a fish" -- while wolf is "a
# species of canid" and owl "A nocturnal bird of prey". Reading the
# definition beats keeping a list of fantasy species, and it stays right
# for species added later.
_HUMANLIKE_WORDS = ("humanoid", "human", "woman", "women", "person",
                    "people", "fantasy race", "girl", "boy", "lady")


_AGENT_NOUNS = {"wrestler", "boxer", "fighter", "swimmer", "runner", "singer", "dancer", "painter", "writer",
                "driver", "rider", "surfer", "skater", "gamer", "streamer", "hiker", "jogger", "diver",
                "climber", "cyclist", "golfer", "archer", "hunter", "gardener", "baker", "brewer", "fisher",
                "player", "performer", "juggler", "acrobat", "magician", "sculptor", "photographer"}

_PERSON_EXTRA = {"patient", "customer", "customers", "guest", "guests", "stranger", "passenger", "tourist",
                 "visitor", "spectator", "bystander", "pedestrian", "client", "tattoo artist", "performer",
                 "musician", "audience member", "opponent", "rival", "companion", "partner", "elder"}


def _is_humanlike(t):
    """-> True when this subject noun names somebody rather than something"""
    if t in (_PERSON_NOUNS or ()) or t in _PERSON_EXTRA or t in _AGENT_NOUNS:
        return True
    # THE CURATED PERSON WORDS ARE THE OWNER OF THIS QUESTION (the author,
    # 2026-09-21, on 'a student in a classroom' coming out `no humans`): a role
    # the booru has no TAG for -- student, intern, apprentice, journalist --
    # has no gloss to read, so every gloss-based test said 'not somebody'.
    # slots.HUMAN_SUBJECT is the studio's own list of words that name a person;
    # it answers here too, and the three lists stop disagreeing.
    if sm.HUMAN_SUBJECT.fullmatch(str(t or "").lower().strip()):
        return True
    if t.endswith("s") and (t[:-1] in _PERSON_EXTRA or t[:-1] in _AGENT_NOUNS or t[:-1] in (_PERSON_NOUNS or ())):
        return True
    # a word built on -ist names a practitioner (pianist, florist) unless
    # the booru knows it as a thing (mist, wrist are too short to match)
    if re.fullmatch(r"[a-z]{4,}ist", t):
        try:
            _fl = set(_gloss_of(t)[1] or ())
        except Exception:
            _fl = set()
        if not (_fl & {"object", "clothing", "scenery", "location", "effect", "food"}):
            return True
    # a gendered noun names a person by definition -- 'daughter' and 'son'
    # have no gloss to read, and were being thrown out as things
    if t in _FEMALE_NOUN_LIST or t in _MALE_NOUN_LIST:
        return True
    try:
        from promptstudio.engine import lexicon as _lx
        if t in _lx.people("role") or t in _lx.people("role_det"):
            return True
    except Exception:
        pass
    try:
        g, _fl = _gloss_of(t)
    except Exception:
        return False
    g = (g or "").lower()
    return any(w in g for w in _HUMANLIKE_WORDS)


_OCC_TABLE = {"mtime": 0, "data": None}


def _occupation_table():
    """GENRE x LOCATION -> OCCUPATION (data/library/genre_occupations.json,
    written by tools/build/write_genre_occupations.py), or None."""
    p = _paths.data("genre_occupations.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _OCC_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _OCC_TABLE["data"] = json.load(f)
            _OCC_TABLE["mtime"] = mt
    except Exception:
        return None
    return _OCC_TABLE["data"]


def occupation_entry(leaf, place):
    """-> (roll list, none share) for this leaf x place from the table --
    the most specific level with a roll list wins the list, the most
    specific `none` wins the share -- or None when the leaf has no entry
    at any level (then the old roller stands)."""
    tb = _occupation_table()
    if not tb or not leaf:
        return None
    bucket = (_genre_pool().get(leaf) or {}).get("bucket")
    levels = [tb.get("pairs", {}).get("%s + %s" % (leaf, place)) if place else None,
              tb.get("leaves", {}).get(leaf),
              tb.get("places", {}).get(place) if place else None,
              tb.get("buckets", {}).get(bucket) if bucket else None]
    if not (levels[1] or levels[3]):
        return None                        # this leaf is not in the table
    roll, none = None, None
    for lv in levels + [tb.get("default")]:
        if not lv:
            continue
        if roll is None and lv.get("roll"):
            roll = list(lv["roll"])
        if none is None and lv.get("none") is not None:
            none = float(lv["none"])
    deny = set(tb.get("deny") or ())
    return [t for t in (roll or []) if t not in deny], (0.65 if none is None else none)


_CLOTHES_TABLE = {"mtime": 0, "data": None}
_SPICE_TABLE = {"mtime": 0, "data": None}


def _json_table(cache, name):
    p = _paths.data(name)
    try:
        mt = os.path.getmtime(p)
        if mt != cache["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                cache["data"] = json.load(f)
            cache["mtime"] = mt
    except Exception:
        return None
    return cache["data"]


def _clothes_table():
    return _json_table(_CLOTHES_TABLE, "clothes.json")


def _spice_table():
    return _json_table(_SPICE_TABLE, "spice.json")


# what a level or an act needs SEEN (the coverage graph's other side); a
# garment covering a needed part yields to a modifier from the nudity
# slot, or goes. Stated once, by level.
_NEEDS_SEEN = {"safe": [], "sensitive": [], "nsfw": ["breasts"], "explicit": ["breasts", "crotch"]}
# nudity-slot words that uncover a part (from tag_group:nudity's sections)
_UNCOVER = {"breasts": ["topless", "breasts out", "one breast out", "breast slip", "no bra",
                        "open clothes", "open shirt", "clothes lift", "shirt lift", "bra lift",
                        "bra pull", "breastless clothes"],
            "crotch": ["bottomless", "no panties", "panties aside", "clothing aside", "skirt lift",
                       "panty pull", "panties around one leg", "pants pull", "dress lift",
                       "clothes pull", "clothes down"]}
_SLOT_SHARE = {"legwear": 0.55, "feet": 0.70, "head": 0.30, "neck": 0.30, "hands": 0.15,
               "accessory": 0.30, "eyes": 0.08, "accessory_head": 0.10, "accessory_torso": 0.15,
               "accessory_limb": 0.15, "sleeve": 0.25, "print": 0.15, "modifier": 0.15}


_SLOT_MEDIAN = {"key": None, "data": None}


def _slot_median(slot_max):
    """the median post count of the slot whose max is `slot_max` -- the
    reference the rarity damping measures against (built once per table)"""
    tb = _clothes_table() or {}
    if _SLOT_MEDIAN["key"] != id(tb):
        import statistics
        out = {}
        for sl, items in (tb.get("slots") or {}).items():
            posts = [float(c.get("posts") or 0) for c in items.values() if c.get("posts")]
            if posts:
                out[max(posts)] = statistics.median(posts)
        _SLOT_MEDIAN["key"], _SLOT_MEDIAN["data"] = id(tb), out
    return (_SLOT_MEDIAN["data"] or {}).get(float(slot_max or 0))


_GARMENT_PAIRS = {"mtime": 0, "data": None}
_GARMENT_POSTS = {"key": None, "data": {}}


def _garment_share(tag):
    """a clothes-table item's own rate: its posts over the booru's"""
    slots = (_clothes_table() or {}).get("slots") or {}
    if _GARMENT_POSTS["key"] != id(slots):
        d = {}
        for items in slots.values():
            for t, c in items.items():
                d[t] = max(d.get(t, 0), int((c or {}).get("posts") or 0))
        _GARMENT_POSTS["key"], _GARMENT_POSTS["data"] = id(slots), d
    return float(_GARMENT_POSTS["data"].get(str(tag or "").lower()) or 0) / _BOORU_POSTS


def garment_lift(cand, worn, lo=0.1, hi=6.0):
    """WHAT THE BOORU WEARS TOGETHER (the author, 2026-10-03: "make the clothes
    pick choices a little more coherent / complementing each other"). -> the
    measured lift of `cand` beside what is already worn -- P(cand | worn piece)
    over cand's own rate, read from both pieces' rows of garment_pairs.json
    (tools/build/harvest_garment_pairs.py) and averaged, the geometric mean over
    the worn pieces, capped -- or None where nothing is measured.

    A row is a garment's 500 strongest companions, so it stops near one percent:
    a candidate absent from it counts against the pair only when the candidate
    is common enough that it would have been listed; of a rare one the row says
    nothing."""
    tb = (_json_table(_GARMENT_PAIRS, "garment_pairs.json") or {}).get("garments") or {}
    cand = str(cand or "").lower()
    if not tb or not cand:
        return None

    def _one(a, b):
        """lift of b beside a, from a's row"""
        ra, pb = tb.get(a), _garment_share(b)
        if not ra or not pb:
            return None
        f = (ra.get("with") or {}).get(b)
        if f is None:
            fl = float(ra.get("floor") or 0.0)
            if not fl or pb < fl:
                return None                  # too rare to be listed: no opinion
            f = 0.5 * fl
        return float(f) / pb

    logs = []
    for w in worn or ():
        w = str(w or "").lower()
        if not w or w == cand:
            continue
        est = [x for x in (_one(w, cand), _one(cand, w)) if x is not None]
        if est:
            logs.append(math.log(min(hi, max(lo, sum(est) / len(est)))))
    if not logs:
        return None
    return math.exp(sum(logs) / len(logs))


def seen_states(kind):
    """the states in which the part is actually SEEN (the author, 2026-09-19: from
    nsfw up a dressed subject shows breasts / nipples / pussy / penis "either
    through undress or by reveal tags (see-through / transparency also counts)").
    One set for the clothes roll, which draws one, and for the level's injection,
    which weighs them (2026-10-03)."""
    nd = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
    if str(kind or "female") in ("female", "futanari"):
        return {t for t, c in nd.items() if str(c.get("section") or "") in
                ("whole_breasts_visible", "nipples", "visible parts of the pussy")} | {
            "topless female", "nipple slip", "areola slip", "nipples", "see-through clothes", "see-through shirt",
            "see-through dress", "see-through leotard", "see-through swimsuit", "see-through bodysuit", "wet shirt",
            "breasts out", "one breast out", "breast slip", "bottomless", "pussy", "pussy peek", "panties aside",
            "clothing aside", "panty pull", "no panties", "see-through panties", "bikini bottom aside",
            "swimsuit aside", "leotard aside", "crotchless", "crotchless panties", "nippleless clothes",
            "breastless clothes", "cupless bra", "pasties", "naked apron", "naked shirt", "naked towel"}
    return {"penis", "penis out", "erection", "bottomless", "penis peek", "testicles", "open fly",
            "pants pull", "clothes pull", "naked towel", "see-through clothes"}


def _item_weight(t, c, ctx, slot_max=None):
    """the measured associations as multiplicative leans (never a filter):
    the subject's job, world, place, season, occasion and race; the gender
    share. An item with NO association to this context is damped by its
    rarity within its slot (the accessory-noise rule, 2026-09-05): a
    jingasa or a monocle no longer competes with a hat on a nurse."""
    fem = (ctx.get("kind") or "female") in ("female", "futanari")
    assoc = bool(ctx.get("occupation") and ctx["occupation"] in (c.get("occupation") or {})) \
        or (ctx.get("genre") in (c.get("genre") or {})) or (ctx.get("bucket") in (c.get("genre") or {})) \
        or (ctx.get("place") in (c.get("place") or {})) or (ctx.get("race") in (c.get("race") or {})) \
        or (ctx.get("season") in (c.get("season") or {})) or (ctx.get("event") in (c.get("event") or {}))
    # the count's square root: a hat (1M) outranks a jingasa (5k) fourteen
    # to one, not four to one -- the exotic is rare because it is rare
    w = float(c.get("posts") or 1) ** 0.5
    # THE ACT LEANS THE OUTFIT (2026-09-11, measured): the garment's share
    # beside the scene's act against its share at the level -- the pose
    # rule's own lift -- capped like the other leans: yoga lifts sports
    # bra and yoga pants (x6), damps the skirt and the boots (x0.2); a
    # garment the act harvest does not list is neutral
    _act9 = ctx.get("act")
    if _act9:
        try:
            _lf9 = act_lift(_act9, t, ctx.get("level") or "safe", view=None)
        except Exception:
            _lf9 = None
        if _lf9 is not None:
            w *= min(6.0, max(0.2, float(_lf9)))
            if _lf9 >= 1.0:
                assoc = True
    occ = c.get("occupation") or {}
    if ctx.get("occupation") in occ:
        w *= 6.0
    elif occ and max(occ.values()) < 0.1 and ctx.get("occupation"):
        w *= 0.3                                     # faintly another job's ('armored boots': knight .045) on this job
    elif occ and max(occ.values()) >= 0.1:           # (a tenth, the line a job owns a uniform at; was .3: 'witch hat' is .15)
        # (the author, 2026-09-19: "use specialized clothes / accessories /
        # costumes for appropriate occupations / professions / locations /
        # genres"; the job sweep dressed a nurse in a witch hat and a samurai in
        # a santa hat -- a fifth of the weight was still a hat in six)
        w *= 0.03                                    # a job's garment (maid headdress) on no maid
    gen = c.get("genre") or {}
    if ctx.get("genre") in gen or ctx.get("bucket") in gen:
        w *= 4.0
    elif gen and ctx.get("genre") and ctx.get("genre") not in gen and ctx.get("bucket") not in gen \
            and max(gen.values()) >= 0.15:
        w *= 0.1                                     # a garment strongly of another world
    if ctx.get("place") in (c.get("place") or {}):
        w *= 3.0
    sn = c.get("season") or {}
    # measured as LIFTS from the context side (a coat: winter 3.1) or as
    # shares from the garment side (older data, under 1): the foreign test
    # reads the scale from the source
    _strong = 2.0 if c.get("season_src") == "context" else 0.10
    if ctx.get("season") in sn:
        w *= 2.0
    elif sn and ctx.get("season") and ctx.get("season") not in sn and max(sn.values()) >= _strong:
        w *= 0.3                                     # a swimsuit in winter, unless the pool says so
    if ctx.get("race") in (c.get("race") or {}):
        w *= 2.0
    ev = c.get("event") or {}
    if ctx.get("event") and ctx["event"] in ev:
        w *= 4.0                                     # the occasion's own garment
    elif ev and max(ev.values()) >= (3.0 if c.get("event_src") == "context" else 0.15) \
            and ctx.get("event") not in ev:
        w *= 0.05                                    # a santa hat on no christmas
    # AN ADOPTED GARMENT ROLLS THROUGH ITS MEASURED ASSOCIATIONS (the ruling the
    # author gave the whole garment family, 2026-09-18, stored with each tag):
    # 118 adopted hats doubled the head slot, and a ninja wore a qingdai guanmao,
    # an idol a pith helmet. Where nothing of this context carries it -- no job,
    # world, place, season, occasion, race or act -- it all but stays out.
    if not assoc:
        _acons = (((_json_table(_ADOPTED, "adopted_tags.json") or {}).get("tags") or {}).get(t) or {}).get("constraints") or []
        if any(str(c_).startswith("rolled only through its measured") for c_ in _acons):
            w *= 0.05
    if not assoc and slot_max:
        # THE RARITY DAMPING IS AGAINST THE SLOT'S TYPICAL ITEM, NOT ITS
        # BIGGEST (the author, 2026-09-16: "I never see anything but simple
        # panties"). Measured against the max, the factor is
        # posts/slot_max, which makes the whole weight LINEAR in posts and
        # collapses a slot onto its one giant tag -- 'panties' (818k) beat
        # 'lingerie' (67k) and 'chemise' (3.9k) by 12x and 210x, so the
        # lingerie section never appeared. Against the slot's MEDIAN, an
        # ordinary item is undamped and only the genuinely rare one (a
        # jingasa among hats) is quieted, which is what the rule was for.
        _ref = _slot_median(slot_max)
        if _ref:
            w *= min(1.0, (float(c.get("posts") or 1) / _ref) ** 0.5)
    f, m = float(c.get("female") or 0), float(c.get("male") or 0)
    # THE GENDER SHARE GATES (measured one-girl / one-boy): a garment the
    # booru shows on women seven times in ten and on men under .15 never
    # lands on a man (fishnet pantyhose, handbag), and the reverse; the
    # softer leans stay for the in-between
    if not fem and f >= 0.7 and m <= 0.15:
        return 0.0
    if fem and m >= 0.6 and f <= 0.2:
        return 0.0
    if fem and m > 0 and m > 2.5 * f:
        w *= 0.15                                    # a man's garment on a woman, rarely
    if not fem and f > 0 and f > 4.0 * m:
        w *= 0.15
    return w


# the nudity group's SECTIONS say which part an item exposes
_NUDITY_SECTION_PART = {"breasts": "breasts", "breastsparts": "breasts", "nipples": "breasts",
                        "chest": "breasts", "ass": "crotch", "legs": "crotch", "points": "crotch",
                        "misc": "crotch", "any": "any"}
# ('complete' -- nude, completely nude -- is a state that replaces the
# outfit, never a modifier on it; the caller's nude flag owns it)


_RULED_OUT = {"mtime": 0, "data": None, "key": None}
_YOUTH = {"data": None, "base": None}
# THE YOUTH-CODED TAGS BY NAME (reviewed 2026-09-05): never rolled
_YOUTH_NAMES = {
    "child", "children", "kid", "kids", "baby", "babies", "infant", "toddler", "toddlercon",
    "fetus", "loli", "shota", "oppai loli", "lolibaba", "shotajiji", "onii-shota",
    "kodomo doushi", "aged down", "underage", "preteen", "minor", "young girl", "young boy",
    "little girl", "little boy", "muscular child", "child on child", "child abuse", "pedophile",
    "cherub", "elementary school student",
    "kindergarten", "kindergarten uniform", "kindergarten bag", "school hat", "randoseru",
    "seishou elementary school uniform", "tomoeda elementary school uniform",
    "indonesian elementary school uniform", "hekiho academy school uniform",
    "b.a.b.e.l. uniform", "age comparison", "if they mated", "lost child", "child carry",
    "hajimete no otsukai", "jeffrey epstein", "adam lanza", "angry german kid",
    "zachary gordon"}
# THE SCHOOL FAMILY (the author's 2026-09-06: "ok at nsfw on adults"): never
# rolled (no age is rolled), but typed they pass at every level -- an adult
# in serafuku is an ordinary booru subject
_SCHOOL_TYPED_OK = {"school uniform", "serafuku", "gakuran", "gym uniform", "buruma",
                    "school swimsuit", "school bag", "schoolgirl", "schoolboy", "student"}
# the gloss NET: only a gloss that DEFINES the tagged thing as a child
# (prepubescent, infancy, newborn, a young child, child-like, loli, shota,
# an elementary or kindergarten pupil); a gloss that mentions children as
# the audience or the other party of an adult's action does not count
_YOUTH_GLOSS_RE = re.compile(
    r"\b(prepubescent|preadolescent|pre-?teen|underage|infancy|newborn|recently born|"
    r"toddlers?|\bloli\b|\bshota\b|child-like|childlike|female child|male child|young child|"
    r"small child|younger (?:boy|girl)|child sex|elementary school (?:student|girl|boy|pupil)|"
    r"kindergarten (?:student|pupil|child)|preschool(?:er)?s?)\b", re.I)
_YOUTH_EXEMPT = _SCHOOL_TYPED_OK | {"aged up", "mature female", "mature male", "milf", "old woman", "old man",
                 "childhood friend", "childhood friends", "kid gloves", "lolita fashion",
                 "sweet lolita", "gothic lolita", "classic lolita", "wa lolita", "santa claus",
                 # adult, animal or furniture words the net caught (reviewed 2026-09-05)
                 "crib", "high chair", "ride-on toy car", "puppy", "hatching", "twee fashion",
                 "kindergarten teacher", "good boy (phrase)", "cradle", "potty"}
# SEXUALISED MINORS, BY DEFINITION (the booru's own glosses: sexually
# suggestive artwork of preadolescents): refused at EVERY level, typed or
# not. This is not a lean and not a setting.
# 'nymphet', 'jailbait' (2026-09-17): English words for a sexualised minor,
# no booru tag, so neither the tag list nor the gloss net could see them --
# found when the lexicon's person-noun candidates were checked for youth
_SEXUAL_YOUTH_NAMES = {"loli", "shota", "toddlercon", "oppai loli", "lolibaba", "shotajiji", "onii-shota",
                       "onee-loli", "onee-shota", "lolidom", "shotadom", "kodomo doushi", "child on child",
                       "pedophile", "child abuse", "nymphet", "nymphets", "jailbait"}
# + the lexicon's sexualised-child words (WordNet senses under a child sense
# whose definition is sexual: 'catamite'), found by build_lexicon.py
try:
    from promptstudio.engine import lexicon as _lexicon_people
    _SEXUAL_YOUTH = _SEXUAL_YOUTH_NAMES | set(_lexicon_people.people("sexual"))
except Exception:
    _SEXUAL_YOUTH = set(_SEXUAL_YOUTH_NAMES)
# TYPED ONLY BY RULING (the author's 2026-09-05), not youth: never rolled
# 'sexy no jutsu' (the author's 2026-09-15): the Naruto technique, reached from
# the word 'sexy' in every prompt that said it -- a copyright's own
# technique is typed, never rolled or mapped
# 'null bulge' (the author's 2026-09-16): a genital-less bulge, flagged 'object' by
# the gloss and drawable as one -- typed only, never rolled
# 'big belly' (the author's 2026-09-16): a body state of the spice and body tables
# (full, fat or pregnant) -- typed only, never rolled
# 'virtual youtuber' (the author, 2026-09-03 for occupations; 2026-09-16 everywhere):
# a status, not a look -- it rode in on a persona's canon at .98
# 'spot color', 'greyscale', 'monochrome' (the author, 2026-09-16): a picture
# without colour cannot carry the hair, eye and garment colours every subject
# is described with -- rolled, they contradict the rest of the line ("to avoid
# contradictions i want [them] typed only"). Measured, it holds: 'greyscale'
# beside 'blonde hair' is 213 posts where chance gives ~37,000.
# 'pregnant' (the author, 2026-09-16): a body state the spice table rolled as
# an act -- it reached a woman lying on a dock who was nothing of the kind
# (2026-09-20, the author: "serafuku - I think we can bring it back to rolled
# (the dress itself doesn't necessarily makes subject underaged)". It leaves the
# never-rolled list below and stays in _SCHOOL_TYPED_OK, so the youth refusals,
# the gloss net and the sexualised-minor rule are untouched; no age is rolled.)
_SCHOOL_NEVER_ROLLED = _SCHOOL_TYPED_OK - {"serafuku"}
_RULED_TYPED_ONLY_NAMES = {"umbilical cord", "breast pump", "sexy no jutsu", "null bulge", "big belly",
                           "virtual youtuber", "spot color", "greyscale", "monochrome",
                           "pregnant", "brain", "rectum", "bruised eye", "bloodshot eyes", "severed hair", "third eye",
                           # (2026-09-19: what the floor sweep unblocked in the fast plan's pools -- organs,
                           #  an injury, a part the author ruled typed only)
                           } | _SCHOOL_NEVER_ROLLED
_COLOURLESS = ("spot color", "greyscale", "monochrome")


_CHAR_AGES = {"mtime": 0, "data": None}


def character_age_class(name):
    """-> 'minor' / 'high school' / 'adult' / 'unknown': what the booru wiki's own
    page says of the character (tools/build/harvest_character_ages.py)"""
    tb = _json_table(_CHAR_AGES, "character_ages.json") or {}
    return str(((tb.get("characters") or {}).get(str(name or "").lower()) or {}).get("class") or "unknown")


_YOUNG_CLASSES = ("minor", "high school")
_CHAR_YOUTH = {"mtime": 0, "data": None}
_YOUTH_SHARE = 0.005        # loli / shota share of a character's posts that marks it drawn as a child


def character_youth_share(name):
    """-> the share of the character's posts the booru tags loli (shota), or None
    unmeasured (tools/build/harvest_character_youth.py)"""
    tb = _json_table(_CHAR_YOUTH, "character_youth.json") or {}
    e = (tb.get("characters") or {}).get(str(name or "").lower())
    return None if not e or e.get("share") is None else float(e["share"])


def character_is_young(name, rolled=False):
    """A CHARACTER MAY BE SEXUALISED ONLY WHEN NOTHING MEASURED SAYS IT IS YOUNG
    (the author, 2026-09-27: "I'd rather allow unknown" -- within the rule that no
    minor is ever sexualised): the wiki's class and the booru's own loli / shota
    share of its pictures; unmeasured counts as young. (A franchise test was
    tried and dropped: the wiki states an age mostly for school-age characters,
    so every franchise's classed members skew young -- touhou, final fantasy,
    dc comics read 'young'.)"""
    if character_age_class(name) in _YOUNG_CLASSES:
        return True
    sh = character_youth_share(name)
    if sh is None or sh >= _YOUTH_SHARE:
        return character_age_class(name) != "adult" or (sh is not None and sh >= _YOUTH_SHARE)
    return False


def refuse_youth(base, user_tags, level):
    """-> {"words": [refused], "base": text without them, "tags": typed tags
    without them, "recast": whether a subject word left}. The sexual-youth
    tags are refused at every level; the youth-coded names whenever the
    level is sensitive or above; typed input is otherwise law."""
    words = []
    above = sm.SPICE_ORDER.get(level, 0) > sm.SPICE_ORDER.get("safe", 0)
    youth = youth_tags()
    for t in list(user_tags or []):
        tl = str(t).lower().strip()
        if tl in _SEXUAL_YOUTH or (above and tl in youth):
            words.append(tl)
    low = " " + str(base or "").lower() + " "
    # A CANON MINOR IS A MINOR BY NAME (2026-09-27, the author's live test: a
    # Precure middle-schooler in an explicit prompt): above safe, a typed
    # character whose wiki page says middle school, high school or an age under
    # 18 is refused as the youth words are -- the name goes, the person stays
    if above:
        try:
            for _cn in _typed_characters(base) or []:
                if character_is_young(_cn) and _cn not in words:
                    words.append(str(_cn).lower())
        except Exception:
            pass
    for t in sorted(_SEXUAL_YOUTH | (youth if above else set()), key=len, reverse=True):
        if re.search(r"(?<![a-z0-9-])%s(?![a-z0-9-])" % re.escape(t), low):
            if t not in words:
                words.append(t)
    if not words:
        return {"words": [], "base": base, "tags": list(user_tags or []), "recast": False}
    text = str(base or "")
    for t in sorted(words, key=len, reverse=True):
        text = re.sub(r"(?i)(?<![a-z0-9-])%s(?![a-z0-9-])" % re.escape(t), " ", text)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,")
    tags = [t for t in (user_tags or []) if str(t).lower().strip() not in words]
    return {"words": words, "base": text, "tags": tags, "recast": True}


def youth_tags(lexicon=True):
    """the youth set (never generated, whatever the floor says): the
    stated names plus every glossed tag whose gloss says child, infant,
    toddler, puberty, underage, minor or kid -- the maturity whitelist
    and the adult words that merely mention childhood are exempt -- plus
    the lexicon's person words whose senses are a child's ('lass' shares
    its sense with 'young girl'). lexicon=False is the set without them,
    which is what build_lexicon.py reads them against."""
    if not lexicon:
        youth_tags()
        return _YOUTH["base"]
    if _YOUTH["data"] is None:
        out = set(_YOUTH_NAMES)
        try:
            with open(_paths.data("tag_glosses.json"), encoding="utf-8-sig") as f:
                gl = json.load(f).get("glosses") or {}
            for t, e in gl.items():
                g = str((e or {}).get("g") or "")
                if g and _YOUTH_GLOSS_RE.search(g):
                    out.add(t)
        except Exception:
            pass
        _YOUTH["base"] = out - _YOUTH_EXEMPT
        try:
            from promptstudio.engine import lexicon as _lx
            out = out | set(_lx.people("youth"))
        except Exception:
            pass
        _YOUTH["data"] = out - _YOUTH_EXEMPT
    return _YOUTH["data"]


def ruled_out():
    """everything RULED OUT of the dice (the author's standing rules, stated
    once in the writers): the spice table's never-list and typed-only
    list, and the body table's ruled builds. Never rolled by any emitter;
    typed input is unaffected."""
    sp = _spice_table() or {}
    bd = _body_table() or {}
    key = (id(sp), id(bd))
    if _RULED_OUT["key"] != key:
        out = set(sp.get("never") or []) | set(sp.get("typed_only") or [])
        out |= set(bd.get("ruled_typed_only") or [])
        out |= youth_tags()                         # never rolled, at any level
        out |= _RULED_TYPED_ONLY_NAMES
        # THE AUTHOR'S RULINGS ON THE UNROLLED MAP (unrolled_rulings.json, 2026-09-18
        # / 19): every name ruled typed only there is ruled out of EVERY die -- the
        # measured place draw had 'unworn gag' on offer because the ruling lived
        # in a file no emitter read
        try:
            with open(_paths.data("unrolled_rulings.json"), encoding="utf-8-sig") as _f9:
                out |= {str(t).lower() for t in (json.load(_f9).get("typed_only") or [])}
        except Exception:
            pass
        _RULED_OUT["data"], _RULED_OUT["key"] = out, key
    return _RULED_OUT["data"]


def object_for(place, rng, allowed=None):
    """-> a held object from the tree's holding objects (the activity
    table's registry), weighted by its measured share at THIS place, else
    by its count; None when the table has nothing allowed"""
    tb = _activity_table() or {}
    objs = tb.get("objects") or {}
    cands = []
    for t, c in objs.items():
        if not _floor_measured(t) or (allowed is not None and not allowed(t)):
            continue
        pl = c.get("places") or {}
        w = float(c.get("posts") or 1) ** 0.5
        # an ADOPTED prop rolls through its measured places (the author's rule
        # for every specialised thing; 2026-09-19 a girl in a cafe held a kunai):
        # nothing of this place carries it, so it all but stays out
        if c.get("section") == "adopted" and not (place and place in pl):
            w *= 0.03
        if place and place in pl:
            w *= 1.0 + 20.0 * float(pl[place])       # the place's own object
        elif pl and max(pl.values()) >= 0.15:
            w *= 0.3                                 # another place's object
        cands.append((t, w))
    return _wroll(rng, cands) if cands else None


_GARMENT_GONE = re.compile(r"^(no |unworn )|( removed| only)$|^(topless|bottomless)( (fe)?male)?$")


def _says_garment_gone(state):
    """the state means the covering garment is absent ('topless', 'no bra',
    'shirt removed', 'bikini bottom only'), not opened or moved aside"""
    return bool(_GARMENT_GONE.search(str(state).lower()))


def _uncover_options(part):
    """-> [tags] from spice.json's nudity slot that expose `part` (its
    section decides), else the stated fallback"""
    tb = _spice_table() or {}
    items = (tb.get("slots") or {}).get("nudity") or {}
    out = []
    for t, c in items.items():
        sec = str(c.get("section") or "")
        p = _NUDITY_SECTION_PART.get(sec)
        if p == part or p == "any":
            out.append(t)
    return out or list(_UNCOVER.get(part, []))


def scene_words(ctx):
    """the facts of the scene a costume can hang on, beyond the place itself:
    the season, the weather, the time, the event, the activity, the room's class
    ('bed'), and whether the picture is indoors"""
    c = ctx or {}
    out = [c.get("season"), c.get("weather"), c.get("time"), c.get("event"), c.get("activity"),
           c.get("activity_state"), c.get("place_class")]
    if c.get("place_class") == "bed":
        out += ["bed", "on bed"]
    if c.get("place_class") in ("bed", "lounge", "seated", "counter", "bath", "kitchen"):
        out.append("indoors")
    return [str(x).lower() for x in out if x]


def clothes_for(ctx, rng, allowed=None):
    """-> {"outfit": [tags], "modifiers": [tags], "uncovered": [parts]} for one
    subject. ctx: kind, occupation, genre, bucket, place, season, race,
    level, clothes_class (from the activity / event), nude (bool: the act
    or state says so). Every garment is a measured tag with a measured
    floor the caller's gate allows. Slots: a whole outfit (uniform /
    traditional / swim) OR a dress OR a top and a bottom, then the rest by
    share; the NEEDS-SEEN parts of the level are uncovered by a modifier
    from the nudity slot or by dropping the garment."""
    tb = _clothes_table() or {}
    slots = tb.get("slots") or {}
    _ruled = ruled_out()
    # the body rulings gate the clothes too ('blood on gloves' is horror's, 2026-09-20)
    ok = (lambda t: _floor_measured(t) and t not in _ruled and (allowed is None or allowed(t))
          and part_gate(t, {"genre": ctx.get("genre"), "kind": ctx.get("kind"), "race": ctx.get("race"),
                            "level": ctx.get("level"), "place": ctx.get("place"), "act": ctx.get("act")}))
    out = {"outfit": [], "modifiers": [], "uncovered": []}
    if ctx.get("nude"):
        return out
    level = ctx.get("level") or "safe"

    slot_max = tb.get("slot_max") or {}

    def pick(slot, extra=None):
        items = slots.get(slot) or {}
        cands = [(t, _item_weight(t, c, ctx, slot_max.get(slot)) * (extra(t, c) if extra else 1.0))
                 for t, c in items.items() if ok(t)]
        cands = [(t, w) for t, w in cands if w > 0]
        # EVERY PIECE IS DRAWN BESIDE WHAT IS ALREADY WORN (2026-10-03): its measured
        # lift next to the outfit so far -- canon included -- leans the pick, in
        # every slot (garment_lift); unmeasured is neutral
        _worn9 = [str(x) for x in list(out["outfit"]) + list(ctx.get("canon") or []) if x]
        if _worn9:
            cands = [(t, w * (garment_lift(t, _worn9) or 1.0)) for t, w in cands]
        return _wroll(rng, cands) if cands else None

    cls = ctx.get("clothes_class")
    occ = ctx.get("occupation")

    def _rollable(t, c):
        """school garments never roll (no age is rolled); a costume only
        with an occasion"""
        if c.get("school"):
            return False
        if c.get("costume"):
            # a costume rolls with its own occasion (measured), or with any
            # occasion while its occasions are unmeasured
            ev = c.get("event") or {}
            if not ctx.get("event") or (ev and ctx["event"] not in ev):
                return False
        return True

    def _assoc(c):
        """the item has a measured association with THIS context (the
        occasion included: a santa costume on christmas)"""
        return (occ and occ in (c.get("occupation") or {})) \
            or ctx.get("genre") in (c.get("genre") or {}) or ctx.get("bucket") in (c.get("genre") or {}) \
            or ctx.get("place") in (c.get("place") or {}) or ctx.get("race") in (c.get("race") or {}) \
            or bool(ctx.get("event") and ctx["event"] in (c.get("event") or {}))

    def _stem(w):
        return w[:-3] + "y" if w.endswith("ies") else (w[:-1] if w.endswith("s") and len(w) > 3 else w)

    def _garment_words():
        return _garment_stems()               # one list (the spice injector reads it too)

    _BODY_SLOTS = ("top", "bottom", "swim", "uniform", "traditional")

    def _names_worn(m):
        """a modifier names a garment ('open vest', 'panty pull'): that
        garment must be worn (by stem); a generic one ('clothes down',
        'lapels') needs a garment on the body, not just shoes"""
        gw = _garment_words()
        words = [_stem(w) for w in m.split() if _stem(w) in gw]
        # A GARMENT IS NAMED BY ITS HEAD WORD (the author, 2026-10-03: contradictory
        # clothes -- 'sideless dress' on a 'dress shirt' and pants: every word of
        # every worn tag counted, so the shirt's 'dress' answered for a dress)
        worn = {_stem(t.split()[-1]) for t in out["outfit"] if t.split()}
        # the adoption ruling names the garment where the tag's own words do not
        _needs, _alts = _adopted_requires(m)
        if _alts and not any(_stem(w) in worn for g in _alts for w in g.split()):
            return False
        if not _adopted_fits_body(m, ctx.get("kind")):
            return False
        # a state of the feet's clothes needs the feet in the frame ('toe seam')
        _cons_m = ((( _json_table(_ADOPTED, "adopted_tags.json") or {}).get("tags") or {}).get(str(m).lower()) or {}).get("constraints") or []
        if "requires feet in frame" in _cons_m and str(ctx.get("framing") or "full body") not in ("full body", "wide shot"):
            return False
        # 'requires any clothing' means a garment on the body -- a top, a bottom,
        # a dress, a uniform, a swimsuit -- not underwear or boots alone
        # ('stomach peek' on a woman in boxers and nothing else)
        if _needs and not _alts and not any(
                t in (slots.get(sl) or {}) or any(_stem(g.split()[-1]) == _stem(t.split()[-1]) for g in (slots.get(sl) or {}))
                for t in out["outfit"] for sl in _BODY_SLOTS):
            return False
        if not words:
            return any(t in (slots.get(sl) or {}) for t in out["outfit"] for sl in _BODY_SLOTS + ("sexual",))
        # 'NO PANTIES' SAYS THEY ARE NOT THERE (the author, 2026-10-03: too little of
        # the other exposure states). The rule above asked that the named garment be
        # WORN, so 'no panties' and 'no bra' applied only to a body wearing them and
        # were all but never drawn -- 'no panties' is on 11% of a lone woman's nsfw
        # posts. A state that names a garment to say it is absent needs that garment
        # absent, and clothes on the body to be without it under.
        if m.startswith("no "):
            return not any(w in worn for w in words) and any(
                t in (slots.get(sl) or {}) or any(_stem(g.split()[-1]) == _stem(t.split()[-1]) for g in (slots.get(sl) or {}))
                for t in out["outfit"] for sl in _BODY_SLOTS)
        return all(w in worn for w in words)

    _cast_ct9 = ctx.get("cast") if isinstance(ctx.get("cast"), dict) else None
    if _cast_ct9 is None and (ctx.get("kind") or "female") == "female":
        _cast_ct9 = {"female": 1}

    def _counted9(pairs):
        """[(state, w)] -> the same states at their measured counts for this cast
        at the level, the uncounted at the median; unchanged where none is counted"""
        if not _cast_ct9 or not pairs:
            return pairs
        mc = {u: spice_count(u, level, _cast_ct9) for u, _w in pairs}
        got = sorted(float(n) for n in mc.values() if n is not None)
        if not got:
            return pairs
        med = got[len(got) // 2]
        return [(u, (float(mc[u]) if mc[u] is not None else med)) for u, _w in pairs]

    def _given_worn9(pairs, garments):
        """WHICH STATE, GIVEN WHAT IS WORN (the author, 2026-10-03: 'see-through
        clothes' on 42% of nsfw outfits and 49% of explicit ones, measured on this
        roll). Most states name a garment and apply only when it is worn; the few
        that fit anything -- open clothes, see-through clothes, revealing clothes
        -- were left on every outfit and shared the pick. The garment's own posts
        say which state it carries (garment_pairs.json: a shirt 'open shirt' .053,
        panties 'panty pull' .075, a bikini 'see-through clothes' .047): each state
        weighs P(state | the garments over the part), times its share at the
        level. A state the row does not list stands under the row's floor.
        Unchanged where no worn garment has a row."""
        tb = (_json_table(_GARMENT_PAIRS, "garment_pairs.json") or {}).get("garments") or {}
        rows = []
        for g_ in garments or ():
            g_ = str(g_).lower()
            r_ = tb.get(g_) or tb.get(g_.split()[-1]) if g_ else None
            if r_:
                rows.append(r_)
        if not rows or not pairs:
            return pairs
        out_ = []
        for u, _w in pairs:
            fs = [float((r_.get("with") or {}).get(u, 0.25 * float(r_.get("floor") or 0.01))) for r_ in rows]
            out_.append((u, (sum(fs) / len(fs)) * max(_floor_share(u, level), 0.02)))
        return out_

    whole = None
    picked_body = False
    # WHAT CANON ALREADY WEARS (2026-09-16: a persona's canon 'skirt' and
    # the roll's 'pants' on one body, then 'thighhighs' beside the pants):
    # a canon garment on a body slot fills that slot; the roll fills only
    # the slots canon left empty
    _canon_worn = [str(t) for t in (ctx.get("canon") or []) if t]
    _canon_slots = set()
    for _cw in _canon_worn:
        _canon_slots |= set(_cl_slots_of(slots, _cw))
    if _canon_slots & {"uniform", "traditional", "swim"}:
        picked_body = True
    # 0. A PLACE THAT UNDRESSES (measured: its top share under .2 -- an
    # onsen .05; a bathroom at .33 does not) dresses the subject from its
    # own strongest garments at sensitive and above (naked towel, towel,
    # yukata at the onsen), floors and gates deciding; at safe the
    # ordinary branch dresses as anywhere
    _pk = ctx.get("place") or ""
    if _pk and _pk not in (tb.get("place_garments") or {}):
        _pk = place_parent(_pk) or _pk           # a ruled small place dresses like its parent (2026-09-15)
    _pg = (tb.get("place_garments") or {}).get(_pk) or {}
    _top_share = ((tb.get("slot_share") or {}).get(_pk, {}) or {}).get("top")
    if level != "safe" and _pg and _top_share is not None and _top_share < 0.2:
        # a body garment or a covering state stands as the outfit; an
        # accessory (the onsen's strongest tag is 'hair ornament') never
        _nud = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
        # weighted by the tag's measured rating share AT this level, as the
        # spice injection is: 'nude' is .05 of its posts at sensitive and
        # .36 at nsfw, so a sensitive bather wears the towel, an nsfw one
        # may not; under .15 the tag is not of the level
        cands = [(t, float(f) * _floor_share(t, level)) for t, f in _pg.items()
                 if ok(t) and _floor_share(t, level) >= 0.15
                 and (any(t in (slots.get(sl) or {}) for sl in ("top", "bottom", "swim", "uniform", "traditional", "sexual"))
                      or t in _nud)]
        if cands:
            whole = _wroll(rng, cands)
            picked_body = True
    # 1. a whole outfit when the job, the world or the activity asks for one
    # SWIMSUIT STARTS AT SENSITIVE (the author's 2026-09-05, and the measured
    # floors agree): at safe the water place dresses like anywhere else
    if whole:
        pass
    elif level != "safe" and (cls in ("swimsuit",)
                            or (ctx.get("place_class") == "water" and rng.random() < 0.7)):
        whole = pick("swim", extra=lambda t, c: 1.0 if _rollable(t, c) else 0.0)
    elif cls in ("kimono", "yukata") or (ctx.get("genre") == "ancient japan" and rng.random() < 0.7):
        # the class names the Japanese family: a summer festival's yukata
        # class drew 'chinese clothes' when the rolled genre damped every
        # kimono as foreign; the family the class names is the one to draw
        whole = pick("traditional", extra=lambda t, c: (1.0 if _rollable(t, c) else 0.0)
                     * (3.0 if c.get("japanese") else 0.3))
    elif occ:
        # THE JOB OWNS THE OUTFIT: its own tag or a measured uniform; never
        # a random dress beside a nurse. The job's own garment tag (maid,
        # police uniform) is the identity case a related share cannot see.
        uni = slots.get("uniform") or {}
        own = next((t for t in (occ, occ + " costume", occ + " uniform", occ + " outfit")
                    if t in uni and ok(t)), None)
        if own:
            whole = own
        else:
            # ... a uniform the job wears on a TENTH of its pictures or more (the
            # line picked_body reads below): 2026-09-19, the adopted 'hospital
            # gown' carries 'nurse' at .05 -- the nurse stands beside the patient
            # who wears it -- and as the only uniform with any nurse share it
            # dressed ten nurses of ten
            whole = pick("uniform", extra=lambda t, c: (1.0 if _rollable(t, c) else 0.0)
                         * (3.0 if float((c.get("occupation") or {}).get(occ) or 0) >= 0.1 else 0.0))
        # the job tag dresses the body only when the booru shows it in
        # garments of its own (a nurse: nurse cap .9; a maid: its tag);
        # an engineer or an otaku gets a top and a bottom like anyone
        picked_body = bool(own) or any(
            float((c.get("occupation") or {}).get(occ) or 0) >= 0.1
            for sl in slots.values() for c in sl.values())
    elif level != "safe" and rng.random() < float((tb.get("underwear_share") or {}).get(level) or 0.0):
        # THE UNDERWEAR BASE (measured share per level): bra with panties,
        # or a one-piece (lingerie, babydoll); the level's coverage step
        # still decides what stays on
        sx = slots.get("sexual") or {}
        first = pick("sexual", extra=lambda t, c: 1.0 if _rollable(t, c) else 0.0)
        if first:
            out["outfit"].append(first)
            pair = {"bra": "panties", "panties": "bra"}.get(first)
            if pair and pair in sx and ok(pair):
                out["outfit"].append(pair)
            picked_body = True
    elif rng.random() < (0.5 if ctx.get("event") and any(
            ctx["event"] in (c.get("event") or {}) for c in (slots.get("uniform") or {}).values()) else 0.15):
        # a jobless whole outfit only when something of this context asks
        # for it (the world, the place, the race); at an occasion with
        # garments of its own (halloween, christmas) the branch fires half
        # the time
        # ... and never a job's own garment (a nun's habit on a park
        # visitor): the job owns it, and this subject has no job
        whole = pick("uniform", extra=lambda t, c: 1.0 if _rollable(t, c) and _assoc(c)
                     and not any(v >= 0.1 for v in (c.get("occupation") or {}).values()) else 0.0)
    if whole:
        out["outfit"].append(whole)
    elif not picked_body:
        top = None
        if "top" not in _canon_slots:
            top = pick("top", extra=lambda t, c: 1.0 if _rollable(t, c) else 0.0)
            if top:
                out["outfit"].append(top)
        _canon_top = next((t for t in _canon_worn if "top" in _cl_slots_of(slots, t)), None)
        if "bottom" not in _canon_slots and not any(
                w in (top or _canon_top or "") for w in ("dress", "gown", "robe", "leotard")):
            bottom = pick("bottom", extra=lambda t, c: 1.0 if _rollable(t, c) else 0.0)
            if bottom:
                out["outfit"].append(bottom)
    # THE ACT NAMES ITS GARMENT (2026-09-15: 'shirt lift' rolled on a
    # hoodie, 'open kimono' on a shirt and a skirt): an act that lifts,
    # opens or pulls a garment is done to that garment, so the garment is
    # worn -- the act is a fact of the scene the outfit follows, as it
    # follows the place. The garment comes from the body slots by its own
    # measured weight and displaces what it replaces: a top or a bottom
    # its own slot's item, a whole outfit the whole.
    _gw_act = _garment_words()
    _act_stems = [s for s in (_stem(w) for w in str(ctx.get("act") or "").lower().split()) if s in _gw_act]
    for _s in dict.fromkeys(_act_stems):
        if any(_stem(w) == _s for t in out["outfit"] for w in t.split()):
            continue
        _cands, _cslot = [], None
        for _sl in ("top", "bottom", "sexual", "traditional", "uniform", "swim"):
            for t, c in (slots.get(_sl) or {}).items():
                if _stem(t.split()[-1]) == _s and ok(t) and _rollable(t, c):
                    _cands.append((t, _item_weight(t, c, ctx, slot_max.get(_sl))))
                    _cslot = _cslot or _sl
        _cands = [(t, w) for t, w in _cands if w > 0]
        _g = _wroll(rng, _cands) if _cands else None
        if not _g:
            continue
        _gone = set()
        for _sl in ("top", "bottom", "traditional", "uniform", "swim"):
            if _sl == _cslot or (_cslot in ("top", "bottom") and _sl in ("traditional", "uniform", "swim")) \
                    or (_cslot in ("traditional", "uniform", "swim") and _sl in _BODY_SLOTS):
                _gone.update(t for t in out["outfit"] if t in (slots.get(_sl) or {}))
        out["outfit"] = [t for t in out["outfit"] if t not in _gone] + [_g]
    # 3. the rest by MEASURED share (the table's, per place where the place
    # lifts it, else the baseline; the stated share only without a table)
    _ss = tb.get("slot_share") or {}
    _place_ss = _ss.get(ctx.get("place") or "", {})
    _base_ss = _ss.get("_baseline") or {}
    # THE FRAME (2026-09-06): a slot's chance follows its measured share
    # under the picture's framing over its baseline -- legwear under an
    # upper-body frame is out of the picture
    _frame_ss = (_ss.get("_framing") or {}).get(ctx.get("framing") or "", {})
    _undressing = _top_share is not None and _top_share < 0.2
    # A PLACE THAT UNDRESSED THE SUBJECT LEAVES THE BODY BARE (2026-09-15:
    # 'nude, white socks' at the onsen): a whole nudity state from the
    # place's garments owns the body slots as the undress roll does
    _bare = bool(whole) and _is_whole_nudity(whole)
    for slot, share in _SLOT_SHARE.items():
        if _bare and slot in ("legwear", "neck", "sleeve", "print", "modifier"):
            continue
        share = _place_ss.get(slot) or _base_ss.get(slot) or share
        if _undressing and slot not in _place_ss:
            share *= 0.2                 # shoes on a bather: the place did not measure them
        if _frame_ss and _base_ss.get(slot):
            # absent from the frame's 500 strongest tags = out of frame
            # (no legwear under 'upper body'): half its smallest listed share
            _fs = float(_frame_ss.get(slot) or 0.5 * min(_frame_ss.values()))
            share = min(0.95, share * _fs / float(_base_ss[slot]))
        # THE DETAIL CONTROL SCALES THE EXTRAS (the author, 2026-09-28: "I haven't
        # really noticed the difference between standard and detailed"; measured,
        # 14.7 subject tags a prompt at standard, 14.8 at detailed): the table
        # owns legwear, shoes, headwear, neckwear and the accessories, and the
        # detail-scaled slot draws that once decided them are discarded -- so the
        # scale is applied here, to each extra's measured share
        share = min(0.95, share * float(ctx.get("detail", 1.0)))
        if rng.random() < share:
            if slot in _canon_slots:
                continue                     # canon fills it
            t = pick(slot, extra=lambda t_, c: 1.0 if _rollable(t_, c) else 0.0)
            if not t or t in out["outfit"]:
                continue
            # A GARMENT THE BOORU DOES NOT WEAR WITH WHAT IS ON (2026-09-16:
            # 'thighhighs' beside 'pants', measured pair lift .31; beside a
            # skirt 1.4): the measured pair lift against every body garment
            # worn (canon or rolled) gates the pick; unmeasured is neutral
            # ... and the same question for every other slot (the author's bug
            # hunt, 2026-09-21: a 'garter belt' over a 'string bikini' at the
            # beach, pair lift .63; a 'blindfold' beside it, .47). The rule was
            # written for legwear and is true of anything worn: what the booru
            # does not wear together, the picture does not either.
            if not _pairs_with_worn(t, out["outfit"] + _canon_worn, slots):
                continue
            if not fits_race(t, ctx.get("race")):
                continue                     # a mermaid wears no shoes
            if slot in ("sleeve", "print", "modifier"):
                if _names_worn(t) and (slot != "modifier" or _of_level(t, level)):
                    out["modifiers"].append(t)
            else:
                out["outfit"].append(t)
    # 4. COVERAGE: what the level needs seen is uncovered by a modifier or
    # by dropping the garment that covers it
    # A GARMENT ROLLS ITS VARIANT (2026-09-14, measured): 'striped bikini'
    # stands in for 'bikini' at P(striped bikini | bikini), 'frilled dress'
    # for 'dress'; the variant's floor gates it at the level, an
    # unmeasured variant is typed-only; at most two variants an outfit
    _nvar = 0
    for _i, _g in enumerate(list(out["outfit"])):
        if _nvar >= 2:
            break
        _c = None
        for _sl, _items in slots.items():
            if _g in _items:
                _c = _items[_g]
                break
        _vars = [(v, float(sh)) for v, sh in ((_c or {}).get("variants") or {}).items()
                 if ok(v) and sh > 0]
        if not _vars:
            continue
        _r = rng.random()
        for v, sh in sorted(_vars, key=lambda kv: -kv[1]):
            if _r < sh:
                out["outfit"][_i] = v
                _nvar += 1
                break
            _r -= sh
    # THE DRESSED SUBJECT SHOWS SOMETHING (the author, 2026-09-16: the
    # nudity group's exposure states are "VERY rarely rolled"). They had no
    # path of their own: only the level's coverage rule reached them, so a
    # dressed subject showed cleavage, bare shoulders, a midriff, an open
    # shirt or a slipped strap almost never -- 10 of the group's 148 tags
    # in 60 runs. Drawn here by their measured share of the level's rating
    # universe, lifted by the place, the genre and the occupation, and
    # gated as every modifier is: a state that names a garment needs that
    # garment worn. The pick is share ** 0.5, the same flattening every
    # slot draw uses, so the group's long tail is reachable.
    if level != "safe" and not ctx.get("nude") and out["outfit"]:
        _exp = [(t, w) for t, w in nudity_states(level, ctx.get("place"), ctx.get("genre"),
                                                 ctx.get("occupation"), whole=False, scene=scene_words(ctx))
                if ok(t) and t not in out["outfit"] and t not in out["modifiers"] and _names_worn(t)]
        if _exp and rng.random() < min(0.6, sum(w for _t, w in _exp)):
            _got = _wroll(rng, [(t, w ** 0.5) for t, w in _exp])
            if _got:
                out["modifiers"].append(_got)
    covers = tb.get("covers") or {}

    def _covers_part(t, part):
        """the garment covers the part: by its slot, or -- a rolled VARIANT
        ('hooded jacket', 'print shirt', 'open jacket' stand in for the jacket
        and the shirt and sit in no body slot) -- by the slot of the garment
        its head word names. 2026-09-19: a woman in a hooded jacket and jeans
        was told 'topless', and coverage never saw a variant top at all."""
        head = _stem(re.split(r"\s+(?:on|around|over)\s+", t)[0].split()[-1])
        # UNDERWEAR COVERS WHAT IT IS (2026-09-19): the lingerie slot states
        # 'breasts, crotch' for all of it, so a thong alone "covered" the
        # chest, nothing could uncover a chest a thong was blamed for, and the
        # thong was taken off -- a fifth of nsfw women start from the underwear
        # base, and one in eight ended in boots and gloves with no body garment
        if t in (slots.get("sexual") or {}):
            if re.search(r"\b(pant(y|ies)|thong|g-string|fundoshi|loincloth|briefs|boxers|bloomers|jockstrap|c-string)\b", t):
                return part == "crotch"
            if re.search(r"\b(bra|bandeau|pasties|nipple)\b", t):
                return part == "breasts"
        for sl, items in slots.items():
            if part not in (covers.get(sl) or []):
                continue
            if t in items or any(_stem(g.split()[-1]) == head for g in items):
                return True
        return False
    need = list(_NEEDS_SEEN.get(level, []))
    for part in need:
        blockers = [t for t in out["outfit"] if _covers_part(t, part)]
        if not blockers:
            out["uncovered"].append(part)
            continue
        # MEASURED (2026-09-15): the exposure that uncovers the part is drawn
        # by its share of the level's rating universe, lifted by the place,
        # the genre and the occupation; unmeasured options keep a floor weight
        _uw = dict(nudity_states(level, ctx.get("place"), ctx.get("genre"), ctx.get("occupation"), whole=False,
                                 scene=scene_words(ctx)))
        _floor_w = (min(_uw.values()) if _uw else 0.01) * 0.5
        opts_ = [(u, _uw.get(u, _floor_w)) for u in _uncover_options(part) if ok(u) and _names_worn(u)]
        # EACH STATE AT ITS OWN COUNT FOR THIS CAST AT THE LEVEL (the author,
        # 2026-10-03: "a lot of 'see-through clothes' ... too little other undress
        # / exposure options (check if the measurements are ok)"). They were not:
        # the weight was the state's share of the level's universe -- its 500
        # commonest tags -- so most garment states fell to the floor weight while
        # 'see-through clothes' kept a real one and fits any garment. The count
        # of '<state> <cast> rating:<band>' is each state's own measure
        # (spice_counts_levels.json); an uncounted state stands at the median.
        _base9 = _counted9(opts_)
        opts_ = _given_worn9(_base9, blockers)
        # (where the garment's own row decided the weights the flattening below is
        # not applied: it was ruled in so the long tail stayed reachable when one
        # state beat all the others, and the conditional weights are varied of
        # themselves -- flattened, 'see-through clothes' on a shirt at explicit
        # rose from a tenth of 'open shirt' to a third)
        _flat9 = 0.5 if opts_ is _base9 else 1.0
        # THE PART IS SHOWN BY A STATE OF THE GARMENT BEFORE BY ITS ABSENCE (the
        # author, 2026-09-19: "'panty pull' or 'panties aside' shows pussy in the
        # same way as the absence of the panties in first place"). A state that
        # names the worn garment keeps it on the body -- and with it everything
        # else the outfit can carry -- so it weighs double, and the garment is
        # dropped outright only when no state was drawn (.15, was .30).
        _gw0 = _garment_words()
        opts_ = [(u, w * (2.0 if any(_stem(x) in _gw0 for x in u.split()) and not u.startswith("no ") else 1.0))
                 for u, w in opts_]
        # THE GARMENT GOES ONLY WHEN NOTHING CAN SHOW THE PART THROUGH IT (the
        # author, 2026-09-19: "97 of 106 cases being topless is dull / no
        # diversity -- nsfw should promote nudity but in a varied way").
        # Measured on a plain nsfw brief, a third of the women lost the top and
        # nearly all of those read 'topless': it is the commonest nsfw chest
        # state on the booru, so by share it beat every other option put
        # together. While a state that keeps the garment is available, one of
        # those is drawn -- by the square root of its share, the flattening
        # every slot draw uses, so the long tail is reachable -- and the
        # garment-gone words wait for the outfit that offers nothing else.
        _keep = [(u_, w ** _flat9) for u_, w in opts_ if not _says_garment_gone(u_)]
        if _keep:
            opts_ = _keep
        u = _wroll(rng, opts_) if opts_ else None
        if u and (_keep or rng.random() < 0.85):
            out["modifiers"].append(u)
            # a modifier that names the worn garment opens it ('open
            # shirt'); a generic state ('topless female', 'no bra')
            # replaces it -- the hoodie comes off
            gw = _garment_words()
            # 'no shirt' names the shirt to say it is gone (2026-09-15:
            # 'collared shirt, no shirt' on one boy): a negation takes
            # the garment off like a generic state does
            # ONLY A STATE THAT SAYS THE GARMENT IS GONE TAKES IT OFF (2026-09-19,
            # traced on the author's question whether the drop rules multiply:
            # they did). Every state that named no garment word counted as
            # 'generic' and removed the covering garment -- 'open clothes',
            # 'clothes down', 'underboob', 'sideboob', 'low neckline',
            # 'accidental exposure' all stripped the top, so an nsfw woman came
            # out in a skirt and a scarf, or in nothing at all with no 'nude'
            # tag, and no clothes state had anything left to sit on. An opened,
            # lifted, pulled or slipped garment is still worn.
            if _says_garment_gone(u):
                for t in blockers:
                    out["outfit"].remove(t)
        else:
            for t in blockers:
                out["outfit"].remove(t)
            # a garment taken off is SAID (a body in a skirt and nothing else
            # carried no word for its bare chest)
            _gone = {"breasts": "topless female" if (ctx.get("kind") or "female") in ("female", "futanari") else "topless male",
                     "crotch": "bottomless"}.get(part)
            if _gone and ok(_gone) and _gone not in out["modifiers"]:
                out["modifiers"].append(_gone)
        out["uncovered"].append(part)
    # a modifier chosen before coverage may name a garment coverage took
    # off ('unzipped' on a boy left in leggings): pruned against the
    # final outfit
    out["modifiers"] = [m for m in out["modifiers"] if _names_worn(m)]
    # ONE RESERVED PICK FOR THE MILD SHIFTERS (the author, 2026-09-19). Measured:
    # 96% of dressed nsfw subjects already carry a state, but the slots go to
    # the states most common on nsfw posts (areola slip, bra pull), so 'skin
    # tight', 'strap gap' and 'button gap' almost never appeared there. Raising
    # the overall chance would only stack a third state; this draws one from
    # the shifters alone, by their measured share at the level, AFTER coverage (drawn before it, the
    # pick named garments coverage then took off and was pruned), through the
    # same gates as every modifier (the garment worn, the level, never on a
    # nude subject, never a second copy).
    if level != "safe" and not ctx.get("nude") and out["outfit"]:
        _mods = slots.get("modifier") or {}
        _nud = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
        # (sorted: a set's order changes from run to run, and the weighted pick
        # walks the list -- the same seed drew 'panties' once and 'see-through
        # panties' the next time)
        _sh = [(t, _floor_share(t, level)) for t in sorted(_spice_shifters())
               if (t in _mods or t in _nud) and ok(t) and t not in out["outfit"] and t not in out["modifiers"]
               and sm.SPICE_ORDER.get(((_mods.get(t) or _nud.get(t)) or {}).get("floor"), 99) <= sm.SPICE_ORDER.get(level, 0)
               and _of_level(t, level) and _names_worn(t)]
        _sh = [(t, w) for t, w in _sh if w > 0 and not t.startswith("naked ")]
        if _sh and rng.random() < 0.6:
            _got = _wroll(rng, [(t, w ** 0.5) for t, w in _sh])
            if _got:
                out["modifiers"].append(_got)
    # NSFW SHOWS IT (the author, 2026-09-19: "nsfw level is too tame right now ...
    # depending on the focused subject's gender the nsfw level should include
    # visible breast(s) / nipple(s) / pussy / penis either through undress or by
    # reveal tags (see-through / transparency also counts)"). The level's coverage
    # step drew states that open the clothes without showing anything ('open
    # clothes', 'cleavage', 'off shoulder'); from nsfw up a dressed subject now
    # carries at least ONE state in which the part is actually seen -- drawn by
    # its measured share at the level, through the same gates as every state --
    # and when the outfit offers none, the garment over the chest (a woman) or
    # the hips (a man) comes off and the word says so.
    if level in ("nsfw", "explicit") and not ctx.get("nude") and out["outfit"]:
        _k9 = ctx.get("kind") or "female"
        _nd9 = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
        _SEEN_F, _SEEN_M = seen_states("female"), seen_states("male")
        _seen9 = _SEEN_F if _k9 in ("female", "futanari") else _SEEN_M
        if _k9 == "futanari":
            # her chest is a woman's, her sex is a man's -- unless she is the one
            # being entered, and then her pussy is hers too (the author, 2026-09-21)
            _rcv9 = futa_receives(act=ctx.get("act"), acts=ctx.get("acts") or (),
                                  cast=ctx.get("cast"), things=ctx.get("things") or ())
            _seen9 = {t for t in _seen9 if sex_part_ok(t, "futanari", receiver=_rcv9)} | _SEEN_M
        _has9 = [m for m in out["modifiers"] + out["outfit"] if m in _seen9 or _is_whole_nudity(m)]
        # AN OPENED TOP SHOWS THE CHEST ITSELF (the author, 2026-10-03: "a lot of
        # 'see-through clothes'"). With nothing seen yet, the pick below added a
        # clothes state in which the part is seen -- and for a shirt that is most
        # often 'see-through clothes', so a third of nsfw outfits carried it. The
        # booru shows an opened top otherwise: of a lone woman's nsfw posts with
        # 'open shirt' 43% carry `nipples`, with 'open clothes' 40%, with 'shirt
        # lift' 54% -- and 'see-through clothes' 3-5%. So where a state already
        # opens or lifts what covers the chest, the part is seen through it at that
        # measured rate, said by the body's own word.
        if not _has9 and _k9 == "female" and ok("nipples"):
            _open9 = [m for m in out["modifiers"]
                      if re.search(r"^(?:open|unbuttoned|partially unbuttoned)\b|\b(?:lift|pull|down)$", m)
                      and not re.search(r"\b(?:skirt|pants?|shorts|panty|panties|pantyhose|bloomers|buruma|fly)\b", m)]
            # (.65: the level requires that SOMETHING is seen, so the rate is the
            # nipples' share among the ways an opened top shows the chest -- 43
            # against see-through clothes 4, areola slip 7, breasts out 7 -- not
            # their bare share of the posts)
            if _open9 and rng.random() < 0.65:
                out.setdefault("seen", []).append("nipples")
                _has9 = ["nipples"]
            # ... and at explicit the hips the same way: of a lone woman's explicit
            # posts in a dress 54% carry `pussy` and 6% 'see-through clothes'; in a
            # shirt 19,307 against 1,802. Where a state lifts, pulls, moves aside or
            # removes what covers the hips, the part is seen through it.
            if level == "explicit" and not _has9 and ok("pussy"):
                _hip9 = [m for m in out["modifiers"]
                         if re.search(r"\b(?:skirt|dress|clothes|kimono) lift$|\b(?:panty|pants|shorts|skirt|pantyhose) pull$"
                                      r"|\baside$|^no panties$|^bottomless$", m)]
                if _hip9 and rng.random() < 0.65:
                    out.setdefault("seen", []).append("pussy")
                    _has9 = ["pussy"]
        if not _has9:
            _uw9 = dict(nudity_states(level, ctx.get("place"), ctx.get("genre"), ctx.get("occupation"), whole=False,
                                      scene=scene_words(ctx)))
            _c9 = [(t, float(_uw9.get(t) or 0.002)) for t in sorted(_seen9)
                   if t in _nd9 and ok(t) and _of_level(t, level) and _names_worn(t)
                   and t not in out["modifiers"] and not _is_whole_nudity(t)]
            # THE BOORU'S OWN RATE DECIDES WHICH STATE (the author, 2026-10-03: "a lot of
            # 'see-through clothes' in nsfw prompts and too little other undress /
            # exposure options"). The pick was by a square-rooted share that gave
            # 'topless female' and 'bottomless' 2.3% each against 'see-through
            # clothes' 14.6% -- on a lone woman's nsfw posts they are 16.1%, 9.7% and
            # 14.0%; and 'see-through clothes' fits any outfit while the garment-named
            # states need their garment, so it won far more than its weight. Each
            # state now weighs its count for this cast at the level
            # (spice_counts_levels.json), an uncounted one the median -- flattened
            # by the square root as the coverage step flattens (the author,
            # 2026-09-19: varied, the long tail reachable).
            # (flattened here whatever decided the weights: the garment's row lists
            # one or two of these states and leaves the rest at its floor, so
            # unflattened the listed one -- 'see-through clothes' -- took a fifth)
            _c9 = _given_worn9(_counted9(_c9), list(out["outfit"]))
            _g9 = _wroll(rng, [(t, w ** 0.5) for t, w in _c9 if w > 0]) if _c9 else None
            if _g9:
                out["modifiers"].append(_g9)
            else:
                _part9 = "breasts" if _k9 in ("female", "futanari") else "crotch"
                out["outfit"] = [t for t in out["outfit"] if not _covers_part(t, _part9)]
    # A STATE OF THE CHEST'S OR THE HIPS' CLOTHES NEEDS CLOTHES THERE (2026-09-19,
    # traced: 'skirt, low neckline, topless' and 'plaid skirt, underboob,
    # topless' -- a neckline and an underboob on a body with no top). The
    # section says which part the state belongs to; a state that says the
    # garment is gone needs none.
    _nud_items = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
    _mod_items = slots.get("modifier") or {}

    def _part_of(m):
        sec = str(((_nud_items.get(m) or _mod_items.get(m)) or {}).get("section") or "")
        # the sections that describe CLOTHES OVER THE CHEST (underboob and
        # sideboob are what a top leaves seen; a neckline is a top's)
        # 'breasts out' is a state of CHEST CLOTHES too (the author, 2026-09-19:
        # it "implies there is something on the torso (clothes) like open shirt
        # and topless undermines that idea"): on a body in a thong alone it is
        # the wrong word and 'topless' the right one -- so it needs a garment
        # over the chest like every state of this family, and is pruned without
        if sec in ("clothes_for_breasts", "visible_parts_of_breasts", "whole_breasts_visible",
                   "shirtanddresscollardesigns", "exposed shoulders", "shoulders", "torso"):
            return "breasts"
        # an adopted state of covered breasts ('covered erect nipples', 'clothes
        # between breasts') needs clothing THERE, not anywhere (a g-string)
        _need, _alts = _adopted_requires(m)
        _cons = ((( _json_table(_ADOPTED, "adopted_tags.json") or {}).get("tags") or {}).get(m) or {}).get("constraints") or []
        if _need and "requires part:breasts" in _cons:
            return "breasts"
        p_ = _NUDITY_SECTION_PART.get(sec)
        return p_ if p_ in ("breasts", "crotch") else None

    def _covered(part):
        return any(_covers_part(t, part) for t in out["outfit"] + _canon_worn)
    out["modifiers"] = [m for m in out["modifiers"]
                        if _says_garment_gone(m) or not _part_of(m) or _covered(_part_of(m))]
    # WHAT IS BARE IS SAID (the author, 2026-09-19: with no 'topless', 'nipples'
    # or the like "the decision is left to the image generation model whether to
    # make the character topless or not and that is not ideal"). A dressed body
    # with nothing over the chest or the crotch, and no state that says so,
    # gets the word -- where the level allows it.
    if not ctx.get("nude") and (out["outfit"] or out["modifiers"]):
        _kind = ctx.get("kind") or "female"
        # danbooru has no bare 'topless' (0 posts): the state is 'topless female'
        # / 'topless male', and only a chest with breasts reaches this line
        for _part, _word in (("breasts", "topless female"), ("crotch", "bottomless")):
            if _part == "breasts" and _kind not in ("female", "futanari"):
                continue
            if _covered(_part):
                continue
            _says = set(_uncover_options(_part)) | {_word}
            if any(m in _says for m in out["modifiers"]) or {"nude", "completely nude"} & set(out["modifiers"]):
                continue
            if ok(_word) and _of_level(_word, level):
                out["modifiers"].append(_word)
    # THE STATE THAT NAMES THE GARMENT STANDS FOR IT (the author, 2026-09-19:
    # "'panties aside' is better than 'panties, panties aside' -- the tag
    # 'panties' pushes image gen models to draw panties in their initial state
    # and that dilutes the effect"). The outfit keeps the garment for every
    # rule that reads it; 'emit' is what the prompt carries.
    _nud_states = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
    _gw9 = _garment_words()
    _named = {_stem(w) for m in out["modifiers"] if m in _nud_states and not _says_garment_gone(m)
              for w in m.split() if _stem(w) in _gw9}
    # ... ONLY THE PURE GARMENT WORD (the author, same day: "'frilled skirt'
    # should stay ... only pure garment tags can be dropped entirely like plain
    # 'skirt'"): a tag that says what kind of garment it is carries what the
    # state does not. The descriptor is not moved into the state ('frilled
    # skirt lift'): that would write tags the booru does not have.
    # A WORN THING A RULING TIES TO A VISIBLE PART (the author, 2026-09-20: the
    # nipple jewellery "requires nipple(s) to be visible"): the finished picture
    # decides -- dressed, it goes
    _seen = {str(x).lower() for x in out["outfit"] + out["modifiers"]}
    _ctx_seen = {"bare": _seen, "nude": bool(ctx.get("nude")) or bool({"nude", "completely nude"} & _seen)}
    for _k in ("outfit", "modifiers"):
        out[_k] = [t for t in out[_k] if act_cast_ok(t, {"female": 1}, _ctx_seen)]
    # ONE THING PER POINT OF THE BODY (the author, 2026-09-21: "If user wants
    # some bdsm and add clamps on nipples generator atleast needs to know what
    # nipple clamps are and dont place other accssoris on nipples"). What the
    # user TYPED holds its point first; after that, the order the picture was
    # built in. The points are written for the typed-only vocabulary as much as
    # the rolled one (tools/build/write_anchors.py), which is his rule: a word
    # that is not rolled still has to be understood when he types it.
    _held = {}
    for _t in (ctx.get("typed") or ()):
        _a = anchor_of(_t)
        if _a:
            _held.setdefault(_a, str(_t).lower())
    for _k in ("outfit", "modifiers"):
        _keep = []
        for t in out[_k]:
            _a = anchor_of(t)
            if _a and _held.get(_a, t) != t:
                continue                  # that point is taken
            if _a:
                _held.setdefault(_a, t)
            _keep.append(t)
        out[_k] = _keep
    out["emit"] = [t for t in out["outfit"]
                   if not (_named and len(t.split()) == 1 and _stem(re.split(r"\s+(?:on|around|over)\s+", t)[0].split()[-1]) in _named)] \
        + list(out["modifiers"])
    return out


_BODY_TABLE = {"mtime": 0, "data": None}


def _body_table():
    """BODY / APPEARANCE table (data/library/body.json, tools/build/write_body.py)."""
    p = _paths.data("body.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _BODY_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _BODY_TABLE["data"] = json.load(f)
            _BODY_TABLE["mtime"] = mt
    except Exception:
        return None
    return _BODY_TABLE["data"]


def _lean_weights(items, *leans):
    """[(item, w)] with every lean multiplied in; 'none' in a lean raises
    the no-pick share the caller reads back"""
    w = {t: 1.0 for t in items}
    none = 1.0
    for ln in leans:
        for t, x in (ln or {}).items():
            if t == "none":
                none *= float(x)
            elif t in w:
                w[t] *= float(x)
    return [(t, x) for t, x in w.items()], none


_THEMATIC_LIFT = 4.0        # a trait is the scene's when the scene quadruples its rate
                            # (reading + muscular female: 43 posts, x3 of a tiny base -- noise)


def _place_lift(place, tag, level=None):
    """P(tag | place) / P(tag | the level's universe), from the whole-related
    places harvest (place_related.json 'places_all') and live-cached
    counts; None when either side is unmeasured"""
    if not place or not tag:
        return None
    try:
        pa = _place_row("places_all", place)
        f = pa.get(str(tag).lower())
        if f is None:
            return None
        base = _level_base(tag, level)
        if not base:
            return None
        return min(8.0, float(f) / float(base))
    except Exception:
        return None


def _thematic_items(pairs, ctx, lean=None):
    """-> ([(item, weight * lift)], max lift) for the items the scene
    lifts to _THEMATIC_LIFT or more: the act (live pair counts), the place
    (whole-related harvest) or a stated lean (a warrior's muscles)"""
    # the base is the whole 1girl universe, not the level's: a trait's
    # thematic-ness is not a rating's ('long legs' beside sleeping read
    # x4 against the general-rated base, x1.6 against all posts)
    act, place, level = ctx.get("act"), ctx.get("place"), None
    out, mx = [], 0.0
    for t, w in pairs:
        lifts = [float((lean or {}).get(t) or 0)]
        if act:
            try:
                # CACHE ONLY (the standing rule of 2026-09-22, broken here until
                # 2026-10-03: every body trait of every slot asked the booru for its
                # pair with the act -- 97 requests and 134 s for one prompt beside
                # 'mixed-sex bathing', and a release test that waits on danbooru). A
                # pair never measured here is neutral: no thematic lift.
                lv = act_lift(act, t, level, view=ctx.get("view"), live="cache")
                # ...on a pair the booru has actually seen (_PAIR_MIN_POSTS)
                _n = _count_in_cache(_q(act) + " " + _q(t))
                if _n is not None and _n < _PAIR_MIN_POSTS:
                    lv = None
            except Exception:
                lv = None
            if lv is not None:
                lifts.append(float(lv))
        pl = _place_lift(place, t, level)
        if pl is not None:
            lifts.append(pl)
        lift = max(lifts)
        if lift >= _THEMATIC_LIFT:
            out.append((t, w * lift))
            mx = max(mx, lift)
    return out, mx


_RACE_REL = {"mtime": 0, "data": None}
_FURRY_TABLE = {"mtime": 0, "data": None}
_ANCHORS = {"mtime": 0, "data": None}


def anchor_of(tag):
    """-> the point of the body a thing sits on ('nipple clamps' -> nipples,
    'choker' -> neck), or None when it sits on no single point. Written by
    tools/build/write_anchors.py for the typed-only vocabulary as much as the
    rolled one -- the author, 2026-09-21: "the typed only (not denied) stuff
    should always be measured and be ready to use"."""
    tb = _json_table(_ANCHORS, "anchors.json") or {}
    e = (tb.get("things") or {}).get(str(tag or "").lower())
    return (e or {}).get("anchor")


def anchors_taken(tags):
    """-> {anchor: the word already holding it}"""
    out = {}
    for t in (tags or ()):
        a = anchor_of(t)
        if a:
            out.setdefault(a, str(t).lower())
    return out

# THE ANTHRO FORM (the author, 2026-09-21: "they are different class of subjects
# - so basically all the rules that apply just translate to them - only that
# they have specialized appearance"). A furry is not a race -- the race
# machinery already composes any '<animal> girl' with its own ears and tail. It
# is a FORM the same subject takes, and only the appearance is its own: fur over
# the body, an animal nose or a snout, paw pads, animal hands, digitigrade legs.
# Everything else -- poses, clothes, acts, spice, the focus policy, the gender
# lean -- is the generator's as it stands.
FURRY_WORDS = re.compile(r"\b(?:furry|furries|anthro|anthropomorphic|kemono)\b", re.I)


def furry_table():
    return _json_table(_FURRY_TABLE, "furry.json") or {}


def furry_form(base, tags=()):
    """-> True when the picture is asked for in the anthro form (typed only:
    nothing rolls a furry subject by itself)"""
    if FURRY_WORDS.search(str(base or "")):
        return True
    return any(FURRY_WORDS.search(str(t or "")) for t in (tags or ()))


def furry_cast_word(kind):
    """the count word the form adds beside 1girl / 1boy / 1futa (a futanari
    furry is the booru's 'furry female' plus 'futanari' -- 1,082 posts)"""
    tb = furry_table()
    return list(((tb.get("cast") or {}).get(str(kind or "female").lower()) or []))


def furry_pairing(kinds):
    """'furry with furry' when they all are, 'furry with non-furry' when the
    picture mixes them; nothing for one subject"""
    ks = [bool(k) for k in (kinds or ())]
    if len(ks) < 2:
        return None
    tb = (furry_table().get("pairing") or {})
    return tb.get("all") if all(ks) else (tb.get("mixed") if any(ks) else None)


def furry_parts(kind, rng, allowed=None, taken=(), cap=4, species=None):
    """-> the anthro body words for this subject, each drawn at its own
    measured share (P(part | the form's posts)), the commonest first, up to
    `cap`. A part the species already gave ('cat ears' on a cat girl) is
    skipped by `taken`, a fur colour comes at most once, and ANOTHER ANIMAL'S
    part is never drawn ('cat ears' on a wolf: 2026-09-21) -- where the picture
    names its animal, only that animal's words and the form's own are used."""
    row = ((furry_table().get("parts") or {}).get(str(kind or "female").lower()) or {})
    sp = re.sub(r"\b(girl|boy|woman|man|female|male|anthro|furry)\b", " ",
                str(species or "").lower()).strip()
    out, have_fur = [], False
    for t, share in sorted(row.items(), key=lambda kv: -kv[1]):
        # a word that names an animal: only this picture's animal may say it
        _m = re.match(r"^([a-z' -]+?) (ears|tail|nose|snout|paws|ear fluff)$", t)
        if _m and _m.group(1) not in ("animal", "extra", "fake"):
            if not sp or _m.group(1) != sp:
                continue
        if len(out) >= cap:
            break
        if t in (taken or ()) or t in out:
            continue
        if allowed is not None and not allowed(t):
            continue
        is_fur = t.endswith(" fur")
        if is_fur and have_fur:
            continue
        if rng.random() < float(share):
            out.append(t)
            have_fur = have_fur or is_fur
    return out
_UNROLLED_RULINGS = {"mtime": 0, "data": None}


def is_overlay_race(race):
    """a race that can be laid over another (the author's ruling, 2026-09-19,
    unrolled_rulings.json 'overlay_races'): robot, android, cyborg, skeleton,
    ghost, zombie, vampire, doll, slime -- by the race word or its head
    ('robot girl', 'ghost')"""
    ov = set((((_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("overlay_races") or {})
              .get("overlays")) or [])
    r = str(race or "").lower()
    return r in ov or any(w in ov for w in r.split())




def race_parts_related(race, least=0.20, level=None):
    """-> [parts] the RACE'S OWN POSTS carry, commonest first (P(part | race),
    tools/build/harvest_race_parts.py): a deer girl's antlers .77, a dragon
    girl's horns and tail, a harpy's winged arms, a lamia's scales. The generic
    word yields to the specific one beside it ('animal ears' to 'deer ears',
    'horns' to 'antlers' or 'sheep horns', 'tail' to 'cow tail'), and what
    every body has (nipples, collarbone) is not a race's."""
    # ... per spice band, on solo posts (2026-09-19 rule): a cat girl's ears are
    # in .95 of her explicit posts and her tail in .74; a lamia's scales .27 at
    # safe and .49 at explicit. The band's own row first, then the all-ratings one.
    tb = (_json_table(_RACE_REL, "race_parts_related.json") or {}).get("races") or {}
    rel = (level_row("race_parts_levels.json", race, level) if level else {}) or tb.get(str(race or "").lower()) or {}
    fams = (_body_table() or {}).get("families") or {}
    human = set()
    for f in ("parts", "parts_lower", "breast_desc", "ass_desc", "face_desc"):
        human |= set((fams.get(f) or {}).get("items") or {})
    human |= {"breasts", "nipples", "collarbone", "tongue", "navel", "thighs"}
    # (.20: a lamia's slit pupils are .24; another race's word is not a part)
    got = [t for t, f in sorted(rel.items(), key=lambda kv: -kv[1])
           if f >= least and t not in human and t not in tb and t not in ("monster girl", "taur", "pale skin")]
    for generic, tail in (("animal ears", " ears"), ("tail", " tail"), ("horns", " horns"), ("wings", " wings")):
        if generic not in got:
            continue
        if any(t != generic and (t.endswith(tail) or (generic == "horns" and t == "antlers")) for t in got):
            got.remove(generic)
            continue
        # the specific word under the line still names the part better than the
        # generic one over it ('deer tail' .19 beside 'tail' .25)
        spec = [t for t, f in sorted(rel.items(), key=lambda kv: -kv[1])
                if t != generic and t.endswith(tail) and f >= 0.10 and t not in human]
        if spec:
            got[got.index(generic)] = spec[0]
    return list(dict.fromkeys(got))


def body_for(ctx, rng, allowed=None):
    """-> {"parts": [], "build": tag|None, "skin": tag|None, "marks": [],
    "states": [], "hair_lean": {}, "eyes_lean": {}, "breast_lean": {}}
    for one subject. ctx: kind, race, occupation, activity, activity_state,
    activity_clothes, event, weather, season, place, place_class, genre,
    level. Order (the author's): gender, race, occupation, activity, event,
    weather and place, spice. No age is ever rolled."""
    tb = _body_table() or {}
    kind = ctx.get("kind") or "female"
    fem = kind in ("female", "futanari")
    race = str(ctx.get("race") or "").lower()
    ok = (lambda t: allowed is None or allowed(t))
    out = {"parts": [], "build": None, "skin": None, "marks": [], "states": [],
           "hair_lean": {}, "eyes_lean": {}, "breast_lean": {}}
    # 1. race parts, fixed
    rp = (tb.get("race_parts") or {}).get(race) or {}
    out["parts"] = [t for t in (rp.get("parts") or []) if ok(t)]
    if rp.get("parts_any"):
        pk = rng.choice(rp["parts_any"])
        if ok(pk):
            out["parts"].append(pk)
    if rp.get("parts_male") and not fem:
        out["parts"] += [t for t in rp["parts_male"] if ok(t)]
    if rp.get("maybe") and rng.random() < 0.5:
        pk = rng.choice(rp["maybe"])
        if ok(pk):
            out["parts"].append(pk)
    # A TYPED ANIMAL RACE THE TABLE DOES NOT ROLL (the author, 2026-09-19: "parts
    # like lion tail / ears should roll for them"): a 'lion girl', a 'raccoon
    # girl' or a 'squirrel boy' came out with the race word and a human body.
    # The race names its animal, and the race-parts family holds that animal's
    # parts ('lion ears', 'lion tail', 'lion mane'): the ears and the tail
    # first, then the commonest other part, each through the level gate.
    # WHAT THE RACE'S OWN PICTURES CARRY (the author, 2026-09-19: "'deer girl' -
    # antlers? and other horns for other races. scale skin / fur and so on"):
    # beside the stated parts, up to four in all, for a rolled race and a typed
    # one alike
    for _t in race_parts_related(race, level=ctx.get("level")):
        if len(out["parts"]) >= 4:
            break
        # a generic word adds nothing beside the stated specific one ('tail'
        # beside 'snake tail')
        if _t in ("tail", "horns", "animal ears", "wings") and any(x.endswith(" " + _t.split()[-1]) or
                                                                     (_t == "horns" and "horn" in x) for x in out["parts"]):
            continue
        if _t not in out["parts"] and _floor_measured(_t) and ok(_t):
            out["parts"].append(_t)
    if not out["parts"] and not (tb.get("race_parts_measured") or {}).get(race):
        _m = re.match(r"^([a-z' -]+?) (girl|boy|woman|man)$", race)
        if _m:
            _animal = _m.group(1)
            _rpf = ((tb.get("families") or {}).get("race_parts") or {}).get("items") or {}
            _own = sorted((t for t in _rpf if t.split()[0] == _animal.split()[-1] and len(t.split()) > 1),
                          key=lambda t: (not t.endswith((" ears", " tail")), -int((_rpf[t] or {}).get("posts") or 0)))
            out["parts"] += [t for t in _own[:3] if ok(t) and t not in out["parts"]]
    # THE OVERLAY'S OWN PARTS, beside the typed race's (a robot cat girl: cat
    # ears, cat tail, robot joints): the overlay's word and up to two of its parts
    _ov = str(ctx.get("race_overlay") or "").lower()
    if _ov:
        _orp = (tb.get("race_parts") or {}).get(_ov) or {}
        for _t in [_ov] + list(_orp.get("parts") or [])[:2] + race_parts_related(_ov, level=ctx.get("level"))[:1]:
            if _t and _t not in out["parts"] and ok(_t):
                out["parts"].append(_t)
    pinned_skin = rp.get("skin") or (rng.choice(rp["skin_any"]) if rp.get("skin_any") else None)
    # 2. build
    bd = tb.get("build") or {}
    lean = bd.get("lean") or {}
    items = bd.get("female" if fem else "male") or []
    pairs, none_x = _lean_weights(items, (lean.get("occupation") or {}).get(ctx.get("occupation")),
                                  (lean.get("race") or {}).get(race),
                                  (lean.get("activity_clothes") or {}).get(ctx.get("activity_clothes")))
    # MEASURED (2026-09-13): the build's share is how often a 1girl post
    # carries any build tag, each item's weight its own share ('tall
    # female' .0004 beside 'curvy' .008) -- the marks rule, for every body
    # family; the stated .50 stands only in an unmeasured table
    _bw = bd.get("weights") or {}
    if _bw:
        _bmin = min(float(x) for x in _bw.values())
        pairs = [(t, w * float(_bw.get(t, _bmin))) for t, w in pairs]
    _bs = bd.get("share", 0.5)
    if isinstance(_bs, dict):
        _bs = _bs.get("female" if fem else "male", 0.5)
    share = float(_bs) / max(1.0, none_x)
    # A BUILD ROLLS ONLY WHERE THE SCENE LIFTS IT (2026-09-14): the act,
    # the place or the job must raise the item to twice its usual rate
    # (boxing: muscular female x8; a gym; a warrior's lean); the family
    # then rolls at its measured share times that lift, among the
    # thematic items. Nothing lifts a build here: nothing rolls.
    _occ_lean = (lean.get("occupation") or {}).get(ctx.get("occupation")) or {}
    _act_lean = (lean.get("activity_clothes") or {}).get(ctx.get("activity_clothes")) or {}
    pairs, _mx = _thematic_items(pairs, ctx, {**_occ_lean, **_act_lean})
    if pairs and rng.random() < min(0.5, share * _mx):
        out["build"] = _wroll(rng, [(t, w) for t, w in pairs if ok(t)])
    # 3. breasts lean (the engine's roll reads it)
    if fem:
        out["breast_lean"] = (tb.get("breast_lean") or {}).get(race) or {}
    # 4. skin
    sk = tb.get("skin") or {}
    if pinned_skin and ok(pinned_skin):
        out["skin"] = pinned_skin
    elif rng.random() < float(sk.get("share", 0.25)):
        ln = {}
        if ctx.get("season") == "summer" and ctx.get("place_class") == "water":
            ln = (sk.get("lean") or {}).get("summer_beach") or {}
        pairs, _ = _lean_weights(sk.get("natural") or [], ln)
        out["skin"] = _wroll(rng, [(t, w) for t, w in pairs if ok(t)])
    # 5. marks -- the share measured (P(any mark | 1girl), write_body), the
    # items by their own shares, all scaled by the DETAIL control (the author's
    # 2026-09-06: marks belong to 'detailed'; standard describes the
    # defining parts)
    scale = float(ctx.get("detail", 1.0))
    _act = ctx.get("act")
    _lvl = ctx.get("level")

    def _by_act(pairs):
        """the act's measured lifts on a candidate list: under .6 the item
        is out of the picture (a blowjob describes no pussy), else weighted"""
        if not _act:
            return list(pairs)
        out_ = []
        for t, w in pairs:
            lf = act_lift(_act, t, _lvl, view=ctx.get("view"))
            if lf is not None and lf < 0.6:
                continue
            out_.append((t, w * (lf or 1.0)))
        return out_
    mk = tb.get("marks") or {}
    # MARKS ARE THEMATIC TOO (2026-09-14): a scar on a pirate or a boxer,
    # a tattoo where the act or the place shows them; the family rolls at
    # its measured share times the lift, among the lifted items only
    _mk_lean = ((mk.get("lean") or {}).get("occupation") or {}).get(ctx.get("occupation")) or {}
    _mk_pairs, _ = _lean_weights(mk.get("items") or [], _mk_lean)
    _mw = mk.get("weights") or {}
    _mk_pairs = _by_act([(t, w * float(_mw.get(t, 1.0))) for t, w in _mk_pairs])
    # A MARK WITH A STORY NEEDS THE STORY (the author, 2026-09-19): a scar, a
    # bandage, an eyepatch, stitches or a tattoo was lifted by any act or place
    # that happened to co-occur with it, so a cafe visitor wore a scar. They
    # now roll only for a job or a genre that carries them -- stated (a
    # pirate's eyepatch) or measured from the mark's own posts; moles,
    # freckles, piercings and glasses keep the scene's lifts.
    _story = re.compile(r"\b(scars?|bandages?|bandaid|eyepatch|stitches|tattoo|burn|bruise)\b")
    _mlm = mk.get("lean_measured") or {}

    def _carried(t):
        if t in _mk_lean:
            return True
        for _w, _gs in (mk.get("story_genres") or {}).items():
            if _w in t and (ctx.get("genre") in _gs or ctx.get("bucket") in _gs):
                return True
        e = _mlm.get(t) or {}
        return (ctx.get("occupation") in (e.get("occupation") or {})
                or ctx.get("genre") in (e.get("genre") or {}) or ctx.get("bucket") in (e.get("genre") or {}))
    _mk_pairs = [(t, w) for t, w in _mk_pairs if not _story.search(t) or _carried(t)]
    # A STATED OR RULED LEAN IS THEMATIC WHATEVER ITS SIZE (2026-09-19, traced: a
    # pirate rolled no mark in 300 bodies -- the scene's-own line is a lift of
    # four, her scar and eyepatch are stated at three, so only a warrior's scar
    # (4) and a yakuza's back tattoo (5) ever passed)
    _mk_lean = {t: max(float(v), _THEMATIC_LIFT) for t, v in _mk_lean.items()}
    _by_world = {t for t, _w in _mk_pairs if _story.search(t) and t not in _mk_lean}
    _mk_lean.update({t: _THEMATIC_LIFT for t in _by_world})
    _mk_pairs, _mk_mx = _thematic_items(_mk_pairs, ctx, _mk_lean)
    # the WORLD allows its marks, it does not multiply them: only a job's own
    # lean (or the act's, the place's) raises the chance above the measured share
    _mk_share = float(mk.get("share", 0.3))
    if _mk_pairs and all(t in _by_world for t, _w in _mk_pairs):
        # ... at THEIR OWN measured rate (the family's .18 is mostly glasses
        # and moles; the scars and tattoos of a war-torn world are a few
        # percent of its people, not two in five)
        _mk_mx = 1.0
        _mk_share = min(_mk_share, sum(float(_mw.get(t) or 0.005) for t, _w in _mk_pairs))
    if _mk_pairs and scale > 0 and rng.random() < min(0.5, _mk_share * scale * _mk_mx):
        pairs = _mk_pairs
        n = 2 if rng.random() < float(mk.get("two_share", 0.25)) else 1
        for _ in range(n):
            pk = _wroll(rng, [(t, w) for t, w in pairs if ok(t) and t not in out["marks"]])
            if pk:
                out["marks"].append(pk)
    # 6. states, only with a cause
    st = tb.get("states") or {}
    cands = []
    for t, need in (st.get("items") or {}).items():
        hit = False
        # a state of a part needs the part bare and in the frame ('dirty feet')
        if not all(ctx.get(n) for n in (need.get("_needs") or [])):
            continue
        # (a bed makes a lone girl 'sleepy'; not one in the middle of an act, 2026-09-20)
        if t == "sleepy" and ctx.get("act"):
            continue
        # AN AURA IS THE MOOD'S (the author, 2026-09-20: 'dark aura' contradicted an
        # embarrassed dancer -- the genre alone had drawn it): the mood lists the
        # auras its pictures carry, and nothing else may add one
        if t.endswith(" aura") or t == "aura":
            if t not in (ctx.get("mood_auras") or []):
                continue
        for key, vals in need.items():
            if key == "_needs":
                continue
            if key == "season_place":
                hit = ("%s:%s" % (ctx.get("season"), ctx.get("place_class"))) in vals
            elif key == "level":
                hit = ctx.get("level") in vals
            else:
                hit = str(ctx.get(key) or "") in vals
            if hit:
                break
        if hit and ok(t) and part_gate(t, {"genre": ctx.get("genre"), "kind": ctx.get("kind"),
                                           "race": race, "level": ctx.get("level"), "pose": ctx.get("pose"),
                                           "act": ctx.get("act"), "place": ctx.get("place")}):
            cands.append((t, 1))
    rng.shuffle(cands)
    for t, _w in cands[:int(st.get("max", 2))]:
        if rng.random() < 0.6:
            out["states"].append(t)
    # 8. hair / eyes leans by race
    hl = (tb.get("hair_eye_lean") or {}).get(race) or {}
    out["hair_lean"], out["eyes_lean"] = hl.get("hair") or {}, hl.get("eyes") or {}
    # 9. THE TREE'S FAMILIES (2026-09-04): every descriptor family the
    # body groups' sections give, rolled by its share among items with a
    # MEASURED floor the gate allows; a family that another slot already
    # rolls (hair colour / length / front / style, eye colour, breast size)
    # is left to that slot. Measured race parts join the stated ones.
    fams = tb.get("families") or {}
    rpm = ((tb.get("race_parts_measured") or {}).get(race) or {})
    # ... ordered by HOW MANY of the race's pictures carry the part (the part's
    # share with the race x the part's posts), not by the share alone
    # (2026-09-19: a rare part is almost always on one race, so 'mechanical
    # tail' .4 and 'robot ears' outranked 'cat ears' and a typed cat girl came
    # out with a mechanical tail and no cat ears)
    _fam_posts = {f_t: int((f_c or {}).get("posts") or 0)
                  for f_v in fams.values() for f_t, f_c in ((f_v or {}).get("items") or {}).items()}
    for t in sorted(rpm, key=lambda x: -(rpm[x] * max(_fam_posts.get(x, 0), 1)))[:3]:
        # the generic word adds nothing beside the race's own ('animal ears',
        # 'extra ears' beside 'cat ears')
        if t in _RACE_GENERIC and any(x.endswith((" ears", " tail")) for x in out["parts"]):
            continue
        if t not in out["parts"] and _floor_measured(t) and ok(t):
            out["parts"].append(t)
    # every race marker anywhere in the table (stated or measured) never
    # rolls as a plain descriptor: a tail is a race's, not the dice's
    _race_marks = set()
    for rp_ in (tb.get("race_parts") or {}).values():
        for k in ("parts", "parts_any", "parts_male", "maybe", "skin_any"):
            _race_marks.update(rp_.get(k) or [])
        if rp_.get("skin"):
            _race_marks.add(rp_["skin"])
    for v in (tb.get("race_parts_measured") or {}).values():
        _race_marks.update(v.keys())
    _race_marks.update((fams.get("race_parts") or {}).get("items") or {})
    _race_marks.update({"tail", "scales", "fins", "horns", "wings", "fangs", "claws", "halo"})
    # A BODY STATE OF A PART THE PICTURE CARRIES (2026-09-14): one per
    # subject, at P(state | part) scaled by the detail control -- sharp
    # teeth beside an open mouth's teeth, skindentation beside thighs,
    # animal ear fluff beside animal ears
    out["state"] = None
    _sbp = tb.get("states_by_part") or {}
    if _sbp and scale > 0:
        _carried = set(out["parts"]) | set(ctx.get("parts") or ()) | {"_always"}
        # the feet's details only when the feet are the focus of the composition
        # (the author, 2026-09-19), bare or in open footwear
        _carried.discard("feet")
        if ctx.get("foot_focus") and ctx.get("barefoot"):
            _carried.add("feet")
        # (an ANIMAL's ears: 'cat ears', 'animal ears' -- a human's 'ears' part drew
        # 'animal ear fluff' onto a plain girl, 2026-09-19)
        # ... and an ELF's ears are nobody's animal ears (the author's bug hunt,
        # 2026-09-21: an elf knight wore 'animal ear fluff' beside 'long pointy
        # ears'). The word before 'ears' has to name an animal, and the booru
        # itself says which words do: an animal has a tail tag of its own.
        def _animal_ear(p_, _seen=None):
            # the BOORU's own implications say which ears are an animal's:
            # 'cat ears' -> animal ears, 'long pointy ears' -> pointy ears and
            # no further. No word list, no guess.
            p_ = str(p_).lower()
            if p_ == "animal ears":
                return True
            if not p_.endswith(" ears"):
                return False
            _seen = _seen or set()
            _im = _implications()
            _q = list(_im.get(p_) or ())
            while _q:
                x = str(_q.pop()).lower()
                if x == "animal ears":
                    return True
                if x in _seen:
                    continue
                _seen.add(x)
                _q += list(_im.get(x) or ())
            return False
        if any(_animal_ear(p) for p in _carried):
            _carried.add("animal ears")
        _cands = []
        for p in _carried:
            for t, c in (_sbp.get(p) or {}).items():
                if ok(t) and _floor_measured(t):
                    if c.get("focus_only"):
                        # the details of the part the composition is ABOUT: each at
                        # its share of the focus tag's posts (foot focus: toes .90,
                        # soles .82, toenails .30, toenail polish .16)
                        if rng.random() < float(c.get("share") or 0):
                            out["states"].append(t)
                        continue
                    _cands.append((t, float(c.get("share") or 0)))
        if _cands:
            _cands.sort(key=lambda kv: -kv[1])
            _r = rng.random() / max(scale, 1e-6)
            for t, sh in _cands:
                if _r < sh:
                    out["state"] = t
                    break
                _r -= sh
    # ONE MEMBER PER EXCLUSIVE FAMILY (2026-09-14: an alien with orange,
    # red and grey skin -- the race's pinned skin beside its measured
    # parts): among the parts, the skin and the descriptors, the first of
    # a skin colour, a hair colour, an eye colour or a breast size stands
    _fams = tb.get("families") or {}
    _excl = {}
    for _fk in ("skin", "skin_unnatural", "hair_color", "eye_color", "breast_size", "hair_length"):
        for _t in ((_fams.get(_fk) or {}).get("items") or {}):
            _excl[str(_t).lower()] = "skin" if _fk.startswith("skin") else _fk
    for _t in (sk.get("natural") or []):
        _excl[str(_t).lower()] = "skin"
    _seen_fam = set()
    if out["skin"]:
        _seen_fam.add(_excl.get(out["skin"], "skin"))
    _kept = []
    for _t in out["parts"]:
        _f = _excl.get(str(_t).lower()) or ("skin" if str(_t).lower().endswith(" skin") else None)
        if _f and _f in _seen_fam:
            continue
        if _f:
            _seen_fam.add(_f)
        _kept.append(_t)
    out["parts"] = _kept
    out["desc"] = []
    for fam, ent in fams.items():
        if fam in ("hair_color", "hair_length", "hair_front", "hair_style", "eye_color",
                   "breast_size", "attire", "spice", "spice_expression", "race_parts",
                   "hair_misc", "parts", "parts_lower", "skin", "skin_unnatural", "skin_desc",
                   "face_desc"):
            # face_desc too (the author's 2026-09-14): 'no nose', 'dot nose',
            # 'long nose' are too niche for the anime register -- typed only
            # 'parts' too (2026-09-13): a visible part ('breasts' .51 of
            # 1girl posts, 'collarbone', 'thighs') is the framing's and the
            # focus's to name (fastplan's focused-part descriptors), not a
            # trait; measured, the family fired on half the subjects
            continue
        if fam in ("breast_desc",) and not fem:
            continue
        if fam == "wing_desc" and not any("wing" in p for p in out["parts"]):
            continue
        # THE FAMILY ROLLS AT ITS RATE INSIDE THE LEVEL (the author, 2026-09-19: "I've
        # almost never seen these tags used ... at spice starting from nsfw their
        # descriptions are very welcome (especially for the exposed / focused parts)"
        # -- and: "are you measuring in every spice band separately?"). It was one
        # share over all posts: the breast descriptions are .005 of every picture of
        # a girl, but .0002 of the safe ones and .05 of the nsfw and explicit ones.
        # The writer now measures each family per rating; where the part is bare
        # the rate triples (a stated lean: the booru has no tag for "the focus").
        _lift9 = 1.0
        _ll9 = (ent.get("level_lift") or {}).get(str(ctx.get("level") or "safe"))
        _ch9 = float(ent.get("share") or 0) * scale * float(_ll9 if _ll9 is not None else 1.0)
        if fam in ("breast_desc", "ass_desc") and str(ctx.get("level")) in ("nsfw", "explicit"):
            _lift9 = 3.0 if ctx.get("nude") else 2.0
            _ch9 *= _lift9
        if scale <= 0 or rng.random() >= min(0.6, _ch9):
            continue
        # an item's weight is its measured share of 1girl posts (the
        # writer measures it); the count's fourth root only where unmeasured
        items = [(t, float(c.get("share") or 0) or float(c.get("posts") or 1) ** 0.25)
                 for t, c in (ent.get("items") or {}).items()
                 if _floor_measured(t) and ok(t) and t not in _race_marks
                 and t not in ruled_out()
                 and part_gate(t, {"race": race, "race_overlay": ctx.get("race_overlay"),
                                   "genre": ctx.get("genre"), "level": ctx.get("level"),
                                   "style": ctx.get("style"), "kind": ctx.get("kind"),
                                   "pose": ctx.get("pose"), "act": ctx.get("act"),
                                   "poses": ctx.get("poses"), "weather": ctx.get("weather"),
                                   "place": ctx.get("place")})]              # (the families never read the rulings: 'lactation' rolled)
        if _lift9 > 1.0:
            items = [(t, w ** 0.5) for t, w in items]   # the long tail is reachable ('sagging breasts' took half the draws)
        pick = _wroll(rng, _by_act(items))
        if pick:
            out["desc"].append(pick)
    return out


_EVENT_TABLE = {"mtime": 0, "data": None}


def _events_table():
    """EVENTS table (data/library/events.json, tools/build/write_events.py)."""
    p = _paths.data("events.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _EVENT_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _EVENT_TABLE["data"] = json.load(f)
            _EVENT_TABLE["mtime"] = mt
    except Exception:
        return None
    return _EVENT_TABLE["data"]


def _event_allowed(ent, leaf, bucket):
    g = ent.get("genres") or []
    return "*" in g or leaf in g or bucket in g


def resolve_event(base, leaf, rng):
    """-> (mode, name, entry) or (None, None, None). TYPED FIRST: the event's
    name or an alias in the prompt (a typed event never yields, whatever
    the genre); else a rare roll (share by bucket) among the events the
    genre allows, weighted. An event is an occasion over a genre, never a
    genre (the author's 2026-09-04)."""
    tb = _events_table() or {}
    events = tb.get("events") or {}
    if not events:
        return None, None, None
    low = " " + re.sub(r"[^a-z0-9' ]+", " ", (base or "").lower()) + " "
    best = None
    for name, ent in events.items():
        for form in [name] + list(ent.get("aliases") or []):
            f = re.sub(r"[^a-z0-9' ]+", " ", form.lower()).strip()
            pos = low.find(" " + f + " ")
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, name)
    if best:
        return "typed", best[1], events[best[1]]
    bucket = (_genre_pool().get(leaf) or {}).get("bucket") if leaf else None
    share = (tb.get("share") or {})
    if rng.random() >= float(share.get(bucket, share.get("default", 0.04))):
        return None, None, None
    cands = [(n, float(e.get("weight") or 1.0)) for n, e in events.items()
             if _event_allowed(e, leaf, bucket)]
    name = _wroll(rng, cands)
    if not name:
        return None, None, None
    return "rolled", name, events[name]


_ACT_TABLE = {"mtime": 0, "data": None}


def _activity_table():
    """ACTIVITY table (data/library/genre_activities.json, written by
    tools/build/write_genre_activities.py)."""
    p = _paths.data("genre_activities.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _ACT_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _ACT_TABLE["data"] = json.load(f)
            _ACT_TABLE["mtime"] = mt
    except Exception:
        return None
    return _ACT_TABLE["data"]


def activity_entry(leaf, place):
    """-> (activities, none share) for this leaf x place, or None when the
    leaf has no table (then the old self-action draw stands)."""
    tb = _activity_table() or {}
    lf = (tb.get("leaves") or {}).get(leaf)
    if not lf:
        return None
    cls = (lf.get("place_class") or {}).get(place)
    ent = (lf.get("classes") or {}).get(cls) if cls else None
    roll = list((ent or {}).get("roll") or [])
    roll += [t for t in (tb.get("place_extra") or {}).get(place, []) if t not in roll]
    if not roll:
        return [], 0.55
    return roll, float((ent or {}).get("none", 0.45))


def activity_tag(name):
    """-> the tag an activity emits, or None for a prose-only one (the
    booru has no tag for forging, hunting, sparring...)"""
    reg = (_activity_table() or {}).get("registry") or {}
    ent = reg.get(str(name or "").lower())
    if ent is None:
        return name or None
    return name if ent.get("tag", True) else None


_MEDIAN_ACT = {"mtime": 0, "v": None}


def _median_activity_share():
    """the median, over the measured places, of the summed related share
    of the registry's activities there -- what an unmeasured place takes"""
    tb = _activity_table() or {}
    if _MEDIAN_ACT["v"] is None or _MEDIAN_ACT["mtime"] != _ACT_TABLE.get("mtime"):
        per = {}
        for t, v in (tb.get("registry") or {}).items():
            for p, f in (v.get("places") or {}).items():
                per[p] = per.get(p, 0.0) + float(f or 0.0)
        vals = sorted(per.values())
        _MEDIAN_ACT["v"] = min(0.95, vals[len(vals) // 2]) if vals else 0.35
        _MEDIAN_ACT["mtime"] = _ACT_TABLE.get("mtime")
    return _MEDIAN_ACT["v"]


def stance_vocab():
    """every base stance the pose table knows (its leaves' classes and the
    defaults): the words a typed pose is recognised by"""
    tb = _pose_table() or {}
    out = set()
    for lf in (tb.get("leaves") or {}).values():
        for cls in (lf.get("classes") or {}).values():
            out |= {str(x[0]).lower() for x in cls if x}
    for dflt in (tb.get("default") or {}).values():
        out |= {str(x[0]).lower() for x in dflt if x}
    return out | set(pe.BASE_POSES)


def stance_for_position(tags, level, rng):
    """A POSITION DECIDES THE STANCE IT IS HELD IN (the author, 2026-10-03,
    contradictory tags on one body: 'legs up' -- injected for the level -- beside
    'standing on one leg'). A position that is no stance itself still has one:
    of nsfw 'legs up' posts .61 lie and .13 sit, none stands
    (pose_companions_levels.json). -> a stance drawn at P(stance | position,
    level) for the first position that measures one, or None."""
    voc = stance_vocab()
    pos = set(((_spice_table() or {}).get("slots") or {}).get("position") or {})
    for t in tags or []:
        t = str(t).lower()
        if t in voc or t not in pos:
            continue
        row = level_row("pose_companions_levels.json", t, level) or {}
        cands = [(s9, float(f9)) for s9, f9 in sorted(row.items()) if s9 in voc and float(f9) > 0]
        if cands:
            return _wroll(rng, cands)
    return None


def typed_stance(user_tags):
    """-> the base stance the user typed ('sitting'), or None"""
    voc = stance_vocab()
    return next((str(t).lower() for t in (user_tags or []) if str(t).lower() in voc), None)


def activity_for(leaf, place, rng, allowed=None, pair=False, level=None, prefer=None,
                 occupation=None, being=None, nudge=1.0, stance=None):
    """-> {"activity", "stance", "object", "clothes", "state", "arity"} for
    a main character, None for no activity, False when the leaf has no
    table. At explicit only `with_act` activities stay (the act tables own
    the scene). `allowed` is the calling path's content gate."""
    ent = activity_entry(leaf, place)
    if ent is None:
        return False
    roll, none = ent
    reg = (_activity_table() or {}).get("registry") or {}
    # THE NO-ACTIVITY SHARE IS MEASURED (the author's 2026-09-11: a single
    # character mostly just holds a pose): one minus the summed share of
    # the roll's activities at this place (library: reading .029 +
    # studying .054 -> an activity 8 percent of the time); an unmeasured
    # place takes the median over the measured places (.061); `nudge`
    # scales the activity share (a typed face framing: x .25)
    _shares = [float(((reg.get(t) or {}).get("places") or {}).get(place) or 0.0) for t in roll]
    _act_share = min(0.95, sum(_shares)) if (place and any(_shares)) else _median_activity_share()
    none = max(0.05, 1.0 - _act_share * float(nudge))
    # THE NO-ACTIVITY SHARE IS MEASURED (the author's 2026-09-11: a single
    # character mostly just holds a pose): one minus the summed share of
    # the roll's activities at this place (the registry's related shares:
    # library reading .029 + studying .054 -> an activity 8 percent of the
    # time); the stated share stands only where the place is unmeasured
    _shares = [float(((reg.get(t) or {}).get("places") or {}).get(place) or 0.0) for t in roll]
    if place and any(_shares):
        none = max(0.05, 1.0 - min(0.95, sum(_shares)))
    # an event's activities join the roll and weigh triple (a christmas
    # eats and plays games wherever it lands)
    prefer = [t for t in (prefer or []) if t in reg]
    roll = list(roll) + [t for t in prefer if t not in roll]
    if prefer and rng.random() < 0.5:
        none = none * 0.5
    if not roll or rng.random() < none:
        return None
    fit = []
    for t in roll:
        r = reg.get(t) or {}
        if level == "explicit" and not r.get("with_act"):
            continue
        # A TYPED STANCE EXCLUDES THE ACTIVITIES THAT CANNOT BE DONE IN IT
        # (the author's 2026-09-15: 'sitting' typed, 'walking' rolled beside it):
        # an activity whose registry stance is another stance is out; one
        # with no stance is neutral
        if stance and r.get("stance") and str(r.get("stance")).lower() != stance:
            continue
        if r.get("tag", True) and not _floor_measured(t):
            continue                    # unmeasured is typed-only
        if allowed is not None and r.get("tag", True) and not allowed(t):
            continue
        if pair is False and int(r.get("arity") or 1) > 1 and t in ("mixed-sex bathing",):
            continue
        _w = 3.0 if t in prefer else 1.0
        # a hand-listed activity leans (x3) over the measured rest of the class
        _hl = ((((_activity_table() or {}).get("hand_lists") or {}).get(leaf) or {}))
        if _hl and any(t in v for v in _hl.values()):
            _w *= float((_activity_table() or {}).get("hand_lean") or 1.0)
        _w *= float((((_activity_table() or {}).get("lean_occupation") or {}).get(occupation) or {}).get(t, 1.0))
        _w *= float((((_activity_table() or {}).get("lean_leaf") or {}).get(leaf) or {}).get(t, 1.0))
        _w *= float((((_activity_table() or {}).get("lean_being") or {}).get(being) or {}).get(t, 1.0))
        fit.append((t, _w))
    pick = _wroll(rng, fit)
    if not pick:
        return None
    r = dict(reg.get(pick) or {})
    r["activity"] = pick
    # the period's object (a flintlock for the pirate, wine at the symposium)
    _lo = ((_activity_table() or {}).get("leaf_objects") or {}).get(leaf) or {}
    if _lo.get(pick):
        r["object"] = _lo[pick]
    return r


_POSE_TABLE = {"mtime": 0, "data": None}


def _pose_table():
    """POSE table (data/library/genre_poses.json, tools/build/write_genre_poses.py)."""
    p = _paths.data("genre_poses.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _POSE_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _POSE_TABLE["data"] = json.load(f)
            _POSE_TABLE["mtime"] = mt
    except Exception:
        return None
    return _POSE_TABLE["data"]


def _floor_measured(tag):
    """UNMEASURED IS TYPED-ONLY (2026-09-04): a tree item with no measured
    safety floor never rolls -- the regex fallback read 'groping' as safe
    while its floor was still in the queue. Typed input is unaffected."""
    try:
        return str(tag).lower().strip() in sm.SAFETY_FLOOR
    except Exception:
        return False


def _scrub_stripped(nl, stripped):
    """THE PROSE LOSES WHAT THE VERIFIER STRIPPED (2026-09-11): one owner
    per fact -- the line dropped 'smile' (it excluded the typed act's
    stance) while the prose kept "wide hips, smile and wears". Each
    stripped tag leaves the prose as the exact list item it was: the tag
    itself or its phrase ('is stretching', 'is in the pigeon pose'),
    whole words only, in a comma list or after a closing 'and'."""
    if not nl or not stripped:
        return nl
    forms = []
    for t, _why in stripped:
        t = str(t or "").strip()
        if not t:
            continue
        forms.append(t)
        try:
            ph = _fastplan._phrase(t)
            if ph and ph != t:
                forms.append(ph)
        except Exception:
            pass
    for f in sorted(set(forms), key=len, reverse=True):
        e = re.escape(f)
        # ', f' before another item, a clause end or ' and'
        nl = re.sub(r",\s*" + e + r"(?![a-z-])(?=\s*(?:,|;|\.|\s+and\b))", "", nl, flags=re.I)
        # 'f, ' as the head of a list: a descriptor right after 'has' /
        # 'wears', a verb phrase before the next one ('light smile' is not
        # 'smile': the item must be whole)
        if f.lower().startswith("is "):
            nl = re.sub(r"(?<=\s)" + e + r",\s*(?=is |holding |in )", "", nl, flags=re.I)
        else:
            nl = re.sub(r"(?:(?<=\bhas )|(?<=\bwears ))" + e + r",\s*(?=[a-z])", "", nl, flags=re.I)
        # ' and f' closing a list
        nl = re.sub(r"\s+and\s+" + e + r"(?![a-z-])(?=\s*(?:;|\.))", "", nl, flags=re.I)
    return nl


_PAIR_MIN_POSTS = 50        # an extra beside an act needs this many posts showing both


_SCENERY_FLAGS = {"scenery", "object"}
_SCENERY_NOT = {"location", "view", "camera", "person", "clothing", "body", "act", "pose",
                "expression", "meta", "creature", "race", "count"}
_SCENERY_STOP = frozenset(("simple background", "white background", "blurry background", "outdoors",
                           "indoors", "day", "night", "sky", "blue sky", "cloud", "cloudy sky", "sunlight",
                           "shadow", "depth of field", "blurry", "letterboxed", "border", "dutch angle",
                           "building"))
# THE KIND OF A PLACE IS NOT A THING IN IT (the author, 2026-10-03: "the prompt
# rolled `soapland` and `building` beside it -- I think `building` is unnecessary
# at all"; and 2026-09-21: 'street, building, tree'). Beside a named place the
# bare class word says nothing the place did not: it is no scene detail, and a
# model's copy of it leaves the line unless the user typed it.
_PLACE_CLASS_WORDS = frozenset(("building",))


_RULINGS_TB = {"mtime": 0, "data": None}


def place_parent(place):
    """the pool place whose measured tables a small ruled place inherits
    (pharmacy -> shop, saloon -> tavern, morgue -> hospital; 2026-09-15)"""
    try:
        return ((_location_pool().get(str(place).lower()) or {}).get("parent")) or None
    except Exception:
        return None


# NUDITY IS MEASURED BY LEVEL AND CONTEXT (the author's 2026-09-15: "nsfw tags --
# pussy, pubic hair, states of nudity -- should be measured and used
# correspondingly; pubic hair is in almost every nsfw prompt; the states
# are random, corresponding to neither location, profession nor genre").
# The base is the level's own rating universe (P(tag | rating:questionable)
# for nsfw, P(tag | rating:explicit) for explicit: nude .17 / .38, pubic
# hair .03 / .13, pussy .035 / .38); the context leans are lifts against the
# 1girl universe -- P(tag | onsen) / P(tag | 1girl) = 8 for 'nude', capped --
# from the places harvest (places_all), the genre tags and the occupation
# states harvest. Unmeasured is unmeasured: no base, no roll.
_LEVEL_UNIVERSE = {"safe": "rating:general", "sensitive": "rating:sensitive",
                   "nsfw": "rating:questionable", "explicit": "rating:explicit"}
_AFFINITY_TB = {"mtime": 0, "data": None}


def _universe(name):
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    return (tb.get("universes") or {}).get(name) or {}


def _ctx_rows(place=None, genre=None, occupation=None, level=None):
    """the measured rows a nudity roll leans on: the place's whole related
    tags, the genre's tags, the occupation's states"""
    rows = []
    if place:
        # THE PLACE DRESSES ITS PEOPLE PER BAND (2026-09-19 rule, measured
        # 2026-09-20 on solo posts): a classroom at safe is a school uniform .71,
        # a shirt .63, a skirt .59; at explicit the skirt .49 and panties .41. An
        # onsen at safe is a kimono .52, at explicit nude .77.
        rows.append(level_row("place_clothes_levels.json", place, level) if level else {})
        rows.append(_place_row("places_all", place))
    if genre:
        try:
            rows.append(((_json_table(_AFFINITY_TB, "pool_affinity.json") or {}).get("genre_tags") or {}).get(str(genre).lower()) or {})
        except Exception:
            pass
    if occupation:
        rows.append(((_json_table(_ACT_REL, "place_related.json") or {}).get("occupation_states") or {}).get(str(occupation).lower()) or {})
    return [r for r in rows if r]


def _ctx_lift(tag, rows, lo=0.2, hi=5.0):
    """P(tag | context) / P(tag | 1girl) over every measured context row,
    each capped; 1.0 where nothing is measured"""
    base = float(_universe("1girl").get(tag) or 0.0)
    if not base:
        return 1.0
    w = 1.0
    for row in rows:
        if tag in row:
            w *= max(lo, min(hi, float(row[tag]) / base))
    return w


# A 'NAKED <GARMENT>' OUTFIT NEEDS A CAUSE (the author, 2026-09-20, on a dancer on a
# rooftop given 'naked bandage' to carry the level: "if we are doing everything
# measured / connected then having a cause for outfit is right (genre / location /
# weather / activity should define outfits)"). Such a state replaces the whole
# outfit -- the verifier strips the dress -- so it is a costume, and a costume is
# drawn where the scene's own posts show it: naked apron in .11 of kitchen posts,
# naked towel in .25 of onsen posts; no row of a rooftop or of the supernatural
# genre holds 'naked bandage'. No measured cause: typed only in this scene.
_NAKED_OUTFIT_RE = re.compile(r"^naked (?!female|male)[a-z -]+$")


_COSTUME_PLACES = {"mtime": 0, "data": None}
_LOC_POOL_TB = {"mtime": 0, "data": None}
# a costume is a costume of the scene at this rate or more (the measured floor:
# naked shirt is .010 of 'on bed' pictures and .031 of 'waking up' ones, while
# its share of a bedroom -- the room, most of it not the bed -- is .0006)
_COSTUME_MIN = 0.001


def _location_names():
    try:
        return set(_json_table(_LOC_POOL_TB, "location_pool.json") or {}) - {"_note"}
    except Exception:
        return set()


def costume_scenes(tag):
    """P(costume | scene) from the costume's own posts (costume_places.json,
    tools/build/harvest_costume_places.py)"""
    tb = _json_table(_COSTUME_PLACES, "costume_places.json") or {}
    return dict((tb.get("costumes") or {}).get(str(tag or "").lower()) or {})


def naked_outfit_cause(tag, place=None, genre=None, occupation=None, scene=()):
    """-> P(the costume | the strongest scene fact that shows it), or 0.0.
    Asked from BOTH sides (the author, 2026-09-20: 'naked shirt' belongs in a
    bedroom and no place row held it): the place's own related tags, and the
    costume's own places measured back (P(place | costume) x N(costume) / N(place))."""
    if not _NAKED_OUTFIT_RE.match(str(tag or "").lower()):
        return None                      # not one of these: no opinion
    best = max([float(row.get(tag) or 0.0) for row in _ctx_rows(place, genre, occupation)] or [0.0])

    mine = costume_scenes(tag)
    # A COSTUME THAT BELONGS TO A ROOM BELONGS THERE (2026-09-20: 'naked towel' is
    # .011 of snow pictures -- the outdoor hot spring -- which put a towel on a
    # snowy street): when the costume's own strongest scene is a PLACE and the
    # picture is somewhere else, only place-like hooks count, not the weather's.
    _top = next(iter(mine), None)
    _places = _location_names()
    _room_bound = bool(_top and _top in _places and place and str(place).lower() not in mine)
    for w in [place, genre, occupation] + [str(x).lower() for x in (scene or ()) if x]:
        w = str(w or "").lower()
        if not w or float(mine.get(w) or 0.0) < _COSTUME_MIN:
            continue
        if _room_bound and w not in _places and w not in ("indoors", "outdoors"):
            continue
        best = max(best, float(mine[w]))
    return best


def naked_outfit_has_cause(tag, place=None, genre=None, occupation=None, scene=()):
    c = naked_outfit_cause(tag, place, genre, occupation, scene)
    return True if c is None else c > 0.0


def nudity_states(level, place=None, genre=None, occupation=None, whole=None, scene=()):
    """-> [(tag, weight)] the spice table's nudity states at this level,
    weighted by their measured share of the level's rating universe and
    the context lifts; largest first. The one owner of "which undress"."""
    sp = _spice_table() or {}
    items = (sp.get("slots") or {}).get("nudity") or {}
    uni = _universe(_LEVEL_UNIVERSE.get(level or "safe", "rating:general"))
    base1 = _universe("1girl")
    rows = _ctx_rows(place, genre, occupation, level)
    ruled = ruled_out()
    out = []
    for t, c in items.items():
        fl = c.get("floor")
        if not fl or sm.SPICE_ORDER.get(fl, 99) > sm.SPICE_ORDER.get(level or "safe", 0):
            continue
        if t in ruled or not _floor_measured(t):
            continue
        if whole is not None and _is_whole_nudity(t) != bool(whole):
            continue                     # whole=True: the undress; whole=False: the exposures on an outfit
        if not _of_level(t, level):
            continue
        _nc = naked_outfit_cause(t, place, genre, occupation, scene)
        if _nc is not None:
            # THE SCENE'S OWN RATE IS THE WEIGHT for a costume the universes never
            # rank (naked apron .11 of kitchen posts, naked towel .25 of onsen
            # posts): where the scene does not show it at all, it is not rolled
            if _nc <= 0.0:
                continue
            out.append((t, min(_nc, 0.25)))
            continue
        base = uni.get(t)
        if base is None:
            base = base1.get(t)          # not in the level's top tags: the 1girl share
        if not base:
            continue                     # unmeasured: not rolled
        out.append((t, float(base) * _ctx_lift(t, rows)))
    return sorted(out, key=lambda kv: -kv[1])


def nudity_chance(level, place=None, genre=None, occupation=None):
    """-> the measured chance that this subject is undressed rather than
    dressed: P(nude | the level's rating) lifted by the context (an onsen
    at nsfw: .17 x 5 capped = .85)"""
    uni = _universe(_LEVEL_UNIVERSE.get(level or "safe", "rating:general"))
    base = uni.get("nude")
    if not base:
        return 0.0
    return min(0.9, float(base) * _ctx_lift("nude", _ctx_rows(place, genre, occupation)))


def genital_chance(part, level):
    """-> the measured chance a genital part is described: its share of the
    level's rating universe ('pubic hair' .03 at nsfw, .13 at explicit;
    'pussy' .035 / .38; 'penis' -- / .47)"""
    uni = _universe(_LEVEL_UNIVERSE.get(level or "safe", "rating:general"))
    return float(uni.get(part) or 0.0)


def _place_row(table_key, place):
    """a place's row of place_related.json, or its parent's when the place
    has none of its own"""
    tb = (_json_table(_ACT_REL, "place_related.json") or {}).get(table_key) or {}
    row = tb.get(str(place).lower())
    if not row:
        par = place_parent(place)
        row = tb.get(par) if par else None
    return row or {}


def act_object_specific(act, cls, rng, allowed=None, least=0.01, chance=0.7):
    """-> the SPECIFIC thing an activity's object is, or None (the author,
    2026-09-19: the booru's verbs carry their things -- 'eating' is a cake, a
    burger, ramen; 'playing instrument' a guitar, a violin -- "we've just mostly
    simplified our actions / interactions with objects by holding"). The
    registry names the object's CLASS ('food', 'cup', 'book', 'instrument');
    the act's own posts name the members: every tag of the act's harvested row
    that the booru says IMPLIES the class, drawn by its measured share
    P(thing | act). Nothing is invented: both words are tags of the same
    pictures. Unmeasured acts and classes keep the class word."""
    if not act or not cls:
        return None
    row = ((_json_table(_ACT_REL, "place_related.json") or {}).get("acts") or {}).get(str(act).lower()) or {}
    if not row or rng.random() >= chance:
        return None
    ruled = ruled_out()
    cands = []
    for t, f in row.items():
        t = str(t).lower()
        if f < least or t == cls or t in ruled or not _floor_measured(t):
            continue
        if cls not in implied(t):
            continue
        # a THING, not another act or a state of one ('holding cup', 'playing
        # guitar', 'food on face' imply the class too)
        _fl = set(_gloss_flags(t) or ())
        if _fl & {"act", "pose", "expression", "body", "clothing"} or not _fl & {"food", "object", "weapon"}                 or re.search(r"^(holding|playing|eating|drinking) | (on|in) ", t):
            continue
        if allowed is not None and not allowed(t):
            continue
        cands.append((t, float(f) ** 0.5))
    return _wroll(rng, cands) if cands else None


def fits_hour(tag, time_word):
    """-> False when the booru almost never shows this word at that hour (the
    author's bug hunt, 2026-09-21: 'full moon' beside 'day' in a gym). Measured
    by the pair lift against the drawn time word; unmeasured passes. One owner
    for every path that can put a sight, an effect or an occasion on the line."""
    if not tag or not time_word:
        return True
    try:
        lf = pair_lift(str(time_word), str(tag))
    except Exception:
        return True
    return lf is None or lf >= 0.5


def says_the_place_again(word, place):
    """-> True when a scene word names the place over again ('rainy street' in
    a street, 'cafe interior' in a cafe). The place, the hour, the weather and
    the lighting already say all of it (the author's bug hunt, 2026-09-21)."""
    if not word or not place:
        return False
    w, pl = str(word).lower(), str(place).lower()
    if w == pl:
        return True
    pw = pl.split()
    return bool(pw) and (pw[-1] in w.split() or pl in w)


def scene_details(place, rng, n=2, floor=0.05):
    """-> [tags]: the details a place shows, drawn each at its measured
    share P(tag | place) from the whole-related harvest; scenery and
    object tags only, at least `floor`, at most `n` (2026-09-14)"""
    if not place:
        return []
    try:
        pa = _place_row("places_all", place)
    except Exception:
        return []
    out = []
    try:
        _worn = {t for items in ((_clothes_table() or {}).get("slots") or {}).values() for t in items}
    except Exception:
        _worn = set()
    try:
        _ruled = set(ruled_out())            # the never and typed-only words ('chain' in a dungeon)
    except Exception:
        _ruled = set()
    for t, f in sorted(pa.items(), key=lambda kv: -kv[1]):
        t = str(t).lower()
        if t == str(place).lower() or f < floor or t in _SCENERY_STOP or not _floor_measured(t):
            continue
        # ... AND THE PLACE UNDER ANOTHER NAME IS NOT A DETAIL OF IT (the
        # author's bug hunt, 2026-09-21: 'street, building, tree, rainy
        # street'). A word that contains the place's own noun says the place
        # again -- the weather, the hour and the lighting already say the rest.
        if says_the_place_again(t, place):
            continue
        # a thing of the place, not of the people: no garment, no count
        # word, nobody ('bow', '2girls', 'multiple girls' were drawn)
        if t in _ruled or t in _worn or _is_garment(t) or t.endswith(" focus") or re.match(r"^(\d+\+?(girls?|boys?|others?)|multiple .*|solo)$", t):
            continue
        # no franchise's hardware, no real-world make or model, nothing worn: a
        # scene detail is a thing of the place (2026-09-19: 'funnels (gundam)',
        # 'toyota', 'ak-47', 'orange jumpsuit' had become drawable)
        if re.search(r"\([a-z0-9 .'-]+\)$", t) and not t.endswith(("(object)", "(structure)", "(sky)", "(place)", "(automobile)")):
            continue
        if implies_clothing(t) or re.search(r"\b(adidas|nike|toyota|honda|nissan|ak-47|m1911|mosin-nagant|panzer|neo zeon)\b", t):
            continue
        fl = set(_gloss_flags(t) or ())
        # THE RULED CREATURES ARE THE SCENE'S TOO (the author, 2026-09-19): a
        # generic animal (crow, deer, fox) or a creature-only being (eastern
        # dragon, kaiju, karakasa obake) the place's own posts carry -- never
        # a specifically named one (polar bear, koi), never a race of a subject
        if not (fl & _SCENERY_FLAGS) or fl & _SCENERY_NOT:
            continue
        if rng.random() < float(f):
            out.append(t)
        if len(out) >= n:
            break
    # A CREATURE OF THE PLACE (the author, 2026-09-19). Asked of the PLACE, a
    # deer is half a percent of a forest's posts and never rolled, while a
    # zoo's few posts made a mouse look at home. Asked of the CREATURE -- of a
    # deer's posts, how many are in a forest? (creature_places.json, the held
    # object's own measure) -- the forest has its deer, bear and fox, the ocean
    # its shark and whale. One creature at most, one scene in six, by the
    # creature's share at the place; only the ruled generic animals and
    # creature-only beings.
    try:
        _rl = (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("creatures_roll") or {}
        _okc = set(_rl.get("animals") or ()) | set(_rl.get("beings") or ())
        _cp = (_json_table(_CREATURE_PLACES, "creature_places.json") or {}).get("creatures") or {}
        _pl = str(place).lower()
        _cands = [(c, float((e.get("places") or {}).get(_pl) or 0)) for c, e in sorted(_cp.items())
                  if c in _okc and c not in _ruled and _floor_measured(c)]
        _cands = [(c, w) for c, w in _cands if w >= 0.03]
        if _cands and rng.random() < 1.0 / 6:
            _c = _wroll(rng, _cands)
            if _c and _c not in out:
                out.append(_c)
    except Exception:
        pass
    return out


_CREATURE_PLACES = {"mtime": 0, "data": None}
_EFFECT_CTX = {"mtime": 0, "data": None}


def scene_effects(contexts, rng, allowed=None, chance=0.25, least=0.05):
    """-> [one effect] the scene's own facts carry, or [] (the author's ruling on
    the effects, 2026-09-19: weather / environment effects "measured against
    their weather / locations and used as their additional / decomposed
    tags", light effects as the lighting's). `contexts` are the scene's facts
    -- the place, the weather, the season, the time of day, the lighting tags
    rolled. Each ruled effect is weighed by the largest share of ITS OWN posts
    at any of them (effect_contexts.json): falling petals beside cherry
    blossoms .24, soap bubbles in a bath .21, condensation in rain .29, sparks
    beside fire .15. The fast path rolled no effect at all before this. 'day'
    says nothing (a quarter of every outdoor tag) and is not a context."""
    tb = (_json_table(_EFFECT_CTX, "effect_contexts.json") or {}).get("effects") or {}
    have = {str(c).lower() for c in (contexts or ()) if c} - {"day"}
    if not tb or not have or rng.random() >= chance:
        return []
    ruled = ruled_out()
    # THE AUTHOR'S CAUSES (2026-09-19): condensation needs water and glass (a
    # bath, a shower, a rained-on window, a bottle or a glass), a glint jewelry
    # or a blade, sparks a sparkling aura; a water drop also sits on a wet body.
    # 'only' replaces the measured contexts, 'also' joins them; a word matches
    # a context that contains it ('sword' in 'holding sword').
    _rules = ((_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("effects") or {}).get("rules") or {}

    def _has(word):
        return any(word == c or (" " + word + " ") in (" " + c + " ") for c in have)
    cands = []
    for t, e in sorted(tb.items()):
        _r = _rules.get(t) or {}
        if _r.get("only"):
            _hit = any(_has(x) for x in _r["only"] if x != "window") or                 any(all(_has(x) for x in pr) for pr in (_r.get("pairs") or []))
            w = 0.15 if _hit else 0.0
        else:
            w = max([float(f) for c, f in (e.get("contexts") or {}).items() if c in have] or [0.0])
            if any(_has(x) for x in (_r.get("also") or [])):
                w = max(w, 0.15)
        if w < least or t in ruled or t in have or not _floor_measured(t):
            continue
        if allowed is not None and not allowed(t):
            continue
        cands.append((t, w))
    got = _wroll(rng, cands) if cands else None
    return [got] if got else []


def pose_family(act):
    """-> {pose: measured share under the activity} when the activity's
    wiki lists its poses (registry 'poses', measured by
    tools/build/harvest_pose_families.py), else None"""
    try:
        ent = ((_activity_table() or {}).get("registry") or {}).get(str(act or "").lower()) or {}
        fam = ent.get("poses") or None
        return dict(fam) if fam else None
    except Exception:
        return None


_PLACE_STANCES = {"mtime": 0, "data": None}
_PLACE_STANCE_MIN = 0.10      # a place is measured when its stances sum to this share of its posts


def place_stances(place):
    """P(stance | place) -> {stance: share} from the place's own posts
    (data/library/place_stances.json, tools/build/harvest_place_stances.py)"""
    p = _paths.data("place_stances.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _PLACE_STANCES["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _PLACE_STANCES["data"] = json.load(f).get("places") or {}
            _PLACE_STANCES["mtime"] = mt
    except Exception:
        return {}
    return dict((_PLACE_STANCES["data"] or {}).get(str(place or "").lower()) or {})


def pose_bases(leaf, place, loc_kind, level=None):
    """-> ([(base, w)], place class or None) for this leaf x place: the
    leaf's table by place class, else the default by place kind."""
    bases, cls = _pose_bases_stated(leaf, place, loc_kind)
    # THE PLACE'S OWN POSTS DECIDE (the author, 2026-09-19: "the real problem /
    # contradiction is lying in a cafe (pose / location)"): the class tables
    # are stated weights (a 'nightlife' place lies 5); where the place is
    # measured, a stance is drawn at the share of the place's posts that show
    # it, and a stance they do not show is not drawn there. The leaf's own
    # stances the booru does not file by place (fighting stance, flying) keep
    # their stated weight beside the measured ones.
    # ... PER SPICE BAND, ON SOLO POSTS (the author, 2026-09-19: "I bet that lying in
    # classroom at nsfw / explicit will be MUCH higher than at safe rating"):
    # classroom -- safe: sitting .41, standing .15, nobody lies; explicit: sitting
    # .28, standing .13, lying .07, squatting .06. The band's own row first
    # (place_stances_levels.json; a thin band falls to its neighbour), then the
    # all-ratings table.
    ms = (level_row("place_stances_levels.json", place, level) if level else {}) or place_stances(place)
    # ('straddling' is a stance ON a partner: the pair tables state it, a place does not)
    ms = {b: f for b, f in ms.items() if b in set((_pose_table() or {}).get("bases") or []) and b != "straddling"}
    if len(ms) >= 2 and sum(ms.values()) >= _PLACE_STANCE_MIN:
        tot = sum(ms.values())
        stated_tot = float(sum(w for _, w in bases)) or 1.0
        _ORD = {"standing", "sitting", "lying", "kneeling", "squatting", "walking", "running", "reclining",
                "all fours", "wariza", "seiza", "indian style", "yokozuwari", "straddling", "jumping"}
        out = [(b, 100.0 * f / tot) for b, f in ms.items()]
        out += [(b, 100.0 * w / stated_tot) for b, w in bases if b not in _ORD and b not in ms]
        return out, cls
    return bases, cls


def _pose_bases_stated(leaf, place, loc_kind):
    tb = _pose_table() or {}
    lf = (tb.get("leaves") or {}).get(leaf) or {}
    cls = (lf.get("place_class") or {}).get(place)
    if cls and cls in (lf.get("classes") or {}):
        return [tuple(x) for x in lf["classes"][cls]], cls
    # THE PLACE KEEPS ITS STANCES IN ANY GENRE (the author, 2026-09-19: "the real
    # problem / contradiction is lying in a cafe (pose / location)"): a leaf
    # that has no class for the place ('world war ii' x cafe) fell to the
    # default by kind, which lies down in any building; the everyday table
    # knows the place (cafe: seated) and answers before the default does
    ev = (tb.get("leaves") or {}).get("everyday") or {}
    ecls = (ev.get("place_class") or {}).get(place)
    if ecls and ecls in (ev.get("classes") or {}):
        return [tuple(x) for x in ev["classes"][ecls]], ecls
    dflt = (tb.get("default") or {}).get(loc_kind or "indoor") \
        or (tb.get("default") or {}).get("indoor") or [("standing", 1)]
    return [tuple(x) for x in dflt], None


_POSE_COMP = {"mtime": 0, "data": None}


def pose_companions(tag, level=None):
    """P(companion | pose) -> {tag: share}: which other tags the pose's OWN
    posts carry; {} when unmeasured. With a level, the band's own row first --
    measured on SOLO posts per spice band (pose_companions_levels.json,
    harvest_levels.py --table poses; the author, 2026-09-19: 'claw pose' brings
    hands up at every band and 'spread legs' .23 only at explicit) -- then the
    all-ratings table (pose_companions.json)."""
    if level:
        try:
            row = level_row("pose_companions_levels.json", tag, level)
        except Exception:
            row = {}
        if row:
            return {t: float(f) for t, f in row.items()}
    p = _paths.data("pose_companions.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _POSE_COMP["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _POSE_COMP["data"] = json.load(f).get("poses") or {}
            _POSE_COMP["mtime"] = mt
    except Exception:
        return {}
    return {t: float(v[0]) for t, v in ((_POSE_COMP["data"] or {}).get(str(tag or "").lower()) or {}).items()}


_BOORU_POSTS = 9.0e6        # danbooru's posts, the base of a tag's own rate


def pose_trait_lift(pose, traits):
    """how much the subject's own traits pull a pose: P(trait | pose) over the
    trait's base rate, for the (two) strongest traits the pose's posts carry
    -- a 'claw pose' shows animal ears in .48 of its posts against .20 of all
    posts and a tail in .39 against .13, so a cat girl draws it ~7x as often.
    Only a lift: an unmeasured or absent trait is neutral (1.0)."""
    if not traits:
        return 1.0
    comp = pose_companions(pose)
    if not comp:
        return 1.0
    tree = _tree_tags()
    lifts = []
    for tr in traits:
        f = comp.get(tr)
        n = float((tree.get(tr) or {}).get("danbooru") or 0)
        if f and n >= 1000:
            lifts.append(f / (n / _BOORU_POSTS))
    lifts = sorted((x for x in lifts if x > 1.0), reverse=True)[:2]
    out = 1.0
    for x in lifts:
        out *= x
    return min(out, 6.0)


# THE PARTS A FRAMING LEAVES OUT (the author, 2026-09-19: "never describe
# out-of-frame parts"): a pose detail that occupies one of them is not drawn
_POSE_FRAME_OUT = {"close-up": {"legs", "feet", "torso", "hands", "all"}, "portrait": {"legs", "feet", "torso", "all"},
                   "upper body": {"legs", "feet", "all"}, "cowboy shot": {"feet"}}
# A BOORU POST NAMES A FRACTION OF ITS PICTURE (the author, 2026-09-19: "danbooru
# posts sometimes contain bare minimum tags describing the picture. The aim of
# the generator is to create fully descriptive ... prompts"): a companion's
# measured share is the floor of how often it is really in the pose's
# pictures, so the draw doubles it, capped
_COMPANION_LIFT = 2.0
_COMPANION_CAP = 0.6
_POSE_MAX_EXTRAS = 5


def pose_table_item(tag):
    """-> (slot, constraints) when `tag` is a detail of the pose table (a solo
    slot item or a pair posture), else None"""
    tb = _pose_table() or {}
    t = str(tag or "").lower().strip()
    for sl, en in (tb.get("slots") or {}).items():
        if t in (en.get("items") or {}):
            return sl, en["items"][t]
    it = (tb.get("pairs") or {}).get("items") or {}
    if isinstance(it, dict) and t in it:
        return "pairs", it[t]
    return None


def pose_things(outfit=None, held=None, race=None):
    """-> the things a subject wears / carries / has that a gesture can need
    ('hat tip': headwear, 'adjusting eyewear': eyewear, 'hugging tail': tail,
    'hands on hilt': sword)"""
    out = set()
    tree = _tree_tags()
    if held:
        out.add("*")                    # something is held ('holding with feet')
    for t in list(outfit or []) + [held]:
        t = str(t or "").lower()
        if not t:
            continue
        out.add(t)
        g = (tree.get(t) or {}).get("groups") or {}
        for k in ("headwear", "eyewear"):
            if k in g:
                out.add(k)
        if re.search(r"\b(sword|katana|rapier|saber|sabre|blade)\b", t):
            out.add("sword")
        if re.search(r"\bflowers?\b|\brose\b|bouquet", t):
            out.add("flower")
    if race:
        try:
            _rp = race_parts_related(race, least=0.3) or {}
            out.update(_rp)                 # 'cat ears', 'cat tail': the traits a pose's own posts carry
            out.add(str(race).lower())
            if any("tail" in x for x in _rp):
                out.update({"tail"})
            if any(x.endswith(" ears") for x in _rp):
                out.add("animal ears")
        except Exception:
            pass
    return out


def pose_for(leaf, place, loc_kind, rng, allowed=None, pair=False, base=None, hands_busy=False,
             act=None, view=None, level=None, stance_nudge=None, things=None, busy=None, frame=None,
             typed=None, ctx_creature=None, ctx_parts=None):
    """-> {"base": tag, "extras": [tags]} for ONE main character: the base
    stance always (the author's: every main character has at least a pose),
    the optional slots by their shares under their constraints, `lying`
    with its detail (on back / on side / on stomach). `allowed` is the
    calling path's content gate. Returns the pair postures too when
    `pair` (two main characters)."""
    tb = _pose_table() or {}
    bases, cls = pose_bases(leaf, place, loc_kind, level=level)
    # THE ACT RULES THE POSES (2026-09-06, measured): with an act in the
    # scene the stance is weighted by its lift beside the act (kneeling
    # 2.1 under a pov fellatio, sitting .44) and an extra under .6 is out
    def _lift(t):
        return act_lift(act, t, level, view=view) if act else None
    # A POSE-FAMILY ACTIVITY DRAWS ITS POSE (the author's 2026-09-11): an
    # activity whose wiki lists its poses (yoga: pigeon pose, downward
    # dog, shoulderstand...) is pictured in one of them, drawn by the
    # measured share of each pose under the activity (P(pose | yoga):
    # stretching .39, pigeon pose .19, split .09); the registry stance is
    # one member among them. Floors gate as everywhere (a family pose
    # without a measured floor is typed-only); the caller's pin yields.
    _fam = pose_family(act) if act else None
    if _fam:
        _fc = [(t, float(w)) for t, w in _fam.items()
               if float(w) > 0 and _floor_measured(t) and (allowed is None or allowed(t))]
        _fp = _wroll(rng, _fc)
        if _fp:
            base = _fp
    # (a typed detail does NOT narrow the stance -- the author, 2026-09-19: "claw
    # pose only places 2 hands/arms/fingers in a deliberate position ... hands
    # are occupied - everything else can roll with no restrictions"; 'crossed
    # arms' and 'lying' do not contradict. The place decides the stance.)
    if not base:                       # an activity may have pinned it
        fit = [(bt, w * (_lift(bt) or 1.0) * float((stance_nudge or {}).get(bt, 1.0)))
               for bt, w in bases if allowed is None or allowed(bt)]
        base = _wroll(rng, fit or bases)
    extras = []
    if base == "lying":
        det = _wroll(rng, [(d, 1) for d in (tb.get("lying_detail") or [])
                           if allowed is None or allowed(d)])
        if det:
            extras.append(det)
    nudge = (tb.get("gesture_nudge") or {}).get(cls) or {}
    # ONE BODY, ONE PICTURE (the author, 2026-09-19: "1 pose can be described by
    # combining different tags ... or at least not contradict one another"). The
    # slots drew blind to each other: 'w arms' beside 'hat tip' on a bare head,
    # 'on stomach' with 'knees to chest', and a 'claw pose' 0 times in 2,000.
    # Now every detail is checked against the body it lands on --
    #   OCCUPANCY: a hand, the legs, the feet, the torso, the mouth are each
    #     taken once (two hands: two one-hand gestures or one two-hand pose); a
    #     whole-body pose ('jack-o' challenge', 'dogeza') stands alone;
    #   FRAME: a part the framing leaves out is not posed;
    #   NEEDS: a gesture on a worn thing needs it worn, a pose on a thing of
    #     the place is drawn where its own posts are, furniture is brought;
    #   SEEN TOGETHER: a detail that shares the body's region with one already
    #     drawn must have been seen beside it (either one's own posts carry the
    #     other) --
    # and the drawn pose is then FINISHED with the details its own posts show
    # (P(companion | pose), harvest_pose_companions.py).
    things = set(things or ())
    _out_parts = set(_POSE_FRAME_OUT.get(str(frame or "").lower(), ()))
    _budget = {"hands": 2, "legs": 1, "feet": 1, "torso": 1, "mouth": 1, "head": 1}
    for _b in (busy or ()):
        _budget[_b] = 0
    if hands_busy:
        _budget["hands"] = 0            # a shovel in both hands leaves no hand in a pocket
    _state = {"whole": False, "brings": []}
    _all_items = {}
    for _sl, _en in (tb.get("slots") or {}).items():
        for _t, _c in (_en.get("items") or {}).items():
            _all_items.setdefault(_t, (_sl, _c))

    def _seen_with(a, b):
        ca, cb = pose_companions(a, level), pose_companions(b, level)
        if not ca and not cb:
            return True                 # unmeasured: neutral
        return b in ca or a in cb

    def _item_ok(slot, t, c):
        req, forb = c.get("requires"), c.get("forbids") or []
        if req and base not in req and not (set(req) & set(extras)):
            return False
        if base in forb:
            return False
        if c.get("classes") and cls not in c["classes"]:
            return False
        if not _floor_measured(t) or (allowed is not None and not allowed(t)):
            return False
        occ = c.get("occ") or {}
        if _state["whole"] or (occ.get("all") and [x for x in extras if x not in (tb.get("lying_detail") or [])]):
            return False
        for part, n in occ.items():
            if part in _out_parts:
                return False
            if part != "all" and _budget.get(part, 0) < int(n):
                return False
        if c.get("wears") and c["wears"] not in things:
            return False
        if c.get("needs_act") and not act:
            return False
        if c.get("needs_creature") and not (ctx_creature or []):
            return False
        if c.get("needs_part") and not any(c["needs_part"] in str(x).lower() for x in (ctx_parts or ())):
            return False
        if c.get("held") and not (set(c["held"]) & things):
            return False
        # THE SURFACE HAS TO BE IN THE PICTURE (the author, 2026-09-21, the
        # 'on ...' / 'against ...' family): the pose is drawn where its own
        # posts are (measured `where`), OR where the scene itself holds the
        # thing -- the place's furniture and details are named in the prompt
        # ('bedroom, on bed, bed, pillow'), so a bed in the picture is a bed
        # under the body wherever the place is; furniture is still brought
        # indoors.
        if c.get("thing") and place not in (c.get("where") or []) \
                and not any(str(c["thing"]).lower() in str(x).lower() for x in
                            (list(ctx_creature or ()) + list(things or ()) + [place or ""])) \
                and not (str(c["thing"]).lower() in ("any", "floor", "ground")
                         and (ctx_creature or things)) \
                and not ((c.get("brings") or str(c["thing"]).lower() in _ON_STRUCTURE)
                         and str(loc_kind or "") in ("indoor", "building")):
            return False
        if c.get("indoor") and str(loc_kind or "") not in ("indoor", "building"):
            return False                # 'against door / table / window' (2026-09-21)
        if c.get("thing") and _state.get("thing"):
            return False                # one surface under one body
        # seen beside what shares its region of the body (and beside how she lies)
        for x in extras:
            xo = (_all_items.get(x) or (None, {}))[1].get("occ") or {}
            if x in (tb.get("lying_detail") or []) and (set(occ) & {"legs", "torso", "all"}) and not _seen_with(x, t):
                return False
            # (the legs, the feet and the torso are one region: 'leg up' with
            # 'pigeon-toed', 'bent over' with 'standing split')
            _r1 = {("trunk" if k in ("legs", "feet", "torso") else k) for k in xo}
            _r2 = {("trunk" if k in ("legs", "feet", "torso") else k) for k in occ}
            if (_r1 & _r2) and not _seen_with(x, t):
                return False
        _lf = _lift(t)
        if _lf is not None and _lf < 0.6:
            return False
        return True

    def _beside_act(t):
        # the picked extra, unlisted beside the act, is asked of the
        # booru itself (cached counts); under .6 it is not in the picture
        if act and _lift(t) is None:
            _lv2 = act_lift(act, t, level, view=view, live=True)
            if _lv2 is not None and _lv2 < 0.6:
                return False
            # ...and the booru must have SEEN the pair (2026-09-14: a
            # handstand beside a sleeper passed at 14 posts out of
            # 99,791, a lift of .7 between two rarities); under
            # _PAIR_MIN_POSTS the pair is unmeasured, and an
            # unmeasured extra beside an act is not rolled
            _n2 = _count_cached(_q(act) + " " + _q(t))
            if _n2 is not None and _n2 < _PAIR_MIN_POSTS:
                return False
        return True

    def _take(t, c):
        extras.append(t)
        for part, n in (c.get("occ") or {}).items():
            if part == "all":
                _state["whole"] = True
            else:
                _budget[part] = _budget.get(part, 0) - int(n)
        if c.get("thing"):
            _state["thing"] = c["thing"]
        if c.get("brings") and c["brings"] not in _state["brings"]:
            _state["brings"].append(c["brings"])

    # A TYPED POSE IS ENHANCED, NOT LEFT BARE (the author, 2026-09-19: "the rule to
    # never override the typed is meant to preserve the baseline that user
    # wants - any additions are welcome (this is the purpose of the generator -
    # enhance)"): the typed detail takes its parts first -- so nothing drawn
    # can land on the hands a typed 'claw pose' holds up -- and the stance, the
    # free parts and the pose's own measured companions are drawn around it.
    # The typed tag is the caller's; it is not returned among the extras.
    _typed = [str(x).lower() for x in (typed or ()) if pose_table_item(x)]
    for _tt in _typed:
        _take(_tt, pose_table_item(_tt)[1])
    # THE PAIR'S POSTURE LEADS (2026-09-19): what two bodies do together is
    # drawn first, and each one's own details fit around it -- it takes its
    # hands, keeps its frame and its whole-body rule like any detail (the
    # author: "this becomes even harder with multiple subjects")
    _pair_pick = None
    if pair and not any(pose_table_item(x)[0] == "pairs" for x in _typed):
        pe_ = tb.get("pairs") or {}
        items = pe_.get("items") or {}
        items = items if isinstance(items, dict) else {t: {} for t in items}
        _pc = [(t, max(1.0, float(c.get("posts") or 1)) ** 0.25 * pose_trait_lift(t, things))
               for t, c in items.items()
               if _item_ok("pairs", t, c) and (not act or (_lift(t) or 0.0) >= 0.6)]
        _pw = sum(max(1.0, float(c.get("posts") or 1)) ** 0.25 for c in items.values())
        _pf = sum(max(1.0, float(items[t].get("posts") or 1)) ** 0.25 for t, _ in _pc)
        if _pc and rng.random() < float(pe_.get("share") or 0) * min(1.0, (_pf / max(_pw, 1e-9)) ** 0.5):
            _pair_pick = _wroll(rng, _pc)
            if _pair_pick:
                _take(_pair_pick, items[_pair_pick])
                extras.remove(_pair_pick)
    # (the slots in a drawn order: a named pose -- 'claw pose', 'paw pose' --
    # came last and found the hands already taken)
    _slots = list((tb.get("slots") or {}).items())
    rng.shuffle(_slots)
    for slot, ent in _slots:
        _roll = rng.random()
        if _roll >= float(ent.get("share") or 0):
            continue
        cands = []
        _whole_w = 0.0
        for t, c in (ent.get("items") or {}).items():
            _whole_w += max(1.0, float(c.get("posts") or 1)) ** 0.25
            if t in extras or not _item_ok(slot, t, c):
                continue
            # a slot's items by the fourth root of their posts: the common
            # gesture leads, the rare one still comes
            cands.append((t, (nudge.get(t, 1.0) if slot == "hands" else 1.0) * (_lift(t) or 1.0)
                          * max(1.0, float(c.get("posts") or 1)) ** 0.25 * pose_trait_lift(t, things)))
        # THE SLOT SPEAKS AS OFTEN AS IT HAS SOMETHING TO SAY: with one item
        # left that fits ('hopping', the only named pose of a walker) the
        # slot's whole share fell on it -- a fifth of all walkers hopped. The
        # share is scaled by the root of the fitting items' part of the slot.
        _fit_w = sum(max(1.0, float((ent["items"][t]).get("posts") or 1)) ** 0.25 for t, _ in cands)
        if not cands or _roll >= float(ent.get("share") or 0) * min(1.0, (_fit_w / max(_whole_w, 1e-9)) ** 0.5):
            continue
        pick = _wroll(rng, cands)
        if pick and _beside_act(pick):
            _take(pick, (ent.get("items") or {})[pick])
    # ... FINISHED BY ITS OWN PICTURES: the stance's and each drawn detail's
    # measured companions, strongest first, each at its (lifted) share
    _cp = {}
    for _src in [base] + list(extras):
        for _t, _f in pose_companions(_src, level).items():
            if _t in _all_items and _t not in extras:
                _cp[_t] = max(_cp.get(_t, 0.0), _f)
    for _t, _f in sorted(_cp.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(extras) >= _POSE_MAX_EXTRAS or _state["whole"]:
            break
        if rng.random() >= min(_COMPANION_CAP, _f * _COMPANION_LIFT):
            continue
        _sl, _c = _all_items[_t]
        if _item_ok(_sl, _t, _c) and _beside_act(_t):
            _take(_t, _c)
    extras[:] = [x for x in extras if x not in _typed]
    out = {"base": base, "extras": extras, "brings": list(_state["brings"])}
    if _pair_pick:
        out["pair"] = _pair_pick
    return out


_RACE_TABLE = {"mtime": 0, "data": None}


def _race_table():
    """GENRE -> RACE (data/library/genre_races.json, written by
    tools/build/write_genre_races.py), or None."""
    p = _paths.data("genre_races.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _RACE_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _RACE_TABLE["data"] = json.load(f)
            _RACE_TABLE["mtime"] = mt
    except Exception:
        return None
    return _RACE_TABLE["data"]


def race_entry(leaf):
    """-> (roll list, human share) for this leaf -- leaf, else its bucket --
    or None when neither has an entry (historical, superhero, everyday:
    human unless typed). Races depend on the genre only (the author's)."""
    tb = _race_table()
    if not tb or not leaf:
        return None
    bucket = (_genre_pool().get(leaf) or {}).get("bucket")
    ent = tb.get("leaves", {}).get(leaf) or (tb.get("buckets", {}).get(bucket) if bucket else None)
    if not ent:
        return None
    return list(ent.get("roll") or []), float(ent.get("human", 0.6))


def race_tag(race, kind):
    """-> the tag a race takes for a subject of `kind`, or None when the
    race is locked to another gender. FORMS: the word carries the gender
    (mermaid / merman); LOCKED: one gender only (yuki onna); ALIASES:
    succubus -> demon girl. Futanari takes the female form. No measured
    gate -- the author's: most humanoids can be female, male or futanari."""
    tb = _race_table() or {}
    r = str(race or "").lower().strip()
    r = (tb.get("aliases") or {}).get(r, r)
    if not r:
        return None
    fem = kind in ("female", "futanari")
    lock = (tb.get("locked") or {}).get(r)
    if lock == "female" and not fem:
        return None
    if lock == "male" and fem:
        return None
    form = (tb.get("forms") or {}).get(r)
    if form:
        return form.get("female" if fem else "male") or None
    return r


def race_noun(tag, who):
    """the prose noun for a subject of race `tag`: 'the mermaid' for a
    being whose word IS the being, 'the elf girl' for a people."""
    tb = _race_table() or {}
    base = str(tag or "").strip()
    if not base:
        return who
    stem = base
    for form in (tb.get("forms") or {}).values():
        if base in form.values():
            stem = next(k for k, v in (tb.get("forms") or {}).items() if v is form)
            break
    if stem in set(tb.get("noun_only") or ()) or base in set(tb.get("noun_only") or ()):
        return base
    return "%s %s" % (base, who) if who else base


def race_for(leaf, kind, rng, allowed=None):
    """-> a race tag for a subject of `kind` in this genre leaf, None for
    human (the table's `human` share), or the sentinel False when the leaf
    has no entry. `allowed` is the calling path's content gate."""
    ent = race_entry(leaf)
    if ent is None:
        return False
    roll, human = ent
    # THE NO-RACE SHARE IS MEASURED (the author, 2026-09-16: "races roll too
    # much"): the table's hand share (fantasy .60 human) gives way to the
    # genre's own tag profile -- the share of the genre's posts that carry
    # any race tag (fantasy .13, cyberpunk .64), capped at .9; a genre
    # with no measured row keeps the table's share
    human = measured_human_share(leaf, human)
    if not roll or rng.random() < human:
        return None
    fit = []
    for r in roll:
        t = race_tag(r, kind)
        if t and (allowed is None or allowed(t)):
            fit.append(t)
    return rng.choice(fit) if fit else None


# THE FANTASY LEAVES KEEP THE TABLE'S RACE RATE (the author, 2026-09-16:
# "return fantasy / dark fantasy race rates"): the measured profile puts a
# race on 13% of fantasy posts; the studio's fantasy is meant to be peopled
# by its races, so those two leaves roll at the table's stated share
_RACE_SHARE_STATED = {"fantasy", "dark fantasy"}


def measured_human_share(genre, fallback):
    """-> 1 - P(any race tag | genre) from the genre's measured tag profile
    (pool_affinity.json 'genre_tags'), capped so a race still rolls at most
    nine times in ten; the table's share where the genre is unmeasured or
    where the author ruled the table's share to stand (_RACE_SHARE_STATED)"""
    if str(genre or "").lower() in _RACE_SHARE_STATED:
        return fallback
    try:
        row = ((_json_table(_AFFINITY_TB, "pool_affinity.json") or {}).get("genre_tags") or {}).get(str(genre or "").lower()) or {}
        if not row:
            return fallback
        races = _race_family()
        tot = sum(float(v) for t, v in row.items() if t in races)
        return 1.0 - min(0.9, tot)
    except Exception:
        return fallback


def measured_none_share(place, fallback):
    """-> 1 - P(any occupation tag | place) from the place's measured tag
    profile (places harvest), capped at .9; the table's share where the
    place is unmeasured"""
    try:
        row = _place_row("places_all", place) if place else {}
        if not row or len(row) < 100:
            return fallback
        occ = set((_json_table(_OCC_POOL_TB, "occupation_pool.json") or {}).get("occupations") or {})
        tot = sum(float(v) for t, v in row.items() if t in occ and t not in ruled_out())
        return 1.0 - min(0.9, tot)
    except Exception:
        return fallback


_OCC_POOL_TB = {"mtime": 0, "data": None}


def occupation_tag(role):
    """-> the booru tag for a rolled role, or None when the booru has none
    (the table's `roles` registry: a role is a role, the tag is a property
    -- the author's 2026-09-04). Unknown to the registry: the word itself."""
    roles = (_occupation_table() or {}).get("roles") or {}
    ent = roles.get(str(role or "").lower().strip())
    if ent is None:
        return role or None
    return ent.get("tag")


def occupation_fits(tag, kind):
    """Does this occupation fit a subject of `kind`? ONE rule for every
    roller: a word that names a gender (the table's `gender` map: king,
    princess, nun, butler...) is a fact and decides first; otherwise the
    measured share (occupation_gender.json) refuses only the extremes
    (>= 0.95 female-only, <= 0.05 male-only); unmeasured passes. Futanari
    counts female. 'butler' rolled onto a girl at a picnic (golden case,
    2026-09-04) because the old roller had no such filter."""
    fem = kind in ("female", "futanari")
    g = ((_occupation_table() or {}).get("gender") or {}).get(tag)
    if g == "female":
        return fem
    if g == "male":
        return not fem
    if g == "any":                       # stated neutral: the measure yields
        return True
    sh = _occupation_share(tag)
    if sh is None:
        return True
    if fem and sh <= 0.05:
        return False
    if not fem and sh >= 0.95:
        return False
    return True


def occupation_for(leaf, place, kind, rng, allowed=None):
    """-> an occupation for a subject of `kind` (female / male / futanari /
    other) in this leaf x place, or None for "no profession" -- rolled at
    the table's `none` share, then uniformly among the roles the measured
    gender share permits (occupation_gender.json: >= 0.95 female-only,
    <= 0.05 male-only; futa counts female; unmeasured passes).
    Returns the sentinel False when the leaf is not in the table."""
    ent = occupation_entry(leaf, place)
    if ent is None:
        return False
    roll, none = ent
    # THE NO-PROFESSION SHARE IS MEASURED (the author, 2026-09-16:
    # "occupations roll too much"): the table's hand share (a park .8)
    # gives way to the place's own tag profile -- the share of the place's
    # posts that carry any occupation tag (a hospital .35, a park .05),
    # capped at .9; an unmeasured place keeps the table's share
    none = measured_none_share(place, none)
    if not roll or rng.random() < none:
        return None
    _ruled_o = ruled_out()
    roll = [t for t in roll if t not in _ruled_o]           # 'virtual youtuber' and its kind
    # `allowed` is the content gate of the calling path (slots.content_allowed
    # with the scene's census and level): 'prostitution' is a job on the
    # booru's list and rolls only where the level lets it (2026-09-04)
    fit = [t for t in roll if occupation_fits(t, kind) and (allowed is None or allowed(t))]
    return rng.choice(fit) if fit else None


_OCC_LEAN = {"mtime": 0, "data": {}}


def _occupations_known():
    """the occupation nouns the gender table knows (loaded once)"""
    _occupation_lean("")
    return _OCC_LEAN.get("data") or {}


def _occupation_lean(noun):
    """-> 'female' | 'male' | 'roll' for an occupation noun, from the
    measured 1girl/1boy share (data/library/occupation_gender.json);
    'roll' when unmeasured. Re-read when the file changes."""
    p = _paths.data("occupation_gender.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _OCC_LEAN["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _OCC_LEAN["data"] = json.load(f).get("nouns") or {}
            _OCC_LEAN["mtime"] = mt
    except OSError:
        return "roll"
    return (_OCC_LEAN["data"].get(str(noun or "").lower()) or {}).get("default", "roll")


def _occupation_share(noun):
    """-> measured female share (0..1) for an occupation noun, or None;
    a RULED default (occupation_rulings.json: fisherman, lumberjack,
    trucker lean male, 2026-09-15) stands in where the measure is
    missing: male .05, female .95"""
    _occupation_lean(noun)                       # (re)loads the file
    rec = _OCC_LEAN["data"].get(str(noun or "").lower())
    if not rec:
        return None
    if rec.get("female_share") is not None:
        return float(rec["female_share"])
    return {"male": 0.05, "female": 0.95}.get(rec.get("default"), 0.5)   # unruled, unmeasured: the coin


_PAIRING_CAST = {"futa with female": {"futa": 1, "female": 1}, "futa with male": {"futa": 1, "male": 1},
                 "futa with futa": {"futa": 2}, "yuri": {"female": 2}, "yaoi": {"male": 2},
                 "hetero": {"female": 1, "male": 1}}


def resolve_cast(base, banks, mode, rng, level=None, refused=False):
    """-> (cast, typed tags). Every path below decides the PEOPLE; this
    wrapper keeps the other two ledgers on every path: humanoid beings
    (gnome, goblin, an anthro fox) are `other`, creatures (a girl AND HER
    CAT, a wizard and his owl) are named in `creatures` and take no slot.
    Before, only the occupation path kept them, so 'a girl and her cat'
    lost the cat from the prose while 'a knight and a dragon' dressed the
    dragon. slots.subject_kind is the one classifier."""
    cast, tags = _resolve_cast_people(base, banks, mode, rng, level=level)
    if not isinstance(cast, dict):
        return cast, tags
    # A NAMED CHARACTER IS IN THE PICTURE (2026-09-19: 'cloud strife and 1girl in a
    # bar' came out '1girl, solo' -- the typed count said one girl, and the named
    # man was nobody's): where the cast holds fewer of a gender than the brief
    # names characters of it, the named ones join
    try:
        _by_g = {"female": 0, "male": 0}
        for _n9 in _typed_characters(base):
            _g9 = (_traits().get(_n9) or {}).get("gender")
            if _g9 in _by_g:
                _by_g[_g9] += 1
        for _g9, _k9 in _by_g.items():
            _have9 = int(cast.get(_g9) or 0) + (int(cast.get("futa") or 0) if _g9 == "female" else 0)
            if _k9 > _have9:
                cast[_g9] = int(cast.get(_g9) or 0) + (_k9 - _have9)
    except Exception:
        pass
    # A TYPED PAIRING NAMES THE CAST (the author's 2026-09-06): 'futa with
    # female' is a futa and a woman by definition, 'yuri' two women,
    # 'hetero' a woman and a man. A typed count outranks it ('1girl, futa
    # with female': the partner is out of frame) and so does pov (the
    # partner is the viewer).
    _low_p = " " + re.sub(r"[^a-z0-9 ]+", " ", (base or "").lower()) + " "
    _counted = re.search(r"\b\d+\s*(girls?|boys?|futas?|others?|wom[ae]n|m[ae]n|people)\b|"
                         r"\d(girls?|boys?|futas?)\b|\bsolo\b|\bmultiple\b", base or "", re.I)
    _pov_p = any(str(t).lower() in _POV_VIEWS for t in (tags or [])) or " pov " in _low_p
    if not _counted and not _pov_p:
        for _pt, _min in _PAIRING_CAST.items():
            if (" " + _pt + " ") in _low_p or _pt in [str(t).lower() for t in (tags or [])]:
                for k, v in _min.items():
                    if int(cast.get(k) or 0) < v:
                        cast[k] = v
                cast["source"] = "pairing:" + _pt
                break
    try:
        spans = _subject_spans(base)
    except Exception:
        spans = []
    people, races, humanoids, creatures = 0, [], [], []
    for n, c in spans:
        kind = sm.subject_kind(n, base)
        if kind == "race":
            races += [n] * c
        elif kind == "humanoid":
            humanoids += [n] * c
        elif kind == "creature":
            creatures += [n] * c
        elif kind == "person" or _is_humanlike(n):
            people += c
    # A RACE WORD ALONE IS SOMEBODY, GENDERED BY MEASURE (race_gender.json:
    # 'elf' 95% girl, 'dwarf' 85% man, 'orc' a coin toss, `1other` under 3%
    # everywhere). Only the races the paths above have NOT already counted
    # through a typed gender word are rolled here -- "an elf in the forest"
    # already stands as 1girl, "a goblin and a girl" has only the girl.
    _have = sum(int(cast.get(k) or 0) for k in ("female", "male", "futa", "other"))
    _need = max(0, people + len(races) - _have)
    for n in races[:_need]:
        sh = _race_share(n)
        if sh is None:
            k = "other"
        else:
            k = rng.choices(["female", "male", "other"],
                            weights=[sh["female"], sh["male"], sh["other"]])[0]
        cast[k] = int(cast.get(k) or 0) + 1
        cast["source"] = (cast.get("source") or "") + "+race:%s" % n
    if humanoids and furry_form(base, tags):
        # AN ANTHRO ANIMAL IS SOMEBODY, GENDERED BY MEASURE (2026-09-21): the
        # booru's furry pictures are half female (69,546) and half male
        # (66,732) and carry 1girl / 1boy, not 1other. The form is the
        # appearance; the subject is a subject like any other.
        _cn = (furry_table().get("counts") or {})
        _f9 = float(_cn.get("furry female") or 1), float(_cn.get("furry male") or 1)
        for _ in humanoids:
            if int(cast.get("female") or 0) + int(cast.get("male") or 0) + int(cast.get("futa") or 0)                     >= len(humanoids) + people:
                break
            k = rng.choices(["female", "male"], weights=list(_f9))[0]
            cast[k] = int(cast.get(k) or 0) + 1
            cast["source"] = (cast.get("source") or "") + "+furry:%s" % k
        humanoids = []
    if humanoids:
        # the parser's `1other` may already have counted one of them
        cast["other"] = max(int(cast.get("other") or 0), len(humanoids))
    # a creature named anywhere is in the picture, even as the object of a
    # verb ("a girl riding a dragon"): the typed tags carry it
    for t in tags or ():
        tl = str(t).lower()
        if tl not in creatures and sm.NONHUMAN_SUBJECT.fullmatch(tl) \
                and sm.subject_kind(tl, base) == "creature":
            creatures.append(tl)
    if creatures:
        cast["creatures"] = creatures
    # AN ACT NEEDS A BODY (the author's bug hunt, 2026-09-21: 'a blowjob to the
    # viewer, pov' came out `no humans, fellatio, penis`). The parser read the
    # act and the girl it implies; the female-evidence guard then dropped the
    # count -- rightly, since no word in the text says 'girl' -- and nobody was
    # left to perform it. Where the picture holds a human act and no cast at
    # all, the performer is one subject, gendered by what the booru measures
    # for that act (gender_levels, solo posts, crossdressing discounted).
    # ... BUT NEVER WHERE A SUBJECT WAS REFUSED (2026-09-22, caught by the
    # safety tests): a youth-coded word leaves the cast empty ON PURPOSE, and
    # the act it was doing must not conjure a body back into the picture.
    _refused9 = bool(refused)
    try:
        _refused9 = _refused9 or bool(refuse_youth(base, tags or [], level or "safe").get("words"))
    except Exception:
        pass
    if not _refused9             and not sum(int(cast.get(k) or 0) for k in ("female", "male", "futa", "other"))             and not sm.scene_has_no_people(base or ""):
        _acts9 = [str(t).lower() for t in (tags or ()) if _is_person_act(t)]
        if _acts9:
            _sh9 = None
            for _a9 in _acts9:
                _sh9 = gender_lean(_a9, level)
                if _sh9 is not None:
                    break
            _k9 = "male" if (_sh9 is not None and _sh9 < 0.5) else "female"
            cast[_k9] = 1
            cast["source"] = (cast.get("source") or "") + "+act:%s" % _acts9[0]
    for k in ("female", "male", "futa", "other"):
        cast.setdefault(k, 0)
    return cast, tags


def _is_person_act(tag):
    """-> True when the word names something a person does (the spice acts and
    positions, the activity registry): an act with nobody in the picture is the
    fault this answers"""
    t = str(tag or "").lower()
    try:
        sp = (_spice_table() or {}).get("slots") or {}
        if t in (sp.get("act") or {}) or t in (sp.get("position") or {}):
            return True
        if t in ((_activity_table() or {}).get("registry") or {}):
            return True
        # the widest measured list of acts the studio has: everything the
        # act-anchor table knows (paizuri sits in no other source)
        return t in ((furry_table() and {}) or (_json_table(_ACT_ANCHORS, "act_anchors.json") or {}).get("acts") or {})
    except Exception:
        return False


_RACE_LEAN = {"mtime": 0, "data": {}}


def _race_share(noun):
    """-> {female, male, other} shares for a race word alone, or None."""
    p = _paths.data("race_gender.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _RACE_LEAN["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _RACE_LEAN["data"] = json.load(f).get("races") or {}
            _RACE_LEAN["mtime"] = mt
    except Exception:
        return None
    return _RACE_LEAN["data"].get(str(noun or "").lower())


def _resolve_cast_people(base, banks, mode, rng, level=None):
    """THE GENERATOR NEVER ADDS SUBJECTS (the author's, superseding the old
    silent-prompt roll). Count comes from the prompt alone; no subjects in
    the prompt means none generated -- scenery by intent. cast_weights'
    role here is GENDER ASSIGNMENT: a genderless count ("2 people kissing")
    fixes the total, and the measured distribution CONDITIONED on that total
    decides the gender split at its real ratios."""
    tags, _ = pe.parse_input(base, banks)
    cen = sm.subject_census([str(t).lower() for t in tags])
    # parse_input INVENTS a default 1girl (old-generator assumption), so the
    # census only counts when the text itself mentions subjects -- otherwise
    # "a beautiful sunset over ruins" arrives with a phantom girl.
    hint = re.search(
        r"\b(?:" + _FEMALE_WORDS + "|" + _MALE_WORDS +
        r"|futa\w*|solo|\dgirls?|\dboys?|no humans?)\b",
        pe.claim_gender_modifiers(base or ""), re.I)   # GENDERED words INCLUDING female/male (the
                              # adjective missed 'a female CEO'); person/
                              # people/figures
                              # belong to the genderless path below, where the
                              # conditional distribution assigns the split
    # '1futa' defeats a leading \b (digit-letter is no boundary) -- its own test
    # `futanari pov` names the CAMERA HOLDER, not the subject: "a girl
    # having sex with the viewer, futanari pov" turned the girl into a
    # futa. Every futa-subject test below reads the text with that
    # phrase removed; the viewer's own kind is handled where the pairing
    # is derived.
    # ...and so do `female pov` and `male pov`: "female pov" put a phantom
    # woman beside a solo futa because the word "female" counted as
    # female evidence. A pov qualifier describes the camera holder only.
    base_nf = re.sub(r"\b(?:futanari|female|male) pov\b", " ",
                     base or "", flags=re.I)
    # a gender word before a noun decides the noun ('a female mariner' is
    # one woman) -- the parser reads the same (enhancer.claim_gender_modifiers)
    base_nf = pe.claim_gender_modifiers(base_nf)
    if not hint:
        hint = re.search(r"\d\s*futas?\b|futanari|\bfutas\b", base_nf, re.I)
    # THE PHANTOM GIRL. parse_input inserts a default `1girl` when the
    # prose names no count, so the census can report a female nobody
    # wrote. `hint` was meant to stop that, but it matches ANY
    # gendered word -- so "a wizard and HIS owl" satisfied the hint,
    # the phantom survived, and the wizard came out a girl.
    #
    # A female count is only real when the text carries its own
    # female evidence. Without it the count is dropped here, and the
    # paths below (raw-text counts, characters, subject nouns) get to
    # read the sentence properly instead of inheriting an invention.
    _fem_word = re.search(
        r"\b(?:" + _FEMALE_WORDS + r")\b", base_nf, re.I)
    _girl_count = re.search(
        r"\d\s*girls?\b|\bmultiple girls\b",
        base_nf, re.I)
    if cen["females"] and not _fem_word and not _girl_count:
        cen = dict(cen, females=0,
                   total=max(0, cen["total"] - cen["females"]))

    # the parser can FUSE tokens ('1girl reading' arrives as one tag) and
    # blind the census; when the gendered hint matched but the census sees
    # nothing, the raw text itself is the count authority
    if cen["total"] == 0 and hint and not cen["no_humans"]:
        g = b_ = 0
        for m in re.finditer(r"\b(\d+)\s*(girls?|boys?)\b", (base or ""),
                             re.I):
            if m.group(2).lower().startswith("girl"):
                g = max(g, int(m.group(1)))
            else:
                b_ = max(b_, int(m.group(1)))
        if not g and re.search(r"\b(?:" + _FEMALE_WORDS + r")\b",
                               base_nf, re.I):
            g = 1
        if not b_ and re.search(r"\b(?:" + _MALE_WORDS + r")\b",
                                base_nf, re.I):
            b_ = 1
        if g or b_:
            cen = dict(cen, females=g, males=b_, total=g + b_)
    # NAMES OUTRANK MODIFIER WORDS: with typed characters present, the
    # census return is only authoritative when the text carries an
    # EXPLICIT count ('2girls, hatsune miku'); a bare gendered/futa word
    # is a modifier on the named cast, not a count ('futanari tifa
    # lockhart fucks aerith gainsborough' had census-collapsed to one
    # anonymous futa)
    _typed_early = _typed_characters(base)
    _explicit_count = re.search(
        r"\b\d+\s*(girls?|boys?|futas?|others?|wom[ae]n|m[ae]n|people)\b|"
        r"\d(girls?|boys?|futas?)\b|\bsolo\b|\bmultiple\b",
        (base or ""), re.I)
    if _typed_early and not _explicit_count:
        pass                          # fall through to the character path
    elif (cen["total"] > 0 and hint) or cen["no_humans"]:
        # A GENDER WORD IS A PARTIAL COUNT WHEN A PLURAL NOUN SAYS MORE.
        # "a woman in a threesome" took the gendered path, counted the
        # woman, and stopped -- two people after the partner rule, never
        # three. The noun's total wins; the stated genders become minimums
        # and the rest of the split is rolled from measured weights, the
        # same way "3 people" already is.
        _pt, _pf, _pm = _pair_shape_in(base)
        # ...and so does a list of beings: "a mother and her daughter"
        # is two women, counted by the same noun-phrase reader the
        # occupation path uses. A gender word proves presence, not
        # number; the largest evidence-backed count wins.
        _folk = [(n, c) for n, c in _subject_spans(base_nf)
                 if sm.subject_kind(n, base_nf) == "person"
                 or (sm.subject_kind(n, base_nf) is None and _is_humanlike(n))]
        # beings ADD UP ("a woman and her two sisters" = 3), and gender
        # minimums come from the gendered WORDS inside each entry: the
        # relationships bucket carries "father and son" as one phrase
        # with a count of two and no gender, and the roll gave it a girl
        # plural heads count too: 'elf girls' carries its two girls in
        # the word 'girls', which the singular list did not hold
        def _in(w, lst):
            return w in lst or (w.endswith("s") and w[:-1] in lst) \
                or (w.endswith("ies") and w[:-3] + "y" in lst) \
                or (w == "women" and "woman" in lst) or (w == "men" and "man" in lst)
        # PER BEING (the author's 2026-09-06): 'succubus milf' is one woman
        # with two female nouns in her name -- a span's count is its
        # count, its words only say which gender
        _fw = sum(c for n, c in _folk
                  if any(_in(w, _FEMALE_NOUN_LIST) for w in n.split()))
        _mw = sum(c for n, c in _folk
                  if any(_in(w, _MALE_NOUN_LIST) for w in n.split()))
        _pt = max(_pt, sum(c for _n, c in _folk), _fw + _mw)
        _pf = max(_pf, _fw)
        _pm = max(_pm, _mw)
        if (_pt > cen["total"] and not cen["no_humans"]
                and not cen.get("futa")):
            # A NAMED CHARACTER IN THE PAIR HAS A GENDER OF HER OWN (2026-09-19:
            # 'hatsune miku and 1girl' rolled '1girl, 1boy' and even a futa pair --
            # the pair's second member was treated as unknown, though the traits
            # table says Miku is a woman)
            _cf9 = _cm9 = 0
            try:
                for _n9 in _typed_characters(base):
                    _g9 = (_traits().get(_n9) or {}).get("gender")
                    _cf9 += _g9 == "female"
                    _cm9 += _g9 == "male"
            except Exception:
                pass
            _shape = _roll_cast_shape(
                _pt, min(_pt, max(_pf, cen["females"] + _cf9)), min(_pt, max(_pm, cen["males"] + _cm9)),
                mode, rng, level=level)
            if _shape:
                _cast, _roll = _shape
                _cast["source"] = "gendered+pair:" + _roll
                return _cast, tags
        # TWO FUTA CONVENTIONS: gelbooru's `1futa` counts a DISTINCT person
        # ("1futa and 1girl" = two people); danbooru's bare `futanari` rides
        # on a girl count ("1girl, futanari" = one person). The literal count
        # tag decides which arithmetic applies -- subtracting always cost the
        # girl her place in "1futa and 1girl".
        futa = cen.get("futa", 0)
        # raw text only: parse_input itself aliases futanari -> 1futa, so
        # the parsed tags cannot testify about which convention the USER used
        # word numbers and the bare plural count too: "two futas" carried
        # no digit, matched nothing here or in `fm`, and came out as one
        # man. Same vocabulary _WORDNUM already spells out for people.
        distinct = re.search(
            r"\b(\d+|one|two|three|four|five)\s*(?:futas?|futanari)\b|\bmultiple futa(?:s|nari)?\b|\bfutas\b",
            base_nf, re.I)
        if distinct:
            # the census has no multi-futa vocabulary: '2 futas' arrives as
            # zero futa plus a phantom girl. The raw count is authoritative,
            # and the girl count then needs its own non-futa evidence.
            _raw = (distinct.group(1) or "").lower()
            n = (int(_raw) if _raw.isdigit() else _WORDNUM.get(_raw, 2)) if _raw else 2
            futa = max(futa, n)
            own_girls = re.search(
                r"\b(girls?|wom[ae]n|lady|ladies|she|her|\dgirls?)\b",
                (base or ""), re.I)
            females = cen["females"] if own_girls else 0
        elif _girl_count:
            # danbooru convention: "1girl, futanari" -- the futa RIDES on
            # the explicit girl count, so it is one person
            females = cen["females"] - futa
        else:
            # prose names the woman separately -- "a futanari fucking a
            # woman" -- and subtracting here erased her: two people, one of
            # them the futa. The phantom-girl guard above already removed a
            # count with no female word behind it, so what is left is real.
            females = cen["females"]
        return {"female": max(0, females),
                "male": cen["males"],
                "futa": futa,
                "other": max(0, cen["others"] - cen.get("futa", 0)),
                "source": "prompt"}, tags
    # A CHARACTER NAME IS A SUBJECT, and this path outranks the raw-text
    # futa path: 'futanari tifa lockhart fucks aerith gainsborough' is
    # THREE reading decisions (Tifa exists, Aerith exists, futanari
    # modifies Tifa) -- the futa regex firing first collapsed all of it
    # into one anonymous futa slot that a gender-matched draw then handed
    # to Aerith.
    typed = _typed_characters(base)
    fm = re.search(r"(?:(\d+|one|two|three|four|five)\s*)?(?:futas?|futanari)\b|"
                   r"\bmultiple futa(?:s|nari)?\b", base_nf, re.I)
    if typed:
        cast = {"female": 0, "male": 0, "futa": 0, "other": 0,
                "source": "characters"}
        for n in typed:
            g = _traits()[n].get("gender")
            if g in ("female", "male"):
                cast[g] += 1
            else:
                cast["other"] += 1
        if fm:
            # the futanari word MODIFIES the adjacent name when there is
            # one ('futanari tifa ...' -> Tifa is the futa); with no
            # adjacent name, an unnamed futa JOINS the named cast
            futa_bound = None
            for n in typed:
                first = n.split(" (")[0].split()[0]
                if _traits()[n].get("gender") == "female" and re.search(
                        r"futa\w*\s+(?:the\s+)?" + re.escape(first),
                        base or "", re.I):
                    futa_bound = n
                    break
            if futa_bound:
                cast["female"] -= 1
                cast["futa"] += 1
                cast["_futa_name"] = futa_bound
            else:
                cast["futa"] += (int(fm.group(1)) if fm.group(1) else 1)
        return cast, tags
    # '1futa solo' produces NO census at all -- parse_input never emits the
    # 1futa tag -- so raw-text futa counts get their own path
    if fm:
        # the regex now captures word numbers too ("two futas")
        _fr = (fm.group(1) or "").lower()
        # 'multiple futanari' names more than one without saying how many
        n = (int(_fr) if _fr.isdigit() else _WORDNUM.get(_fr, 1)) if _fr else             (2 if "multiple" in str(fm.group(0) or "").lower() else 1)
        gm = re.search(r"\b(\d*)\s*(girls?|wom[ae]n|lady|ladies)\b",
                       (base or ""), re.I)
        g = (int(gm.group(1)) if gm and gm.group(1) else (1 if gm else 0))
        return {"female": g, "male": cen["males"], "futa": n, "other": 0,
                "source": "prompt-futa"}, tags
    m = _GENDERLESS.search(base or "")
    _pair_total = 0 if m else _pair_count_in(base)
    if m or _pair_total:
        if m:
            raw = m.group(1).lower()
            total = int(raw) if raw.isdigit() else _WORDNUM.get(raw, 1)
        else:
            total = _pair_total
        cw = {"weights": _cast_weights(mode, level)}
        _pf = _pm = 0
        if not m:
            _t9, _pf, _pm = _pair_shape_in(base)
        fits = [(n, w) for n, w in cw["weights"].items()
                if sum(v for k, v in _CAST_CLASS.get(n, {}).items()
                       if k in ("female", "male", "futa", "other")) == total
                and _CAST_CLASS.get(n, {}).get("female", 0)
                + _CAST_CLASS.get(n, {}).get("futa", 0) >= _pf
                and _CAST_CLASS.get(n, {}).get("male", 0) >= _pm]
        if fits:
            roll = rng.choices([n for n, _ in fits],
                               weights=[w for _, w in fits], k=1)[0]
            cast = dict(_CAST_CLASS[roll])
            cast["source"] = ("gendered:" if m else "pair:") + roll
            return cast, tags
        return {"female": total,
                "source": "count-only" if m else "pair-count"}, tags
    # ONE EXCEPTION (the author's): an EMPTY prompt is not "scenery by intent",
    # it is full creative mode -- the only case where the generator rolls a
    # cast on its own, from the full measured joint distribution.
    if not (base or "").strip():
        cw = {"weights": _cast_weights(mode, level)}
        names = list(cw["weights"])
        roll = rng.choices(names, weights=[cw["weights"][n] for n in names],
                           k=1)[0]
        cast = dict(_CAST_CLASS.get(roll, {"female": 1}))
        cast["source"] = "creative:" + roll
        return cast, tags
    # PERSON-NOUN / OCCUPATION SUBJECT (the author's systemic fix: the cast
    # reader missed 'a horny female CEO' -> emitted no-humans while the
    # NL described her). A prompt naming a human by OCCUPATION carries a
    # subject even with no gender word. Data-driven: the library's
    # `person` flag marks occupation tags (nurse, office lady, ...), so
    # any person-flagged word present = one human. Gender from a gender
    # word if any, else genderless -> conditional roll (total 1).
    occ_list = _subject_nouns(base)
    occ = occ_list[0] if occ_list else None
    if occ:
        # people and animals are counted on separate ledgers: the
        # gendered roll applies to the people, the animals land in
        # `other`, which is what "a wizard and his owl" means.
        _spans = _subject_spans(base)
        # THREE LEDGERS (the author's, 2026-09-03): a person gets the gendered
        # roll; a HUMANOID being with no gender in its name (gnome, goblin,
        # an anthro fox) is `other`, a character who may wear clothes; a
        # CREATURE (dragon, owl, horse) is in the picture and not in the
        # cast -- it stays a tag, takes no slot, no hair and no shrug.
        # slots.subject_kind is the one classifier; "dragon girl" and
        # "wolf boy" are persons by their gender head and keep their
        # clothes and features when typed (we never roll such kinds).
        _folk = []
        for n, c in _spans:
            kind = sm.subject_kind(n, base)
            if kind == "person" or (kind is None and _is_humanlike(n)):
                _folk += [n] * c
        if not _folk:
            return {"female": 0, "male": 0, "futa": 0, "other": 0,
                    "source": "no-person:%s" % occ}, tags
        occ = _folk[0]
        # HOW MANY, not merely whether. A single gender word cannot
        # speak for a cast of two, so the gendered shortcut applies
        # only when the text describes one being; two or more go to
        # the measured conditional roll for that total -- the same
        # path "2 people kissing" already takes.
        n_subj = len(_folk)
        gword = re.search(r"\b(?:" + _FEMALE_WORDS + r")\b", base_nf, re.I)
        mword = re.search(r"\b(?:" + _MALE_WORDS + r")\b", base_nf, re.I)
        if n_subj == 1 and gword and not mword:
            return {"female": 1, "male": 0, "futa": 0,
                    "other": 0, "source": "occupation:" + occ}, tags
        if n_subj == 1 and mword and not gword:
            return {"female": 0, "male": 1, "futa": 0,
                    "other": 0, "source": "occupation:" + occ}, tags
        # THE WORD ITSELF LEANS, MEASURED. 'a maid cleaning the room' once
        # rolled a man: an occupation was a genderless roll however the
        # booru actually uses the word. occupation_gender.json holds each
        # noun's share of solo posts tagged 1girl against 1boy; at or
        # above 0.90 it is female (maid 0.99, nurse 0.98, waitress 0.98),
        # at or below 0.10 male, in between it stays the measured roll
        # (knight 0.63, butler 0.50, doctor 0.67). Data decides, and the
        # user's own gender word above still outranks it.
        # THE MEASURED SHARE IS THE PROBABILITY. A threshold made 'butler'
        # (0.50) fall to the global cast roll, which is ~92% female for a
        # lone subject -- eleven of twelve butlers were women. When the
        # noun is measured, its own share decides: nurse 0.98 is almost
        # always a woman, butler 0.50 is a coin toss, knight 0.63 leans
        # female the way the booru does. Unmeasured nouns keep the roll.
        _share = _occupation_share(occ)
        if n_subj == 1 and _share is not None and not (gword or mword):
            _fem = rng.random() < _share
            return {"female": 1 if _fem else 0, "male": 0 if _fem else 1,
                    "futa": 0, "other": 0,
                    "source": "occupation:%s (share %.2f)" % (occ, _share)}, tags
        cw = {"weights": _cast_weights(mode, level)}
        fits = [(n, w) for n, w in cw["weights"].items()
                if sum(v for k, v in _CAST_CLASS.get(n, {}).items()
                       if k in ("female", "male", "futa", "other"))
                   == n_subj]
        roll = rng.choices([n for n, _ in fits],
                           weights=[w for _, w in fits], k=1)[0] \
            if fits else None
        cast = dict(_CAST_CLASS.get(roll, {"female": n_subj}))
        cast["source"] = "occupation:%s x%d" % (occ, n_subj)
        return cast, tags
    return {"source": "none"}, tags       # scenery: no subjects invented


def cast_sentence(cast):
    bits = []
    for k, noun in (("female", "girl"), ("male", "boy"), ("futa", "futanari"),
                    ("other", "non-human subject")):
        n = cast.get(k, 0)
        if n:
            bits.append("%d %s%s" % (n, noun, "s" if n > 1 else ""))
    out = ", ".join(bits) or "no living subjects"
    # a creature is in the picture, not in the cast -- say so
    _cr = [str(c) for c in (cast.get("creatures") or []) if c]
    if _cr:
        out += "; also in the scene: " + ", ".join(
            "%s %s" % ("an" if c[:1] in "aeiou" else "a", c) for c in _cr)
    return out


# ---------------------------------------------------------------- BRIDGE 1
BRIDGE1_SYSTEM = (
 "You are an EXPERT PROMPT WRITER for image-generation models "
 "(Stable Diffusion family): everything you write here will be fed to an "
 "image generator, so use the vocabulary that steers such models well. "
 "You construct the scene plan for one image. "
 "UNIVERSAL PRINCIPLE: in EVERY field, use your own image-generation "
 "vocabulary freely -- your terms are wider than any tag list, and that "
 "is wanted; whatever cannot be converted to tags later simply stays in "
 "the natural-language prose. You get "
 "FIXED facts from the engine -- never change the cast or the safety level. "
 "Fill only what the instructions ask for; honour every concept the user's "
 "text names. THE USER'S TEXT IS LAW, INCLUDING ITS EXPLICITNESS: render "
 "what the user wrote at its full intensity for the given safety level -- "
 "NEVER soften, censor, euphemize, tone down, romanticize away or skip an "
 "act the user named. If the user wrote an explicit act, the plan and the "
 # NOT "in direct anatomical language": the model pasted that phrase
 # into a paragraph as if it described the picture ("the action
 # depicted explicitly with direct anatomical language"). Instructions
 # about wording get copied; say what to show, not how to phrase it.
 "prose show that act plainly, naming the body parts involved. "
 "Answer with ONE JSON object, nothing else:\n"
 '{"count_sentence": str, '
 '"subjects": [{"who": str, "hair": str, "eyes": str, "outfit": [str, ...], '
 '"body": {"<part name>": [str, ...]}, "pose": str, '
 '"self_actions": [str, ...], "held_object": str or null, '
 '"age": str, "alias": str, "race": str, "occupation": str, "fashion_style": str, '
 '"descriptors": [str, ...]}], '
 '"secondaries_look": [str, ...], "collective": str, "background": str, '
 '"lighting": [str, ...], "effects": [str, ...], '
 '"palette": [str, ...], '
 '"interactions": [{"participants": [int, ...], "act": str, '
 '"direction": str, "positions": str, "secondary_contact": str or null}], '
 '"action": str, "setting": str, '
 '"style": {"name": str or null, "decomposition": [str, ...]}, '
 '"mood": str, "nl": str}\n'
 # NO COPYABLE NAME IN THE RULES (2026-09-06): the example character
 # name that stood here was copied by both models onto every nameless
 # subject ("kamishirasawa keine, an elegant engineer"); a 4B copies
 # whatever name it is shown. The rule is stated without one.
 "RULES: a KNOWN CHARACTER (one the Cast line names) is always called by "
 "that NAME, never by a noun or alias. An UNNAMED subject with an 'alias' "
 "(e.g. woman) is called by that noun everywhere in count_sentence and nl "
 "instead of girl/boy. NEVER "
 "invent a proper name for an unnamed subject -- naming is not "
 "yours to do; use descriptive references instead. Every subject is "
 "named in full (noun or alias) in every sentence that mentions them; a "
 "pronoun may appear only in a sentence that already names its subject "
 "-- an image model does not resolve pronouns on its own. "
 "Setting and action are SHORT noun/verb phrases, 2-5 words each "
 "-- never sentences. The count_sentence states how many subjects and names known "
 "characters. Outfit items are short noun phrases. DECOMPOSITION RULE: if the "
 "style has a name (an artist, studio, movement or work), decomposition MUST "
 "give 2-4 simpler descriptive phrases conveying the same look; if the style "
 "is generic, decomposition may be empty. The 'nl' field renders the whole "
 "plan as natural language: one small story of the image in plain, "
 "concrete, flowing sentences -- a description for an image generator, "
 "not a list. PRESENT TENSE ONLY: the image is one moment that IS -- 'the "
 "woman sits', 'is holding', 'wears' -- never 'sat', 'was holding', 'wore', "
 "'had'. COVER: each subject named in full with their appearance and "
 "EXACT clothing; what each subject is DOING (the specific pose/position/"
 "act -- if the plan says a sex position like doggystyle or girl on top, "
 "SAY it plainly); the SPATIAL RELATIONS (who is where relative to whom, "
 "in front/behind/on top/beside); the setting; and the light. Every fact "
 "that is in the plan appears in the nl -- above all the act the user "
 "asked for, exactly as asked; never replace it, never leave it out. Start "
 "with the count sentence. Content must fit the given safety level: safe = no "
 "sexualisation at all; sensitive = mild allure allowed; nsfw = nudity "
 "allowed; explicit = sexual acts allowed, described plainly.")


# --------------------------------------------------- SUBJECTS section
# The v2 split: the ENGINE rolls the structure (which slots each subject
# gets, from subject_rules' chance classes scaled by the detail control,
# gates respected), the LLM fills the content of exactly those slots, and
# a persona's canon traits arrive LOCKED -- copied, never re-invented.

CHANCE = {"core": 1.0, "common": 0.6, "uncommon": 0.25, "rare": 0.08}
DETAIL_SCALE = {"minimal": 0.0, "standard": 1.0, "detailed": 1.8}

# attire slots (distilled from subject_rules.json): slot, chance class, gate
_SLOT_MENU = [
    ("hair",      "core",     None),        # colour + length, one of each
    ("eyes",      "core",     None),        # one colour
    ("hair_style","common",   None),
    ("outfit",    "core",     "clothes"),   # the gen-clothes checkbox gates
    ("neckwear",  "uncommon", "clothes"),   # at most one (rules cluster)
    ("headwear",  "uncommon", "clothes"),
    ("legwear",   "common",   "clothes+female"),
    ("accessory", "rare",     "clothes"),
]

# CLOTHES MODE (the author's): a three-way DROPBOX, priority initial prompt >
# option. 'none' = only prompt-described clothes are used -- it does NOT make
# anyone naked, it just generates nothing; 'random' = generate freely, never
# overriding prompt choices (a known character's canon garments are NOT used);
# 'canon' (valid only when a known character is present) = canon outfits for
# known characters, random for the rest. ALL modes -- canon included -- obey
# the awareness rules: gender, location, occupation/profession, genre/setting
# and spice. Nudity states (nude/topless/bottomless) belong to THIS
# subsection as clothing states; naked subjects can still wear accessories.

# INDIVIDUAL POSE / ACTION / OBJECT-INTERACTION (the author's 7 points).
# Solo-involvement only -- anything needing another subject belongs to the
# group-actions section, and the verifier enforces it by ARITY on the mapped
# tags. Composition is BY BODY PART: one overall pose plus a few part-level
# actions, each claiming DIFFERENT parts, so combinations multiply without
# contradicting. Own-body touches ('hand on own hip') and own-clothes
# interactions live here; objects are location/genre/gender/spice-aware.
# WORD THE INTENT, NOT THE CLICHE (the author's: the 8B over-favours 'lying',
# 'posing', 'hand on own hip' -- because the OLD ladder literally named
# them as the template). Describe the ALLURE LEVEL and demand a specific,
# scene-motivated pose; do not hand the model ready-made stock poses.
_ACTION_LADDER = {
    "safe": "casual and non-sexual -- a SPECIFIC action that fits what "
            "this subject is actually doing in this scene; not a generic "
            "stock pose",
    "sensitive": "lightly alluring, but a SPECIFIC pose motivated by the "
                 "scene -- avoid the stock 'lying down' / 'leaning "
                 "forward' / 'hand on hip' defaults",
    "nsfw": "provocative and confident, a SPECIFIC body position that "
            "suits the setting and character -- avoid the stock 'posing "
            "to display' / 'hand on own hip' / 'lying on back' defaults",
    "explicit": "an overtly sexual SOLO act, chosen to fit the character "
                "and setting rather than the generic default",
}
_PART_ACTION_CHANCE = {"safe": 0.35, "sensitive": 0.45, "nsfw": 0.6,
                       "explicit": 0.75}
# THE PAIR LADDER. _ACTION_LADDER describes what ONE subject does on their
# own; the interaction brief quoted it for two, and at the explicit level
# the model dutifully wrote "X and Y are performing an explicit solo act".
_PAIR_LADDER = {
    "safe": "a casual, non-sexual shared activity specific to this scene",
    "sensitive": "lightly intimate contact motivated by the scene",
    "nsfw": "provocative contact between them, short of a sex act",
    "explicit": "an explicit sexual act between them -- the one the user "
                "typed if there is one, otherwise chosen to fit the pair",
}


# AGE / OCCUPATION / GENERAL DESCRIPTIONS (the author's, 2026-08-29): flavor,
# not necessity -- low chances, detail-scaled. Age may be years, a modifier
# ("young") or implied ("woman"), and for an UNNAMED subject it may change
# the ALIAS the natural language uses ("woman" instead of "girl" to imply
# maturity) -- the TAG part keeps booru count conventions regardless.
# Occupation is genre/setting/location-aware. General descriptors are
# spice-aware and NEVER negative unless the user's own text asked for it.
# THE OCCUPATION IS ALWAYS OFFERED (2026-09-16): whether a subject has a
# profession is decided once, by the measured no-profession share of the
# place (a hospital .62, a park 1.0 -- occupation_for), not by a hand
# chance layered on top of it
_FLAVOR_CHANCE = {"age": 0.30, "occupation": 1.0, "general": 0.45,
                  "fashion_style": 0.20}

# AGE -> TAG is a FIXED WHITELIST (the author's), engine-mapped, never fuzzy:
# retrieval on age words once reached 'adult baby'. Only maturity-coded tags
# exist here, gender-matched; anything not matching stays NL-only (the alias
# noun still carries it). Youth-coded tags are deliberately absent.
_AGE_TAGS = {
    "female": [(r"\b(milf)\b", "milf"),
               (r"\b(old|elderly|grandmother|granny)\b", "old woman"),
               (r"\b(mature|adult|middle-?aged|woman)\b", "mature female")],
    "futanari": [(r"\b(old|elderly)\b", "old woman"),
                 (r"\b(mature|adult|middle-?aged|woman)\b", "mature female")],
    "male": [(r"\b(old|elderly|grandfather)\b", "old man"),
             (r"\b(mature|adult|middle-?aged|man)\b", "mature male")],
}


_AGE_TAG_SET = {t for rows in _AGE_TAGS.values() for _rx, t in rows}


_TYPED_OLD = re.compile(r"\b(old (?:wom[ae]n|m[ae]n|lady|ladies)|elderly|grandmother|granny|grandma|"
                        r"grandfather|grandpa)\b", re.I)


def map_age(age_text, alias, kind, typed=""):
    """AN OLD WOMAN IS OLD ONLY WHEN THE USER SAYS SO (the author, 2026-10-03:
    'old woman' rolled on 'mature' and 'milf' prompts). The model's age field
    reached it two ways -- '35 years old' holds the bare word 'old', and an age
    of sixty or more added it. The model's age text gives 'milf' or 'mature'
    at most; 'old woman' / 'old man' need the user's own words (`typed`)."""
    blob = ((age_text or "") + " " + (alias or "")).lower()
    m = re.search(r"\b(\d{2})\b", blob)
    if m and int(m.group(1)) >= 30:
        blob += " mature"
    _old_ok = bool(_TYPED_OLD.search(str(typed or "")))
    for pat, tag in _AGE_TAGS.get(kind, []):
        if tag in ("old woman", "old man") and not _old_ok:
            continue
        if re.search(pat, blob):
            return tag
    return None
_GENERAL_LADDER = {
    "safe": "wholesome flavor words (beautiful, cute, elegant, graceful)",
    "sensitive": "may include attractive, alluring, charming",
    "nsfw": "may include sexy, seductive, sultry",
    "explicit": "may include overtly sexual flavor words (lewd, erotic)",
}


# SUBJECT TIERS (the author's): main / secondary / background. Image models hold
# TWO identities reliably, three marginally -- that is the main cap. The
# more subjects, the simpler each description (one budget, spread thinner):
#   N=1: 1 main            N=4:  2 main + 2 secondary
#   N=2: 2 main            N=5:  2 main + 3 secondary
#   N=3: 2 main + 1 sec    N=6+: 2 main + 3 secondary + OVERFLOW COLLECTIVE
# Named characters are ALWAYS main (the user named them deliberately; naming
# four makes four mains and every description auto-simplifies to pay for it).
# Main-count caps the detail level: 1 main = user's choice, 2 mains <=
# standard, 3+ mains = minimal. Secondaries get ONE look phrase, no
# pipeline. The BACKGROUND entity (crowd) is outside the count entirely:
# detected from the user's text, never rolled, emits crowd-family tags plus
# solo focus instead of count tags.
MAIN_CAP = 3
SECONDARY_CAP = 3
_BACKGROUND_RE = re.compile(
    r"\b(crowd(ed)?|busy street|audience|onlookers|spectators|passers.?by|"
    r"bystanders|people in( the)? background|crowded)\b", re.I)
_GROUP_ACT_RE = re.compile(r"\b(orgy|gangbang|group sex|gang.?bang)\b", re.I)


def assign_tiers(n_subjects, n_named, base, detail_name):
    """-> (n_main, n_secondary, collective, background, capped_detail,
    all_secondary)"""
    background = bool(_BACKGROUND_RE.search(base or ""))
    all_secondary = bool(_GROUP_ACT_RE.search(base or "")) and n_subjects >= 4
    if all_secondary:
        return 0, min(n_subjects, 6), n_subjects > 6, background, \
               "minimal", True
    n_main = max(1, min(2, n_subjects))
    if n_named >= 3:
        n_main = min(n_named, MAIN_CAP + 1)   # 4 named = 4 mains, all pay
    elif n_named > n_main:
        n_main = min(n_named, MAIN_CAP)
    n_main = min(n_main, n_subjects)
    n_sec = min(n_subjects - n_main, SECONDARY_CAP)
    collective = n_subjects - n_main - n_sec > 0
    cap = {1: detail_name,
           2: ("standard" if detail_name == "detailed" else detail_name)}
    capped = cap.get(n_main, "minimal")
    return n_main, n_sec, collective, background, capped, False


def plan_flavor(kind, detail, rng):
    if kind not in ("female", "male", "futanari"):
        return []
    return [f for f, p in _FLAVOR_CHANCE.items()
            if rng.random() < min(0.9, p * max(detail, 0.4))]


# A WHOLE-BODY ACT OWNS THE POSE: the gloss says which tags are acts or
# poses, and a pose tag ('standing') or an act the whole body performs
# ('dancing', 'running', 'swimming') leaves no room for a second posture.
_POSTURE_ACT = re.compile(
    r"^(danc\w*|run\w*|walk\w*|jump\w*|swim\w*|fly\w*|fall\w*|sit\w*|stand\w*|kneel\w*|lying|lie\w*|"
    r"crouch\w*|squat\w*|leap\w*|climb\w*|stretch\w*|pos(e|ing)\w*|bend\w*|lean\w*|straddl\w*|"
    r"riding|bath\w*|sleep\w*|rest\w*)\b")


def _whole_body_act(act):
    """True when the scene's act already says what the whole body does"""
    for t in ([act] if isinstance(act, str) else list(act or ())):
        t = str(t or "").strip().lower()
        if not t:
            continue
        if _POSTURE_ACT.match(t) or (_gloss_flags(t) or set()) & {"pose"}:
            return True
    return False


def plan_actions(kind, level, detail, rng, act=None):
    """the engine's action BUDGET for one subject: pose yes/no, how many
    part-level actions, whether an object may be involved. Content is the
    LLM's; the budget and the arity check are the engine's.

    THE SCENE'S ACT IS THE POSE (the author, 2026-09-18: "dancing became
    running"). A whole-body act the user typed or the engine drew already
    says what the body does; a pose rolled beside it contradicts it --
    'dancing' with 'standing' in 78 of 100 rolls, and the prose then picks
    one of the two. The pose slot is skipped when the scene has such an act;
    part-level actions (hands, gaze) still roll, since they layer over it."""
    if kind not in ("female", "male", "futanari"):
        return None
    plan = {"pose": rng.random() < 0.7 and not _whole_body_act(act)}
    n = 0
    p = _PART_ACTION_CHANCE[level] * max(detail, 0.4)
    while n < 2 and rng.random() < p:
        n += 1
        p *= 0.5
    plan["n_part_actions"] = n
    plan["object_ok"] = rng.random() < 0.4
    return plan if (plan["pose"] or n) else None

# wearables, for stripping canon garments under 'none'/'random'
_GARMENT_WORDS = {
    "shirt", "skirt", "dress", "uniform", "jacket", "coat", "sweater",
    "swimsuit", "bikini", "leotard", "serafuku", "vest", "shorts", "pants",
    "kimono", "yukata", "armor", "apron", "sleeves", "necktie", "bowtie",
    "socks", "thighhighs", "pantyhose", "kneehighs", "shoes", "boots",
    "heels", "gloves", "hat", "cap", "hood", "hoodie", "scarf", "choker",
    "collar", "hairband", "headband", "ornament", "ribbon", "bow",
    "top", "bra", "panties", "underwear", "footwear", "headwear",
}


_WORN_WORDS = {"data": None}


def names_garment(tag):
    """-> True when the tag is a garment or names one ('shirt lift', 'no bra',
    'see-through clothes', 'panties around ankles'): it belongs to the clothes'
    sentence. 'underboob', 'one breast out', 'cleavage' name the body."""
    t = str(tag or "").lower().strip()
    if not t:
        return False
    if _WORN_WORDS["data"] is None:
        try:
            items = {x for it in ((_clothes_table() or {}).get("slots") or {}).values() for x in it}
        except Exception:
            items = set()
        _WORN_WORDS["data"] = (items, {x for x in items if " " not in x} | set(_GARMENT_WORDS)
                               | {"clothes", "clothing", "outfit"})
    items, words = _WORN_WORDS["data"]
    return t in items or _is_garment(t) or any(w in words for w in t.split())


def _is_garment(tag):
    head = tag.split()[-1] if tag.split() else str(tag)
    return head in _GARMENT_WORDS or \
           (len(tag.split()) > 1 and tag.split()[-2] in _GARMENT_WORDS) or \
           any(head.endswith(w) for w in _GARMENT_WORDS if len(w) >= 4)   # sundress, miniskirt, undershirt

# BODY DESCRIPTION MENU (the author's 16-point spec, 2026-08-29). Each part:
# (name, gate, base chance, spice_scaled). DESCRIPTION ONLY -- no
# interactions, no poses; that is the actions section's territory. Priority
# per part: user text > canon > LLM free choice. Chances are per-part and
# multiply with the detail control; spice_scaled parts also multiply with the
# level (frequency AND instructed depth rise with spice).
_BODY_MENU = [
    # part          gate            base   spice-scaled
    ("face",        None,           0.60,  False),
    ("ears",        "distinct",     0.06,  False),  # elf ears etc: only when
    ("nose",        "distinct",     0.04,  False),  # a distinct characteristic
    ("eyebrows",    "expression",   0.06,  False),  # when the face needs them
    ("facial hair", "male",         0.08,  False),
    ("breasts",     "female_futa",  0.40,  True),   # not only size: shape too
    ("ass",         None,           0.30,  True),
    ("body shape",  None,           0.40,  False),  # waist, build, height
    ("skin",        None,           0.15,  True),   # colour AND texture
    ("hands",       None,           0.06,  False),
    ("legs and feet", None,         0.08,  False),
    # marks (freckles, moles, scars, tattoos) are the BODY TABLE's alone
    # (body_for: measured share, detail-scaled) -- the menu's own 'markings'
    # line drew them a second time (2026-09-06)
    ("pubic hair",  "genital",      0.00,  True),   # measured: genital_chance
    ("pussy",       "pussy",        0.00,  True),
    ("penis",       "penis",        0.00,  True),
]
_SPICE_MULT = {"safe": 0.5, "sensitive": 1.0, "nsfw": 1.5, "explicit": 2.0}


def plan_body(kind, has_pussy, level, detail, rng, suppress=frozenset()):
    """which body parts get described for one subject. FUTANARI HAS BREASTS
    AND A PENIS -- and sometimes a pussy too (several futa types; the engine
    rolls the type once per subject and the flag arrives here)."""
    # HUMANOID SUBJECTS ONLY (the author's): an 'other' subject -- creature,
    # robot, monster -- gets a free-form appearance from the LLM, never this
    # anatomy menu. Monster GIRLS are female subjects and come through
    # normally.
    if kind not in ("female", "male", "futanari"):
        return []
    female_body = kind in ("female", "futanari")
    parts = []
    for part, gate, base, scaled in _BODY_MENU:
        if part in suppress:
            continue                 # out of frame: a portrait needs no feet
        if gate == "male" and kind != "male":
            continue
        if gate == "female_futa" and not female_body:
            continue
        if gate == "penis" and kind not in ("male", "futanari"):
            continue
        if gate == "pussy" and not (kind == "female" or
                                    (kind == "futanari" and has_pussy)):
            continue
        if gate in ("distinct", "expression"):
            # only when the scene gives a reason; the LLM is told the
            # condition and may add it, the dice never force it
            if rng.random() >= base * detail:
                continue
            parts.append(part + " (only if it is a distinct feature)")
            continue
        if gate == "genital" or part in ("pussy", "penis"):
            # MEASURED (2026-09-15): the part's share of the level's rating
            # universe, not a hand chance ('pubic hair' was .5 at nsfw and
            # .9 at explicit against the booru's .03 and .13)
            p = genital_chance(part, level)
        else:
            p = base * detail * (_SPICE_MULT[level] if scaled else 1.0)
        if rng.random() < min(0.95, p):
            parts.append(part)
    return parts

_TRAITS = None


def _traits():
    global _TRAITS
    if _TRAITS is None:
        try:
            with open(_paths.data("character_traits.json"),
                      encoding="utf-8-sig") as f:
                _TRAITS = json.load(f)["characters"]
        except Exception:
            _TRAITS = {}
    return _TRAITS


_SURNAMES = None


def _lev2(a, b):
    """edit distance, capped reasoning: only <=2 matters here"""
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_COMMON_WORD = {}


def _species_phrase(bare):
    """'bunny girl' names a KIND, not a person -- an original character
    whose name is literally 'bunny girl (yuuhagi (...))' matched the
    typed species and dressed the subject in that character's canon."""
    b = (bare or "").lower().strip()
    if sm.subject_kind(b) == "person" and re.search(
            r"\b(girls?|boys?|wom[ae]n|m[ae]n|lady)$", b):
        head = b.rsplit(" ", 1)[0] if " " in b else ""
        return bool(head) and (sm.subject_kind(head) in ("creature", "humanoid")
                               or head in ("bunny", "monster", "slime", "cow",
                                           "sheep", "mouse", "bat", "bee"))
    return False


def _common_word(name, min_tags=4):
    """-> True when `name` is ordinary vocabulary rather than an identity.

    Measured, not a hand-list: count how many tags in the emittable
    vocabulary use the word. 'knight' and 'neon' appear in many ('neon
    lights', 'neon trim', 'armored knight'), 'hatsune miku' in none.
    A multi-word name is treated as identity -- two words colliding with
    ordinary vocabulary is vanishingly rare and the check would be slow.
    EXCEPT when the extra word is only an ARTICLE: danbooru carries
    characters literally named 'a knight (reverse:1999)', and the bare
    phrase 'a knight' sailed past the multi-word bail-out, so "a knight in
    a castle" cast a gacha character. Stripping a leading article and
    re-testing the remaining single word closes that without touching real
    multi-word names ('black rock shooter' keeps every word).
    """
    w = str(name).strip().lower()
    if not w:
        return False
    if " " in w:
        # A MULTI-WORD NAME THAT IS ITSELF A GENERAL TAG IS A WORD FIRST
        # (2026-09-16: 'an old man fishing' cast 'old man (guin guin)', an
        # Arknights character; 'old man' is a danbooru tag on 100k posts).
        # The same test the artist rule uses, before the identity bail-out.
        if w in _vocab():
            return True
        head = w.split(" ", 1)
        if head[0] in ("a", "an", "the") and " " not in head[1]:
            w = head[1]
        else:
            return False
    if w in _COMMON_WORD:
        return _COMMON_WORD[w]
    if w in _vocab():
        _COMMON_WORD[w] = True
        return True
    # the WORD INDEX, built once: how many vocabulary tags carry each word
    # as a whole token (each tag counted once). The scan this replaces
    # split the whole vocabulary per name -- 129 million splits on the
    # first generation of every studio session (the author's 40 s, 2026-09-05).
    _COMMON_WORD[w] = _word_tag_count().get(w, 0) >= min_tags
    return _COMMON_WORD[w]


_WORD_TAG_COUNT = {"data": None, "n": -1}


_LEFTOVER_STOP = set("""a an the and or but of to in on at by for with from into onto over under
her his their its she he they them it is are was were be being been as while during through
across along near beside behind before after above below between among around against without
within this that these those there here some any each every all both few many much more most
very just only also still then than so too such own same other another one two three four five
six seven eight nine ten first second third who whom whose which what where when how up down out
off away back again about like doing being having getting
him me us you i we my your our mine yours ours hers theirs himself herself themselves itself""".split())


STATUS = {"stage": "", "detail": "", "t0": 0.0, "calls": 0}


def _stage(name, detail=""):
    """the studio polls this: what the engine is doing right now"""
    STATUS.update(stage=name, detail=detail)


def exclusion_terms(opts):
    """the user's 'what should not appear' terms: lowercase, comma or
    newline separated, at most 30"""
    raw = (opts or {}).get("exclude") or []
    if isinstance(raw, str):
        raw = re.split(r"[,;\n]+", raw)
    out = []
    for t in raw:
        t = str(t).lower().strip(" .")
        if t and t not in out and len(t) <= 60:
            out.append(t)
    return out[:30]


def _hits_exclusion(text, terms):
    """whole-word: 'hat' hits 'witch hat' and 'hats', not 'that'; the
    booru alias of a term hits too"""
    low = str(text).lower()
    try:
        from promptstudio.engine import vocab as _vc
        al = _vc._load_aliases()
    except Exception:
        al = {}
    for term in terms:
        if al.get(term) == low or al.get(low) == term:
            return True
        stem = term[:-1] if term.endswith("s") and len(term) > 3 else term
        if re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?:s|es)?(?![a-z0-9])", low):
            return True
    return False


def _coverage(user_tags, final_tags, opts, unmapped):
    """every typed concept and its fate on the line"""
    final = {str(t).lower() for t in (final_tags or [])}
    drop = opts.get("_dropped_unknown") or {}
    phr = set(opts.get("_phrases") or [])
    rows = []
    for t in user_tags or []:
        t2 = str(t).lower()
        gone = [m for m, lst in drop.items() if t2 in [str(x).lower() for x in (lst or [])]]
        _count = bool(re.match(r"^(\d(girls?|boys?|others?|futas?)|multiple (girls|boys)|solo|no humans)$", t2))
        # A TAG ANOTHER TAG ALREADY SAYS IS STILL A TAG (the author, 2026-09-19:
        # "'pants -> prose', 'sweater -> prose', 'breasts -> prose' - are those not
        # recognised as tags?"). They are: the line says each fact once, and
        # 'black pants' says 'pants', 'white sweater' says 'sweater', 'large
        # breasts' says 'breasts' (the booru's own implications, or the same
        # head word). The card now says which tag carries it.
        _by = None
        if t2 not in final and not _count and not gone:
            try:
                _by = next((f for f in sorted(final) if t2 in implied(f)), None)                     or next((f for f in sorted(final) if f != t2 and f.endswith(" " + t2)), None)
            except Exception:
                _by = None
        rows.append({"concept": t2, "fate": "tag" if (t2 in final or _count)
                     else ("tag (said by '%s')" % _by if _by
                           else ("dropped on " + ", ".join(gone) if gone else "prose"))})
    for c in sorted(_LEFTOVER_CONCEPTS):
        rows.append({"concept": c, "fate": "phrase" if c in phr else "prose, stored for review"})
    return rows


def negative_line(mode, opts=None):
    """the standard negative plus the user's exclusions"""
    terms = exclusion_terms(opts)
    return negative_for(mode) + ((", " + ", ".join(terms)) if terms else "")


_LEFTOVER_CONCEPTS = set()       # this generation's typed leftovers: never fuzzy-mapped
_STEM_STOP = _LEFTOVER_STOP | {"hair", "eyes", "eye", "style", "color", "colour"}


def _garment_stem(w):
    """the clothes roll's stem: 'panties' -> 'panty', 'gloves' -> 'glove'"""
    w = str(w).lower()
    return w[:-3] + "y" if w.endswith("ies") else (w[:-1] if w.endswith("s") and len(w) > 3 else w)


_ADOPTED = {"mtime": 0, "data": None}


def _adopted_requires(tag):
    """-> (needs clothing, [garment alternatives]) the author's adoption ruling
    states for a tag (adopted_tags.json 'constraints'): 'upshirt' names no
    garment as a word and 'hadanugi dousa' none at all, yet one needs a shirt
    and the other a kimono (2026-09-18: both were drawn onto bodies wearing
    neither). The ruling is data; the name rule stays for everything else."""
    tb = _json_table(_ADOPTED, "adopted_tags.json") or {}
    cons = ((tb.get("tags") or {}).get(str(tag).lower()) or {}).get("constraints") or []
    garments = [g.strip() for c in cons if c.startswith("requires garment:")
                for g in c.split(":", 1)[1].split("|") if g.strip()]
    return bool(garments) or "requires any clothing" in cons, garments


_GARMENT_SLOTS = ("top", "bottom", "swim", "uniform", "traditional", "legwear", "feet", "head", "hands", "neck",
                  "sexual")
_BODY_GARMENT_SLOTS = ("top", "bottom", "swim", "uniform", "traditional", "sexual")
_PART_NEEDS_SHARE = 0.4      # a part shown with its garment on this share of its own posts needs it


_CLOTHING_GLOSS_RE = re.compile(r"\b(cloth\w*|garment\w*|zipper\w*|unzip\w*|button\w*|outfit\w*|fabric)\b", re.I)


def is_clothing_state(tag):
    """a state of clothes that names no garment ('unbuttoned', 'unzipped', 'open
    clothes'): the gloss flags it clothing, or its gloss speaks of clothes"""
    t = str(tag or "").lower().strip()
    if not t:
        return False
    _fl = _gloss_flags(t) or set()
    if "clothing" in _fl or set(t.split()) & {"clothes", "clothing", "outfit"}:
        return True
    if "body" in _fl:
        return False                    # 'tanlines', 'bare shoulders': the body's, whatever the gloss recalls
    try:
        return bool(_CLOTHING_GLOSS_RE.search(str((_gloss_of(t) or ("",))[0] or "")))
    except Exception:
        return False


def _garment_head(tag):
    """the stem of the word that names the thing: 'panties on head' -> panty,
    'hat feather' -> feather, 'open shirt' -> shirt"""
    ws = re.split(r"\s+(?:on|around|over|under|between)\s+", str(tag or "").lower().strip())[0].split()
    return _garment_stem(ws[-1]) if ws else ""


def garment_decorations():
    """the decorations of a garment: the clothes table's accessories section (hat
    bow, hat feather, hat flower, hat ribbon, hat tassel, the hat ornaments)"""
    cs = (_clothes_table() or {}).get("slots") or {}
    return {t for t, e in (cs.get("print") or {}).items() if (e or {}).get("section") == "accessories"}


def lone_garment_states(groups, keep=(), lacking=None):
    """-> [(tag, reason)]: THE PART OR THE STATE OF A GARMENT ITS SUBJECT DOES NOT
    WEAR (the author's live tests, 2026-10-03: `unbuttoned` on a woman in a hat
    feather and gloves, `hat feather` with no hat). The clothes roll checks its
    own modifiers (_names_worn); the accessories, the level's states and the
    adopted tags came by other doors. One check on the settled line.

    A STATE THAT NAMES ITS GARMENT IS THE GARMENT (the author, same day: "we try
    to use 'skirt lift' instead of 'skirt, skirt lift' -- the tagline shouldn't
    name the garment twice if it is already included with the act"). The booru
    says 'skirt lift' implies `skirt`, 'bra lift' `bra`, 'hood up' `hood`, 'hat
    ribbon' `hat`: the line drops the bare garment beside such a tag (each fact
    once), and the tag standing alone says she wears it. So an implied garment
    is never asked for a second time -- the first cut of this check did, and
    stripped 'pants pull' and 'bra lift' from bodies whose only pants and bra
    they were. It counts as worn, for the tag itself and for every other check.

    What is judged, of the tags that are not the user's or a persona's:
      - a state whose garment is one of two that cannot both be worn: 'skirt
        lift' on a body in pants, 'pants pull' on one in a skirt (one bottom);
      - an adoption ruling's garment ('bra strap' requires a bra, 'wind lift' a
        skirt or a dress) -- worn, or implied by another tag of hers;
      - a table part the booru does NOT tie to its garment by implication,
        whose own posts show that garment (garment_pairs: 'hat feather' with a
        hat .94, 'hat ornament' .96; 'ribbon trim' with a ribbon .29 does not);
      - a state of clothing that names none ('unbuttoned', 'unzipped', 'open
        clothes', a print) needs a garment on the body, not a hat and gloves.
    `groups` is {subject: [her tags]} -- a tag two subjects share is in both
    lists, and it is lone only where it is lone for every subject that has it.
    `keep` is never judged -- but a kept DECORATION whose garment is missing is
    reported in `lacking` as (subject, decoration, garment), for the caller to
    put the garment on: the user's 'hat ribbon' is law, and it sits on a hat."""
    cs = (_clothes_table() or {}).get("slots") or {}
    gw = _garment_stems()
    gp = (_json_table(_GARMENT_PAIRS, "garment_pairs.json") or {}).get("garments") or {}
    nud = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
    keep = {str(k).lower() for k in keep}
    in_gs = {t for sl in _GARMENT_SLOTS for t in (cs.get(sl) or {})}
    states = set(cs.get("modifier") or {}) | set(cs.get("print") or {})
    bottoms = set(cs.get("bottom") or {})
    # A DECORATION IS NOT AN ACT (the author, 2026-10-03: "both hat feather and hat
    # ribbon cannot exist without a hat -- they are hat decorations"). The wiki's
    # accessories list (hat bow, hat feather, hat flower, hat ribbon, hat tassel,
    # the hat ornaments): each sits ON a garment and says nothing of wearing it,
    # whatever the booru's implication. The garment its name carries must be worn
    # as a garment -- by a hat of any kind, never by another decoration.
    decor = garment_decorations()
    # THE GARMENT NOUNS: a one-word garment of the table, or a head word two or
    # more of its garments share ('cap', 'bow', 'top'). The stems list also holds
    # 'up' (pants rolled up), 'food' (food on head) and 'flower' (ear flower).
    _hc = {}
    for t in in_gs:
        if " " in t:
            _hc[_garment_head(t)] = _hc.get(_garment_head(t), 0) + 1
    gn = {_garment_stem(t) for t in in_gs if " " not in t} | {h for h, c in _hc.items() if c >= 2}
    gn -= {"clothe", "clothing", "outfit", "hand", "head", "neck", "foot", "leg", "arm", "body", "skin"}

    def _garments_of(u):
        """the garments a tag puts on its subject: itself, and what it implies"""
        out = set()
        if u in in_gs or _garment_head(u) in gn:
            out.add(u)
        for x in implied(u):
            if x in in_gs or (len(str(x).split()) == 1 and _garment_head(x) in gn):
                out.add(x)
        return out

    _one = {_garment_stem(t): t for t in in_gs if " " not in t}

    def _named(u):
        """the garment a state names in its own words where the booru files no
        implication ('shirt lift' -> shirt): worn, as far as the others can tell"""
        if u not in states and u not in nud:
            return set()
        return {_one[_garment_stem(x)] for x in u.split() if _garment_stem(x) in _one and _garment_stem(x) in gn}

    verdict = {}                      # tag -> [reason or None, per subject that has it]
    for sj, own in groups.items():
        own = [str(u).lower() for u in own]
        gone = set()

        def _worn(skip):
            return {_garment_head(g) for u in own if u != skip and u not in gone and u not in decor
                    for g in (_garments_of(u) | _named(u))}

        def _dressed(skip):
            for u in own:
                if u == skip or u in gone or u in decor or u.startswith(("no ", "unworn ")):
                    continue
                for g in (_garments_of(u) | _named(u)):
                    if any(g in (cs.get(sl) or {}) for sl in _BODY_GARMENT_SLOTS) or (_cl_slots_of(cs, g) - {"legwear"}):
                        return True
            return False

        for tl in list(own):
            _kept = tl in keep
            if (_kept and tl not in decor) or tl.startswith(("no ", "unworn ", "naked ")) or _is_whole_nudity(tl):
                continue
            head = _garment_head(tl)
            fl = _gloss_flags(tl) or set()
            _any, _alts = _adopted_requires(tl)
            # A GARMENT IS NOT A PART OF ITS KIND ('miniskirt' implies `skirt`): only
            # a part or a state is judged -- a table modifier or accessory, a state
            # of undress, an adopted tag, a tag the gloss calls clothing that no
            # garment slot holds
            _is_state = tl in states or tl in nud or _any or bool(_alts)
            if tl in in_gs or not (_is_state or ("clothing" in fl and head not in gn)):
                continue
            why = None
            if tl in decor:
                _on = [w for w in dict.fromkeys(_garment_stem(x) for x in tl.split())
                       if w in _one and w in gn and w != head]
                _has = _worn(tl)
                _lack = [w for w in _on if w not in _has]
                if _kept:
                    if _lack and lacking is not None:
                        lacking.append((sj, tl, _one[_lack[0]]))
                    continue
                verdict.setdefault(tl, []).append(
                    "a decoration of the %s, which this subject does not wear" % _lack[0] if _lack else None)
                if _lack:
                    gone.add(tl)
                continue
            _mine = _garments_of(tl) - {tl}
            # one bottom: the state's own garment against the one she wears
            for g in _mine:
                if g in bottoms:
                    for u in own:
                        if u == tl or u in gone or u not in in_gs or "bottom" not in _cl_slots_of(cs, u):
                            continue
                        if _garment_head(u) != _garment_head(g) and g not in implied(u) and u not in implied(g):
                            why = "%s is on these hips, not the %s it speaks of" % (u, g)
            if _alts and head in {_garment_head(a) for a in _alts}:
                _alts = []                      # 'spiked armor' is the armor it needs
            if why is None and _alts and not ({_garment_head(a) for a in _alts} & (_worn(tl) | {_garment_head(g) for g in _mine})):
                why = "needs %s, which this subject does not wear" % " or ".join(_alts)
            # a table part the booru does not tie to its garment, whose posts show it
            if why is None and tl in states and head not in gn and not _mine:
                row = (gp.get(tl) or {}).get("with") or {}
                for w in dict.fromkeys(_garment_stem(x) for x in tl.split()):
                    if w in gw and w != head and max([f for b, f in row.items() if b in in_gs
                                                      and _garment_head(b) == w] or [0.0]) >= _PART_NEEDS_SHARE \
                            and w not in _worn(tl):
                        why = "a part of the %s, which this subject does not wear" % w
            # a state of clothing that names no garment
            if why is None and not _mine and not _alts and (tl in states or tl in nud or _any) \
                    and not any(_garment_stem(x) in gn for x in tl.split()):
                _acc = ((cs.get("print") or {}).get(tl) or {}).get("section") == "accessories"     # a ribbon, a bell
                if (_any or (is_clothing_state(tl) and not _acc)) and not _dressed(tl):
                    why = "a state of clothes on a body with none"
            verdict.setdefault(tl, []).append(why)
            if why:
                gone.add(tl)
    return [(t, next(w for w in whys if w)) for t, whys in verdict.items() if whys and all(whys)]


def _adopted_fits_body(tag, kind):
    """the ruling's body conditions ('testicle peek' requires a penis, 'sidepec'
    a male, 'clothes between breasts' breasts): 2026-09-19, 'testicle peek' was
    drawn onto a woman. kind: female / male / futanari / other."""
    tb = _json_table(_ADOPTED, "adopted_tags.json") or {}
    cons = set(((tb.get("tags") or {}).get(str(tag).lower()) or {}).get("constraints") or [])
    kind = kind or "female"
    if "requires male" in cons and kind != "male":
        return False
    if "requires female" in cons and kind not in ("female", "futanari"):
        return False
    if "requires part:penis" in cons and kind not in ("male", "futanari"):
        return False
    if ({"requires part:breasts", "requires part:pussy"} & cons) and kind not in ("female", "futanari"):
        return False
    if "requires partner" in cons or "requires furry" in cons:
        return False                      # a lone clothes roll cannot promise either
    return True


def _spice_shifters():
    """the states the author ruled in as cheap spice ('revealing / impossible /
    taut clothes ... cost almost nothing in calculation of contradictions'):
    the adopted tags whose ruling carries 'shifts spice up'"""
    tb = _json_table(_ADOPTED, "adopted_tags.json") or {}
    return {t for t, a in (tb.get("tags") or {}).items() if "shifts spice up" in (a.get("constraints") or [])}


_GARMENT_STEMS = {"key": None, "data": set()}


def _garment_stems():
    """the head words of every garment the clothes table knows, by stem --
    the one list that says whether a state names a garment ('shirt lift',
    'panty pull', 'open kimono')"""
    slots = (_clothes_table() or {}).get("slots") or {}
    if _GARMENT_STEMS["key"] == id(slots):
        return _GARMENT_STEMS["data"]            # (rebuilt on every call before: 4s in 500 calls)
    ws = set()
    for sl in ("top", "bottom", "swim", "uniform", "traditional", "legwear", "feet", "head", "hands", "neck", "sexual"):
        for t in (slots.get(sl) or {}):
            # 'panties on head', 'bandana around neck': the garment is the
            # word before the preposition, never the body part
            head = re.split(r"\s+(?:on|around|over)\s+", t)[0].split()[-1]
            ws.add(_garment_stem(head))
    ws = ws - {"clothe", "clothing", "outfit", "hand", "head", "neck", "foot", "leg", "arm", "body", "skin"}
    _GARMENT_STEMS["key"], _GARMENT_STEMS["data"] = id(slots), ws
    return ws


def _stem(w):
    w = w.lower()
    for suf in ("ies", "es", "s", "ed", "ing"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)] + ("y" if suf == "ies" else "")
    return w


def _shares_stem(concept, tag):
    """the MEANING GATE for a pick no model judged (the fast path's rank-1):
    the tag shares a content-word stem with the concept, or the booru's
    own alias table says the concept IS the tag ('blowjob' -> fellatio).
    'dais' -> 'ao dai' fails; 'crimson hair' -> 'red hair' fails and stays
    a phrase; 'pauldron' -> 'pauldrons' passes."""
    c = str(concept).lower().strip()
    t = str(tag).lower().strip()
    if c == t:
        return True
    try:
        from promptstudio.engine import vocab as _vc
        if _vc._load_aliases().get(c) == t:
            return True
    except Exception:
        pass
    cw = {_stem(w) for w in re.findall(r"[a-z0-9]+", c) if w not in _STEM_STOP}
    tw = {_stem(w) for w in re.findall(r"[a-z0-9]+", t) if w not in _STEM_STOP}
    # A MECHANICAL PICK MAY DROP THE USER'S MODIFIERS, NEVER ADD ITS OWN
    # (2026-09-15: 'sexy pose' became 'paw pose', 'drawn art' 'pixel art',
    # 'exaggerated ... proportions' 'bad proportions' -- each shared a
    # word and brought a meaning nobody asked for): every content word of
    # the tag must be one of the concept's ('cluttered desk' -> 'desk').
    # The qualifier in parentheses is booru bookkeeping, not a word.
    tw -= {_stem(w) for w in re.findall(r"\(([a-z0-9 ]+)\)", t) for w in w.split()}
    return tw <= cw if cw and tw else c.split()[0] == t.split()[0]


def _meaning_ok(concept, tag):
    """the meaning gate for a pick a model judged: a shared stem, the booru
    alias, or a concept word in the tag's gloss ('blowjob' -> fellatio by
    alias; 'crimson hair' -> 'red hair' by the gloss 'hair that is red'?
    no -- by the anchor path, which the caller exempts)"""
    if _shares_stem(concept, tag):
        return True
    try:
        g = str(_gloss_of(tag)[0] or "").lower()
    except Exception:
        g = ""
    if not g:
        return False
    gw = {_stem(w) for w in re.findall(r"[a-z0-9]+", g)}
    cw = {_stem(w) for w in re.findall(r"[a-z0-9]+", str(concept).lower()) if w not in _STEM_STOP}
    return bool(cw & gw)


_PLACE_PREP = re.compile(r"\b(in|inside|at|through|across|along|into|under|beneath|onto|by|near|beside|"
                         r"behind|outside|within|atop|amid|among)\s+(?:(?:an|a|the|this|that|his|her|their|my|some)\s+)?")


_IRREGULAR_ING = {"sat": "sitting", "stood": "standing", "knelt": "kneeling",
                  "lay": "lying", "lain": "lying", "slept": "sleeping", "swam": "swimming"}

# ---------------------------------------------------------------------------
# THE BRIEF'S GRAMMAR, AS FAR AS THE TAGS NEED IT (2026-09-16). A brief is a
# sentence, and three things were read from it word by word: 'the drying
# hull' gave the woman the act 'drying'; 'a rusted steam engine hisses' gave
# her 'hissing'; 'sun-warmed' gave 'warming'; and the runs of words no tag
# uses -- the candidates for new concepts -- came out as 'sun-warmed wooden'
# (cut off from its dock) and 'hisses nearby' (a verb and an adverb, no
# thing at all). What decides each case is closed-class English, stated
# once here: determiners, prepositions, clause joiners, adverbs, and word
# shapes. Nothing below is a list of content.
_DETERMINERS = set("""a an the her his its their my your our this that these those some each every
another any no""".split())
_PREPOSITIONS = set("""in on at by beside near with of from under over inside outside behind before after
into onto across along through against among between around atop upon toward towards beneath below
above during without within past like""".split())
_CLAUSE_JOINERS = set("""and as while when whenever where because until though although but then so whilst
once since""".split())
_CLOSED_ADVERBS = set("""nearby away around together behind inside outside above below beneath overhead
afar again still there here apart aside along back down up off out forward forwards backward backwards
ahead underneath somewhere everywhere nowhere anywhere alone too very now then already always never
often sometimes soon later""".split())
_LIVE_PRONOUNS = {"she", "he", "they", "i", "we", "you", "someone", "somebody", "everyone", "everybody"}
# the words that name the picture itself, not a thing in it: a candidate's
# head stops before them ('cozy christmas scene' is 'cozy christmas')
_PICTURE_NOUNS = set("""scene picture image shot moment illustration artwork drawing setting atmosphere
ambience ambiance vibe vibes mood aesthetic""".split())
_ADJECTIVE_SHAPE = re.compile(r"[a-z]{2,}(?:y|ous|ful|al|ic|ive|less|ed|en|ish|ary|able|ible)$")


def _brief_tokens(text):
    """-> the brief as words and punctuation marks, hyphenated compounds whole"""
    return re.findall(r"[a-z0-9][a-z0-9'-]*|[,.;:!?]", str(text or "").lower())


def _ing_forms(w):
    """-> the -ing spellings a conjugated word can come from. THE LEXICON
    FIRST (Open English WordNet, 2026-09-16): its verb lemmas and the forms
    it lists ('sat' -> sitting, 'lies' -> lying). A word the lexicon knows as
    no verb has none. Only a word it does not know at all falls back to the
    spelling rules below."""
    lx = _lex.ing_forms(w)
    if lx is not None:
        return lx
    if _lex.pos(w) is not None:
        return []                                   # known, and not a verb
    forms = []
    if w in _IRREGULAR_ING:
        forms.append(_IRREGULAR_ING[w])
    if w.endswith("ed"):
        forms += [w[:-2] + "ing", w[:-3] + "ing", w[:-1] + "ing"]
    elif w.endswith("s") and not w.endswith("ss"):
        forms += [w[:-1] + "ing", w[:-2] + "ing"]
        if w.endswith("ies"):
            forms.append(w[:-3] + "ying")
        _st = w[:-1]
        if re.search(r"(?:^|[^aeiou])[aeiou][bdgklmnprt]$", _st):
            forms.append(_st + _st[-1] + "ing")
    return forms


def _irregular_past(w):
    """-> True for a verb form that inflects by no rule ('sat', 'stood',
    'knelt', 'lay'): the lexicon lists it as a form of a verb and it carries
    no regular ending"""
    if w.endswith(("s", "ed", "ing")):
        return False
    listed = (_lex._load() or {}).get("forms") or {}
    if listed:
        return any(k.endswith(":v") for k in listed.get(w) or ())
    return w in _IRREGULAR_ING


def _is_adverb(w):
    """closed-class adverbs, and words the lexicon knows only as adverbs or
    adjectives-and-adverbs ('nearby', 'slowly'), never as a noun or verb"""
    if w in _CLOSED_ADVERBS:
        return True
    p = _lex.pos(w)
    if p is not None:
        return "r" in p and "n" not in p and "v" not in p
    return w.endswith("ly") and len(w) >= 6


def _noun_before(toks, i):
    """-> True when the word before position i can be the subject a verb
    follows: a word, not a determiner, preposition, joiner or adverb, that
    can be a noun ('breeze mixes' yes, 'oily fumes' no -- 'oily' is only an
    adjective)"""
    if i <= 0:
        return False
    p = toks[i - 1]
    if not (p[0].isalpha() and p not in _DETERMINERS and p not in _PREPOSITIONS
            and p not in _CLAUSE_JOINERS and not _is_adverb(p)):
        return False
    if p in _LIVE_PRONOUNS or p == "it":
        return True
    ps = _lex.pos(p)
    if ps is not None:
        return "n" in ps or bool(_lex.someone(p))
    # a word the lexicon does not know ('rain-slicked'): the shape of its
    # last part decides -- a compound ending in -ed is a modifier
    return not _ADJECTIVE_SHAPE.fullmatch(p.split("-")[-1]) or _live_word(p)


_VERB_FOLLOWERS = _PREPOSITIONS | _DETERMINERS | _CLAUSE_JOINERS | _LIVE_PRONOUNS | {
    "it", "him", "them", "me", "us"}


def _verb_here(toks, i, ing=True):
    """-> True when the word at position i is a verb in this sentence. It
    must follow something that can be its subject; then
      * right after a subject pronoun it is the verb ('she tells');
      * a noun the lexicon lists with the word before ('coffee cups') is a
        compound, not a verb;
      * an -ed word after a noun is a past verb ('a girl hugged');
      * an -ing word is a verb, except inside a noun phrase ('stone
        building', where the head passes ing=False);
      * a word that can only be a verb is one ('tells', 'hugged');
      * a word that can be a noun or a verb ('mixes', 'houses', 'stands')
        is a verb when what follows is what follows a verb -- a
        preposition, an article, a pronoun, an adverb, a pause, the end --
        or when its clause's subject is someone ('a knight in chrome
        pauldron stands guard'); otherwise it is a plural noun ('brick
        houses line the street')."""
    w = toks[i]
    if not _noun_before(toks, i):
        return False
    if toks[i - 1] in _LIVE_PRONOUNS or toks[i - 1] == "it":
        return True
    if _lex.is_compound_noun(toks[i - 1], w):
        return False
    if ing and w.endswith("ed") and len(w) > 4:
        return True
    if w.endswith("ing") and len(w) > 4:
        return ing
    p = _lex.pos(w)
    if p is None:
        return any(_vocab().get(f) for f in _ing_forms(w))
    if "v" not in p:
        return False
    if "n" not in p:
        return True
    # a noun-or-verb word without a verb ending is a noun, unless a plural
    # subject stands right before it ('two women sit'): 'stone bridge',
    # 'rain-slicked asphalt'
    if not w.endswith(("s", "ed")) and not _irregular_past(w):
        prev = toks[i - 1]
        _pl = [l for l in _lex.lemmas(prev, "n") if l != prev]
        if not (_pl and _live_word(prev)):
            return False
    nxt = toks[i + 1] if i + 1 < len(toks) else ""
    if not nxt or not nxt[0].isalpha() or nxt in _VERB_FOLLOWERS or _is_adverb(nxt):
        return True
    _subj = _clause_subject(toks, i)
    return bool(_subj and _live_word(_subj))


def _live_word(w):
    """-> True when the word names somebody or some creature: a personal
    pronoun; a noun whose main meaning the lexicon files under person or
    animal ('worker', 'cat'); a person noun or a glossed person, creature
    or race of the booru's ('robot' is a creature there). Only a word the
    lexicon does not know falls back to the agent-noun shape ('-er')."""
    w = str(w or "").lower()
    if w in _LIVE_PRONOUNS:
        return True
    lx = _lex.someone(w)
    if lx:
        return True
    for form in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None):
        if not form:
            continue
        if _is_humanlike(form):
            return True
        fl = set(_gloss_flags(form) or ())
        if fl & {"person", "creature", "race"}:
            return True
        if lx is None and not fl and re.fullmatch(r"[a-z]{3,}(?:er|or|ess|ist|ian)", form):
            return True
    return False


def _thing_word(w):
    """-> True when the word names something and nobody: the lexicon's main
    meaning is no person or animal ('engine', 'hull', 'breeze'), or the
    booru glosses it and nothing makes it someone"""
    if _live_word(w):
        return False
    w = str(w or "").lower()
    if _lex.thing(w):
        return True
    return any(set(_gloss_flags(f) or ()) for f in (w, w[:-1] if w.endswith("s") else None) if f)


def _clause_subject(toks, i, depth=0):
    """-> the head word of the subject of the verb at position i: the first
    noun phrase of its clause, cut at its first preposition ('a woman in a
    red dress sits' -> woman); a clause that opens on its verb ('she kneels
    and prays') shares the subject before it"""
    j = i - 1
    while j >= 0 and toks[j] not in _CLAUSE_JOINERS and toks[j][0].isalpha():
        j -= 1
    np_ = []
    for t in toks[j + 1:i]:
        if t in _PREPOSITIONS:
            break
        np_.append(t)
    np_ = [t for t in np_ if t not in _DETERMINERS and not _is_adverb(t)]
    if np_:
        return np_[-1]
    if j >= 0 and depth < 3 and toks[j] in ("and", ","):
        return _clause_subject(toks, j, depth + 1)
    return None


def attributive_acts(base, user_tags):
    """-> the parsed act words the brief only uses to describe a THING:
    'the drying hull' is a hull, not a drying woman. An -ing act word right
    after a determiner, before a noun that names no one, belongs to that
    noun; if every use in the brief is like that, it is no act of the cast
    ('a sleeping girl' keeps 'sleeping')."""
    toks = _brief_tokens(base)
    out = set()
    for t in {str(x).lower() for x in (user_tags or [])}:
        if " " in t or not t.endswith("ing") or t not in toks:
            continue
        if not set(_gloss_flags(t) or ()) & {"act", "pose"}:
            continue
        uses = [k for k, w in enumerate(toks) if w == t]
        thing_only = True
        for k in uses:
            before = toks[k - 1] if k > 0 else ""
            after = []
            for w in toks[k + 1:k + 4]:
                if not w[0].isalpha() or w in _PREPOSITIONS or w in _DETERMINERS or w in _CLAUSE_JOINERS:
                    break
                after.append(w)
            if not (before in _DETERMINERS and after) or _live_word(after[-1]):
                thing_only = False
                break
        if uses and thing_only:
            out.add(t)
    return out


_ALIASES_TB = {"mtime": 0, "data": None}


def _alias_of(tag):
    """-> the tag danbooru's alias table sends `tag` to, else the tag
    ('gold hair' is danbooru's alias for 'blonde hair')"""
    from promptstudio.engine.aliases import booru_aliases
    al = booru_aliases()
    t = str(tag or "").lower().strip()
    return str(al.get(t) or t).replace("_", " ")


def lexicon_tags(base, user_tags, banks=None):
    """-> (tags to add, words claimed, tags to drop): the booru's tag for a
    word the parser did not know (2026-09-17, Open English WordNet).
    A word said through a BROADER term ('orchard' -> garden, 'steed' ->
    horse) stays a concept candidate in the user's own words as well; a
    synonym ('frock' -> dress) says it whole. lexicon_tags.broader holds
    those words after a call.

    'crimson hair' is no booru tag and 'crimson' no booru word, so the brief
    said nothing: WordNet says crimson is red, and 'red hair' is a tag.
    Likewise 'azure eyes' -> blue eyes, 'a frock' -> dress, 'a brook' ->
    stream, 'a steed' -> horse. Two passes over the brief's words:

      PAIRS  a modifier and its noun ('crimson hair', 'scarlet gown'): each
             word as written, its lemma, or what it can be said as; the
             combination must be a booru tag with at least 100 posts. Fewer
             substitutions win, then more posts ('red gown' before 'red
             dress' if both exist).
      NOUNS  a noun the parser left unclaimed and the booru has no tag for:
             what it can be said as, from the senses of its main meaning
             ('blade' is mostly an artifact, so sword, not leaf), the
             closest term first, then the most posts.

    A person word is left to the cast ('lass'). A tag the booru's alias
    table produced from a word now claimed by a pair goes: 'ebony' alone is
    danbooru's alias for 'very dark skin', and 'ebony hair' is black hair.
    """
    if not _lex.available():
        return [], set(), []
    v = _vocab()
    have = {str(t).lower() for t in (user_tags or [])}
    have_words = {w for t in have for w in t.split()}
    toks = _brief_tokens(base)
    try:
        free = set(re.findall(r"[a-z0-9][a-z0-9'-]*", pe.unclaimed_text(base, banks).lower())) if banks else None
    except Exception:
        free = None
    # the words that name the picture itself are not mapped either way
    # ('cozy christmas scene' gained 'shot', the film sense of scene)
    closed = _LEFTOVER_STOP | _DETERMINERS | _PREPOSITIONS | _CLAUSE_JOINERS | _PICTURE_NOUNS
    add, claimed, drop = [], set(), []
    broader = set()                     # words claimed through a broader term

    def forms(word, head=False):
        out = [(word, 0)]
        if head:
            out += [(l, 0) for l in _lex.lemmas(word, "n") if l != word]
        out += [(t, 1 + d) for t, d in _lex.related(word) if " " not in t and d <= 1]
        return out

    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if not (a[0].isalpha() and b[0].isalpha()) or a in closed or b in closed:
            continue
        if "%s %s" % (a, b) in have or v.get("%s %s" % (a, b), 0) >= 100:
            continue
        best = None
        for a2, ca in forms(a):
            for b2, cb in forms(b, head=True):
                if a2 == a and b2 == b:
                    continue
                t = _alias_of("%s %s" % (a2, b2))
                n = v.get(t, 0)
                if n < 100:
                    continue
                # fewer words changed first, then the booru's own weight: a
                # modifier and its noun are one picture, and 'red dress'
                # (144,054 posts) is what 'scarlet gown' means more than
                # 'red robe' (2,754)
                cost = (ca > 0) + (cb > 0)
                key = (cost, -n)
                if best is None or key < best[0]:
                    best = (key, t, max(ca, cb) >= 2)
        if best:
            # the words are claimed even when their tag is already there
            # (a second reading of the same brief must agree with the first)
            if best[1] not in have:
                add.append(best[1])
                have.add(best[1])
            claimed |= {a, b}
            if best[2]:
                broader |= {a, b}
    # a tag the alias table made of a word a pair now owns
    try:
        from promptstudio.engine.aliases import booru_aliases
        al = booru_aliases()
        low = str(base or "").lower()
        for w in claimed:
            t = str(al.get(w) or "").replace("_", " ")
            if t and t in have and t not in add and t not in low:
                drop.append(t)
    except Exception:
        pass
    for w in toks:
        if not w[0].isalpha() or w in closed or w in claimed or w in have_words or len(w) < 3:
            continue
        if free is not None and w not in free:
            continue
        if v.get(w, 0) >= 100 or "n" not in (_lex.pos(w) or ()):
            continue
        cls = _lex.main_classes(w)
        if not cls or "person" in cls:
            continue
        # A WORD IS RESAID ONLY WHEN IT HAS ONE MEANING, AND NEVER A WORD OF THE BODY
        # (the author's live test, 2026-10-03: '1girl, looking away, biting lip, but
        # body leaning in' put `border, dead body` on every line -- 'lip' said as an
        # edge, 'body' as a corpse). Two tests the pass lacked:
        #  - an ambiguous word is nobody's to guess (the standing rule): 'lip' has
        #    five main classes -- a mouth's, a cup's, a flower's;
        #  - the body's own words are the body tables' and the user's: the
        #    dictionary's other senses of 'body' are a torso, a trunk and a corpse.
        #    'brook', 'steed', 'frock', 'blade' are things, and are translated.
        if len(cls) > 1 or "body" in cls:
            continue
        best = None
        for t, d in _lex.related(w, classes=cls):
            t = _alias_of(t)
            n = v.get(t, 0)
            if n < 100 or t in _PICTURE_NOUNS:
                continue
            # ... and it is said as a tag the booru describes and lets one use: not
            # a picture format, a meme or text, not one its wiki says to avoid, not
            # a token only another booru's list holds ('dead body')
            if not (_gloss_of(t) or ("",))[0] or pe._never_from_prose(t) \
                    or ((_gloss_flags(t) or set()) and (_gloss_flags(t) or set()) <= {"format", "meme", "text"}):
                continue
            # A NOUN NAMES A THING (2026-09-17: 'rivulet' -> 'run', which
            # danbooru aliases to the act 'running'): a target the booru files
            # as an act, a pose or an expression is another sense
            if (_gloss_flags(t) or set()) & {"act", "pose", "expression"}:
                continue
            key = (d, -n)
            if best is None or key < best[0]:
                best = (key, t)
        if best:
            if best[1] not in have:
                add.append(best[1])
                have.add(best[1])
            claimed.add(w)
            if best[0][0] >= 1:
                broader.add(w)
    lexicon_tags.broader = broader
    return add, claimed, drop


def inflected_acts(base, user_tags):
    """the user's inflected verbs the parser passed over ('spanked',
    'hugged', 'danced'): their -ing form when it is a booru act or pose
    the user did not already type -- typed is law even when conjugated.
    The verb must BE a verb and must be the cast's: a hyphenated compound
    is an adjective ('sun-warmed'), a word after a determiner or an
    adjective shape is a noun or a modifier ('the exposed legs'), and a
    verb whose subject is a thing is the thing's ('a rusted steam engine
    hisses')."""
    have = {str(t).lower() for t in (user_tags or [])}
    v = _vocab()
    out = []
    low = str(base or "").lower()
    toks = _brief_tokens(low)
    _stances = stance_vocab()
    for i, w in enumerate(toks):
        if len(w) < 3 or not w[0].isalpha() or "-" in w:
            continue
        # A WORD IS ITS OWN TAG ONLY WHERE THE REQUEST COULD TYPE IT: the
        # typed floor (100 posts, typed_direct's). 'lies' is a 71-post tag,
        # and that alone kept "lies on a dock" from ever meaning lying.
        if w in have or v.get(w, 0) >= 100:
            continue
        forms = _ing_forms(w)
        _irr = _irregular_past(w)
        if not forms or (len(w) < 4 and not _irr):
            continue
        # 'lay' is also the present of 'to lay': a stance only before a preposition
        if w == "lay" and not re.search(r"\blay\s+(?:on|in|across|beside|against|along|"
                                        r"upon|atop|under|back|down|face)\b", low):
            continue
        if not _irr and not (_noun_before(toks, i) or (i > 0 and toks[i - 1] in _LIVE_PRONOUNS)):
            continue
        _subj = _clause_subject(toks, i)
        if _subj and _thing_word(_subj):
            continue                                   # the engine hisses, not the woman
        for f in forms:
            if f in have or not v.get(f):
                continue
            try:
                fl = set(_gloss_of(f)[1] or ())
            except Exception:
                fl = set()
            if not fl & {"act", "pose"}:
                continue
            # AN '-S' STANCE NEEDS TO BE A VERB. '-s' is also the plural
            # ('tells lies to everyone'), so a stance read from an -s word
            # needs what makes it a verb after it: where it happens ('lies
            # on', 'sits by', 'runs through'), a pause ('kneels, cleaning'),
            # 'and', or the end of the sentence. The -ed and irregular pasts
            # are verbs already ('knelt', 'danced').
            # an irregular past takes the same test ('she ran her fingers
            # through her hair' is no running)
            if f in _stances and (w.endswith("s") or _irr) and not re.search(
                    r"\b%s(?:\s*[,.;:!?]|\s*$|\s+(?:and|on|in|at|by|beside|near|against|"
                    r"across|along|upon|atop|under|inside|outside|before|behind|down|"
                    r"up|back|over|around|between|among|through|into|onto|toward|"
                    r"towards|past|next|face|alone|still|quietly|together|with|there|"
                    r"here)\b)" % re.escape(w), low):
                break
            out.append(f)
            have.add(f)
            break
    return out


_PLACE_ABSTRACT = set("""hurry love time moment silence style way mood manner order case fact spite turn
front back middle end half distance rain snow fog sunlight moonlight dark darkness light shadow shadows
morning evening afternoon night noon dawn dusk winter summer spring autumn heat cold wind air water
motion action pose position hand hands arms arm lap knee knees shoulder shoulders back mouth face eyes
mirror reflection""".split())


def _is_artist_name(name):
    """A WORD THE REQUEST NAMES AS AN ARTIST IS NOT A PLACE (2026-09-15:
    'a girl drawn by wlop' read 'by wlop' as a place phrase, made wlop the
    location and opened the prose "In the wlop, the girl ..."). One owner
    per word: an artist of the measured pool (at the generation floor) or
    of the external registry (kind artist, pending or not) is an artist,
    the mirror of the artist scan refusing words the request claimed as
    places."""
    low = str(name or "").lower().replace("_", " ").strip()
    if not low:
        return False
    try:
        pool, _ = pe._artist_pool()
        if pool.get(low, 0) >= pe.ARTIST_FLOOR:
            return True
    except Exception:
        pass
    try:
        from promptstudio.library import external as _ext
        rec = _ext.get(low) or _ext.get(_ext.canonical(low) or "")
        return bool(rec) and (rec or {}).get("kind") == "artist"
    except Exception:
        return False


def place_phrases(base, leftovers, user_tags=()):
    """the places the text names that the parser did not take: a place
    preposition, up to two modifiers and a noun that is no tag of the
    request ('in a cluttered cottage', 'at a forge', 'through an
    orchard'); the picture's place in the user's own words"""
    have = {str(t).lower() for t in (user_tags or [])}
    have_words = {w for t in have for w in t.split()}
    out = []

    _THING_FLAGS = {"object", "vehicle", "clothing", "person", "act", "pose", "body", "food",
                    "expression", "race", "creature", "weapon"}
    # a word the lexicon gave a booru tag is said by that tag ('near a
    # rivulet' is the tag 'stream', not a place called 'rivulet')
    try:
        _lx_claimed = lexicon_tags(base, user_tags)[1]
    except Exception:
        _lx_claimed = set()

    def _ok_head(head, determined=False, described=False):
        # a bare word the lexicon gave a tag is said by it ('rivulet' ->
        # stream); a described one keeps the user's words ('cluttered
        # cottage' is more than 'house')
        if head in _lx_claimed and not described:
            return False
        if (head in _LEFTOVER_STOP or head in _PLACE_ABSTRACT or head in _FEMALE_NOUN_LIST
                or head in _MALE_NOUN_LIST or head.endswith("ing")
                or head in _PERSON_EXTRA or head in _AGENT_NOUNS or _is_humanlike(head)
                or head in _pair_nouns() or (head.endswith("s") and _is_humanlike(head[:-1]))):
            return False                # 'surrounded by friends' names people
        # A NAME TAKES NO ARTICLE (2026-09-16: 'in a cluttered cottage' drew
        # '@cottage' again). 'drawn by wlop' names the artist, but a word
        # after 'a' or 'the' is a common noun: the cottage is the place, and
        # the artist scan then leaves it alone, as ruled on 2026-09-13.
        if not determined and _is_artist_name(head):
            return False
        # a thing, a garment, a person or an act heads no place, parsed or
        # not ('studio' yes, 'couch' no: the object's own place is measured
        # elsewhere; 'in chrome pauldron' is armour, not a room)
        for form in (head, head + "s", head[:-1] if head.endswith("s") else None):
            if form and (_gloss_flags(form) or set()) & _THING_FLAGS:
                return False
        return True
    text = _sans_camera(str(base or "").lower())
    strong, weak = [], []
    for segment in re.split(r"[,.;:!?()]+", text):
        low = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'-]+", " ", segment)) + " "
        for m in re.finditer(_PLACE_PREP.pattern, low):
            out = strong if m.group(1) in ("in", "inside", "within", "through", "across", "along", "into", "amid", "among") else weak
            rest = low[m.end():].split()
            # an article right after the preposition was eaten by the
            # pattern; it still says the head is a common noun
            _pre = low[:m.end()].split()
            determined = bool(_pre) and _pre[-1] in _DETERMINERS
            # the longest fitting phrase first: up to three words, the head
            # a noun the parser did not take -- and never a verb ('in chrome
            # pauldron stands guard' has no place called 'stands')
            for n in (3, 2, 1):
                words = rest[:n]
                if len(words) < n:
                    continue
                head = words[-1]
                if n > 1 and _verb_here(words, n - 1, ing=False):
                    continue
                if len(rest) > n:
                    nxt = rest[n]
                    if (nxt[:1].isalpha() and nxt not in _LEFTOVER_STOP and nxt not in _PREPOSITIONS
                            and nxt not in _CLAUSE_JOINERS and not _is_adverb(nxt)
                            and not _verb_here(rest, n)):
                        continue            # the noun phrase goes on past this word
                _desc = len([x for x in words if x not in _LEFTOVER_STOP]) > 1
                if not _ok_head(head, determined or (bool(words) and words[0] in _DETERMINERS), _desc):
                    continue
                while words and (words[0] in _LEFTOVER_STOP or words[0] in have_words):
                    words = words[1:]
                phrase = " ".join(words)
                if phrase and phrase not in out and phrase not in have:
                    out.append(phrase)
                break
    # 'in a mirrored studio' outranks 'at the barre': the containing
    # preposition names the place, the touching one names a thing
    return strong + [p for p in weak if p not in strong]


def typed_leftovers(base, user_tags):
    """-> the runs of the user's words that no booru tag uses (stem
    tolerant: 'pauldron' is covered by 'pauldrons'), stop words and
    count/gender words aside; each run is one concept in the user's own
    words. Quoted lettering and weights are left alone."""
    if not base:
        return []
    text = re.sub(r'''"[^"]*"|(?<![a-z])'[^']*'(?![a-z])|\([^()]*:[0-9.]+\)''', " ", str(base).lower())
    text = re.sub(r"'s\b", "", text)
    wtc = _word_tag_count()
    gendered = set(_FEMALE_NOUN_LIST) | set(_MALE_NOUN_LIST) | {"woman", "women", "man", "men", "girl",
                                                               "girls", "boy", "boys", "person", "people"}

    # A SUBJECTIVE WORD IS THE USER'S, WITH ITS NOUN (the author's 2026-09-15:
    # "sexy girl should be the phrase 'sexy girl' in the prompt"): the
    # booru's subjective words are modifiers, never tags, so no tag claims
    # them -- 'sexy' counted as known only because 'sexy no jutsu' carries
    # the word. A run of subjective words keeps the noun it modifies, the
    # gendered head included ('sexy girl', 'cute elegant woman'), and lands
    # as a phrase in the user's own words.
    try:
        from promptstudio.library import concepts as _clx
        _subjective = set(_clx.subjective_words())
    except Exception:
        _subjective = set()

    try:
        _lx_claimed = lexicon_tags(base, user_tags)[1] - getattr(lexicon_tags, "broader", set())
    except Exception:
        _lx_claimed = set()

    def known(w):
        if w in _lx_claimed:
            return True                  # said whole by its booru synonym ('frock' -> dress)
        if w in _subjective:
            return False
        if w in _LEFTOVER_STOP or w in gendered or w.isdigit():
            return True
        for form in (w, w + "s", w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None,
                     w[:-3] + "y" if w.endswith("ies") else None, w + "es"):
            if form and wtc.get(form):
                return True
        return False
    out = []

    def emit(run):
        # A CANDIDATE ENDS ON ITS THING: trailing and leading adverbs are
        # how something is done, not what is there ('hisses nearby')
        while run and (_is_adverb(run[-1]) or run[-1] in _PICTURE_NOUNS):
            run = run[:-1]
        while run and _is_adverb(run[0]):
            run = run[1:]
        if len(run) == 1 and run[0].endswith("ing"):
            run = []                                # a lone participle ('creating') names nothing
        if run:
            out.append(" ".join(run))

    for segment in re.split(r"[,.;:!?\n()]+", text):       # a run never crosses punctuation
        toks = re.findall(r"[a-z0-9][a-z0-9'-]*", segment)
        run = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            if known(tok):
                if run:
                    # A MODIFIER KEEPS ITS WHOLE HEAD (2026-09-16): 'chrome'
                    # before 'pauldron' is 'chrome pauldron', and 'sun-warmed'
                    # before 'wooden dock' is 'sun-warmed wooden dock' -- the
                    # head runs on through the noun phrase (up to three words,
                    # never into an act, a pose or a person), where the old
                    # rule stopped at its first word and left 'sun-warmed
                    # wooden' with no dock. A subjective run keeps even a
                    # gendered head ('sexy girl').
                    _all_subj = all(w in _subjective for w in run)
                    j = i
                    while j < len(toks) and j - i < 3:
                        h = toks[j]
                        if not known(h) or h in _LEFTOVER_STOP or h.isdigit():
                            break
                        if h in gendered and not _all_subj:
                            break
                        if h in _PICTURE_NOUNS:
                            break
                        if j > i and set(_gloss_flags(h) or ()) & {"act", "pose"}:
                            break
                        if _verb_here(toks, j, ing=False):
                            break                   # breeze mixes, pauldron stands
                        run.append(h)
                        j += 1
                        if h in gendered:
                            break
                    emit(run)
                    run = []
                    i = max(j, i + 1)
                    continue
                run = []
                i += 1
                continue
            # A VERB IS NO PART OF A CANDIDATE (2026-09-16: 'salty breeze
            # mixes', 'hisses nearby', 'tracing'): a conjugated word after a
            # noun, or an -ing word after one, closes the noun phrase before
            # it; the verb and what hangs off it are how the scene moves,
            # which the prose says
            if _verb_here(toks, i):
                emit(run)
                run = []
                i += 1
                continue
            run.append(tok)
            i += 1
        emit(run)
    seen, kept = set(), []
    for r_ in out:
        r_ = r_.strip(" '-")
        if len(r_) < 3 or r_ in seen:
            continue
        seen.add(r_)
        kept.append(r_)
    # A CONCEPT THE EXTERNAL REGISTRY OWNS IS NO LEFTOVER (2026-09-15): a
    # prose-only lighting term ('soft lighting') is said by its own path,
    # never carried as a bare phrase on the line
    try:
        from promptstudio.library import external as _extl
        _own = {str(n).lower() for n in _extl.all_typed_names()}
        kept = [x for x in kept if str(x).lower().strip() not in _own]
    except Exception:
        pass
    return kept[:6]


def _word_tag_count():
    vocab = _vocab()
    if _WORD_TAG_COUNT["data"] is None or _WORD_TAG_COUNT["n"] != len(vocab):
        counts = {}
        for t in vocab:
            for word in set(str(t).split()):
                counts[word] = counts.get(word, 0) + 1
        _WORD_TAG_COUNT["data"], _WORD_TAG_COUNT["n"] = counts, len(vocab)
    return _WORD_TAG_COUNT["data"]


_EXTERNAL_NAMES = None


def _external_name(bare):
    """-> True when this bare name belongs to a non-booru CONCEPT.

    the author's: the checkpoints know concepts booru never tagged, and those
    names collide with booru characters -- 'van gogh' is not a danbooru
    artist but 'van gogh (fate)' IS a Fate servant, so asking for the
    painter cast the servant. The concept wins the BARE name; anyone who
    means the character still gets it by writing the qualified form, which
    matches on the branch above this one.
    """
    global _EXTERNAL_NAMES
    if _EXTERNAL_NAMES is None:
        try:
            from promptstudio.library import external as _ext
            _EXTERNAL_NAMES = _ext.all_typed_names()
        except Exception:
            _EXTERNAL_NAMES = set()
    return str(bare).strip().lower() in _EXTERNAL_NAMES


def _sans_characters(base):
    """-> the text with every typed character name blanked. 'aerith
    GAINSBOROUGH' matched the painter concept 'Thomas Gainsborough' and put
    'Thomas Gainsborough style' on the line; a surname inside a character's
    name is the character's, never a concept's."""
    low = str(base or "")
    try:
        names = _typed_characters(base)
    except Exception:
        names = []
    for n in names:
        for form in (n, n.split(" (")[0], _display_name(n)):
            form = str(form or "").strip()
            if len(form) >= 3:
                low = re.sub(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])",
                             " ", low, flags=re.I)
                # each word of a multi-word name is claimed on its own too
                for w in form.split():
                    if len(w) >= 5:
                        low = re.sub(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])",
                                     " ", low, flags=re.I)
    return low


def _typed_characters(base):
    """character names the user actually wrote, found in the traits table.
    TYPO TOLERANCE: a UNIQUE surname (>=5 chars) whose preceding word sits
    within edit distance 2 of the real first name recovers 'claude
    strife' -> cloud strife; a unique surname is identification enough."""
    global _SURNAMES
    raw_low = " " + (base or "").lower() + " "
    low = " " + re.sub(r"[^a-z0-9() ]", " ", (base or "").lower()) + " "

    # A HYPHEN IS NOT A WORD BOUNDARY FOR IDENTITY. The normalisation above
    # turns punctuation into spaces so 'miku, dancing' still matches -- but
    # that also splits 'two-tone hair' into 'two' and 'tone', and there is
    # a kancolle shipgirl named Tone. Measured on the 21,996-name table:
    # 175 ordinary hyphenated tags hijacked a character this way, every one
    # sampled leaking -- 'two-tone hair' cast Tone, 'left-handed' cast Left
    # (atomic heart), 'chain-link fence' cast Link. Those are everyday tags,
    # so this was firing constantly.
    #
    # A single-word name only counts when it stands alone in the ORIGINAL
    # text; glued to a hyphen it is part of another word. Multi-word names
    # are unaffected (nothing hyphenates 'hatsune miku').
    def _hyphen_bound(word):
        return (("-" + word + " ") in raw_low or ("-" + word + "-") in raw_low
                or (" " + word + "-") in raw_low)
    # A CHARACTER NAMED AFTER A COMMON WORD MUST NOT HIJACK THE WORD.
    # pick_artists has held this rule for artists since the beginning ("an
    # artist named after a common word must not hijack the word"); the
    # character matcher never did, and it only became visible when the
    # traits table grew past the top 2,000. At 6,050 characters it already
    # holds 'knight (hollow knight)', 'neon (nikke)', 'angel (kof)',
    # 'doctor (arknights)' and 'king (aotu world)', so "a knight resting by
    # a campfire" cast two Hollow Knight characters and "cyberpunk street,
    # neon" cast a NIKKE.
    #
    # The test is the same one the artist rule uses: if the bare name is
    # ALSO a general danbooru tag, it is a word first and a character
    # second -- it only counts when the user wrote the qualified form.
    out, bare_hit = [], {}
    for n in _traits():
        bare = n.split(" (")[0]
        if " " + n + " " in low:
            # a ONE-WORD name has no qualifier to vouch for it, so the
            # hyphen rule applies here too: 'chain-link fence' is not Link
            if " " not in n and _hyphen_bound(n):
                continue
            out.append(n)                       # fully qualified: certain
        elif (" " + bare + " " in low and not _common_word(bare)
                and not (" " not in bare and _hyphen_bound(bare))
                and not _external_name(bare)
                and not _species_phrase(bare)):
            # remember the candidates per bare name; resolve after
            bare_hit.setdefault(bare, []).append(n)
    # drop a bare name that is merely part of a longer name we matched:
    # 'hatsune' inside 'hatsune miku' is the same person, not a second one
    for bare in list(bare_hit):
        if any(bare != o.split(" (")[0] and bare in o.split(" (")[0].split()
               for o in out):
            del bare_hit[bare]
    for bare, cands in bare_hit.items():
        if bare in out:
            # the canonical character already matched on its own name --
            # its qualified variants need their qualifier to be meant
            continue
        if any(c == bare for c in cands):
            # the canonical character exists -- the qualified variants
            # ('hatsune miku (nt)') need their qualifier to be meant
            out.append(bare)
        elif " " not in bare and not _qualifier_corroborated(bare, cands, low,
                                                            out):
            # A ONE-WORD NAME DANBOORU ONLY EVER WRITES QUALIFIED is not
            # an identity on its own. 'a blonde elf ranger and a
            # red-haired dwarf blacksmith' cast `ranger (ragnarok
            # online)`, dragged in the series, and collapsed two people
            # into one -- because `ranger` matched the bare half of a
            # qualified character.
            #
            # The evidence is danbooru's own: it declined to give that
            # character the bare word, precisely because the word alone
            # does not identify anyone. So the qualifier has to be MEANT
            # -- the series named, or a castmate already present. A
            # multi-word bare name ('tifa lockhart') is untouched, and
            # the fully-qualified form always works.
            continue
        elif len(cands) == 1:
            out.append(cands[0])
        else:
            # ambiguous bare name with no canonical entry: the most drawn
            # one is the one a reader would mean
            out.append(max(cands, key=lambda c: (_traits()[c] or {})
                           .get("posts", 0)))
    if _SURNAMES is None:
        idx = {}
        for n in _traits():
            parts = n.split(" (")[0].split()
            if len(parts) == 2 and len(parts[1]) >= 5:
                idx.setdefault(parts[1], []).append(n)
        # NOT filtered by _common_word: a surname is allowed to be an
        # ordinary word, because the whole point of this path is that the
        # FIRST name vouches for it. Screening surnames against the
        # vocabulary looked right and silently broke real characters --
        # 'hakurei' is a tag ('hakurei shrine') and 'strife' is one too,
        # so it lost Reimu Hakurei and the documented 'claude strife' ->
        # cloud strife recovery. The article guard below is what actually
        # closed the 'a demon' -> 'sea demon (fate)' hole.
        _SURNAMES = {s: ns[0] for s, ns in idx.items() if len(ns) == 1}
    words = low.split()
    for wi in range(1, len(words)):
        owner = _SURNAMES.get(words[wi])
        if not owner or owner in out:
            continue
        # AN ARTICLE IS NOT A MISSPELLED FIRST NAME. 'a' sits within edit
        # distance 2 of 'sea', which is how "a demon" became "sea demon";
        # likewise "the nagant". A determiner carries no identity.
        prev = words[wi - 1]
        if prev in ("a", "an", "the", "and", "of", "in", "with"):
            continue
        if _hyphen_bound(words[wi]):
            continue                    # 'mosin-nagant' is one word
        first = owner.split(" (")[0].split()[0]
        if _lev2(prev, first) <= 2:
            out.append(owner)
    return out


def _series_scope(base):
    """-> the series the text actually names.

    THIS WAS A BARE SUBSTRING MATCH, and it scoped the entire
    character draw. Series `c (control)` matched the letter "c" in
    any text at all, and `it (stephen king)` matched the "it" inside
    "with" -- so an ordinary prompt narrowed the candidate pool to
    those two series, and every generated character came from their
    two or three members. the author's: "I see it choosing same characters
    again and again - mashu (control) and c (control) in every
    generation".

    56 series are unsafe to match on their bare name: 7 are one or
    two letters, and 49 more are ordinary words -- fate, cyberpunk,
    doom, alien, city, another. "A cyberpunk street" must not decide
    which world the cast comes from.

    So: whole words only, and a bare name that is ordinary
    vocabulary needs its qualifier written out ("c (control)").
    `_common_word` is the same measured predicate the character
    matcher already uses for exactly this.
    """
    low = " " + re.sub(r"[^a-z0-9() ]+", " ", (base or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    hits = set()
    for v in _traits().values():
        sr = v.get("series")
        if not sr or sr in hits:
            continue
        if " " + sr + " " in low:      # fully qualified: certain
            hits.add(sr)
            continue
        bare = sr.split(" (")[0]
        if len(bare) < 3 or _common_word(bare):
            continue
        if " " + bare + " " in low:
            hits.add(sr)
    return hits


# ONE-OF FAMILIES: a body has exactly one value per family. Defined once,
# used by the per-subject dedup at assembly AND by the persona draw.
ONE_OF_FAMILIES = (
    {"flat chest", "small breasts", "medium breasts", "large breasts",
     "huge breasts", "gigantic breasts"},
    {"very long hair", "long hair", "medium hair", "short hair",
     "very short hair"},
    {"blue eyes", "red eyes", "green eyes", "brown eyes", "purple eyes",
     "yellow eyes", "pink eyes", "aqua eyes", "orange eyes", "grey eyes",
     "black eyes", "golden eyes", "amber eyes", "silver eyes",
     "white eyes"},
    {"blonde hair", "brown hair", "black hair", "blue hair",
     "purple hair", "pink hair", "red hair", "white hair", "grey hair",
     "silver hair", "green hair", "orange hair", "aqua hair",
     "light brown hair", "light blue hair", "dark blue hair",
     "lavender hair", "platinum blonde hair"},
    {"small penis", "large penis", "huge penis", "gigantic penis"},
    # ONE BODY POSTURE PER SUBJECT (the author's: anal can be missionary, standing
    # or on hands and knees, "but not all at the same time for one subject")
    {"standing", "all fours", "on hands and knees", "lying", "on back",
     "on stomach", "on side", "kneeling", "sitting", "squatting",
     "top-down bottom-up", "prone", "reclining", "handstand", "headstand"},
    # ONE SKIN TONE PER BODY (the author, 2026-10-03, contradictory tags on one
    # body part: 'pale skin, tan' in 2 prompts of 46)
    {"pale skin", "tan", "dark skin"},
    # the route: a typed `anal` must not sit beside a mapped `vaginal`
    # (the model wrote "penetrates her deeply" and the mapper answered
    # vaginal); typed precedence decides, and a typed `double penetration`
    # carries both by its own arity
    {"anal", "vaginal"},
    # time-of-day is a scene one_of: 'golden hour' next to a typed
    # 'night' is a contradiction (transitional words like dusk and
    # evening stay free -- they legitimately co-tag)
    {"day", "night", "golden hour", "midnight", "noon"},
)


_RACE_FAM = {"key": None, "data": None}


def _race_family():
    """the kemonomimi and fantasy races the body table knows, each with its
    parts: {race: {race, parts...}}; the union is the family a typed race
    is one-of in (2026-09-16: 'a bunny girl' cast a lion girl). Built once
    per body table (the vocabulary walk is 50k tags; the veto runs per
    candidate persona)."""
    tb = _body_table() or {}
    if _RACE_FAM["key"] == id(tb) and _RACE_FAM["data"] is not None:
        return _RACE_FAM["data"]
    fam = {}
    for race, ent in (tb.get("race_parts") or {}).items():
        parts = set((ent or {}).get("parts") or [])
        fam[str(race).lower()] = {str(race).lower()} | {str(p).lower() for p in parts}
    for race, ent in (tb.get("race_parts_measured") or {}).items():
        fam.setdefault(str(race).lower(), {str(race).lower()})
        fam[str(race).lower()] |= {str(p).lower() for p in (ent or {})}
    # every '<animal> girl' / '<animal> boy' the vocabulary pairs with
    # '<animal> ears' is a kemonomimi race too (lion girl, tiger girl, deer
    # girl...): the table lists the rolled ones, the booru names the rest
    try:
        voc = _vocab()
        for t in voc:
            ws = str(t).split()
            if len(ws) == 2 and ws[1] in ("girl", "boy") and (ws[0] + " ears") in voc:
                fam.setdefault(t, {t}).update({ws[0] + " ears", ws[0] + " tail", ws[0] + " girl", ws[0] + " boy"})
    except Exception:
        pass
    _RACE_FAM["key"], _RACE_FAM["data"] = id(tb), fam
    return fam


_RACE_GENERIC = {"animal ears", "animal ear fluff", "tail", "kemonomimi mode", "extra ears", "fake animal ears"}


_TREE_VOCAB = {"mtime": 0, "data": None}


def _tree_tags():
    """the booru's own vocabulary per layer (tree_vocab.json 'tags'): each
    tag's tag groups and sections, its post counts and related shares"""
    return (_json_table(_TREE_VOCAB, "tree_vocab.json") or {}).get("tags") or {}


def _count_1girl(t, universe="1girl"):
    """-> posts carrying 1girl and `t`. The booru refuses to count the very
    large queries ('1girl holding' -- about 1.4 million -- comes back empty),
    which left every pair with a common tag UNMEASURED (2026-09-16: 'heart
    hands, holding' was never checked). The tree table carries the tag's
    own count and its measured 1girl share, which is the same number."""
    q = universe + " " + _q(t)
    tb = _pair_counts()
    if q in tb:
        return tb[q]
    rec = _tree_tags().get(t) or {}
    share = float((rec.get("related") or {}).get(universe) or 0.0)
    posts = float(rec.get("danbooru") or 0.0)
    est = int(posts * share) if posts and share else 0
    # THE BOORU REFUSES THE BIG ONES, SO THEY ARE NOT ASKED (2026-09-16):
    # every refused count cost a ~4 s round trip and was asked again on the
    # next line -- the harness went from 8 minutes to over 40. At half a
    # million posts and up the tree table's own measure is the answer.
    if est >= 500000:
        return est
    n = _count_cached(q)
    return n if n else (est or n)


_HAND_POSTURE_SECTIONS = {"hand", "hands", "basicarm", "specificarm", "carry",
                          "hug", "hugone", "hugtwo", "rest"}
_HAND_SKIP_SECTIONS = {"see also", "strange hands", "other"}


_IMPL_TB = {"mtime": 0, "data": None}
_IMPL_CLOSURE = {"key": None, "data": {}}


def _implications():
    """danbooru's active tag implications, {antecedent: [consequents]}
    (data/tags/tag_implications.json, tools/build/build_implications.py)"""
    return (_json_table(_IMPL_TB, "tag_implications.json") or {}).get("implies") or {}


def _general_tag(t):
    """-> True for a general booru tag (not an artist, copyright, character
    or meta tag)"""
    try:
        from promptstudio.engine import vocab as _vc
        return _vc._tag_cat(t) == 0
    except Exception:
        return True


def implied(tag):
    """-> every tag the booru says `tag` implies, the whole chain ('holding
    gun' -> holding weapon, gun, holding, weapon), never the tag itself"""
    imp = _implications()
    if _IMPL_CLOSURE["key"] != id(imp):
        _IMPL_CLOSURE["key"], _IMPL_CLOSURE["data"] = id(imp), {}
    cache = _IMPL_CLOSURE["data"]
    t = str(tag or "").lower().strip()
    if t in cache:
        return cache[t]
    out, stack = set(), list(imp.get(t) or ())
    while stack:
        c = stack.pop()
        if c == t or c in out:
            continue
        out.add(c)
        stack.extend(imp.get(c) or ())
    cache[t] = frozenset(out)
    return cache[t]


def _is_holding(t):
    """a grip on an object: 'holding' and every tag the booru says implies
    it ('holding gun', 'dual wielding', 'between fingers'). 'holding hands'
    implies no 'holding' -- it is a hand on another person. The name test
    stands only when the implication table is missing."""
    if t == "holding":
        return True
    if _implications():
        return "holding" in implied(t)
    return t.startswith("holding ") and t != "holding hands"


def _hand_claim(t):
    """-> True when the tag occupies a hand: the booru files it under the
    'hands' or 'gestures' tag group (not their 'see also', 'strange hands'
    or miscellany sections), under an arm or hand section of 'posture', or
    it is a holding verb"""
    if _is_holding(t):
        return True
    for g, sec in ((_tree_tags().get(t) or {}).get("groups") or {}).items():
        if g in ("hands", "gestures") and sec not in _HAND_SKIP_SECTIONS:
            return True
        if g == "posture" and sec in _HAND_POSTURE_SECTIONS:
            return True
    return False


def pair_lift(a, b, universe="1girl"):
    """-> P(a,b) / (P(a)P(b)) in the 1girl universe (or 'solo', where one
    body is all there is), or None when either side is unmeasured. Under 1
    the booru shows the two together less than chance; under .5 it barely
    shows them at all."""
    try:
        tot = float(_count_cached(universe) or 0)
        na = _count_1girl(a, universe)
        nb = _count_1girl(b, universe)
        nab = _count_cached(universe + " " + _q(a) + " " + _q(b))
        if not (tot and na and nb) or nab is None:
            return None
        return (nab / tot) / ((na / tot) * (nb / tot))
    except Exception:
        return None


def fits_race(tag, race, floor=0.5):
    """-> False when the booru does not dress this race in this garment (the
    author's bug hunt, 2026-09-21: a MERMAID in a microskirt and shoes). A
    body that is not shaped like a human's wears what its own pictures wear:
    mermaid + microskirt .24, + shoes .11, centaur + pants .40, against
    mermaid + bikini top 12.5, elf + dress 1.16. Unmeasured is neutral."""
    if not race or not tag:
        return True
    lf = _pair_lift_cached(str(race).lower(), str(tag).lower())
    return lf is None or lf >= floor


def _pairs_with_worn(t, worn, slots, floor=0.5):
    """-> False when the booru measures this garment beside a worn body
    garment at a pair lift under `floor` (P(a,b) / P(a)P(b) in the 1girl
    universe); True when unmeasured or fine"""
    try:
        tot = float(_count_in_cache("1girl") or 0)
        if not tot:
            return True
        ha = _head_garment(slots, t)
        na = _count_in_cache("1girl " + _q(ha))
        if not na:
            return True
        for w in worn:
            if not _cl_slots_of(slots, w) or _cl_slots_of(slots, w) == _cl_slots_of(slots, t):
                continue
            hb = _head_garment(slots, w)
            nb = _count_in_cache("1girl " + _q(hb))
            nab = _count_in_cache("1girl " + _q(ha) + " " + _q(hb))
            if not nb or nab is None:
                continue
            if (nab / tot) / ((na / tot) * (nb / tot)) < floor:
                return False
    except Exception:
        return True
    return True


def _cl_slots_of(slots, t):
    """the clothes-table body slot(s) a canon garment belongs to, by the tag
    or its head word ('black thighhighs' -> legwear)"""
    return frozenset(sl for sl in ("bottom", "legwear", "top", "uniform", "traditional", "swim")
                     if t in (slots.get(sl) or {}) or t.split()[-1] in (slots.get(sl) or {}))


def _head_garment(slots, t):
    """the measured garment behind a coloured canon tag: 'black thighhighs'
    counts as 'thighhighs' when the colour form is not a table item"""
    for sl in ("bottom", "legwear", "top", "uniform", "traditional", "swim"):
        if t in (slots.get(sl) or {}):
            return t
    return t.split()[-1]


def _canon_companion(tag):
    """-> True when this canon tag is another BEING in the picture rather
    than a trait of the subject (the author, 2026-09-16: 'pokemon
    (creature)' rode in on akari's canon at .49, and her Pokemon is not
    her appearance). A creature-flagged tag is a companion unless it is a
    race the subject can BE -- the race family knows those ('cat girl',
    'cat ears', 'tail', 'monster girl'); 'wolf', 'dog', 'pokemon
    (creature)' are not in it. The cast owns who else is in the frame."""
    t = str(tag or "").lower().strip()
    if not (_gloss_flags(t) & {"creature"}):
        return False
    fam = _race_family()
    return t not in set().union(*fam.values()) if fam else False


def _canon_contradicts(v, want_tags):
    """-> True when this character's canon fights a typed one-of trait.

    "blonde milf" drew a pink-haired persona, and the line then carried
    both colours on one head. Typed wins over canon at assembly, but the
    right place to honour it is the DRAW: a character whose canon holds a
    different value in a family the user typed is not this subject.
    """
    traits = set(v.get("traits") or {})
    for fam in ONE_OF_FAMILIES:
        typed = want_tags & fam
        if not typed:
            continue
        if (traits & fam) - typed:
            return True
    # A TYPED RACE IS ONE-OF TOO (2026-09-16): a bunny girl is not a lion
    # girl; a persona whose canon carries another race or its parts is out
    races = _race_family()
    typed_races = {t for t in want_tags if t in races}
    if typed_races:
        allowed = set().union(*[races[r] for r in typed_races]) | _RACE_GENERIC
        whole = set().union(*races.values())
        if (traits & whole) - allowed:
            return True
    return False


def _character_fit(v, want_tags, genre):
    """-> a small multiplier for how well a character suits the prompt.

    the author's asked the first draw to match "prompt traits + genre/style".
    Traits are the concrete half: if the user wrote 'a blonde girl', a
    blonde character should be likelier. The genre half is deliberately
    weak -- a series is not a genre and pretending otherwise would invent
    a mapping we have not measured.
    """
    if not want_tags:
        return 1.0
    traits = set((v.get("traits") or {}))
    hit = len(want_tags & traits)
    if not hit:
        return 1.0
    return 1.0 + min(hit, 3) * 1.5


def assign_personas(cast, base, opts, rng):
    """personas for existing subjects only -- never adds one. Gender-matched
    on ORIGINAL gender, futa personas drawn from female characters, 'other'
    subjects never named."""
    picks = []
    typed = _typed_characters(base) if base else []
    if not opts.get("gen_characters") and not typed:
        return picks

    # SERIES SCOPE: what the prompt names, plus the series every typed
    # character belongs to. Several series is fine -- generated characters
    # may then come from any of them.
    scope = set(_series_scope(base) or ())
    for n in typed:
        sr = (_traits().get(n) or {}).get("series")
        if sr:
            scope.add(sr)

    # what the prompt says about appearance, for the first draw's nudge
    # WITH THE BANKS: parse_input without them cannot run the vocabulary
    # scan, so 'blonde milf' yielded no `blonde hair` here and the draw
    # could neither prefer a blonde nor veto a pink-haired canon
    want_tags = set()
    try:
        want_tags = {str(t).lower() for t in
                     (pe.parse_input(base, pe.load_all_banks())[0] or [])}
    except Exception:
        pass
    genre = (opts.get("_genre") or (None, None))[1]

    # A PERSONA IS A PERSON. The traits table lists every character tag,
    # and some are devices, mecha and mascots with a gender field --
    # "bardiche (riot zanber stinger) (nanoha)" is a weapon, and it was
    # cast as the blonde milf. A drawable person has canon hair or eyes;
    # a thing does not. Data, not a list.
    # A RANDOM PERSONA NEEDS DECISIVE GENDER EVIDENCE AND A REAL NAME.
    # 'sekiutsu maria tarou' (a girl) was drawn as the goblin BOY on a
    # 0.27 boy / 0.22 girl split over 179 mostly-group posts, and
    # 'priest (dq3)' -- a class, not a person -- as the elf milf. The
    # dominant share must be at least 0.45 and twice the other; a bare
    # name that is a common word never stands in for a person.
    def _decisive(v):
        g, b = float(v.get("girl_share") or 0), float(v.get("boy_share") or 0)
        hi, lo = max(g, b), min(g, b)
        return hi >= 0.45 and hi >= 2 * lo

    pool = [(n, v) for n, v in _traits().items()
            if v.get("gender") in ("female", "male")
            and _decisive(v)
            and not _common_word(n.split(" (")[0])
            and not _species_phrase(n.split(" (")[0])
            and any(("hair" in t or "eyes" in t)
                    for t in (v.get("traits") or {}))]
    used = set(typed)
    bound = cast.get("_futa_name")
    n_futa = cast.get("futa", 0)
    # set by the first GENERATED pick when the prompt named no series --
    # everyone drawn after it comes from the same world
    locked_series = None

    for kind, want in (("female", cast.get("female", 0) + cast.get("futa", 0)),
                       ("male", cast.get("male", 0))):
        for slot_i in range(want):
            if typed:
                cand = [n for n in typed
                        if _traits()[n].get("gender") == kind]
                # futa slots come FIRST in the female bucket
                # (plan_subjects' build order) -- the futanari-bound name
                # must land exactly there, and nowhere else
                if kind == "female" and bound in cand:
                    if slot_i < n_futa:
                        cand = [bound]
                    else:
                        cand = [c for c in cand if c != bound] or cand
                if cand:
                    picks.append((kind, cand[0], _traits()[cand[0]]))
                    typed.remove(cand[0])
                    continue
            if not opts.get("gen_characters"):
                picks.append((kind, None, None))
                continue
            # A ROLLED CHARACTER ABOVE SAFE IS A STATED ADULT (2026-09-27, the author's
            # live test: 'kasai amane' -- a Precure middle-schooler -- was rolled into
            # an explicit anal prompt as the milf). The booru wiki's own page is read
            # for each character (character_ages.json); above safe only a character
            # it states an adult may be drawn -- unknown is not enough.
            _adult_only = sm.SPICE_ORDER.get(str(opts.get("_level") or "safe").lower(), 0) >= sm.SPICE_ORDER["sensitive"]
            eff = scope or ({locked_series} if locked_series else None)
            cand = [(n, v) for n, v in pool
                    if v.get("gender") == kind and n not in used
                    and (not eff or v.get("series") in eff)
                    and not _canon_contradicts(v, want_tags)
                    and (not _adult_only or not character_is_young(n, rolled=True))]
            if not cand:
                # the scope has nobody left of this gender: UNNAMED, not
                # filled from another world
                picks.append((kind, None, None))
                continue
            w = [max(1.0, float(v.get("posts") or 1)) ** 0.5
                 * _character_fit(v, want_tags, genre) for _n, v in cand]
            n, v = rng.choices(cand, weights=w)[0]
            used.add(n)
            if locked_series is None and not scope and v.get("series"):
                locked_series = v["series"]
            picks.append((kind, n, v))
    return picks


# Which words actually STATE a hair/eye colour. pe.COLORS covers the plain
# ones; these are the multi-colour statements that genuinely conflict with
# a plain colour (a character is not both 'two-tone' and 'blue'-only).
# Everything else ending in ' hair' -- drill, spiked, antenna, tentacle,
# curly, messy, wavy, medium -- is a STYLE or LENGTH and must not compete.
_CANON_COLOURS = set(pe.COLORS) | {
    "gradient", "multicolored", "two-tone", "split-color", "streaked",
    "colored", "rainbow", "platinum", "strawberry", "light", "dark",
    "pale", "crystal", "starry"}


def _is_identity_trait(tag):
    """-> True for the canon that survives a RANDOM appearance.

    Hair and eye COLOUR only: they are what makes a named character
    recognisable at a glance. Length, style, garments, accessories and body
    detail are all re-rolled. Uses the same colour vocabulary as the canon
    one-of cluster, so 'spiked hair' is a style (rolled) while 'blonde
    hair' is identity (kept).
    """
    t = str(tag).lower()
    for suffix in (" hair", " eyes"):
        if t.endswith(suffix) and len(t.split()) == 2 \
                and t.split()[0] in _CANON_COLOURS:
            return True
    return False


_SUBJECT_LAYER_SLOTS = ("hair", "body", "clothing", "gaze", "expression", "pose")


_SENT_CACHE = {"key": None, "data": None}
_SUBJECT_SENTENCE = re.compile(
    r"^(the\s+)?(first|second|third|fourth|other|another|1st|2nd|3rd|one|1girl|1boy|1futa|1other|"
    r"(girl|boy|woman|man|subject|character)\s*(one|two|three|\d))\b", re.I)


def brief_sentences(base):
    """the brief cut at its full stops (never inside a number)"""
    return [x.strip() for x in re.split(r"(?<![0-9])\.(?![0-9])\s*|[;\n]+", str(base or "")) if x.strip()]


def structured_scene(base):
    """-> the SCENE SENTENCE of a structured brief, else None. A brief is
    structured when a later sentence opens on a subject ('First girl ...',
    '1girl, ...', 'The other woman ...'): its first sentence alone says how
    many are in the picture (2026-09-19: 'Two girls playing ... First girl
    ... Second girl ...' counted FOUR girls)."""
    sents = brief_sentences(base)
    if len(sents) >= 3 and sum(1 for x in sents[1:] if _SUBJECT_SENTENCE.match(x)) >= 2:
        return sents[0]
    return None



def structured_subjects(base):
    """-> the per-subject tag lists of a structured brief (as many as it has
    subject sentences), else None"""
    if not structured_scene(base):
        return None
    n = sum(1 for x in brief_sentences(base)[1:] if _SUBJECT_SENTENCE.match(x))
    return subject_sentences(base, n)


def subject_sentences(base, n_total):
    """-> [[tags of subject 0's sentence], [subject 1's], ...] when the brief is
    written THE STRUCTURED WAY (the author's proposal, 2026-09-19, the manual's
    advice for several subjects): the first sentence is the scene and says how
    many, then ONE SENTENCE PER SUBJECT, then whatever else (style, light) --
    "Two girls playing volleyball in the ocean. First girl has big breasts and
    a red bikini. Second girl has blue eyes. Sunny weather", tags or prose or
    both. None when the brief is not written that way (fewer sentences than
    subjects + 1): the older reading by the nearest subject noun stands."""
    if n_total < 2 or not base:
        return None
    key = (base, n_total)
    if _SENT_CACHE["key"] == key:
        return _SENT_CACHE["data"]
    sents = brief_sentences(base)
    out = None
    if len(sents) >= n_total + 1:
        try:
            banks = pe.load_all_banks()
            out = []
            # the subjects' sentences are the ones that open on a subject, in
            # order; without such openers, simply the ones after the scene
            _subj = [x for x in sents[1:] if _SUBJECT_SENTENCE.match(x)]
            _use = _subj[:n_total] if len(_subj) >= n_total else sents[1:1 + n_total]
            # THE SENTENCES IN THE CAST'S ORDER (2026-10-03: 'A man and a woman in a
            # cafe. The man has a beard. The woman wears a red dress.' gave the
            # woman the beard): the cast lists women, then men, then futanari, and
            # the list returned is read by subject index. Where every sentence
            # opens on a being of a known sex, they are put in that order -- two of
            # the same sex keep the order they were written in.
            def _sex_rank(sn):
                for w in re.sub(r"[^a-z0-9 ]+", " ", str(sn).lower()).split()[:4]:
                    if w in _FEMALE_NOUN_LIST or w in ("1girl", "she", "her") or re.search(r"(girl|woman)$", w):
                        return 0
                    if w in _MALE_NOUN_LIST or w in ("1boy", "he", "his") or re.search(r"(boy|man)$", w):
                        return 1
                    if w.startswith("futa") or w == "1futa":
                        return 2
                return None
            _ranks = [_sex_rank(sn) for sn in _use]
            if None not in _ranks and len(set(_ranks)) > 1:
                _use = [sn for _r, _i, sn in sorted(zip(_ranks, range(len(_use)), _use))]
            for sn in _use:
                got = pe.parse_input(sn, banks)
                tl = got[0] if isinstance(got, tuple) else got
                out.append([str(t).lower() for t in (tl or [])])
        except Exception:
            out = None
    _SENT_CACHE["key"], _SENT_CACHE["data"] = key, out
    return out


def beings_in_subject_order(base, kinds):
    """-> [the subject index of the j-th being the text names], None for a being
    no subject is left for. THE TEXT'S ORDER IS NOT THE CAST'S (2026-10-03: 'a man
    and a woman, the man has a beard, the woman wears a red dress' gave the woman
    the beard and the man the dress): the cast lists the women first, and every
    reading by position took the first being written for the first subject. A
    being takes the first free subject of its own sex ('man', 'boy', 'he' a male;
    'woman', 'girl' a female); a noun that names no sex ('nurse', 'doctor') takes
    what is left, in order."""
    kinds = [str(k or "") for k in (kinds or [])]
    nouns = []
    for n, c in _subject_spans_text(base):
        nouns += [str(n).lower()] * max(1, int(c or 1))

    def _sex(n):
        if n in _FEMALE_NOUN_LIST or re.search(r"(girl|woman|women|female)s?$", n):
            return "female"
        if n in _MALE_NOUN_LIST or re.search(r"(boy|man|men|male)s?$", n):
            return "male"
        if re.search(r"^futa", n):
            return "futanari"
        return None
    out, free = [None] * len(nouns), list(range(len(kinds)))
    for j, n in enumerate(nouns):
        sx = _sex(n)
        i = next((i for i in free if sx and (kinds[i] == sx or (sx == "futanari" and kinds[i].startswith("futa")))), None)
        if i is not None:
            out[j] = i
            free.remove(i)
    for j, n in enumerate(nouns):
        if out[j] is None and _sex(n) is None and free:
            out[j] = free.pop(0)
    return out


def _being_mentions(base, n_total, kinds=None):
    """-> (the text, lowered and spaced; [(position, subject index)] for every
    mention of a being): its noun, the noun again, and the pronoun of its sex"""
    low = " " + re.sub(r"[^a-z0-9'()-]+", " ", (base or "").lower()) + " "
    spans = _subject_spans_text(base) if n_total > 1 else []
    starts = []
    if n_total > 1:
        pos = 0
        for noun, _c in spans:
            k = low.find(" " + noun + " ", pos)
            starts.append(k if k >= 0 else pos)
            pos = max(pos, k + 1) if k >= 0 else pos
    # A SUBJECT NAMED AGAIN OWNS WHAT FOLLOWS (the author, 2026-10-03: 'a girl and a
    # boy, the girl has a hat ribbon' put the ribbon on the boy). Only the first
    # mention of each being was a start, so every word after the second being was
    # his, whoever the sentence went back to. Every mention counts: the noun again
    # where no other being shares it, and 'she / her', 'he / his / him' where the
    # text has exactly one being of that sex.
    mentions = [(st, j) for j, st in enumerate(starts) if st >= 0]
    # the being's subject: by its sex where the cast is known (the cast's order is
    # women first, the text's is the user's), else by its place in the text
    _subj_of_span = list(range(len(spans)))
    if n_total > 1 and kinds:
        _order = beings_in_subject_order(base, kinds)
        _k, _subj_of_span = 0, []
        for _n, _c in spans:
            _subj_of_span.append(_order[_k] if _k < len(_order) else None)
            _k += max(1, int(_c or 1))
    if n_total > 1:
        _nouns = [n for n, _c in spans]

        def _all(word, j):
            k = low.find(" " + word + " ")
            while k >= 0:
                mentions.append((k, j))
                k = low.find(" " + word + " ", k + 1)
        for j, n in enumerate(_nouns):
            if _nouns.count(n) == 1:
                _all(n, j)
        _fem = [j for j, n in enumerate(_nouns) if n in _FEMALE_NOUN_LIST or n.endswith("girl")]
        _mal = [j for j, n in enumerate(_nouns) if n in _MALE_NOUN_LIST or n.endswith("boy")]
        if len(_fem) == 1:
            for w in ("she", "her", "hers"):
                _all(w, _fem[0])
        if len(_mal) == 1:
            for w in ("he", "his", "him"):
                _all(w, _mal[0])
    return low, [(st, (_subj_of_span[j] if j < len(_subj_of_span) else j)) for st, j in mentions]


def _typed_subject_tags(base, user_tags, si, n_total, persona=None, kinds=None):
    """the typed tags that belong to subject `si`: a tag of a subject
    layer (clothes, body, hair, expression, pose, a nudity state), owned
    by the only subject, else by the subject whose noun or name stands
    last before the tag in the text; a tag the text does not carry (a
    tag-list prompt) goes to subject 0 when there is one subject."""
    out = []
    low, mentions = _being_mentions(base, n_total, kinds)
    try:
        banks = pe.load_all_banks()
    except Exception:
        banks = {}
    for t in user_tags:
        t = str(t).lower().strip()
        if not t or re.match(r"^\d*(?:girls?|boys?|futas?|others?)$", t):
            continue
        try:
            slot = sm.slot_of(t, banks)
        except Exception:
            slot = None
        fl = set()
        try:
            fl = set(_gloss_flags(t) or ())
        except Exception:
            pass
        # a garment by its head word too ('red scarf' glosses as an object);
        # never an act or a pose -- the action plan owns those
        _head = t.split()[-1]
        _garment = _is_garment(t) or any(_head in items for items in
                                          ((_clothes_table() or {}).get("slots") or {}).values())
        if fl & {"act"} or slot == "pose" or "pose" in fl:
            continue
        # a typed job or race is the subject's too ('pirate captain',
        # 'cyborg girl', 'a nurse and a doctor': the doctor is the second)
        # AN AGE TAG DESCRIBES THE SUBJECT (2026-09-16: a typed 'milf' has
        # no gloss flags at all, so it belonged to nobody and sat in the
        # scene band): the age whitelist is the one list of maturity words
        _role = (t in _AGE_TAG_SET
                 or (bool(fl & {"person", "race"}) and t not in _FEMALE_NOUN_LIST and t not in _MALE_NOUN_LIST))
        if not (slot in ("hair", "body", "clothing", "gaze", "expression") or _garment or _role
                or fl & {"clothing", "body", "expression"}
                or t in ("nude", "topless", "bottomless", "completely nude", "no panties", "no bra")):
            continue
        if n_total <= 1:
            out.append(t)
            continue
        # THE STRUCTURED BRIEF: the subject's own sentence owns its words
        _ss = subject_sentences(base, n_total)
        if _ss is not None:
            if si < len(_ss) and t in _ss[si]:
                out.append(t)
            continue
        p = low.find(" " + t + " ")
        if p < 0:
            continue                             # a tag-list word: nobody's, with several subjects
        owner, _at = None, -1
        for st, j in mentions:
            if _at <= st <= p:               # a subject noun that IS the tag owns itself ('doctor')
                owner, _at = j, st
        if persona:
            pp = low.rfind(str(persona).lower().split(" (")[0], 0, p)
            if pp >= 0 and (owner is None or pp > _at):
                owner = si
        if owner == si:
            out.append(t)
    return out


def _slots_covered_by(tags):
    """the subject slots a set of typed garments fills: a body garment
    covers the outfit, legwear the legwear, neckwear and headwear theirs"""
    covered = set()
    try:
        tb = _clothes_table() or {}
        slots = tb.get("slots") or {}
    except Exception:
        slots = {}
    _BODY = ("top", "bottom", "swim", "uniform", "traditional", "sexual")
    for t in tags:
        t = str(t).lower()
        head = t.split()[-1] if t.split() else t
        found = [sl for sl, items in slots.items() if t in items] or \
                [sl for sl, items in slots.items() if head in items]
        for sl in found:
            if sl in _BODY:
                covered.add("outfit")
            elif sl == "legwear":
                covered.add("legwear")
            elif sl == "neck":
                covered.add("neckwear")
            elif sl in ("head", "accessory_head"):
                covered.add("headwear")
        if not found and head in ("shirt", "skirt", "dress", "uniform", "jacket", "coat", "sweater",
                                  "top", "pants", "shorts", "swimsuit", "bikini", "leotard", "kimono"):
            covered.add("outfit")
        if t in ("nude", "completely nude", "topless", "bottomless", "no panties", "no bra"):
            covered.add("nudity state")
    return covered


_NUDITY_TAGS = {"data": None}


def _is_nudity_tag(tag):
    """the spice table's nudity slot (nude, topless, naked shirt...)"""
    if _NUDITY_TAGS["data"] is None:
        try:
            _NUDITY_TAGS["data"] = {str(k).lower() for k in
                                    (((_spice_table() or {}).get("slots") or {}).get("nudity") or {})}
        except Exception:
            _NUDITY_TAGS["data"] = set()
        _NUDITY_TAGS["data"] |= {"nude", "completely nude", "topless", "bottomless"}
    return str(tag or "").lower().strip() in _NUDITY_TAGS["data"]


# WHAT ONLY A DRESSED BODY CAN HAVE (the author, 2026-09-16: "combining
# the raw tag 'nude' with clothes is not ok on one subject"). The clothes
# table's own slots answer it: top, bottom, swim, uniform, traditional,
# legwear and sexual are garments; sleeve, modifier and print are parts or
# states OF a garment ('short sleeves', 'see-through clothes', 'striped
# shirt' -> print) and imply one; head, hands, neck, feet and the accessory
# slots are accessories, which a naked subject may wear (a hat, gloves,
# earrings). A word naming clothes at all ('open clothes', 'clothes lift')
# implies them wherever it sits.
_CLOTHING_SLOTS = ("top", "bottom", "swim", "uniform", "traditional", "legwear",
                   "sexual", "sleeve", "modifier", "print")
_CLOTHES_WORDS = ("clothes", "clothing", "outfit")


_EYE_WORD_RE = re.compile(r"\b(?:eyes?|eyed|pupils?)\b")
_SHUT_EYES = {"closed eyes"}


def is_eye_state(tag):
    """-> True when the tag says something about the eyes themselves: a
    colour, a pupil, a gaze, an eye state. Hair that merely mentions them
    ('hair between eyes', 'hair over eyes') is hair, and 'eyelashes' names
    no eye at all."""
    t = str(tag or "").lower().strip()
    if not _EYE_WORD_RE.search(t):
        return False
    fl = _gloss_flags(t) or set()
    if "hair" in fl:
        return False
    return bool(fl & {"expression", "body"}) or any(t in fam for fam in ONE_OF_FAMILIES)


def implies_clothing(tag, _chain=True):
    """-> True when this tag can only be true of a dressed body"""
    t = str(tag or "").lower().strip()
    if not t or _is_whole_nudity(t) or t.startswith("naked "):
        return False
    if any(w in t.split() for w in _CLOTHES_WORDS):
        return True
    # ANATOMY IS NOT A GARMENT whatever its head word ('clitoral hood'
    # shares 'hood' with a garment): the gloss says which it is
    _fl = _gloss_flags(t) or set()
    if "body" in _fl and "clothing" not in _fl:
        return False
    # A GARMENT'S STATE IS THE GARMENT (2026-09-16: 'hood down' beside
    # 'nude'): the booru says 'hood down' implies 'hood', and a hood is
    # clothing; the chain is read one level deep so a cycle cannot loop
    if _chain:
        for a in implied(t):
            # only an ancestor the booru calls clothing: 'rose' implies
            # 'flower', which sits in a print slot and is no garment
            if "clothing" in (_gloss_flags(a) or set()) and implies_clothing(a, _chain=False):
                return True
    slots = (_clothes_table() or {}).get("slots") or {}
    for sl in _CLOTHING_SLOTS:
        items = slots.get(sl) or {}
        if t in items or t.split()[-1] in items:
            return True
    # a garment the table names under another qualifier ('alzano school
    # uniform' for 'school uniform', 'blue serafuku'): the head word of a
    # clothing-slot item is a garment word, but only on a tag the gloss
    # calls clothing -- 'patterned hair' sits in the print slot, and
    # 'blonde hair' is hair whatever its head word
    if "clothing" not in (_gloss_flags(t) or set()):
        return False
    return _garment_stem(t.split()[-1]) in _clothing_stems()


_CLOTHING_STEMS = {"key": None, "data": None}


def _clothing_stems():
    """the head words of every item on a CLOTHING slot (accessories are
    not here: a hat, gloves and a scarf may be worn naked)"""
    tb = _clothes_table() or {}
    if _CLOTHING_STEMS["key"] == id(tb) and _CLOTHING_STEMS["data"] is not None:
        return _CLOTHING_STEMS["data"]
    slots = tb.get("slots") or {}
    ws = set()
    for sl in _CLOTHING_SLOTS:
        for t in (slots.get(sl) or {}):
            ws.add(_garment_stem(t.split()[-1]))
    ws -= {"clothe", "clothing", "outfit"}
    _CLOTHING_STEMS["key"], _CLOTHING_STEMS["data"] = id(tb), ws
    return ws


def strip_body_garments(outfit):
    """-> the outfit with every garment on a body slot gone when a whole
    nudity state is in it (2026-09-16: 'completely nude, bodysuit' on one
    subject -- the persona's canon bodysuit under the model's nudity);
    accessories stay, a naked subject may wear them"""
    items = [str(x) for x in (outfit or []) if x]
    if not any(_is_whole_nudity(x) for x in items):
        return items
    return [x for x in items if not implies_clothing(x)]


def _is_whole_nudity(tag):
    """a state that REPLACES the outfit (2026-09-15): the nudity slot's
    'complete' section (nude, completely nude), topless / bottomless, and
    the naked compounds (naked apron: the apron is the whole outfit). A
    partial exposure (open shirt, cleavage, see-through clothes, no bra)
    is a modifier on a worn outfit -- the clothes roll's, never a reason
    to drop the outfit slots"""
    t = str(tag or "").lower().strip()
    if not t:
        return False
    if t in ("nude", "completely nude", "topless", "bottomless", "topless female", "topless male", "naked"):
        return True
    if t.startswith("naked "):
        return True
    ent = (((_spice_table() or {}).get("slots") or {}).get("nudity") or {}).get(t) or {}
    return ent.get("section") == "complete"


def plan_subjects(cast, base, opts, rng):
    """-> list of per-subject dicts: kind, persona, locked canon, slots to
    fill. The engine decides the STRUCTURE; the LLM only fills content."""
    detail = DETAIL_SCALE.get(opts.get("detail", "standard"), 1.0)
    # APPEARANCE (the author's point 8): automatic (the prompt, then canon, then
    # the tables) or random (the tables, identity locked). The old
    # 'clothes_mode' option is gone (2026-09-06).
    appearance = str(opts.get("appearance") or "automatic").lower()
    if appearance in ("canon", "auto"):
        appearance = "automatic"
    if appearance == "none":
        appearance = "random"     # 'none' retired: the prompt says that
    personas = assign_personas(cast, base, opts, rng)
    cmode = "canon" if appearance == "automatic" else "random"
    if cmode == "canon" and not any(p[1] for p in personas):
        cmode = "random"          # canon needs a known character
    clothes = True
    subjects = []
    i = 0
    order = ([("female", cast.get("female", 0) + cast.get("futa", 0)),
              ("male", cast.get("male", 0)),
              ("other", cast.get("other", 0))])
    futa_left = cast.get("futa", 0)
    for kind, n in order:
        for _ in range(n):
            is_futa = kind == "female" and futa_left > 0
            if is_futa:
                futa_left -= 1
            persona = personas[i] if i < len(personas) else (kind, None, None)
            name, tv = persona[1], persona[2]
            locked = {}
            if tv:
                # canon = identity only. Explicit content obeys the level
                # gate later; DOUJIN RIDERS are cut here -- 'comic',
                # 'monochrome' and 'green background' are how Meiling gets
                # DEPICTED, not what she is.
                rider = re.compile(
                    r"\b(comic|monochrome|greyscale|4koma|sketch|background|"
                    r"speech bubble|translated|check translation|"
                    r"looking at viewer|open mouth|closed mouth|smile|blush|"
                    r"aged down|aged up|child|focus|solo|"
                    r"uncle|aunt|niece|nephew|siblings|brothers|sisters|"
                    r"[0-9]\+?(girls?|boys?)|multiple (girls|boys))\b")
                # CANON IS A BODY AND A LOOK, NEVER A PAIRING. A character's
                # traits are mined from her posts, and a character who is
                # usually drawn with a man carries `hetero` in them -- so
                # "a mother and her daughter" cast maishima yuri and got
                # `hetero` on two women at safe level. Who is paired with
                # whom is the scene's fact (bridge._PAIRING); any act that
                # needs a second person is likewise not canon.
                # A BACKDROP IS NOT A TRAIT (the author's 2026-09-15: "coloured /
                # simple backgrounds leak into prompts with locations"):
                # 2,626 characters carry 'simple background' or 'white
                # background' in their canon bundle because their posts do;
                # the picture's place is the location layer's, one owner
                # THE RULED WORDS APPLY TO CANON TOO (2026-09-16: 'virtual
                # youtuber', denied as an occupation since 2026-09-03, rode
                # in on shishiro botan's canon): the never and typed-only
                # list gates every emitter, a persona's bundle included
                _ruled_c = ruled_out()
                locked = {t: f for t, f in (tv.get("traits") or {}).items()
                          if not sm.EXPLICIT_RE.search(t)
                          and not rider.search(t)
                          and t not in sm.PAIRING_TAGS
                          and (sm.arity_of(t) or 1) < 2
                          and not str(t).endswith(" background")
                          and not (_gloss_flags(t) & {"location", "scenery"})
                          and not _canon_companion(t)
                          and t not in _ruled_c}
                # NOT-HEAD HAIR IS NOT CANON HAIR. 'facial hair' was
                # winning the hair slot outright for bearded characters.
                for t in list(locked):
                    if t in _NOT_HEAD_HAIR:
                        locked.pop(t, None)
                # canon-internal one-of conflicts: an ensemble-polluted bundle
                # can carry TWO hair colours (kariya: purple hair + white
                # hair) -- keep the highest-frequency one per cluster.
                #
                # ONLY COLOURS COMPETE FOR THE COLOUR SLOT. This used to
                # treat any two-word '<x> hair' tag as a candidate, so a
                # HAIRSTYLE could win and the real colour was dropped --
                # 49 of our 2,000 characters, including 'bakugou katsuki'
                # (kept 'spiked hair', discarded 'blonde hair') and
                # 'callie (splatoon)' ('tentacle hair' over 'black hair').
                # A style and a colour are different slots, not
                # alternatives; both stay.
                for suffix in (" hair", " eyes"):
                    clr = [(f, t) for t, f in locked.items()
                           if t.endswith(suffix) and len(t.split()) == 2
                           and t.split()[0] in _CANON_COLOURS]
                    for _, t in sorted(clr)[:-1]:
                        locked.pop(t, None)
                # hair LENGTH is its own one-of cluster (kariya carried both
                # long hair and short hair from different-era depictions)
                lengths = [(locked.get(t, 0), t) for t in
                           ("very long hair", "long hair", "medium hair",
                            "short hair", "very short hair") if t in locked]
                for _, t in sorted(lengths)[:-1]:
                    locked.pop(t, None)
                # A PARENT TAG IS IMPLIED BY ITS OWN CHILD. tagnet's
                # NEVER_PROPOSE already refuses bare 'breasts'/'hair'/'eyes'
                # for exactly this reason -- "implied by any specific value,
                # so emitting them adds nothing and invites 'breasts' next
                # to 'large breasts'" -- but it only ever guarded tags the
                # NETWORK proposed. Canon never passed through it, so 65% of
                # characters carried a bare parent beside its child (1,881
                # tags: breasts+large breasts, tail+horse tail, ribbon+hair
                # ribbon, halo+blue halo...).
                #
                # A rule about tag SHAPE, not a list of families, so it also
                # covers whatever the next data refresh introduces.
                #
                # THE RULE IS SUFFIX CONTAINMENT, and a bare parent is only
                # its trivial case. Restricting it to single words missed
                # the multi-word parents: 'school uniform' beside 'tracen
                # school uniform', 'school uniform' beside "little busters!
                # school uniform". If A is a proper suffix of B on a word
                # boundary, B says everything A says and more.
                #
                # It stays LEXICAL on purpose. 'bat wings' + 'head wings'
                # (Morrigan has both) and 'colored skin' + 'pink skin' are
                # not suffix pairs and are left alone -- deciding those
                # needs meaning, not shape, and guessing would cost real
                # canon.
                for t in sorted(locked, key=len):
                    if t not in locked:
                        continue
                    for other in list(locked):
                        if other != t and other.endswith(" " + t):
                            locked.pop(t, None)
                            break
                # CANON PASSES THE SAME GATE EVERY ROLLED TAG PASSES: a
                # character's measured traits carried `everyone` and
                # `6+girls` (ensemble bookkeeping) onto her subject block
                from promptstudio.engine.tagnet import NEVER_PROPOSE as _NP9
                locked = {t: f for t, f in locked.items()
                          if not _NP9.match(t)
                          and not ({"meme", "text", "format", "emote",
                                    "symbol", "theme"}
                                   & set(_gloss_of(t)[1] or ()))}
                # canon GARMENTS survive only in canon mode: under 'none'
                # only the prompt dresses anyone, and under 'random' the
                # dice do -- Miku keeps her hair either way, not her necktie
                if cmode != "canon":
                    # RANDOM APPEARANCE KEEPS IDENTITY. Hair and eye
                    # COLOUR are who the character is, not what they are
                    # wearing -- strip everything else and let the dice
                    # dress them.
                    locked = {t: f for t, f in locked.items()
                              if _is_identity_trait(t)}
            # canon owns its slots: Miku's aqua eyes must not be re-filled
            # with a rolled colour ('blue eyes' rode in exactly that way)
            # TYPED SUBJECT TAGS ARE THIS SUBJECT'S CANON (2026-09-14)
            _typed_mine = _typed_subject_tags(base, opts.get("_user_tags") or [], i,
                                              sum(n_ for _k, n_ in order), name,
                                              kinds=[_k for _k, n_ in order for _x in range(n_)])
            for _t in _typed_mine:
                locked.setdefault(_t, "typed")
            # CANON GARMENTS THE BOORU DOES NOT WEAR TOGETHER (2026-09-16:
            # 'pants, black thighhighs' from one persona's costumes): two
            # canon garments of different body slots whose measured pair
            # lift is under .5 (pants x thighhighs .26; skirt x thighhighs
            # 1.4) are not one outfit -- the rarer canon garment goes
            try:
                _cl_slots = (_clothes_table() or {}).get("slots") or {}
                _gs = [t for t in locked if any(t in (_cl_slots.get(sl) or {}) or t.split()[-1] in (_cl_slots.get(sl) or {})
                                               for sl in ("bottom", "legwear", "top", "uniform", "traditional", "swim"))]
                _tot = float(_count_cached("1girl") or 0)
                for i9 in range(len(_gs)):
                    for j9 in range(i9 + 1, len(_gs)):
                        a9, b9 = _gs[i9], _gs[j9]
                        if a9 not in locked or b9 not in locked:
                            continue
                        ha = _cl_slots_of(_cl_slots, a9); hb = _cl_slots_of(_cl_slots, b9)
                        if not ha or not hb or ha == hb:
                            continue
                        na = _count_cached("1girl " + _q(_head_garment(_cl_slots, a9)))
                        nb = _count_cached("1girl " + _q(_head_garment(_cl_slots, b9)))
                        nab = _count_cached("1girl " + _q(_head_garment(_cl_slots, a9)) + " " + _q(_head_garment(_cl_slots, b9)))
                        if not (_tot and na and nb) or nab is None:
                            continue
                        lift = (nab / _tot) / ((na / _tot) * (nb / _tot))
                        if lift < 0.5:
                            locked.pop(a9 if locked.get(a9, 0) < locked.get(b9, 0) else b9, None)
            except Exception:
                pass
            canon_join = " " + " ".join(locked) + " "
            covered = set(_slots_covered_by(_typed_mine))
            if " hair" in canon_join or "hair " in canon_join:
                covered |= {"hair", "hair_style"}
            if " eyes" in canon_join:
                covered.add("eyes")
            if any(w in canon_join for w in (" thighhighs ", " pantyhose ",
                                             " kneehighs ", " socks ")):
                covered.add("legwear")
            garmenty = sum(1 for t in locked if t.split()[-1] in
                           ("shirt", "skirt", "dress", "uniform", "jacket",
                            "sleeves", "necktie", "coat", "sweater", "top"))
            if garmenty >= 3:
                covered.add("outfit")
            # canon necktie means the neckwear slot is TAKEN -- offering it
            # anyway added a neck ribbon beside Miku's necktie
            if any(t.split()[-1] in ("necktie", "bowtie", "choker", "scarf",
                                     "necklace", "ribbon", "collar")
                   for t in locked):
                covered.add("neckwear")
            if any(t.split()[-1] in ("hat", "cap", "hood", "headband",
                                     "hairband", "crown", "beret")
                   for t in locked):
                covered.add("headwear")
            slots = []
            for slot, cls, gate in _SLOT_MENU:
                if slot in covered:
                    continue
                if gate and "clothes" in gate and not clothes:
                    continue
                if gate and "female" in gate and kind != "female":
                    continue
                p = CHANCE[cls] * (detail if cls != "core" else 1.0)
                if detail == 0.0 and cls != "core":
                    continue
                if rng.random() < min(1.0, p):
                    slots.append(slot)
            # NUDITY IS A CLOTHING STATE of this subsection, spice-gated;
            # a naked subject can still wear accessories. The roll runs in
            # EVERY clothes mode -- the author's: even under 'none' the level of
            # undress may still be specified by spice.
            _nud_req = [t for t in (opts.get("_required_content") or []) if _is_whole_nudity(t)]   # a partial exposure keeps the outfit
            # the slots the nudity state owns: topless keeps the bottoms,
            # bottomless the tops -- the compound is the whole outfit
            _drop = ("outfit", "legwear", "neckwear")
            if _nud_req and all(t == "topless" for t in _nud_req):
                _drop = ("outfit",)
            elif _nud_req and all(t == "bottomless" for t in _nud_req):
                _drop = ("legwear",)
            # A NUDITY STATE IS NOT A GARMENT, AND AN ACCESSORY IS NO
            # CONFLICT (2026-09-15: a typed 'nude' covers the 'nudity state'
            # layer slot, so it met itself here as "a typed garment", 'naked
            # nude' measured nothing and the nudity was dropped -- 'a naked
            # girl' rolled a skirt; 'nude, straw hat' did the same through
            # the hat). Only a typed garment on a slot the nudity owns meets
            # it; a hat, gloves or a belt are worn naked, as the roll allows.
            _typed_body = [t for t in _typed_mine
                           if not _is_nudity_tag(t) and (_slots_covered_by([t]) & set(_drop))]
            if _nud_req and _typed_body:
                # a typed garment meets an injected nudity: the booru's
                # compound ('naked shirt', 'naked apron') when measured,
                # else the garment wins and the nudity goes
                _g = _typed_body[0]
                _comp = "naked " + _g.split()[-1]
                if _floor_measured(_comp):
                    locked.pop(_g, None)
                    locked[_comp] = "typed"
                    _typed_mine = [t for t in _typed_mine if t != _g] + [_comp]
                    opts["_required_content"] = [t for t in opts["_required_content"]
                                                 if not _is_nudity_tag(t)] + [_comp]
                    # THE COMPOUND STANDS FOR ITS PARTS (2026-09-22, caught by
                    # the harness: 'a nude girl in a shirt' came out `nude,
                    # shirt` -- the compound was made here and the two typed
                    # words went on to the line by their own channel). One
                    # owner for the fact: the parts leave the typed words.
                    try:
                        _ut9 = [str(t) for t in (opts.get("_user_tags") or [])]
                        _parts9 = {str(_g).lower(), }
                        _parts9 |= {str(t).lower() for t in (opts.get("_required_content") or []) if False}
                        _ut9 = [t for t in _ut9 if t.lower() != str(_g).lower()
                                and not _is_whole_nudity(t.lower())]
                        if _comp not in _ut9:
                            _ut9.append(_comp)
                        opts["_user_tags"] = _ut9
                    except Exception:
                        pass
                else:
                    opts["_required_content"] = [t for t in opts["_required_content"]
                                                 if not _is_nudity_tag(t)]
                _nud_req = []
            if _nud_req:
                # an injected or typed nudity state drops the body slots as
                # the roll does (2026-09-14)
                slots = [s2 for s2 in slots if s2 not in _drop]
                if "nudity state" not in slots:
                    slots.append("nudity state")
            elif not any(implies_clothing(t) for t in _typed_mine) \
                    and rng.random() < nudity_chance(
                    opts.get("_level", "sensitive"),
                    place=(opts.get("_location") or (None, None, None))[1],
                    genre=(opts.get("_genre") or (None, None))[1],
                    occupation=next((t for t in _typed_mine if t in _occupations_known()), None)):
                # MEASURED (2026-09-15): P(nude | the level's rating), lifted by
                # the place, the genre and the occupation -- an onsen undresses,
                # a hospital does not
                slots = [s2 for s2 in slots
                         if s2 not in ("outfit", "legwear", "neckwear")]
                slots.append("nudity state")
            skind = "futanari" if is_futa else kind
            has_pussy = is_futa and rng.random() < 0.35   # futa type roll
            _cam = opts.get("_camera") or (None, None, None)
            _sup = _FRAME_SUPPRESS.get(_cam[0] or "", {})
            slots = [s2 for s2 in slots
                     if s2 not in (_sup.get("slots") or set())]
            # CANON BODIES ARE NOT EMBELLISHED (the author's: a canon
            # character must not get 'huge ass' unprompted -- canon
            # defines the body; sizes come from the locked traits or
            # not at all). The genital slot stays: a typed futanari is
            # the user's own out-of-canon addition.
            _sup_body = set(_sup.get("body") or frozenset())
            if name:
                _sup_body |= {"ass", "breasts"}
            # THE ACT DECIDES WHAT IS IN THE PICTURE (measured, the author's
            # 2026-09-06: a blowjob described by the pussy pulls the eye
            # off the face): a part whose share beside the act falls under
            # .6 of its share at the level is not described
            _act9 = opts.get("_act")
            if _act9:
                for _pt in ("pussy", "penis", "pubic hair", "ass", "breasts", "feet", "hands"):
                    _lf = act_lift(_act9, _pt, opts.get("_level", "sensitive"), view=_cam[1] or "")
                    if _lf is not None and _lf < 0.6:
                        _sup_body.add("legs and feet" if _pt == "feet" else _pt)
            body_parts = plan_body(skind, has_pussy,
                                   opts.get("_level", "sensitive"),
                                   max(detail, 0.4), rng,
                                   suppress=frozenset(_sup_body))
            subjects.append({
                "kind": skind, "has_pussy": has_pussy if is_futa else None,
                "persona": name, "series": (tv or {}).get("series"),
                "locked_canon": locked, "slots": slots,
                "body_parts": body_parts,
                "action_plan": plan_actions(skind,
                                            opts.get("_level", "sensitive"),
                                            max(detail, 0.4), rng,
                                            act=opts.get("_act")),
                "flavor": plan_flavor(skind, max(detail, 0.4), rng)})
            i += 1

    # ---- TIER PASS (the author's): demote beyond the quotas. Typed-named
    # subjects claim main slots first; secondaries lose the whole pipeline
    # and keep a one-phrase look; everyone past SECONDARY_CAP merges into
    # ONE collective entity.
    typed_names = set(_typed_characters(base))
    n_named = sum(1 for s in subjects if s.get("persona") in typed_names)
    n_main, n_sec, collective, background, capped_detail, all_sec = \
        assign_tiers(len(subjects), n_named, base,
                     opts.get("detail", "standard"))
    # MAIN-SLOT PRIORITY (the author's): user-typed names rank above
    # checkbox-assigned personas (the checkbox names everyone, which says
    # nothing about who the USER cares about), and among typed names the one
    # the user DESCRIBED most wins -- their own words are the attention
    # signal. Unnamed subjects fall back to mention order.
    try:
        from promptstudio.engine import enhancer as _pe2
        said = _pe2.character_attributes(base or "", list(typed_names), {})
    except Exception:
        said = {}

    # MENTION ORDER is the attention signal (the author's: the user describes
    # the subject they care about FIRST -- 'Cloud walks ... not noticing
    # Tifa fucking Aerith' means Cloud is the focus, mentioned first and
    # the main-clause subject; the old build-order tiebreak sank him
    # below the females). First-mention position drives priority AND
    # presentation; description count is only the secondary tiebreak.
    low_base = (base or "").lower()

    def _mention_pos(p):
        if not p:
            return 10 ** 6
        best = 10 ** 6
        for w in re.findall(r"[a-z]{3,}", str(p).lower()):
            i = low_base.find(w)
            if 0 <= i < best:
                best = i
        return best

    def _prio(j):
        s = subjects[j]
        p = s.get("persona")
        if p in typed_names:
            return (0, _mention_pos(p), -len(said.get(p, [])), j)
        if p:
            return (1, _mention_pos(p), 0, j)   # checkbox-assigned persona
        return (2, 0, 0, j)                      # unnamed: build order
    order2 = sorted(range(len(subjects)), key=_prio)
    for rank, j in enumerate(order2):
        s = subjects[j]
        if all_sec or rank >= n_main:
            s["tier"] = "secondary"
            for k in ("slots", "body_parts", "locked_canon"):
                s[k] = [] if k != "locked_canon" else {}
            s["action_plan"] = None
            s["flavor"] = []
            if rank < n_main:      # all-secondary keeps personas off
                s["persona"] = None
        else:
            s["tier"] = "main"
    # PRESENTATION FOLLOWS PRIORITY, not build order (the author's: Cloud
    # first). order2 is already mains-then-secondaries in mention order,
    # so keep it as-is instead of re-sorting back to females-first.
    keep = [subjects[j] for j in order2[:n_main + n_sec]]
    dropped = len(subjects) - len(keep)
    subjects = keep
    tier_meta = {"n_main": 0 if all_sec else n_main,
                 "n_secondary": (len(subjects) if all_sec
                                 else len(subjects) - (0 if all_sec
                                                       else n_main)),
                 "collective": dropped if (collective or dropped) else 0,
                 "background": background,
                 "all_secondary": all_sec,
                 "capped_detail": capped_detail}
    return subjects, tier_meta


# LOCATION + SETTING/GENRE resolution (the author's, 2026-08-29).
# LOCATION: typed pool name > checkbox roll FROM THE CURATED POOL (280
# entries -- where a good pool exists, the dice roll the pool; the LLM
# invents only for novel typed places, which then go the learning way) >
# free (describe only what the text gives). GENRE: typed > checkbox roll
# from genre_pool > none (contemporary implied, not emitted). GENRE GATES
# ARE SOFT -- an astronaut in a fantasy world is legitimate. NO time-of-day
# roll here: subjects can be in the office at night (day/night/weather
# belong to the lighting/atmosphere section).
_LOC_POOL = None
_GENRES = None


_GENRE_LOCS = {"mtime": 0, "data": {}}


def _genre_locations(leaf):
    """-> {"roll": [...], "typed": [...]} for a genre leaf from the agreed
    GENRE x LOCATION hierarchy (data/library/genre_locations.json, written
    by tools/build/write_genre_locations.py), or None when the leaf has no
    entry yet -- then the measured-affinity roll over the whole pool
    stands, as before."""
    p = _paths.data("genre_locations.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _GENRE_LOCS["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _GENRE_LOCS["data"] = json.load(f).get("leaves") or {}
            _GENRE_LOCS["mtime"] = mt
    except Exception:
        return None
    return _GENRE_LOCS["data"].get(str(leaf or "").lower())


def _location_pool():
    global _LOC_POOL
    if _LOC_POOL is None:
        try:
            with open(_paths.data("location_pool.json"),
                      encoding="utf-8-sig") as f:
                _LOC_POOL = json.load(f)
        except Exception:
            _LOC_POOL = {}
        # THE RULINGS ARE HONOURED AT LOAD (2026-09-16): a name ruled generic
        # ('water', 'home') or discarded is no place of the pool, whatever the
        # pool file still carries between rebuilds -- 'a girl taking a bath'
        # drew the place 'water' from the activity's measured places
        try:
            _lr = _json_table(_RULINGS_TB, "location_rulings.json") or {}
            for _n in list(_LOC_POOL):
                if _n in (_lr.get("generic") or {}) or _n in (_lr.get("discard") or []):
                    _LOC_POOL.pop(_n, None)
        except Exception:
            pass
    return _LOC_POOL


_GENRE_BUCKETS = None


def _genre_pool():
    global _GENRES, _GENRE_BUCKETS
    if _GENRES is None:
        try:
            with open(_paths.data("genre_pool.json"),
                      encoding="utf-8-sig") as f:
                _blob = json.load(f)
            _GENRES = _blob["genres"]
            _GENRE_BUCKETS = _blob.get("buckets") or {}
        except Exception:
            _GENRES, _GENRE_BUCKETS = {}, {}
    return _GENRES


def _genre_buckets():
    """the two-level pool's global genres (the author's rework 2026-09-03):
    {bucket: {"hint", "rollable"}}; leaves carry their bucket name."""
    _genre_pool()
    return _GENRE_BUCKETS or {}


def _genre_anchors(leaf):
    """-> the booru tags a leaf emits ('ancient egypt' -> ['ancient
    egyptian']); a plain leaf emits its own name."""
    rec = _genre_pool().get(leaf) or {}
    return [str(t) for t in (rec.get("anchors") or [leaf]) if t]


def genre_menu():
    """-> [{"leaf", "bucket"}] for the studio's dropdown, by bucket"""
    out = []
    for g, e in sorted(_genre_pool().items(), key=lambda kv: ((kv[1].get("bucket") or kv[0]), kv[0])):
        if str(g).startswith("_"):
            continue
        out.append({"leaf": g, "bucket": e.get("bucket") or g})
    return out


def _genre_rollable(leaf):
    rec = _genre_pool().get(leaf) or {}
    if rec.get("rollable") is False:
        return False
    bk = _genre_buckets().get(rec.get("bucket") or "") or {}
    return bk.get("rollable", True) is not False


_ARTIFICIAL_BG_RE = re.compile(
    r"\b(simple|white|black|grey|gray|blue|red|pink|green|yellow|purple|"
    r"brown|orange|aqua|two-tone|gradient|abstract|pattern(ed)?|checkered|"
    r"polka.?dot|striped|vertical-striped|diagonal-striped|halftone|grid|"
    r"heart|argyle|sparkle|plaid|sunburst|dotted|colorful|floral|photo|"
    r"star symbol|text|3d|screenshot|game screenshot|paw print|"
    r"greyscale with colored|monochrome|blank|plain|studio)\s+background\b", re.I)


def _backdrop_vocab():
    """{tag: posts (thousands)} from the hierarchy file."""
    p = _paths.data("genre_locations.json")
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f).get("backdrop_vocab") or {}
    except Exception:
        return {}


def _roll_backdrop(leaf, rng):
    """-> a backdrop TAG for this genre leaf, from its agreed roll list,
    log-weighted by posts; None when the leaf lists none (the model then
    invents one, as before)."""
    h = _genre_locations(leaf) if leaf else None
    names = list(((h or {}).get("backdrops") or {}).get("roll") or [])
    if not names:
        return None
    import math as _math
    vocab = _backdrop_vocab()
    w = [_math.log(max(2.0, float(vocab.get(n) or 1))) for n in names]
    return rng.choices(names, weights=w)[0]


_INDOOR_KINDS = ("indoor", "building")
_OUTDOOR_KINDS = ("outdoor-natural", "outdoor-manmade", "building")


def place_lean(base, banks, min_edge=0.06, ratio=2.0):
    """-> 'indoor' | 'outdoor' | None -- where the prompt already puts us.

    Measured from the tag co-occurrence network rather than a keyword list,
    so it covers whatever the user types: campfire and snow pull outdoors,
    bed and sofa pull indoors, knight and sword pull neither. 'building'
    counts for both because a castle is a place you can be inside or in
    front of.
    """
    net = (banks or {}).get("_net")
    if not net or not (base or "").strip():
        return None
    try:
        tags = pe.parse_input(base, banks)[0] or []
    except Exception:
        return None
    out = ind = 0.0
    for t in tags:
        try:
            out += net.edge(str(t).lower(), "outdoors")
            ind += net.edge(str(t).lower(), "indoors")
        except Exception:
            continue
    if max(out, ind) < min_edge:
        return None                      # nothing in the prompt says
    if out > ind * ratio:
        return "outdoor"
    if ind > out * ratio:
        return "indoor"
    return None


def resolve_location(base, opts, rng, typed=()):
    """-> (mode, name, kind). AN ARTIFICIAL BACKGROUND IS A SPECIAL TYPE OF
    LOCATION (the author's): '1girl, simple white background' draws no place --
    the backdrop IS the location. Typed artificial backdrops win; the
    gen-location roll yields an artificial backdrop 25% of the time
    (danbooru reality: ~29%% of all posts are simple background)."""
    low = " " + re.sub(r"[^a-z0-9() ]", " ", (base or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    m = _ARTIFICIAL_BG_RE.search(base or "")
    if m:
        return "backdrop-typed", m.group(0).lower(), "artificial"
    pool = _location_pool()
    # A RULED ALIAS IS THE POOL PLACE (2026-09-15, location_rulings.json):
    # 'tomb' -> graveyard, 'salon' -> saloon, 'gym' -> fitness gym, 'motel'
    # -> love hotel -- the discarded or renamed word in the user's text
    # names the place that stands for it
    try:
        _ral = (_json_table(_RULINGS_TB, "location_rulings.json") or {}).get("alias") or {}
    except Exception:
        _ral = {}
    if _ral:
        _low_r = " " + re.sub(r"[^a-z0-9 ]+", " ", (base or "").lower()) + " "
        _low_r = re.sub(r"\s+", " ", _low_r)
        for _w, _to in sorted(_ral.items(), key=lambda kv: -len(kv[0])):
            if " " + _w + " " in _low_r and _to in pool:
                return "typed-alias", _to, pool[_to].get("kind")
    # ... and the words that name a place ONLY after a place preposition
    # (2026-09-21): 'at a ball' is the ballroom, 'playing with a ball' is the
    # toy. English decides which reading it is; the ruling file lists the pairs.
    if _ral is not None:
        _rp = (_json_table(_RULINGS_TB, "location_rulings.json") or {}).get("alias_after_preposition") or {}
        if _rp:
            _low_p = " " + re.sub(r"[^a-z0-9 ]+", " ", (base or "").lower()) + " "
            _low_p = re.sub(r"\s+", " ", _low_p)
            for _w, _to in sorted(_rp.items(), key=lambda kv: -len(kv[0])):
                if _to in pool and re.search(
                        r"\b(?:at|in|inside|during|attending)\s+"
                        r"(?:(?:the|a|an|this|that|his|her|their|my|our)\s+)?"
                        r"(?:[a-z]+\s+){0,2}" + re.escape(_w) + r"\b", _low_p):
                    if isinstance(opts, dict):
                        opts["_place_claimed"] = _w     # the word is the PLACE, not a thing
                    return "typed-alias", _to, pool[_to].get("kind")
    # A WORD INSIDE AN OCCUPATION NAMES THE PERSON, NOT THE PLACE: 'a tired
    # office lady on the train' put the scene in an office. The measured
    # occupation table is the one list of such phrases.
    _occupation_lean("")
    for _occ in list(_OCC_LEAN["data"]):
        if " " in _occ and " " + _occ + " " in low:
            low = low.replace(" " + _occ + " ", " ")
    # a pool name 'X style' answers to a typed 'X' too ('film noir' in a
    # style hint is the film noir style), 2026-09-11
    # ...read without the phrases the parser claimed ('field' inside 'depth
    # of field' is no place, 2026-09-15)
    low = _sans_claimed(low)
    hits = [n for n in pool if " " + n + " " in low
            or (n.endswith(" style") and " " + n[:-6] + " " in low)]
    if hits:
        name = max(hits, key=len)
        return "typed", name, pool[name].get("kind")
    # THE PARSER'S PLACE IS THE PLACE. The parse already recognised the
    # location ('on the train' -> `train interior`, 'summer festival'), and
    # the concept library's `location` flag is the one owner of "is this a
    # place" -- so a typed tag flagged location IS the scene, whether or not
    # this pool spells it the same way. Before, text-only matching against
    # the pool's own names rolled 'space' for a train ride and 'ocean' for
    # a summer festival while the typed place sat on the tag line.
    _typed_low = [str(t).lower() for t in (typed or ())]
    for t in _typed_low:
        if t in pool and not _names_people(t):
            return "typed", t, pool[t].get("kind")     # a crowd is a background of people
    for t in _typed_low:
        # THE LEAD FLAG IS THE CLASS (2026-09-21): 'under covers' is a POSE the
        # classifier also tagged a location, and membership alone made the
        # scene 'under covers'. The class a curated pool assigned leads the
        # list (apply_flag_overrides), so that is what is read.
        if (list(_gloss_of(t)[1] or [None]) or [None])[0] == "location"                 and not _is_humanlike(t) and not _names_people(t):
            return "typed", t, None            # a crowd names people, not a place
    # ...a SCENERY-flagged tag is a place too ('on a rock': the rock, not
    # a rolled closet), and a place-like LEFTOVER phrase the booru has no
    # tag for ('in a cluttered cottage', 'through an orchard', 'at a
    # forge') is the place in the user's own words (2026-09-12)
    for t in _typed_low:
        if (list(_gloss_of(t)[1] or [None]) or [None])[0] == "scenery"                 and not _is_humanlike(t) and not _names_people(t):
            return "typed", t, None
    # A GENERIC PLACE WORD IS THE TYPED PLACE (2026-09-15, location_rulings
    # 'generic'): 'a girl in a room' is indoors in a room, never a rolled
    # city; the word stands with its stated kind
    try:
        _gen = (_json_table(_RULINGS_TB, "location_rulings.json") or {}).get("generic") or {}
    except Exception:
        _gen = {}
    for _w, _kind in sorted(_gen.items(), key=lambda kv: -len(kv[0])):
        if " " + _w + " " in low and _w not in pool:
            # ... AND THE USER'S OWN PHRASE FOR IT IS THE PLACE (the author's
            # bug hunt, 2026-09-21: 'a man walking down a rainy street' said
            # 'street' AND 'rainy street'). A phrase that contains the generic
            # word is the same place in his words: it stands, the bare word
            # does not.
            for _ph9 in ((opts or {}).get("_place_phrases") or []):
                if _w in str(_ph9).lower().split():
                    (opts or {}).setdefault("_phrases_typed", []).append(_ph9)
                    return "typed-phrase", _ph9, _kind
            (opts or {}).setdefault("_phrases_typed", []).append(_w)
            return "typed-generic", _w, _kind
    for ph in ((opts or {}).get("_place_phrases") or []):
        # ... unless the phrase is really a THING the studio has measured a
        # place for (2026-09-21: 'two girls in bed together' kept the fragment
        # 'bed together' as the scene; a bed is furniture, and its posts say
        # bedroom). A phrase names the place only where no word of it is a
        # thing that lives somewhere.
        try:
            _thp9 = (_json_table(_THING_PLACES, "thing_places.json") or {}).get("things") or {}
            _b9, _f9 = None, 0.04
            for _w9 in str(ph).lower().split():
                for _p9, _s9 in ((_thp9.get(_w9) or {}).get("places") or {}).items():
                    if _p9 in pool and not pool[_p9].get("typed_only") and float(_s9) >= _f9:
                        _b9, _f9 = _p9, float(_s9)
            if _b9:
                return "typed-thing", _b9, pool[_b9].get("kind")
        except Exception:
            pass
        (opts or {}).setdefault("_phrases_typed", []).append(ph)
        return "typed-phrase", ph, None
    # A TYPED OCCUPATION IMPLIES ITS PLACE FIRST (an office lady is in an
    # office), THEN A TYPED OBJECT (a bed is in a bedroom): P(place | tag)
    # from the harvests, the best pool place at .05 or more (2026-09-12)
    _tb_rel = _json_table(_ACT_REL, "place_related.json") or {}
    # A TYPED ACTIVITY DRAWS ITS MEASURED PLACE (2026-09-15: 'an old man
    # fishing at dawn' rolled a hallway and a bedroom): the registry holds
    # every activity's places (fishing: ocean .11, river .06, forest .04,
    # beach .04), read at the occupation floor
    try:
        _reg9 = (_activity_table() or {}).get("registry") or {}
        _best9, _best_f9 = None, 0.08
        for t in _typed_low:
            for p_, f_ in ((_reg9.get(t) or {}).get("places") or {}).items():
                if p_ in pool and not pool[p_].get("typed_only") and float(f_) >= _best_f9:
                    _best9, _best_f9 = p_, float(f_)
        if _best9:
            return "typed-activity", _best9, pool[_best9].get("kind")
    except Exception:
        pass
    # A TYPED THING LIVES SOMEWHERE (the author's bug hunt, 2026-09-21: 'two
    # girls in bed together' had no bed and no bedroom -- a bed is furniture,
    # not a place, so the scene fell through to an artificial backdrop).
    # P(place | thing) measured on the thing's own posts
    # (tools/build/harvest_thing_places.py), the same question the creatures
    # and the held objects already answer.
    try:
        _thp = (_json_table(_THING_PLACES, "thing_places.json") or {}).get("things") or {}
        # the booru rarely tags the room at all, so even a strong tie reads low:
        # a bed is in a bedroom on .087 of its posts, 'under covers' on .043
        _bt, _bf = None, 0.04
        for t in _typed_low:
            for p_, f_ in ((_thp.get(t) or {}).get("places") or {}).items():
                if p_ in pool and not pool[p_].get("typed_only") and float(f_) >= _bf:
                    _bt, _bf = p_, float(f_)
        if _bt:
            return "typed-thing", _bt, pool[_bt].get("kind")
    except Exception:
        pass
    for _key, _mode, _min in (("occupations", "typed-occupation", 0.08), ("objects", "typed-object", 0.05)):
        _rel = _tb_rel.get(_key) or {}
        _best, _best_f = None, _min          # a witch's .068 'moon' is a sky, not her place
        for t in _typed_low:
            for p_, f_ in (_rel.get(t) or {}).items():
                if p_ in pool and not pool[p_].get("typed_only") and float(f_) >= _best_f:
                    _best, _best_f = p_, float(f_)
        if _best:
            return _mode, _best, pool[_best].get("kind")
    # A LOCATION-SHAPED GENRE LEAF IS THE PLACE (the author's home / leisure /
    # work buckets): a rolled 'onsen' or 'bedroom' sets the scene there
    # instead of rolling a second, unrelated place beside it. Only when
    # the anchor is a place this pool knows; else the genre just colours.
    _g = (opts or {}).get("_genre") or (None, None)
    if _g[1] and _genre_pool().get(_g[1], {}).get("location_shaped"):
        for _anc in _genre_anchors(_g[1]):
            if _anc in pool:
                opts["_genre_anchor"] = _anc
                return "genre", _anc, pool[_anc].get("kind")
    # a non-figure SCENE TYPE is a place by definition: it always rolls a
    # location, constrained to its own kinds, and never a backdrop
    kinds = opts.get("_scene_kinds")
    # TYPED WEATHER IS OUTDOORS (2026-09-12): 'in the rain' rolls no
    # ballroom -- the rolled place keeps an outdoor kind
    # A TYPED SKY HOUR IS OUTDOORS TOO (2026-09-15: 'at dawn' in a bedroom):
    # dawn, sunrise, sunset, dusk and twilight are the sky's, not a room's
    if (_WEATHER_RE.search(base or "") or _SKY_TIME_RE.search(base or "")) and pool:
        _out_kinds = {pool[n].get("kind") for n in pool if str(pool[n].get("kind") or "").startswith("outdoor")}
        kinds = [k for k in (kinds or _out_kinds) if str(k).startswith("outdoor")] or sorted(_out_kinds)
    if kinds and pool:
        fit = sorted(n for n in pool if pool[n].get("kind") in kinds)
        if fit:
            name = rng.choice(fit)
            return "rolled", name, pool[name].get("kind")
    # ALWAYS RESOLVED (the author's rework): no location checkbox. An
    # artificial backdrop is still a legitimate outcome a quarter of the
    # time -- danbooru is ~29% simple background -- and the place itself is
    # drawn weighted by the genre, so a fantasy scene leans castle/forest
    # and a school scene leans classroom without either being forced.
    if pool:
        _g = (opts or {}).get("_genre") or (None, None)
        # AN EVENT PREFERS ITS PLACES (events.json, place_override share):
        # a christmas lands in a living room or a lit street, a hanami in
        # a park; the rest of the time the genre's own roll stands
        _ev = (opts or {}).get("_event") or (None, None, None)
        if _ev[2] and _ev[2].get("places"):
            _tb = _events_table() or {}
            if rng.random() < float(_tb.get("place_override", 0.7)):
                _hier = _genre_locations(_g[1]) if _g[1] else None
                _roll = set((_hier or {}).get("roll") or ())
                _evp = [n for n in _ev[2]["places"] if n in pool]
                _pref = [n for n in _evp if n in _roll] or _evp
                if _pref:
                    _pick = rng.choice(_pref)
                    return "event", _pick, pool[_pick].get("kind")
        _bh = ((_genre_locations(_g[1]) or {}).get("backdrops") or {}) if _g[1] else {}
        _share = float(_bh.get("share", 0.25)) if _bh else 0.25
        if rng.random() < _share:
            # A BACKDROP IS A TAG (the author's: the location class the first
            # pass forgot). The leaf's agreed list decides which; with no
            # list the model still invents one.
            return "backdrop-rolled", _roll_backdrop(_g[1], rng), "artificial"
        names = sorted(pool)
        genre = (opts or {}).get("_genre") or (None, None)
        # THE HIERARCHY DECIDES WHAT MAY ROLL (the author's, 2026-09-03): for a
        # leaf with an agreed entry the dice see only its roll list; every
        # other place in the pool is typed-only for that leaf. The measured
        # affinity still orders the draw inside the list.
        _hier = _genre_locations(genre[1]) if genre[1] else None
        if _hier and _hier.get("roll"):
            _allowed = [n for n in names if n in set(_hier["roll"])]
            if _allowed:
                names = _allowed
        # TWO-STAGE, because a multiplier is invisible across 280 places.
        # Weighting alone moved 'castle' for a fantasy prompt from 0.36% to
        # 0.7% -- technically a nudge, practically nothing, and the author's
        # asked locations to "try to match" the genre (stronger language
        # than the "minimal" he wanted for style).
        #
        # So most of the time the draw happens among the places this genre
        # is actually measured with, weighted by that affinity; the rest of
        # the time it is uniform over the whole pool. Every location stays
        # reachable, and an unmeasured genre falls through to uniform.
        # a rolled place must not contradict what the prompt already
        # implies (see place_lean); typed places are never touched
        lean = (opts or {}).get("_place_lean")
        if lean:
            ok_kinds = _INDOOR_KINDS if lean == "indoor" else _OUTDOOR_KINDS
            kept = [n for n in names if pool[n].get("kind") in ok_kinds]
            if kept:
                names = kept
        # AN EXTERNAL GENRE CARRIES ITS OWN LOCATION LIST. There is no
        # measured genre_location row for a non-booru genre -- related_tag
        # cannot see it -- so its reasoned `locations` field stands in,
        # entering at exactly the same point and the same 65% share as a
        # measured genre. That is what makes the reasoning pass worth
        # doing: the field is consumed, not merely stored.
        # UNMEASURED IS TYPED-ONLY, for places too (2026-09-14): 22 pool
        # names carry no booru count ('gym' is not a danbooru tag; 'tomb'
        # has 45 posts under the count file's floor) -- they stay typed
        # places and never roll
        # UNMEASURED OR SMALL IS TYPED-ONLY (the author's 2026-09-11/15): a place
        # under the rulings' floor (100 posts) never rolls unless the ruling
        # binds it to genres; a genre-bound place (slums: dark fantasy,
        # horror) rolls inside those leaves only
        try:
            _rulf = (_json_table(_RULINGS_TB, "location_rulings.json") or {})
            _floor_p = int(_rulf.get("floor") or 100)
        except Exception:
            _floor_p = 100
        _leaf_now = ((opts or {}).get("_genre") or (None, None))[1]
        names = [n for n in names
                 if not (pool[n] or {}).get("typed_only")
                 and ((pool[n] or {}).get("genres") and _leaf_now in (pool[n] or {}).get("genres")
                      or (not (pool[n] or {}).get("genres") and float((pool[n] or {}).get("posts") or 0) >= _floor_p))]
        _xg = (opts or {}).get("_external_genre") or {}
        _xlocs = [n for n in (_xg.get("locations") or []) if n in pool]
        if _xlocs:
            fit, w = _xlocs, [1.0] * len(_xlocs)
        else:
            fit = ([] if not genre[1] else
                   [n for n in names
                    if _affinity.score("genre_location", genre[1], n) > 0])
            # DAMPED: the raw score let 'dungeon' (fantasy 0.25 against
            # 0.036 for the next place) take 39% of fantasy rolls -- one
            # place owning a genre, the same staleness the genre roll had.
            # A square root keeps the measured order and caps the spread.
            w = [_affinity.score("genre_location", genre[1], n) ** 0.5 for n in fit]
        # THE AFFINITY SHARE FOLLOWS ITS COVERAGE: a thin measured row (dark
        # fantasy: hallway, nature, forest once the typed-only places are
        # out) took the full 65% and three places owned the leaf. Fewer than
        # eight measured places shrink the share in proportion; the rest of
        # the draws are uniform over the leaf's roll list.
        # a place the rulings bind to this leaf joins the fit list at the
        # median fit weight (slums in dark fantasy / horror, 2026-09-15)
        _bound = [n for n in names if _leaf_now and _leaf_now in ((pool[n] or {}).get("genres") or [])]
        if _bound:
            _medw = sorted(w)[len(w) // 2] if w else 1.0
            for n in _bound:
                if n not in fit:
                    fit.append(n)
                    w.append(_medw)
        _aff_share = 0.65 * min(1.0, len(fit) / 8.0) if fit else 0.0
        if fit and rng.random() < _aff_share:
            name = rng.choices(fit, weights=w)[0]
        else:
            name = rng.choice(names)
        return "rolled", name, pool[name].get("kind")
    return "free", None, None


# CAMERA / COMPOSITION / FRAMING (construction order: PLANS EARLIEST -- the
# framing gates which body parts even need describing; a portrait needs no
# feet). Small closed vocabulary -> the ENGINE rolls it, weighted by the
# measured corpus distribution (full body 1.31M > upper body 1.18M > cowboy
# shot 844k...). Typed camera words are honoured first via slot_model's own
# probes. The framing suppresses out-of-frame body parts and attire slots
# HARD; viewpoint stays soft (from behind + looking back still shows a face).
_FRAMINGS = [("full body", 1309075), ("upper body", 1181838),
             ("cowboy shot", 844210), ("portrait", 142646),
             ("close-up", 67229), ("wide shot", 23606)]
_VIEWPOINTS = [("from behind", 5), ("from above", 4), ("from below", 3),
               ("from side", 3), ("dutch angle", 2), ("pov", 3)]
_FRAME_SUPPRESS = {
    "close-up":  {"body": {"breasts", "ass", "pussy", "penis", "pubic hair",
                           "legs and feet"},
                  "slots": {"legwear", "outfit"}},
    "portrait":  {"body": {"breasts", "ass", "pussy", "penis", "pubic hair",
                           "legs and feet"},
                  "slots": {"legwear"}},
    "upper body": {"body": {"ass", "pussy", "penis", "pubic hair",
                            "legs and feet"},
                   "slots": {"legwear"}},
    "cowboy shot": {"body": {"legs and feet"}, "slots": set()},
    "full body": {"body": set(), "slots": set()},
    "wide shot": {"body": set(), "slots": set()},
}


# FOCUS: the camera's THIRD axis (framing = how much, viewpoint = from
# where, focus = what the eye lands on), and it reaches into every sphere:
# a body-part focus FORCES that part's description past the framing gates
# and the chance rolls (ass focus means the ass gets described, whatever
# the dice said); solo/gender focus steers subject attention; object focus
# elevates a prop to the star. Typed focus always honoured; rolled focus is
# small, and only for a part already being described.
_PART_FOCUS = {"ass": "ass focus", "breasts": "breast focus",
               "legs and feet": "foot focus", "eyes": "eye focus",
               "hands": "hand focus", "face": "portrait"}
_FOCUS_TAGS = {"ass focus", "breast focus", "foot focus", "eye focus",
               "hand focus", "hair focus", "back focus", "armpit focus",
               "crotch focus", "male focus", "female focus", "solo focus",
               "object focus", "clothes focus", "food focus",
               "reflection focus"}


def resolve_focus(base):
    low = (base or "").lower()
    return [f for f in _FOCUS_TAGS if f in low]


# WHOSE ATTENTION IS NOT A PART OF THE BODY (the author, 2026-09-21: "going
# 'male focus' in anal scene with female as main subject would be awkward at
# best"). 'solo focus', 'male focus' and 'female focus' say which BODY the
# picture is about -- the cast owns that, not the focus draw -- and the booru's
# act rows are thick with them ('footjob' carries solo focus on .48 of its
# posts). They stay measured in the table and are never drawn from it.
_CAST_FOCUS = {"solo focus", "male focus", "female focus"}
_PART_FOCUS_TAGS = _FOCUS_TAGS - _CAST_FOCUS


# THE FOCUS MUST BE A PART THE FOCUSED SUBJECT CAN HAVE (the author,
# 2026-09-21: "if you already decided the main subject (the who is focused)
# then just exclude the opposite gender's body parts as a focus target (for
# example in paizuri the main focus is female so even if the penis is present
# it can be excluded as the focus target because females can't have penis)").
# Good advice: it needs no measurement at all, and it catches what a measured
# row cannot -- the row counts the PICTURE's tags, both bodies together.
_FOCUS_PART_NOT = {"female": {"penis"}, "male": {"pussy", "breasts"}, "futanari": set(), "other": set()}


def focused_kind(cast):
    """-> whose picture this is, by his own pairing rules: a woman if one is
    there, else a futanari, else a man"""
    c = cast or {}
    if int(c.get("female") or 0):
        return "female"
    if int(c.get("futa") or 0) or int(c.get("futanari") or 0):
        return "futanari"
    if int(c.get("male") or 0):
        return "male"
    return None


def focus_part_ok(part, kind):
    """-> False when the focused subject cannot own that part"""
    if not part or not kind:
        return True
    return str(part).lower() not in _FOCUS_PART_NOT.get(str(kind).lower(), set())


def rolled_focus(subjects, framing, level, rng, act=None, view=None, cast=None):
    """a rolled body-part focus only lands on a part already described;
    with an act in the scene, the act's OWN focus (measured): the focus
    tags it lifts twice or more, at their summed share beside it.

    NOTHING IS BETTER THAN THE WRONG THING (the author, 2026-09-21: "not
    mentioning a framing / camera direction / focus is better than mentioning a
    wrong / contradictory one ... if the generator cant rule any of them
    decidedly it will fallback to no framing / camera direction / focus at all
    -- this way the image generator will decide on its own what is more
    common"). So a focus is said only where something in the picture MEASURES
    it: the act, or the framing it would sit in.

    ... AND THE ACT'S ANCHOR IS SUCH A MEASURE (the author, same day): where the
    act is done to a part, that part IS the focus of the picture and the booru's
    rare focus tags do not get a vote -- unless they measure against it."""
    if act:
        _an = act_anchor(act)
        _k9 = focused_kind(cast)
        if _an and not focus_part_ok(_an.get("anchor"), _k9):
            _an = None          # the anchor names the OTHER body's part, not hers
        if _an and _an.get("focus") and _an["focus"] in _FOCUS_TAGS:
            _lf = act_lift(act, _an["focus"], level, view=view)
            if _lf is None or _lf >= 0.5:
                if rng.random() < 0.8:          # sure, but not on every picture
                    return _an["focus"]
                return None
    if act:
        # the BAND's own row for this act, where the harvest has one
        _band = {f: act_camera_share(act, f, level) for f in _PART_FOCUS_TAGS}
        if any(v is not None for v in _band.values()):
            fc = [(f, v) for f, v in _band.items() if v]
            if fc and rng.random() < min(0.9, sum(w for _, w in fc)):
                return _wroll(rng, fc)
            return None                 # the act is measured and is about none of them
        fc = [(f, act_share(act, f, view)) for f in _FOCUS_TAGS
              if (act_lift(act, f, level, view=view) or 0.0) >= 2.0 and act_share(act, f, view) > 0]
        if fc and rng.random() < min(0.9, sum(w for _, w in fc)):
            return _wroll(rng, fc)
        # the act IS measured and lifts none of them: the picture says no focus
        # ('paizuri' no longer takes 'ass focus' from a blind draw)
        if any(act_lift(act, f, level, view=view) is not None for f in _FOCUS_TAGS):
            return None
    if rng.random() > (0.10 + 0.08 * sm.SPICE_ORDER.get(level, 1)):
        return None
    parts = set()
    for s in subjects:
        if s.get("tier") == "main":
            parts.update(p.split(" (")[0] for p in
                         (s.get("body_parts") or []))
    cands = [f for p, f in _PART_FOCUS.items()
             if p in parts and f != "portrait"]
    if framing in ("portrait", "close-up"):
        cands.append("eye focus")
    if not cands:
        return None
    # with no act, the FRAMING is what can measure it: a focus the booru shows
    # beside this framing less than chance is not drawn, and where nothing is
    # measured at all nothing is said
    if framing:
        lifted = [(f, pair_lift(framing, f)) for f in dict.fromkeys(cands)]
        known = [(f, lf) for f, lf in lifted if lf is not None]
        if not known:
            return None
        cands = [f for f, lf in known if lf >= 1.0]
        if not cands:
            return None
    return rng.choice(cands)


# LIGHTING / ATMOSPHERE (the author's brief, 2026-08-29). The booru wiki gives
# tags, not comprehension of HOW light is built -- and the LLM's lighting
# vocabulary (volumetric, rim light, practicals, motivated lighting...) is
# far wider than booru tags, ESPECIALLY valuable in anima NL. So the engine
# NEVER restrains the terms; it guides SANITY through a reasoning chain:
#   natural or artificial? -> if natural, what time of day? -> main source
#   (+ maybe secondary; they need not be enumerated in the prompt but they
#   determine the light's qualities) -> direction -> weather (outdoors) and
#   its effect -> QUALITY, soft vs hard (photography: a large/close/diffused
#   source is soft with gentle shadow gradients; a small/distant/bare source
#   is hard with crisp shadows and high contrast) -> effect on colour.
# The engine rolls the anchor dice (time, weather) because an unguided LLM
# mean-regresses to golden hour forever; typed words always win. Time of
# day is location-INDEPENDENT (the office at night is sacred).
_TIMES = [("day", 40), ("night", 25), ("sunset", 12), ("evening", 8),
          ("morning", 8), ("dawn", 4), ("dusk", 3)]
_WEATHERS = [("clear", 50), ("overcast", 14), ("rain", 10), ("cloudy", 10),
             ("snow", 6), ("fog", 5), ("storm", 3), ("wind", 2)]
_SKY_TIME_RE = re.compile(r"\b(?:dawn|sunrise|sunset|dusk|twilight|golden hour)\b", re.I)
_TIME_RE = re.compile(
    r"\b(night|midnight|sunset|sunrise|dawn|dusk|noon|midday|morning|"
    r"evening|golden hour|daytime|day)\b", re.I)
_WEATHER_RE = re.compile(
    r"\b(rain(y|ing)?|snow(y|ing)?|overcast|cloudy|fog(gy)?|mist(y)?|"
    r"storm(y)?|blizzard|drizzle|hail|clear sky|sunny|wind(y)?)\b", re.I)


_WEATHER_TABLE = {"mtime": 0, "data": None}


def _weather_table():
    """SEASON / WEATHER / TIME / LIGHTING lookups (data/library/
    weather_lighting.json, written by tools/build/write_weather.py)."""
    p = _paths.data("weather_lighting.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _WEATHER_TABLE["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _WEATHER_TABLE["data"] = json.load(f)
            _WEATHER_TABLE["mtime"] = mt
    except Exception:
        return None
    return _WEATHER_TABLE["data"]


def _wroll(rng, pairs, nudge=None):
    """weighted choice over [(name, w)] or {name: w}, nudged by {name: x}"""
    items = list(pairs.items()) if isinstance(pairs, dict) else list(pairs)
    items = [(n, float(w) * float((nudge or {}).get(n, 1.0))) for n, w in items]
    items = [(n, w) for n, w in items if w > 0]
    if not items:
        return None
    return rng.choices([n for n, _ in items], weights=[w for _, w in items], k=1)[0]


def _place_class(tb, loc_name, loc_kind):
    """-> (sky class, time class): sky class is one of nosky / windowed /
    unwindowed / outdoor-natural / outdoor-manmade / building / None;
    time class one of nightlife / workday / evening / None."""
    nosky = set((tb.get("climate_by_place") or {}).keys())
    nosky = {p for p in nosky if tb["climate_by_place"][p] == "nosky"}
    tcls = (tb.get("place_class") or {}).get(loc_name)
    if loc_name in nosky:
        return "nosky", tcls
    if loc_kind == "indoor":
        return ("windowed" if loc_name in set(tb.get("windowed") or ()) else "unwindowed"), tcls
    if loc_kind in ("outdoor-natural", "outdoor-manmade", "building"):
        return loc_kind, tcls
    # AN UNCLASSIFIED PLACE ASKS THE BOORU (the author, 2026-09-21: "the
    # generator is bugged: it places weather like fog or a night in indoor
    # locations still"). A place the location pool does not classify -- a gym, a
    # karaoke, an attic, anything the user typed -- used to fall through to the
    # open-air draw and take cloudy skies and snow indoors with it. Its own
    # posts say where it is; where even they say nothing, nothing is said about
    # the weather (better no word than the wrong one, his rule of the same day).
    if not loc_kind:
        _row = _place_row("places_all", loc_name) or {}
        _in, _out = float(_row.get("indoors") or 0.0), float(_row.get("outdoors") or 0.0)
        if _in or _out:
            return ("windowed" if _in >= _out else "outdoor-manmade"), tcls
        return "unknown", tcls
    return None, tcls


def place_side(name, kind, rng, lean=None):
    """A 'building' IS SHOWN FROM ONE SIDE IN ONE PICTURE (the author's live test,
    2026-10-03: a soapland with `building` beside it, day, sunlight and falling
    leaves). The pool's 'building' kind says a place can be seen from inside or
    from the street, and every later step read it as both at once: the open-air
    lights, a weather, a sky -- and the model, told 'location kind: building',
    wrote the word into the setting. The place's own posts say which side they
    show (places harvest: soapland indoors .24 / outdoors none, library .73 /
    none, cafe .48 / .13, shrine .03 / .51, rooftop none / .55), so the side is
    drawn in that proportion, once, and the rest of the prompt reads 'indoor' or
    'outdoor-manmade'. What the prompt's own words lean to decides first; a
    place with neither share measured stays the open class."""
    if kind != "building" or not name:
        return kind
    if lean == "indoor":
        return "indoor"
    if lean == "outdoor":
        return "outdoor-manmade"
    row = _place_row("places_all", name) or _place_row("places_all", place_parent(name) or "") or {}
    _in, _out = float(row.get("indoors") or 0.0), float(row.get("outdoors") or 0.0)
    if not (_in or _out):
        return kind
    return "indoor" if rng.random() < _in / (_in + _out) else "outdoor-manmade"


def _kind_said(kind):
    """the pool's kind as the model is told it: 'building' is a class of the
    pool, and a model that reads the bare word writes it into the setting"""
    return {"building": "a building, shown from inside or from outside -- the kind is not a thing in the picture",
            "outdoor-manmade": "outdoors, man-made", "outdoor-natural": "outdoors, natural",
            "indoor": "indoors"}.get(str(kind or ""), kind)


def _pairs(x):
    """[(name, weight)] from a dict, a list of pairs or a list of lists"""
    if isinstance(x, dict):
        return [(str(k), float(v)) for k, v in x.items()]
    return [(str(p[0]), float(p[1])) for p in (x or []) if len(p) >= 2]


def _place_reweight(cands, place, vocab, keep=frozenset(), strict=False):
    """-> the (name, weight) candidates reweighted by the place's measured
    shares (places harvest, 'places_all'): a candidate the place shows is
    weighted by its share there; one the place never shows is dropped
    (a measured row is complete); `keep` names stay at
    their own weight (the unsaid 'clear'). Unmeasured place: unchanged."""
    if not place:
        return cands
    row = _place_row("places_all", place)
    if not row or len(row) < 100:
        return cands                     # unmeasured place: the class table
    # a measured row is the place's whole top of related tags: a weather or
    # a time absent from it is absent from the place (a library shows no
    # fog, no dawn), not merely unmeasured
    # the kept name ('clear', the unsaid weather) is the complement: the
    # share of the place's pictures that show none of the vocabulary
    _rest = max(0.0, 1.0 - sum(float(row.get(t) or 0.0) for t in vocab if t not in keep))
    out = []
    for name, w in cands:
        if name in keep:
            out.append((name, _rest))
        elif name in row:
            out.append((name, float(row[name])))
    # strict: a measured place that shows none of the vocabulary has none of it
    # (a soapland's 500 related tags name no hour at all -- and 'day' was drawn
    # for it from the class table); the caller says nothing
    return out if (out or strict) else cands


_HOUR_FLOOR = 0.02          # a place's hour is on at least this share of its posts
_LOOK_YIELD_UNDER = 0.5    # a rolled style whose pairing with a typed look is under half of chance yields to it


def look_clash(rolled, looks, tagset=()):
    """-> (the typed look, its lift) when the rolled style or medium does not
    carry it, else None. Two booru tags are measured by their pair lift; anything
    else by how the artists' descriptions pair them; unmeasured is neutral.
    (Lifted out of generate() 2026-09-20 so the rule can be tested without the
    live counts -- a harvest saturating the booru made the test fail, not a
    change in the rule.)"""
    try:
        from promptstudio.library import looks as _lkm
    except Exception:
        _lkm = None
    for lk in looks or ():
        if lk in (tagset or ()) and _count_1girl(rolled, "1girl") and _count_1girl(lk, "1girl"):
            lf = pair_lift(rolled, lk)
        else:
            lf = _lkm.description_lift(rolled, lk) if _lkm else None
        if lf is not None and lf < _LOOK_YIELD_UNDER:
            return lk, lf
    return None


def resolve_lighting(base, loc_kind, rng, loc_name=None, genre=None, season=None, mood=None):
    """-> {season (never voiced), time, weather, lighting [tags], climate,
    place_class, sources}. the author's (2026-09-04): the season is rolled by
    the climate of the genre and the place, steered by the prompt's own
    words and never said; the weather follows the season within the
    climate; the time follows the place; lighting follows all three and
    the genre. Table-driven -- see tools/build/write_weather.py."""
    # an ARTIFICIAL BACKDROP has no sky, no weather, no time of day --
    # studio light only; anchors would smuggle a world onto a white void
    if loc_kind == "artificial":
        # ... BUT NOT ALWAYS THE SAME STUDIO (the author's live tests, 2026-09-19:
        # "bright studio lighting ... I meet them very often"). With no light given,
        # the model wrote 'bright studio lighting' on every plain background. The
        # pool's own sky-less lights are rolled for it instead -- a quality, and
        # half the time a direction -- so a white void is lit a different way each
        # time and the words are the pool's.
        _BACKDROP_LIGHTS = ["soft lighting", "rim lighting", "dramatic lighting", "dim lighting", "high contrast",
                            "chiaroscuro", "cinematic lighting", "ambient lighting", "warm lighting", "moody lighting",
                            "soft shadows", "hard shadows", "gentle shadows", "spotlight", "bloom", "colored shadow",
                            "drop shadow", "volumetric lighting"]
        _lp9 = _lighting_pool2() if "_lighting_pool2" in globals() else {}
        _q9 = [t for t in _BACKDROP_LIGHTS if not _lp9 or t in _lp9]
        _ml9 = set(mood_lighting(mood)) if mood else set()
        _lt9 = [_wroll(rng, [(t, 3.0 if t in _ml9 else 1.0) for t in _q9])] if _q9 else []
        if rng.random() < 0.5:
            _lt9.append(rng.choice(["backlighting", "sidelighting", "underlighting", "overlighting"]))
        return {"time": None, "time_src": "none", "weather": None,
                "weather_src": "none", "artificial": True, "season": None,
                "lighting": _lt9}
    tb = _weather_table() or {}
    low = " " + re.sub(r"[^a-z0-9' ]+", " ", (base or "").lower()) + " "
    bucket = (_genre_pool().get(genre) or {}).get("bucket") if genre else None
    nudge = (tb.get("genre_nudge") or {}).get(genre) or {}
    sky, tcls = _place_class(tb, loc_name, loc_kind)
    # CLIMATE: the genre's, unless the place has no sky; else the place's;
    # else temperate
    climate = None
    if sky == "nosky":
        climate = "nosky"
    elif genre and (tb.get("genre_climate") or {}).get(genre):
        climate = tb["genre_climate"][genre]
    elif loc_name and (tb.get("climate_by_place") or {}).get(loc_name):
        climate = tb["climate_by_place"][loc_name]
    climate = climate or "temperate"
    # SEASON: the prompt's words first, then a pinned one (an event), then
    # the climate's weights
    pinned = season
    season, s_src = None, "rolled"
    for sn, words in (tb.get("season_words") or {}).items():
        if any((" " + w + " ") in low for w in words):
            season, s_src = sn, "typed"
            break
    if season is None and pinned:
        season, s_src = pinned, "event"
    if season is None:
        season = _wroll(rng, (tb.get("seasons") or {}).get(climate)
                        or (tb.get("seasons") or {}).get("temperate") or {"summer": 1},
                        nudge.get("season"))
    # TIME: typed, else by the place's class, none where there is no sky
    m = _TIME_RE.search(base or "")
    t_src = "typed" if m else "rolled"
    if m:
        tod = m.group(1).lower()
    elif sky == "nosky":
        tod, t_src = None, "none"
    else:
        # THE PLACE'S OWN HOURS (the author's 2026-09-15: "fog and dawn tags
        # appear indoors"): the places harvest measured every place's
        # related tags, times included (a library: day .034, night .013,
        # nothing else), so the draw is reweighted by the place's shares and
        # a time the place never shows is not drawn there. The class table
        # stands where the place is unmeasured.
        _cands = _pairs((tb.get("time_by_class") or {}).get(tcls) or tb.get("time_default") or _TIMES)
        _cands = _place_reweight(_cands, loc_name, {"dawn", "morning", "day", "noon", "afternoon", "evening",
                                                   "dusk", "sunset", "night", "midnight", "twilight"},
                                 strict=True)
        # ... and an hour on fewer than one post in fifty of a measured place is not
        # the place's hour (a soapland: day .014 of 418 posts, its only one, was
        # drawn every time)
        _row9 = _place_row("places_all", loc_name) if loc_name else None
        if _row9 and len(_row9) >= 100:
            _cands = [(n9, w9) for n9, w9 in _cands if float(_row9.get(n9) or 0.0) >= _HOUR_FLOOR]
        tod = _wroll(rng, _cands, nudge.get("time")) if _cands else None
        if tod is None:
            t_src = "none"
    tod = {"midnight": "night", "daytime": "day", "midday": "noon",
           "sunrise": "dawn", "golden hour": "sunset"}.get(tod, tod)
    # WEATHER: typed, else by season within the climate; unseen indoors
    # unless the room has a window (and then only sometimes)
    w = _WEATHER_RE.search(base or "")
    w_src = "typed" if w else "rolled"
    weather = None
    if w:
        weather = w.group(0).lower()
        weather = {"rainy": "rain", "raining": "rain", "snowy": "snow", "snowing": "snow",
                   "foggy": "fog", "misty": "fog", "mist": "fog", "stormy": "storm",
                   "windy": "wind", "cloudy": "cloudy sky", "sunny": None,
                   "clear sky": None, "drizzle": "rain", "hail": "storm"}.get(weather, weather)
    elif sky in ("nosky", "unwindowed", "unknown"):
        weather, w_src = None, "none"
    elif sky in ("windowed", "building") and rng.random() >= 0.35:
        weather, w_src = None, "none"   # seen through a window, sometimes
    else:
        table = ((tb.get("weather") or {}).get(climate) or {}).get(season) \
            or ((tb.get("weather") or {}).get("temperate") or {}).get(season) or []
        # the same for the weather: fog where the place shows fog (a forest
        # .017), none where it shows none (a library)
        table = _place_reweight(_pairs(table), loc_name, set((tb.get("weather_tag") or {}).keys()) | {"clear"},
                                keep={"clear"})
        weather = _wroll(rng, table, nudge.get("weather"))
        if weather == "clear":
            weather = None            # unremarkable weather goes unsaid
    weather_tag = (tb.get("weather_tag") or {}).get(weather, weather) if weather else None
    # LIGHTING: every matching rule offers its tags; one or two are drawn
    offers = []
    for rule in (tb.get("lighting") or []):
        def _ok(key, val):
            want = rule.get(key)
            return want is None or val in want
        if not (_ok("time", tod) and _ok("weather", weather)
                and _ok("bucket", bucket) and _ok("genre", genre)):
            continue
        want_pl = rule.get("place")
        if want_pl is not None and sky not in want_pl \
                and not ("evening-class" in want_pl and tcls == "evening") \
                and not ("nightlife-class" in want_pl and tcls == "nightlife"):
            continue
        offers += [tuple(x) for x in rule.get("tags") or []]
    lighting = []
    # ONE OWNER (2026-09-06): a tag the gloss flags as a VIEW (silhouette)
    # is the camera's, not the light's -- it made a featureless figure of
    # a subject in a face act; and a light tag rolls only with a measured
    # floor, like every other rolled tag
    offers = [(t, w) for t, w in offers
              if "view" not in _gloss_flags(t) and _floor_measured(t)]
    # THE LIGHT FOLLOWS THE MOOD (the author, 2026-09-20: "lighting I think on the
    # contrary is dependant on the tone (as well as on location / weather)"): the
    # place, the time and the weather say what light is POSSIBLE; among those, the
    # mood's own lights are three times as likely
    _ml = set(mood_lighting(mood)) if mood else set()
    if _ml:
        offers = [(t, w * (3.0 if t in _ml else 1.0)) for t, w in offers]
    if offers:
        n = 2 if rng.random() < 0.4 else 1
        for _ in range(n):
            pick = _wroll(rng, [(t, w) for t, w in offers if t not in lighting])
            if pick and (_vocab().get(pick, 0) >= 100):
                lighting.append(pick)
    return {"time": tod, "time_src": t_src, "weather": weather_tag, "weather_src": w_src,
            "season": season, "season_src": s_src, "climate": climate,
            "place_class": sky, "lighting": lighting, "offers": offers}


def relight(lt, rng, mood):
    """the same possible lights, re-picked with the mood's own three times as
    likely (the author, 2026-09-20: the light follows the mood, the place and
    the weather decide what light is possible)"""
    offers = list((lt or {}).get("offers") or [])
    ml = set(mood_lighting(mood)) if mood else set()
    if not (offers and ml):
        return lt
    out = []
    for _ in range(len(lt.get("lighting") or []) or 1):
        pick = _wroll(rng, [(t, w * (3.0 if t in ml else 1.0)) for t, w in offers if t not in out])
        if pick and (_vocab().get(pick, 0) >= 100):
            out.append(pick)
    if out:
        lt = dict(lt)
        lt["lighting"] = out
    return lt


def _lighting_brief(light, loc_kind, style_name, genre_name):
    if light.get("artificial"):
        return ("LIGHTING: an artificial backdrop, no sky. The light is: %s. Put those "
                "words in 'lighting' and describe THAT light (its quality, direction, mood); "
                "do not write 'studio lighting'. NO sky, NO weather, NO time of day, NO "
                "named sources." % (", ".join(light.get("lighting") or []) or "soft lighting"))
    facts = ["time of day: %s" % (light.get("time") or "unseen (no sky here)")]
    if light.get("weather"):
        facts.append("weather: %s" % light["weather"])
    if light.get("lighting"):
        facts.append("the light: %s" % ", ".join(light["lighting"]))
    if loc_kind:
        facts.append("location kind: %s" % _kind_said(loc_kind))
    return (
        "LIGHTING. Reason through this chain SILENTLY -- it is for your "
        "thinking, not for the output:\n"
        " Facts (fixed): %s.\n"
        " 1. Natural or artificial light (or mixed)? An interior needs a "
        "window for natural light; night favours artificial sources.\n"
        " 2. What is the main source, and maybe a secondary? Sources "
        "decide everything below.\n"
        " 3. What direction does the light come from?\n"
        " 4. Soft or hard? Large/close/diffused source = SOFT (gentle "
        "shadow edges, low contrast); small/distant/bare = HARD (crisp "
        "shadows, high contrast).\n"
        " 5. What does it do to colour (golden hour warms, neon tints, "
        "overcast desaturates)?\n"
        " THEN OUTPUT (2-4 descriptors in 'lighting', echoed in the "
        "prose): describe THE LIGHT ITSELF -- its quality, direction, "
        "colour, mood -- NEVER its source, unless that source should be "
        "DRAWN in the scene. An image generator paints what you name: "
        "'soft light from the overcast sky' in an office puts a sky in "
        "the office. Say 'a warm soft light from the left in a dark "
        "office' instead. USE YOUR OWN image-generation lighting "
        "vocabulary freely -- dramatic lighting, cinematic lighting, rim "
        "lighting, volumetric light, chiaroscuro, and any term you know "
        "beyond these; unmappable terms simply stay in the prose. Every "
        "descriptor must agree with your chain -- no soft dreamy haze "
        "under a bare noon sun.%s%s"
        % ("; ".join(facts),
           (" The style (%s) and its habits should show in the light."
            % style_name) if style_name else "",
           (" The genre (%s) suggests its own sources." % genre_name)
           if genre_name else ""))


def resolve_camera(base, cast, rng, act=None, level=None):
    """-> (framing, viewpoint_or_None, source). With an act in the scene
    the rolled framing and viewpoint follow its measured lifts."""
    # WHOLE WORDS. `bust` matched "robust" and `pov` matched "poverty",
    # and a spurious `pov` does not stop at the camera: it makes the
    # viewer a partner and changes the cast. Same class of fault as the
    # series-scope substring match.
    low = " " + re.sub(r"[^a-z0-9'()-]+", " ", (base or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    framing = next((t for t in sm.FRAMING_TAGS
                    if " " + t + " " in low), None)
    view = next((t for t in sorted(sm.VIEWPOINT_TAGS, key=len, reverse=True)
                 if " " + t + " " in low), None)
    # THE PARSER'S PHRASE READING IS THE SAME FACT. "a blowjob to the
    # viewer" yields the tag `pov` through VIEWPOINT_MAP, but this reader
    # only saw the literal word -- so the camera rolled its own view and
    # the prose could describe a POV act from the side. One owner: what
    # the parser read as a viewpoint or framing is typed here too.
    if not view or not framing:
        try:
            from promptstudio.engine import enhancer as _pe
            _pt = [str(t).lower() for t in (_pe.parse_input(base or "", _pe.load_all_banks())[0] or [])]
            if not view:
                view = next((t for t in _pt if t in sm.VIEWPOINT_TAGS), None)
            if not framing:
                framing = next((t for t in _pt if t in sm.FRAMING_TAGS), None)
        except Exception:
            pass
    source = "typed" if (framing or view) else "rolled"
    # THE VIEWPOINT FIRST, then the framing under it (the author's 2026-09-06:
    # through the viewer's eyes a blowjob is the face; from the side the
    # same act may be framed whole): the act's measured lifts, split by
    # pov, decide both
    # THE ACT'S ANCHOR AIMS THE CAMERA (the author, 2026-09-21): where the act
    # is done to a part, the framings and the directions of THAT part's own
    # pictures decide -- 'ass focus' is seen from behind on .56 of its posts,
    # 'foot focus' full body on .22, 'breast focus' upper body or close-up.
    # Dense data, unlike the focus tags beside the act itself. One picture in
    # six still says nothing, so a series does not become all close-ups.
    _anc = act_anchor(act) if act else None
    if _anc and (not view) and _anc.get("views"):
        _vv = [(v, w) for v, w in _anc["views"].items() if v in dict(_VIEWPOINTS)]
        if _vv:
            _vv.append(("none", max(0.15, 1.0 - sum(w for _v, w in _vv))))
            view = _wroll(rng, _vv)
            if view == "none":
                view = None
    if _anc and (not framing) and _anc.get("framings"):
        _ff = list(_anc["framings"].items())
        _ff.append(("none", max(0.15, 1.0 - sum(w for _f, w in _ff))))
        framing = _wroll(rng, _ff)
        if framing == "none":
            framing = None
    if not view:
        # MEASURED, WITH 'NONE' AMONG THE OPTIONS (the author, 2026-09-21: a
        # camera direction nothing decides is better left unsaid -- the image
        # model then does what is commonest). The share of pictures that carry
        # each viewpoint tag at this level, and the rest that carry none; the
        # act's lift steers it where the act is measured.
        vps = _viewpoint_shares(level)
        if cast.get("viewer"):
            vps = [(v, w * (4.0 if v == "pov" else 1.0)) for v, w in vps]
        _lifts = {v: (act_camera_lift(act, v, level) if act_camera_lift(act, v, level) is not None
                      else act_lift(act, v, level)) for v, _w in _VIEWPOINTS}
        vps = [(v, w * (_lifts.get(v) or 1.0)) for v, w in vps]
        # ... and where the ACT decides one (it shows it twice as often as the
        # level does), the camera is decided: 'none' leaves the pool
        if any((_lifts.get(v) or 0) >= 2.0 for v in _lifts):
            vps = [(v, w) for v, w in vps if v != "none"]
        view = _wroll(rng, vps)
        if view == "none":
            view = None
    if not framing:
        _fw = [(f, w * ((act_camera_lift(act, f, level) if act_camera_lift(act, f, level) is not None
                         else act_lift(act, f, level, view=view or "")) or 1.0))
               for f, w in _framing_shares(level)]
        framing = rng.choices([f for f, _ in _fw], weights=[w for _, w in _fw], k=1)[0]
        if framing == "none":
            framing = None
    # 'profile' (the author, 2026-09-19: "can be rolled (it's a face portrait from
    # side, why not)"): the side view of a face framing, half the time
    if source == "rolled" and view == "from side" and framing in ("portrait", "close-up", "upper body")             and rng.random() < 0.5:
        view = "profile"
    return framing, view, source


_VIEWPOINT_SHARES = {}


def _viewpoint_shares(level=None):
    """[(viewpoint, share)] at the level's universe, live counts cached, plus
    ('none', the rest) -- the measured chance that a picture names no camera
    direction at all (the author, 2026-09-21). The hand weights stand in when
    the counts are unavailable, with the same measured framed share."""
    u = _LEVEL_UNIVERSE.get(level or "", "1girl")
    if u in _VIEWPOINT_SHARES:
        return _VIEWPOINT_SHARES[u]
    out = []
    try:
        tot = float(_count_cached(u) or 0)
        tot1 = float(_count_cached("1girl") or 0)
        for v, _w in _VIEWPOINTS:
            n = _count_cached(u + " " + _q(v))
            if not (tot and n):
                n1 = _count_cached("1girl " + _q(v))
                n, tot_ = (n1, tot1) if (tot1 and n1) else (None, 0.0)
            else:
                tot_ = tot
            if tot_ and n:
                out.append((v, n / tot_))
    except Exception:
        out = []
    if len(out) < len(_VIEWPOINTS):
        _tot = float(sum(w for _v, w in _VIEWPOINTS))
        out = [(v, w / _tot * 0.20) for v, w in _VIEWPOINTS]
    out.append(("none", max(0.0, 1.0 - sum(sh for _v, sh in out))))
    _VIEWPOINT_SHARES[u] = out
    return out


_FRAMING_SHARES = {}


def _framing_shares(level=None):
    """[(framing, share)] under the level's universe, live counts cached,
    plus ('none', the rest): the measured chance that a picture carries no
    framing tag (2026-09-14). The hand counts stand when offline."""
    u = _LEVEL_UNIVERSE.get(level or "", "1girl")
    if u in _FRAMING_SHARES:
        return _FRAMING_SHARES[u]
    out = []
    try:
        tot = float(_count_cached(u) or 0)
        tot1 = float(_count_cached("1girl") or 0)
        for f, _w in _FRAMINGS:
            n = _count_cached(u + " " + _q(f))
            if not (tot and n):
                # the count endpoint times out on the biggest searches
                # ('rating:general full_body' answers null): the 1girl
                # universe's share stands in for that framing
                n1 = _count_cached("1girl " + _q(f))
                n, tot_ = (n1, tot1) if (tot1 and n1) else (None, 0.0)
            else:
                tot_ = tot
            if tot_ and n:
                out.append((f, n / tot_))
    except Exception:
        out = []
    if len(out) < len(_FRAMINGS):
        _tot = float(sum(w for _f, w in _FRAMINGS))
        out = [(f, w / _tot * 0.32) for f, w in _FRAMINGS]     # the measured framed share
    out.append(("none", max(0.0, 1.0 - sum(sh for _f, sh in out))))
    _FRAMING_SHARES[u] = out
    return out


# the camera's FOURTH axis: composition extras, measured (danbooru) --
# the author's: 'where is the camera work?' -- framing+viewpoint alone read
# flat. Rolled at 30%, one tag, count-weighted.
# 'dark background' rides here too (the author's): like blurry, a MODIFIER of a
# real place, never a backdrop of its own
_CAM_EXTRAS = [("blurry background", 184928), ("dark background", 16000),
               ("dutch angle", 162068),
               ("depth of field", 124936), ("foreshortening", 69171),
               ("blurry foreground", 41651)]


_DARK_LIGHT_RE = re.compile(
    r"(?<![a-z])(night|dark|dim|dimly|low[- ]key|moonlight|moonlit|candlelight|"
    r"midnight|dusk|evening|twilight|silhouette|backlighting|shadow|shadows|"
    r"nighttime|unlit|gloom|gloomy)(?![a-z])", re.I)


# 'perspective' (the author, 2026-09-19: "add it to the camera / framing
# vocabulary. it can be paired with a lot of framings / views - it just adds
# more depth distortion (especially for close framings - so it is better paired
# with those + some angle, like 'low angle' / 'view from below' + 'close-up' +
# 'perspective'"): an extra like foreshortening, at its own posts, lifted
# where the author says it works -- a close framing, an angled view, or both
_PERSPECTIVE_POSTS = 9769
_CLOSE_FRAMINGS = {"close-up", "portrait", "upper body", "cowboy shot"}
_ANGLED_VIEWS = {"from below", "from above", "dutch angle", "from side", "pov"}


def roll_camera_extra(rng, framing=None, viewpoint=None):
    if rng.random() < 0.30:
        _lift = (3.0 if str(framing or "") in _CLOSE_FRAMINGS else 1.0)             * (3.0 if str(viewpoint or "") in _ANGLED_VIEWS else 1.0)
        ex = list(_CAM_EXTRAS) + [("perspective", _PERSPECTIVE_POSTS * _lift)]
        return rng.choices([t for t, _ in ex], weights=[w for _, w in ex], k=1)[0]
    return None


_STYLE_POOL2 = None


def _style_pool2():
    global _STYLE_POOL2
    if _STYLE_POOL2 is None:
        _STYLE_POOL2 = {}
        for fname, key in (("style_pool.json", None),
                           ("cultural_styles.json", "pool")):
            try:
                with open(_paths.data(fname),
                          encoding="utf-8-sig") as f:
                    d = json.load(f)
                d = d.get(key) if key else d
                for n, v in (d or {}).items():
                    _STYLE_POOL2[n.lower()] = v if isinstance(v, dict) else {}
            except Exception:
                pass
    return _STYLE_POOL2


def resolve_style(base, opts=None, rng=None):
    """-> (name, record). Typed wins; otherwise one is DRAWN.

    the author's rework: the style checkbox is gone, so a prompt that names no
    style still gets one. The draw is weighted by the resolved genre, but
    only where the measured link is strong -- his instruction was that the
    "style dependency from genre should stay minimal (only strongly evident
    cases like the cyberpunk that you mentioned should really matter)".
    affinity.THRESHOLD['style_genre'] is what enforces that: cyberpunk ->
    science fiction (0.379) steers, cyberpunk -> fantasy (0.014) does not,
    and a style with no measurement keeps a neutral weight so it stays
    reachable.

    rng=None keeps the old typed-only behaviour for callers that just want
    to read a style out of text.
    """
    low = " " + re.sub(r"[^a-z0-9() ]", " ", (base or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    pool = _style_pool2()
    hits = [n for n in pool if " " + n + " " in low]
    # THE QUALIFIED FORM, AS THE TAG SCANNER ALREADY DOES. Booru writes
    # `western comics (style)` and `1980s (style)`; people type "western
    # comics style" and "1980s style". Matching pool names literally meant
    # NONE of the 13 qualified styles could ever be typed -- so asking for
    # western comics style resolved `afrofuturism` off the dice, and the
    # free medium roll (which only defers to a TYPED style) then added
    # `pixel art` on top. Three competing looks, and the pile-up I was
    # sent to fix was only the visible half of this.
    #
    # Only the "<name> style" spelling is accepted, never the bare stem:
    # `1980s (style)` must not be summoned by "a 1980s bar".
    for n in pool:
        if "(" not in str(n) or n in hits:
            continue
        plain = str(n).replace(" (style)", " style").replace(" (medium)",
                                                             " medium")
        if plain != n and " " + plain + " " in low:
            hits.append(n)
    # the style box's own style name (see generate): typed, and whole
    _hn = (opts or {}).get("_style_hint_name")
    if _hn and _hn in pool:
        hits.append(_hn)
    if hits:
        name = max(hits, key=len)
        if opts is not None:
            opts["_style_mode"] = "typed"
        return name, pool[name]
    if rng is None:
        return None, {}
    # A TYPED EXTERNAL LOOK IS A TYPED STYLE. The pool does not know Van
    # Gogh, so the roll went ahead and put `low poly` or `emo art` beside
    # him; the user's look is the style, and nothing is drawn against it.
    if _typed_external_look(base):
        if opts is not None:
            opts["_style_mode"] = "typed-external"
        return None, {}
    # A TYPED MEDIUM IS THE TYPED LOOK (2026-09-15, the coherence triangle):
    # 'watercolor painting style' reaches watercolor (medium) through its
    # alias; no style rolls beside a medium the user chose
    try:
        _pt0 = [str(t).lower() for t in (pe.parse_input(base or "", pe.load_all_banks())[0] or [])]
        if any(is_medium_tag(t) for t in _pt0):
            if opts is not None:
                opts["_style_mode"] = "typed-medium"
            return None, {}
    except Exception:
        pass
    if opts is not None:
        opts["_style_mode"] = "rolled"
    # A NICHE STYLE IS TYPED-ONLY (the author's 2026-09-14: afrofuturism, 21
    # posts, rolled): a tagged style rolls only from 1,000 posts (the body
    # writer's own floor); a pool-only name (no booru tag: 'pixar style')
    # is the user's curated look and stays rollable
    names = sorted(n for n in pool if not str(n).startswith("_")
                   and (float((pool[n] or {}).get("posts") or 0) >= 1000
                        or not (pool[n] or {}).get("posts")))
    genre = (opts or {}).get("_genre") or (None, None)
    # A STYLE THAT IS A GENRE IS A SETTING, AND THE GENRE OWNS THE SETTING (the
    # author, 2026-09-27: 'ancient japan' selected, a ninja on a rooftop came
    # out "Rendered in the style of steampunk"). The affinity nudge could not
    # stop it -- steampunk's measured links reach only the broad genres. A
    # rolled style that is a genre leaf's own name (cyberpunk, steampunk) rolls
    # only in its own genre; a typed one stands as always.
    if genre[1]:
        _leaf = {g.split(" (")[0].lower() for g in _genre_pool()}
        _own = str(genre[1]).split(" (")[0].lower()

        def _setting(n):
            s = str(n).lower().replace(" (style)", "")
            s = s[:-len(" style")] if s.endswith(" style") else s
            return s if s in _leaf else None
        names = [n for n in names if _setting(n) in (None, _own)]
    if not names:
        return None, {}
    w = ([1.0] * len(names) if not genre[1] else
         _affinity.weights_for("style_genre", genre[1], names, invert=True))
    # THE LEVEL (2026-09-06): a style rolls by its measured share at the
    # level (art nouveau: none of its posts are explicit); an unmeasured
    # style takes the measured median
    # THE ROLL IS WEIGHTED BY THE STYLE'S MEASURED SIZE (2026-09-13): the
    # pool's posts, square-rooted like a garment's count (the exotic is
    # rare because it is rare) -- cubism (137 posts) rolled as often as
    # '1990s (style)' (73,516). A pool name without a booru count (the 40
    # prose-defined styles) takes the measured median: reachable, never
    # favoured.
    _ps = [float((pool[n] or {}).get("posts") or 0) for n in names]
    _pm = sorted(x for x in _ps if x > 0)
    _pmed = _pm[len(_pm) // 2] if _pm else 1.0
    # ... THE FOURTH ROOT (the author's live test, 2026-09-19: "1990s style ... I
    # meet them very often"): measured over 3,000 nsfw rolls the square root gave
    # '1990s (style)' a quarter of all pictures -- among some fifty rollable
    # styles one giant still owned the draw. The order by size is kept, the
    # gap between 73,000 posts and 2,000 shrinks from six times to two and a half.
    w = [wi * ((x if x > 0 else _pmed) ** 0.25) for wi, x in zip(w, _ps)]
    _lv = (opts or {}).get("_level")
    if _lv:
        # THE LEVEL'S LIFT WITHIN THE STYLE (2026-09-06): how much of the
        # style's own work is at this rating, over the median style --
        # pinup .19 and the 2000s style .22 of their posts are explicit,
        # art nouveau .013 (the median), the 1920s style none. An explicit
        # card draws the styles whose work is explicit; a style with no
        # tag to measure (the pool's own names) stands at the median
        _sh = [(_floor_share(n, _lv) if _floor_measured(n) else None) for n in names]
        _meas = sorted(x for x in _sh if x is not None)
        _med = max(_meas[len(_meas) // 2], 1e-3) if _meas else 1.0
        w = [wi * ((x if x is not None else _med) / _med) for wi, x in zip(w, _sh)]
    name = rng.choices(names, weights=w)[0]
    return name, pool[name]


def _sans_claimed(low):
    """the lowered text with every claimed multi-word phrase blanked: a
    normalised phrase ('shallow depth of field' -> depth of field), a
    camera phrase, so 'field' inside 'depth of field' is no place
    (2026-09-15)"""
    try:
        from promptstudio.engine import enhancer as _pe0
        out = low
        for ph in sorted((k for k in _pe0.NORMALIZE if " " in k), key=len, reverse=True):
            if " " + ph + " " in out:
                out = out.replace(" " + ph + " ", "  ")
        for ph in ("depth of field", "field of view"):
            out = out.replace(" " + ph + " ", "  ")
        return _sans_camera(out) if "_sans_camera" in globals() else out
    except Exception:
        return low


def _typed_place(base, banks=None):
    """-> the place the prompt names (a location-pool name in the text, or
    a typed tag the pool knows), or None. The cheap half of
    resolve_location's typed branch, callable before the genre resolves."""
    low = " " + re.sub(r"[^a-z0-9() ]", " ", (base or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    pool = _location_pool()
    hits = [n for n in pool if " " + n + " " in _sans_claimed(low)]
    if hits:
        return max(hits, key=len)
    try:
        for t in (pe.parse_input(base or "", banks or pe.load_all_banks())[0] or []):
            if str(t).lower() in pool:
                return str(t).lower()
    except Exception:
        pass
    return None


def resolve_genre(base, opts, rng, style_typed=False, banks=None):
    # WHOLE WORDS, AND NOT INSIDE A LONGER CONCEPT. Plain substring matching
    # made "western comics style" resolve the genre `western`, so a request
    # for western COMICS produced a cowboy scene -- and then drew artists to
    # match a genre the user never asked for. The style phrase owns those
    # words; a genre may only claim what is left.
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", (base or "").lower()) + " "
    low = re.sub(r"\s+", " ", low)
    # WHICH MULTI-WORD TAGS CONSUMED THEIR WORDS. Only KNOWN vocabulary
    # claims: a short input is parsed as a tag list, so "cyberpunk street"
    # came back as one 'tag' of its own and claimed the whole text --
    # every genre word vanished and the roll fired instead.
    _known = []
    try:
        from promptstudio.engine import enhancer as _pe
        _vocab = (banks or {}).get("_scanvocab") or {}
        for t in (_pe.parse_input(base or "", banks or {})[0] if banks else []):
            t = str(t).lower()
            if " " in t and (t in _vocab or _gloss_of(t)[0]):
                flat = re.sub(r"\s+", " ",
                              re.sub(r"[^a-z0-9 ]+", " ", t)).strip()
                if flat:
                    _known.append(flat)
    except Exception:
        _known = []
    # A LEAF ANSWERS TO ITS NAME, ITS ALIASES, ITS ANCHORS AND ITS MEMBERS
    # ('ancient rome', 'toga' and 'greek mythology' all mean the
    # greco-roman leaf). A form counts only when it is not merely part of
    # a longer known tag ('western' inside 'western comics'); the form
    # that appears FIRST in the text wins ("ww2 nurse" is the war, with a
    # nurse in it), longest on a tie.
    _best, _best_key = None, None
    # A SELECTED GENRE IS USED (the author's 2026-09-06): with the dropdown set,
    # only a leaf's own name or alias in the text outranks it; an anchor or
    # member word ('succubus' belongs to supernatural) is evidence for the
    # dice, not an order over the user's choice -- a succubus in the
    # everyday world is what was asked for
    _sel0 = str(opts.get("genre") or "").strip().lower()
    _sel0 = _sel0 if (_sel0 and _sel0 not in ("random", "auto") and _sel0 in _genre_pool()) else ""
    for g, rec in _genre_pool().items():
        # a leaf's own name and aliases outrank a member that merely
        # belongs to it: 'kitchen' is the home leaf, not the cafe's kitchen
        forms = [(0, g.split(" (")[0])] \
            + [(0, x) for x in (rec.get("aliases") or [])] \
            + [(1, x) for x in (rec.get("anchors") or [])] \
            + [(2, x) for x in (rec.get("members") or [])]
        for prio, form in forms:
            if _sel0 and prio > 0:
                continue
            form = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ",
                                              str(form).lower())).strip()
            if not form:
                continue
            pos = low.find(" " + form + " ")
            if pos < 0:
                continue
            if any(k != form and (" " + form + " ") in (" " + k + " ")
                   for k in _known):
                continue                      # inside a longer known tag
            key = (pos, prio, -len(form))
            if _best_key is None or key < _best_key:
                _best, _best_key = g, key
    if _best:
        return "typed", _best
    # A TYPED EXTERNAL GENRE IS STILL A TYPED GENRE. Once its dependency
    # fields are filled (locations, occupations, hint) it can drive the
    # downstream axes exactly like a measured one -- which is the whole
    # point of filling them. Before this it was emitted as a loose tag
    # while a DIFFERENT genre was rolled and furnished the scene:
    # "action movie" arrived next to science fiction and a spacecraft.
    # Only `ready` concepts reach here; pending ones are withheld.
    try:
        from promptstudio.library import external as _ext
        for _n, _rec in _ext.find_typed(_sans_characters(base) or "", kinds=("genre",)):
            opts["_external_genre"] = _rec
            return "typed-external", _n
    except Exception:
        pass
    pool = _genre_pool()
    # SELECTED IN THE STUDIO (the author's 2026-09-06): the dropdown beats the
    # dice and rides the typed path downstream (it is the user's choice);
    # a genre word in the text beats the dropdown; 'random' is the dice
    _sel = str(opts.get("genre") or "").strip().lower()
    if _sel and _sel not in ("random", "auto") and _sel in pool:
        opts["_genre_selected"] = _sel
        return "typed", _sel
    # ALWAYS RESOLVED (the author's rework): there is no genre checkbox any
    # more. A prompt that names no genre gets one anyway, because the genre
    # is what the style, location and cast decisions lean on.
    #
    # Holidays stay typed-only: a holiday IS a genre, but a random
    # christmas sprung onto a beach prompt is not wanted.
    # holidays roll now (the author's: one bucket among thirteen, spread inside)
    rollable = sorted(g for g in pool if _genre_rollable(g))
    # A TYPED PLACE NARROWS THE GENRE (the hierarchy read backwards): 'a
    # girl in a park' rolled the space leaf, the park won the location and
    # `space` sat on the line beside it. Only leaves whose roll list holds
    # the typed place may roll; a leaf with no list stays eligible.
    _tp = _typed_place(base, banks)
    if _tp:
        _fit = [g for g in rollable
                if not _genre_locations(g) or _tp in (_genre_locations(g).get("roll") or [])]
        if _fit:
            rollable = _fit
    # A TYPED STYLE OUTRANKS AN INVENTED GENRE. The user typing 'cyberpunk'
    # is real evidence; the genre is about to be made up. Without this the
    # roll produced 'western' for a cyberpunk street and then furnished it
    # with a well. Same strong-links-only threshold as everywhere else, so
    # a style with no measured genre keeps the draw uniform.
    # THE NUDGES FEED THE TWO-STAGE ROLL below instead of replacing it:
    # returning a flat roll over every leaf here bypassed the capped
    # bucket shares (measured: fantasy 4%, holidays 10% on 'a girl').
    _nudge = {}
    if style_typed:
        w = _affinity.weights_for("style_genre", style_typed, rollable)
        if any(x > 1.0 for x in w):
            _nudge = dict(zip(rollable, w))
    # THE PROMPT'S OWN WORDS NARROW THE FIELD. Naming no genre is not the
    # same as giving no evidence: 'a maid cleaning the floor' used to roll
    # uniformly and could land on science fiction, which then furnished
    # itself with a spacecraft. The prompt's tags now nudge the roll -- a
    # nudge with a hard cap, because a garment narrows the plausible genres
    # and must never pick one (the author's on kimono/new year). Words the data
    # has never seen change nothing, so the roll stays uniform for them.
    if base and banks is not None:
        try:
            terms = pe.parse_input(base, banks)[0] or []
        except Exception:
            terms = []
        if terms and not _nudge:
            w = _affinity.genre_seed_weights(terms, rollable,
                                             (banks or {}).get("_net"))
            if any(x > 1.0 for x in w):
                _nudge = dict(zip(rollable, w))
    # A TYPED OCCUPATION NAMES ITS WORLDS (2026-09-04): 'a geisha in a
    # tatami room' rolled superhero. The occupation tables know which
    # leaves list a job; those leaves are lifted (x8: the bucket-first roll
    # dilutes a leaf lift), a nudge on top of
    # whatever the words already gave, never a pick.
    _occ_flat = False
    try:
        _occ = _person_noun_in(base or "")
    except Exception:
        _occ = None
    if _occ:
        _lift = {}
        for g in rollable:
            ent = occupation_entry(g, None)
            if ent and _occ in set(ent[0]):
                _lift[g] = 8.0
        if _lift and len(_lift) < len(rollable):
            _nudge = {g: _nudge.get(g, 1.0) * _lift.get(g, 1.0) for g in rollable}
            _occ_flat = True
    # DAMPED BY POST COUNT, not uniform and not raw. Uniform gave ten
    # genres ~10% each, so `western` (a rare genre in the data) turned up
    # as often as `contemporary` -- the author's: "I dont get why it so loves
    # western setting (ALOT of generations with it)". Raw post count
    # would swing the other way ('science fiction' 58k crowding out
    # 'post-apocalypse' 1.8k, the variety cost the old comment feared).
    # Square-root damping (the character draw's posts ** 0.5) still let
    # `military` (106k, the pool's broadest tag) take 24% of rolls -- one
    # dominant genre instead of another. Log damping keeps the measured
    # ORDER but caps the spread: 106k against 1.8k becomes 11.6 against
    # 7.5, so the commonest genre is a little commoner and nothing owns
    # a quarter of the draws. Measured (360 rolls): western 10.3% under
    # uniform, 5.3% under sqrt, ~8% under log.
    import math as _math
    # TWO STAGES (the author's rework): a BUCKET first, then a LEAF inside it.
    # Bucket shares are CAPPED -- equal, with a mild tilt by the log of the
    # bucket's posts -- so 'leisure' (beach 154k) cannot own the roll the
    # way raw counts would let it; inside a bucket the leaves keep the
    # log-damped order the flat roll used.
    _bk = {}
    for g in rollable:
        _bk.setdefault(pool[g].get("bucket") or g, []).append(g)
    # WITH A TYPED OCCUPATION THE LEAVES COMPETE DIRECTLY: through the
    # bucket stage a two-leaf bucket gained more from the lift than a
    # nine-leaf one ('geisha' went to supernatural three times as often as
    # to ancient japan), so the roll is flat over the leaves, log-damped
    # posts times the lift
    if _occ_flat:
        w = [_math.log(max(2.0, float(pool[g].get("posts") or 1))) * _nudge.get(g, 1.0)
             * float(pool[g].get("weight") or 1.0) for g in rollable]
        return "rolled", rng.choices(rollable, weights=w)[0]
    if len(_bk) > 1:
        _sums = {b: sum(float(pool[g].get("posts") or 1) for g in gs)
                 for b, gs in _bk.items()}
        _mx = max(_math.log(max(2.0, v)) for v in _sums.values())
        # a nudge lifts the BUCKET by its best-nudged leaf, then the leaf
        # inside it; the cap on the un-nudged shares stays
        # a STATED bucket share wins when the pool carries one (everyday is
        # the big default; superhero a one-leaf bucket); else the equal cap
        _shares = _genre_buckets()
        _bw = [(float((_shares.get(b) or {}).get("share") or 0)
                or (1.0 + 0.5 * _math.log(max(2.0, _sums[b])) / _mx))
               * max([_nudge.get(g, 1.0) for g in gs] or [1.0])
               for b, gs in _bk.items()]
        bucket = rng.choices(list(_bk), weights=_bw)[0]
        leaves = _bk[bucket]
    else:
        leaves = rollable
    w = [_math.log(max(2.0, float(pool[g].get("posts") or 1))) * _nudge.get(g, 1.0)
         * float(pool[g].get("weight") or 1.0) for g in leaves]
    return "rolled", rng.choices(leaves, weights=w)[0]


# MEDIUM -- the 'how it was made' axis (the author's taxonomy: genre = what,
# medium = how made, style = aesthetic system). DEFAULT IS ABSENCE: the
# mode's native digital-illustration look is never tagged; a medium tag
# marks the departure from it (the artificial-backdrop philosophy).
# How often a medium is rolled when no style implies one -- see
# resolve_medium. Raised 0.06 -> 0.12 by the author's once the pool actually had
# the vocabulary to spend it on: rebuilding from danbooru's *_(medium)
# family took the rollable set from 13 mostly-traditional entries to 33,
# including pixel art, 3d, photo and painting. The tool names (photoshop,
# clip studio) stay map-only, so a bigger share buys variety rather than
# noise.
MEDIUM_ROLL_SHARE = 0.12

_MEDIUM_POOL = None


def _medium_pool():
    global _MEDIUM_POOL
    if _MEDIUM_POOL is None:
        with open(_paths.data("medium_pool.json"),
                  encoding="utf-8-sig") as f:
            _MEDIUM_POOL = json.load(f)
    return _MEDIUM_POOL


_MEDIUM_VOCAB = None


def medium_vocabulary():
    """every word that names a MEDIUM, in one place.

    The medium axis is deliberately rare (see resolve_medium), which only
    holds if nothing else can emit a medium behind its back. It kept
    happening: the flavour banks carried 'lineart' as a member of
    'monochrome', and style_pool's `techniques` listed it as a technique.
    Each was fixed where it was noticed, which is how it got fixed four
    times. This is the shared vocabulary those guards share.
    """
    global _MEDIUM_VOCAB
    if _MEDIUM_VOCAB is None:
        mp = _medium_pool()
        v = {t.lower() for t in (mp.get("roll") or {})
             if not str(t).startswith("_")}
        v |= {str(x).lower() for x in (mp.get("typed_surfaces") or {}).values()}
        v |= {str(k).lower() for k in (mp.get("typed_surfaces") or {})
              if not str(k).startswith("_")}
        _MEDIUM_VOCAB = v
    return _MEDIUM_VOCAB


def is_medium_tag(tag):
    """-> True when this tag belongs to the medium axis and to no other."""
    try:
        return str(tag).lower() in medium_vocabulary()
    except Exception:
        return False


def _typed_external_look(base):
    """-> True when the text names an external artist or style (midlibrary
    concepts, pending or not). "in the style of van gogh" is a typed look
    exactly as "western comics style" is; the free medium roll must defer
    to it the same way, or a rolled `low poly` fights the user's Van Gogh."""
    try:
        from promptstudio.library import external as _ext
        return any(True for _n, _r in _ext.find_typed(
            base or "", kinds=("artist", "style"), include_pending=True))
    except Exception:
        return False


def resolve_medium(base, opts, rng):
    """-> (mode, tag). typed > style-implied > dice, per the COHERENCE
    TRIANGLE (the author's): whichever of genre/style/medium is defined first,
    the generated others adapt. A resolved style rolls its MEASURED
    mediums at their real shares (sumi-e pulls ink; a style that pulls
    nothing is digitally native and keeps the default absence); only a
    style-free roll draws from the whole pool, share-weighted."""
    mp = _medium_pool()
    low = " " + re.sub(r"[,.]", " ", (base or "").lower()) + " "
    hits = [(s, t) for s, t in mp["typed_surfaces"].items()
            if not s.startswith("_") and " " + s + " " in low]
    if hits:
        return "typed", max(hits, key=lambda x: len(x[0]))[1]
    # The checkbox is gone (the author's rework): medium is resolved
    # automatically like every other axis. An explicit False from a caller
    # or a saved CLI invocation is still honoured.
    if opts.get("gen_medium") is False:
        return "none", None
    st = opts.get("_style") or (None, {})
    if st[0]:
        meds = (st[1] or {}).get("mediums") or {}
        r = rng.random()
        for t, share in sorted(meds.items(), key=lambda x: -x[1]):
            if r < share:
                return "style", t
            r -= share
        # A style that pulls no medium is digitally native, so its default
        # really is absence -- but it FALLS THROUGH to the rare roll below
        # rather than returning here. Since the rework resolves a style on
        # every prompt, returning at this point made the roll unreachable
        # and left the medium rate entirely at the mercy of style evidence
        # (~1.8%), with no tunable knob at all.
        #
        # EXCEPT WHEN THE USER NAMED THE STYLE. The fall-through treats a
        # typed style exactly like a rolled one, so "western comics style"
        # went on to draw `pixel art` out of the whole pool -- three
        # competing looks in one prompt (pixel art + western comics +
        # anime coloring), with the one the user actually asked for
        # outnumbered by the two the dice added.
        #
        # The coherence triangle in this docstring already says typed
        # beats dice. A typed style's own measured mediums are the only
        # ones it licenses; when it licenses none, absence is the answer
        # and the dice do not get a second opinion. A ROLLED style keeps
        # the fall-through, because then both are the engine's own choices
        # and MEDIUM_ROLL_SHARE stays the knob it is meant to be.
        #
        # Measurement cannot settle this one: conditioned on solo,
        # `pixel art` + `western comics (style)` expects 1.1 posts, so
        # observing 0 is no evidence at all and the contradiction table
        # rightly refuses it. Precedence answers what statistics cannot.
        if opts.get("_style_mode") in ("typed", "typed-external"):
            return "none", None
    # A STYLE-FREE ROLL IS RARE ON PURPOSE. the author's: "medium should stay
    # absent by default, only roll it rarely (we will add different medium
    # concepts later so the pull will be bigger in time like '3d render',
    # 'anime screencap' and so on)". The pool is still only 13 entries and
    # mostly traditional surfaces, so rolling freely would put paint on
    # every image and contradict the default-is-absence philosophy above.
    #
    # RAISE THIS as the pool grows into the digital media that are the
    # common case rather than the exception. It is the one number to touch;
    # the style-implied branch above needs no change, because it already
    # fires at each style's own measured share (~1.8% overall).
    if opts.get("_style_mode") == "typed-external":
        return "none", None           # the user's look is the whole look
    if rng.random() >= MEDIUM_ROLL_SHARE:
        return "none", None
    roll = {t: v for t, v in mp["roll"].items() if not t.startswith("_")}
    names = sorted(roll)
    weights = [roll[t]["posts"] for t in names]
    return "rolled", rng.choices(names, weights=weights)[0]


# SCENE TYPE -- the master switch the author's compositional genres expose:
# portraiture/landscape/still life are not sections, they are CAST x
# CAMERA x LOCATION outcomes. Typed subjects force 'figures'; the dice
# below only run on an EMPTY prompt (full creative mode) with the scenes
# checkbox on, making subjectless scenes reachable by dice at all.
_SCENE_TYPES = {
    # type: (weight, location kinds allowed (None=any), scene tags)
    "figures":    (0.80, None, []),
    "landscape":  (0.06, ("outdoor-natural",),
                   ["no humans", "scenery", "landscape"]),
    "cityscape":  (0.04, ("outdoor-manmade", "building"),
                   ["no humans", "scenery", "cityscape"]),
    "interior":   (0.04, ("indoor",), ["no humans", "indoors"]),
    "still life": (0.04, ("indoor",), ["no humans", "still life"]),
    "macro":      (0.02, None, ["no humans", "close-up"]),
}


def resolve_scene_type(base, opts, rng):
    """-> (type, scene_tags). Only an EMPTY prompt may roll away from
    figures (any text keeps the typed reading; the creative cast roll is
    zeroed by a non-figure outcome, which removes nothing typed)."""
    if (base or "").strip():
        return "figures", []
    # THE GENRE DECIDES (the author's). The scenes checkbox is gone; on an empty
    # prompt the chance of a subject-less scene is the genre's own measured
    # share of 'no humans' posts -- science fiction 0.36, fantasy 0.11,
    # school 0.04, western 0.00. A genre that is never peopleless will not
    # produce one, and no fixed 20% is invented for any of them.
    genre = (opts or {}).get("_genre") or (None, None)
    share = _affinity.no_humans_share(genre[1]) if genre[1] else 0.0
    if not share or rng.random() >= share:
        return "figures", []
    names = sorted(n for n in _SCENE_TYPES if n != "figures")
    weights = [_SCENE_TYPES[n][0] for n in names]
    pick = rng.choices(names, weights=weights)[0]
    return pick, list(_SCENE_TYPES[pick][2])


def plan_interactions(subjects, tier_meta, level, rng):
    """THE GROUP SECTION PLANS FIRST (construction order != presentation
    order): edges exist only among MAINS, near-mandatory when 2+ mains are
    present -- and GAZE COUNTS as an interaction, so satisfying that is
    cheap. Zero-edge scenes only at <=2 mains (10%). All-secondary scenes
    replace the edge graph with ONE group action uniting everyone."""
    if tier_meta.get("all_secondary"):
        return [{"participants": list(range(1, len(subjects) + 1)),
                 "group": True}]
    mains = [i + 1 for i, s in enumerate(subjects)
             if s.get("tier") == "main"]
    # a TEXT-MARKED bystander ('hiding from clueless X') never joins an
    # edge -- the 30% group roll had united all three in the act the
    # user explicitly kept X out of
    aware = [i for i in mains if not subjects[i - 1].get("bystander")]
    if len(aware) >= 2:
        mains = aware
    if len(mains) < 2:
        return []
    if len(mains) <= 2 and rng.random() < 0.10:
        return []                              # legitimate parallel existence
    if len(mains) >= 3 and rng.random() < 0.30:
        return [{"participants": mains[:3], "group": True}]
    pair = rng.sample(mains, 2)
    return [{"participants": sorted(pair), "group": False}]


def _interaction_brief(subjects, interactions, level, base="", typed_act="",
                       fixed_position=None):
    if not interactions:
        return ""
    if base and not typed_act:
        # the same reader the fast path uses for its own action sentence
        try:
            typed_act = _fastplan._typed_action(base, {}, pe.load_all_banks())
            typed_act = typed_act.replace(" to the viewer", "").strip()
        except Exception:
            typed_act = ""
    facts = []
    for i, s in enumerate(subjects, 1):
        kd = s["kind"]
        if kd == "futanari":
            kd += " (breasts and penis%s)" % (
                ", and a pussy" if s.get("has_pussy") else "")
        facts.append("subject %d is %s" % (i, kd))
    # ONE NAMING SCHEME. This used to ask for "a short DESCRIPTIVE alias
    # built from their look ('the lavender-haired girl')" while the RULES
    # block asks for the subject's own alias noun (woman, milf) -- two
    # schemes for the same person, in the same prompt. The alias rule is
    # the one the post-processors enforce, so it is the one stated here.
    lines = ["INTERACTIONS (presented AFTER all subjects; refer to named "
             "characters by NAME and to unnamed ones by their noun or "
             "alias, the same one used everywhere else in the plan -- "
             "never pronouns, never 'girl1'; anatomy facts, fixed: %s). "
             "BODY BUDGET: a body part claimed by an interaction is BUSY -- "
             "a girl holding hands has only ONE free hand, so give no "
             "participant a pose or self-action needing a part the "
             "interaction already uses (no 'hands clasped behind her back' "
             "while holding hands):"
             % "; ".join(facts)]
    for e in interactions:
        who = " and ".join("subject %d" % p for p in e["participants"])
        if e.get("group"):
            lines.append(
                " One GROUP action uniting %s: say what they do together, "
                "with an explicit spatial arrangement (who is where). At "
                "this safety level: %s." % (who,
                _PAIR_LADDER.get(level, "keep it neutral")))
        else:
            lines.append(
                " ONE primary interaction between %s: direction matters -- "
                "say WHO does WHAT to WHOM by subject number (gaze or talk "
                "counts; physical contact is not required). Give an "
                "explicit spatial arrangement (who is where relative to "
                "whom). Optionally ONE light secondary contact using body "
                "parts NOT used by the primary. At this safety level: %s."
                % (who, _PAIR_LADDER.get(level, "keep it neutral")))
            # THE TEXT'S OWN FACTS, STATED: who does it to whom, and which
            # positions the booru actually pairs with the typed act. The
            # model reversed "goblin boy fucking elf milf" and invented a
            # boy standing cross-legged on her back; the direction is in
            # the sentence and the postures are measured (act_posture).
            extra = []
            _dir = _typed_direction(base, subjects)
            if _dir:
                extra.append("the user's text says: %s" % _dir)
            if fixed_position:
                _post = [t for t, _w in sorted(
                    _act_posture_row(fixed_position).items(), key=lambda kv: -kv[1])
                    if t in _BODY_POSTURES][:4]
                extra.append("the sex position is FIXED by the engine: %s -- "
                             "put both bodies in it, name no other position. "
                             "Body postures the booru shows in it (give each "
                             "body ONE of these, never two for the same body): %s"
                             % (fixed_position, ", ".join(_post) or "as the position implies"))
            elif typed_act:
                _pos = _typed_positions(typed_act)
                if _pos:
                    extra.append("positions the booru pairs with %s (pick one, "
                                 "do not invent another): %s"
                                 % (typed_act, ", ".join(_pos)))
            if extra:
                lines.append(" FIXED BY THE PROMPT: " + "; ".join(extra) + ".")
    return "\n".join(lines)


_SEX_VERB_RE = re.compile(
    r"\b(fuck\w*|pound\w*|rail\w*|penetrat\w*|ramm\w*|thrust\w*|slamm\w*|"
    r"suck\w*|lick\w*|eat\w* out|rid\w*|kiss\w*|hug\w*|grop\w*|spank\w*|"
    r"finger\w*|giv\w+ (?:a )?(?:blowjob|handjob|footjob|titjob|paizuri)|"
    r"tak\w+ it|carr\w*|lift\w*|hold\w*)\b")


def _typed_direction(base, subjects):
    """-> "subject 2 (the boy) does it to subject 1 (the girl)" from the
    sentence shape ACTOR VERB RECEIVER, or "" when the text does not say.
    'X getting fucked by Y' is the passive: receiver first."""
    low = " " + re.sub(r"[^a-z0-9 ]", " ", re.sub(r"'s\b", "", (base or "").lower())) + " "
    m = _SEX_VERB_RE.search(low)
    if not m:
        return ""
    after = low[m.end():]
    spans = _subject_spans(base or "")
    if len(spans) < 2:
        return ""

    def _kind_of(noun):
        n = noun.lower()
        if re.search(r"\b(futanari|futa|shemale|dickgirl)\b", n):
            return "futanari"
        if re.search(r"\b(" + _MALE_WORDS + r")\b", n):
            return "male"
        if re.search(r"\b(" + _FEMALE_WORDS + r")\b", n):
            return "female"
        return None

    def _pos(noun):
        i = low.find(" " + noun.lower() + " ")
        return i if i >= 0 else None

    ordered = sorted(((_pos(n), n) for n, _c in spans if _pos(n) is not None))
    if len(ordered) < 2:
        return ""
    first, second = ordered[0][1], ordered[1][1]
    passive = bool(re.search(r"\b(getting|being|gets|get|is|was)\s+\w+ed\b", low)) \
        or " by " in after[:40]
    actor, receiver = (second, first) if passive else (first, second)

    def _index(noun):
        k = _kind_of(noun)
        for i, sub in enumerate(subjects, 1):
            sk = sub.get("kind")
            if k and (sk == k or (k == "female" and sk == "futanari" and
                                  not any(_kind_of(n) == "futanari" for n, _ in spans))):
                return i
        return None
    ia, ir = _index(actor), _index(receiver)
    if not ia or not ir or ia == ir:
        return ""
    return "subject %d (the %s) does it to subject %d (the %s)" % (ia, actor, ir, receiver)


_ACT_POSTURE_TBL = {"mtime": 0, "data": {}}

# body postures a sex position is measured against (act_posture.json,
# positions measured 2026-09-03)
_BODY_POSTURES = frozenset((
    "standing", "all fours", "bent over", "lying", "on back", "on stomach",
    "on side", "kneeling", "sitting", "squatting", "top-down bottom-up",
    "legs up", "spread legs", "leaning forward", "arched back",
    "on hands and knees", "prone", "reclining"))
# acts whose receiver has a prostate
_MALE_RECEIVER_ACTS = frozenset(("prostate milking", "pegging", "male penetrated",
                                 "prostate stimulation"))


def _act_posture_row(act):
    """-> {pose: lift} the table measures with `act` (or a position)."""
    p = _paths.data("act_posture.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _ACT_POSTURE_TBL["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _ACT_POSTURE_TBL["data"] = json.load(f).get("acts") or {}
            _ACT_POSTURE_TBL["mtime"] = mt
    except Exception:
        return {}
    return dict((_ACT_POSTURE_TBL["data"].get(str(act or "").lower()) or {}).get("with") or {})


def _typed_positions(act):
    """-> the sex positions act_posture.json measures with `act`, by lift.
    Only for a PENETRATIVE act: 'fellatio' co-occurs with every position
    in group scenes, and offering 'sex from behind' for a blowjob would
    be a measured falsehood."""
    act = str(act or "").lower().strip()
    if not act or (act not in ("sex", "anal", "vaginal", "group sex",
                               "double penetration", "hetero")
                   and act not in _SEX_POSITIONS):
        return []
    p = _paths.data("act_posture.json")
    try:
        mt = os.path.getmtime(p)
        if mt != _ACT_POSTURE_TBL["mtime"]:
            with open(p, encoding="utf-8-sig") as f:
                _ACT_POSTURE_TBL["data"] = json.load(f).get("acts") or {}
            _ACT_POSTURE_TBL["mtime"] = mt
    except Exception:
        return []
    row = (_ACT_POSTURE_TBL["data"].get(str(act).lower()) or {}).get("with") or {}
    pos = [(t, v) for t, v in row.items() if t in _SEX_POSITIONS]
    pos.sort(key=lambda kv: -kv[1])
    return [t for t, _v in pos[:5]]


_KIND_NOUN = {"male": "boy", "female": "girl", "futanari": "futanari"}
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]


def _display_name(persona):
    """'ereshkigal (fate)' -> 'Ereshkigal': the name a person would use.

    One definition for the brief, the scrubber and the pronoun enforcer,
    so no path hands the model or the reader a booru qualifier as a name.
    """
    bare = str(persona or "").split(" (")[0].strip()
    return " ".join(w[:1].upper() + w[1:] for w in bare.split()) if bare         else ""


_FEM_NOUN_RE = re.compile(
    r"\b(girls?|wom[ae]n|lad(?:y|ies)|milfs?|futanaris?|futas?|mothers?|"
    r"daughters?|sisters?|wife|wives|princess(?:es)?|queens?|maids?|"
    r"nurses?|witch(?:es)?|goddess(?:es)?|mermaids?|succub(?:us|i))\b", re.I)
_MASC_NOUN_RE = re.compile(
    r"\b(boys?|m[ae]n|guys?|fathers?|sons?|brothers?|husbands?|princes?|"
    r"kings?|butlers?|knights?)\b", re.I)
_OBJ_STOP = frozenset(("a", "an", "the", "and", "to", "with", "from", "as",
                       "while", "on", "in", "at", "by", "of", "into", "onto",
                       "for", "or", "but", "so", "then", "before", "after",
                       "up", "down", "over", "under", "off", "out", "again",
                       "firmly", "gently", "slowly", "deeply", "hard", "there",
                       "here", "closer", "away", "back"))


_SIZE_FAMILIES = (
    ("flat chest", "small breasts", "medium breasts", "large breasts",
     "huge breasts", "gigantic breasts"),
    ("small penis", "large penis", "huge penis", "gigantic penis"),
)


def _agree_typed_sizes(nl, subjects, plan, tag_subj, typed):
    """-> nl with a named subject's body size replaced by its typed one."""
    typed = {str(t).lower() for t in (typed or ())}
    # who is called what in the prose
    refs = []
    for si, s2 in enumerate(subjects):
        forms = []
        if s2.get("persona"):
            dn = _display_name(s2["persona"])
            if dn:
                forms.append(dn)
                if len(dn.split()[0]) >= 4:
                    forms.append(dn.split()[0])
        try:
            al = _subject_alias(subjects, si, plan)
        except Exception:
            al = ""
        if al:
            forms.append(al)
        refs.append([re.compile(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", re.I)
                     for f in forms if f])
    # each subject's typed size per family
    wants = {}
    for fam in _SIZE_FAMILIES:
        for t in fam:
            if t in typed and t in tag_subj:
                wants.setdefault(tag_subj[t], {})[fam] = t
    if not wants:
        return nl
    out = []
    for sent in re.findall(r"[^.!?]*[.!?]?\s*", nl):
        if not sent:
            continue
        named = [si for si, rx in enumerate(refs) if any(r.search(sent) for r in rx)]
        if len(named) == 1 and named[0] in wants:
            for fam, want in wants[named[0]].items():
                for other in fam:
                    if other != want:
                        sent = re.sub(r"(?<![a-z0-9])" + re.escape(other) + r"(?![a-z0-9])",
                                      want, sent, flags=re.I)
        out.append(sent)
    return "".join(out)


def _agree_pronouns(nl, subjects, plan=None):
    """-> nl with he/she/his/her/him agreeing with the cast's gender.

    CAST-SCOPED, NOT SENTENCE-SCOPED. In a mixed cast "the goblin boy
    grips her hips" is a legitimate reference to the other subject, so
    nothing can be corrected there without knowing who is meant. When
    every subject shares one gender (a solo girl, two girls, two boys),
    a singular pronoun of the other gender is the model's slip -- 'his'
    in a yuri scene -- and every such pronoun is corrected. A mixed cast
    is left to the model."""
    genders = {("male" if s2.get("kind") == "male" else "female")
               for s2 in subjects if s2.get("kind")}
    if len(genders) != 1:
        return nl
    g = next(iter(genders))
    pron = re.compile(r"\b([Hh]e|[Ss]he|[Hh]is|[Hh]er|[Hh]im)\b")

    def rep(m):
        w = m.group(1)
        low, cap = w.lower(), w[0].isupper()
        nxt = nl[m.end():].lstrip()
        nxt_word = re.match(r"([A-Za-z']+)", nxt)
        nxt_word = nxt_word.group(1).lower() if nxt_word else ""
        if g == "female":
            out = {"he": "she", "his": "her", "him": "her"}.get(low, w)
        else:
            if low == "she":
                out = "he"
            elif low == "her":
                # possessive before a noun-ish word, object otherwise
                out = "his" if (nxt_word and nxt_word not in _OBJ_STOP
                                and not nxt.startswith((",", ".", ";", ":"))) \
                    else "him"
            else:
                out = w
        return out[0].upper() + out[1:] if cap and out != w else out
    return pron.sub(rep, nl)


def _subject_alias(subjects, si, plan=None):
    """What an unnamed subject is called: 'the woman', 'the second boy'.

    THE SUBJECT'S OWN ALIAS WINS. The model picks one per subject -- woman,
    milf, young woman -- and the brief has always asked it to use that
    noun "everywhere in count_sentence and nl instead of girl/boy". The
    scrubber ignored it and substituted a bare kind noun, so a stripped
    name turned a woman into "the girl" and threw away the one piece of
    vocabulary that was keeping subjects distinguishable.
    """
    if plan:
        try:
            al = str((plan.get("subjects") or [])[si].get("alias") or "").strip()
        except Exception:
            al = ""
        if al:
            al = al.lower()
            for lead in ("the ", "a ", "an "):
                if al.startswith(lead):
                    al = al[len(lead):]
            if al:
                return "the " + al
    kinds = [_KIND_NOUN.get(s.get("kind"), "figure") for s in subjects]
    kn = kinds[si] if si < len(kinds) else "figure"
    same = [j for j in range(len(kinds)) if kinds[j] == kn]
    if len(same) == 1 or si not in same:
        return "the " + kn
    return "the %s %s" % (_ORDINALS[min(same.index(si), len(_ORDINALS) - 1)],
                          kn)


_REFUSAL_WORDS = ("i cannot", "i can't", "i am unable", "i'm unable",
                  "i will not", "i won't", "as an ai", "i'm not able",
                  "i am not able", "cannot write", "cannot create",
                  "cannot provide", "cannot generate",
                  # the other shape of not-describing: complaining about
                  # the input ("I have not been provided with the specific
                  # JSON data...") -- shipped as a paragraph once
                  "i have not been", "i haven't been", "i have not received",
                  "i need the", "i need a", "please provide", "i do not have",
                  "i don't have", "i was not given", "no json", "the json")


_PROSE_NUDE_RE = re.compile(r"\b(?:completely |fully |entirely |stark )?(?:nude|naked)\b|\bbare[- ]?(?:breasted|chested|bottomed)\b"
                            r"|\btopless\b|\bbottomless\b|\bwithout (?:any )?clothes\b|\bnothing on\b", re.I)
_PROSE_STYLE_RE = re.compile(r"(?:rendered |drawn |painted |illustrated )?in (?:the style of |an? )?([a-z][a-z0-9' -]{1,40}?) style\b"
                             r"|in the style of ([a-z][a-z0-9' -]{1,40}?)(?=[.,;]|$)", re.I)


_PAST_RE = re.compile(r"\b(was|were|had|wore|stood|sat|lay|knelt|held|looked|gazed|smiled|leaned|walked|"
                      r"stared|rested|hung|seemed|appeared|felt|glanced|turned|reached|pressed|clung)\b", re.I)
_PRESENT_RE = re.compile(r"\b(is|are|has|have|wears|wear|stands|stand|sits|sit|lies|lie|kneels|holds|hold|"
                         r"looks|look|gazes|smiles|leans|walks|stares|rests|hangs|seems)\b", re.I)


def prose_in_past_tense(nl):
    """-> True when the paragraph tells the picture as something that happened:
    two or more past forms of the verbs a description is made of ('was', 'wore',
    'stood', 'sat') and more of them than of their present forms. A participle
    ('dressed in', 'lit by') is no past tense and is not counted."""
    t = str(nl or "")
    past = len(_PAST_RE.findall(t))
    return past >= 2 and past > len(_PRESENT_RE.findall(t))


def _prose_contradicts(nl, plan):
    """-> the first way the model's paragraph contradicts the plan it was
    written from, else None (2026-09-15: "Yuki Miku is completely nude"
    over a shirt lift and a skirt lift, and "Rendered in Arknights style"
    with no such style anywhere in the plan). A paragraph that undresses
    a dressed subject or names a style the plan does not carry is dirty:
    rendered again from the plan, and the template's if the model
    insists. The plan is the one owner of the facts; the prose says
    them."""
    text = str(nl or "")
    if not text.strip() or not isinstance(plan, dict):
        return None
    subjects = [ps for ps in (plan.get("subjects") or []) if isinstance(ps, dict)]
    # a lift or a pull is exposure, not nudity: only the whole states
    # (nude, topless, bottomless, the naked compounds) let the prose say
    # 'nude'
    _whole = set(_fastplan._NUDE_STATES) | {"completely nude", "topless female", "topless male"}
    plan_nude = any(str(x).lower() in _whole or str(x).lower().startswith("naked ")
                    for ps in subjects for x in (ps.get("outfit") or []))
    if subjects and _PROSE_NUDE_RE.search(text) and not plan_nude:
        return "nude-in-prose"
    try:
        blob = json.dumps(plan, ensure_ascii=False).lower()
    except Exception:
        blob = ""
    for m in _PROSE_STYLE_RE.finditer(text):
        name = (m.group(1) or m.group(2) or "").strip().lower()
        if name and name not in blob and not any(w in blob for w in name.split() if len(w) > 3):
            return "style-in-prose:" + name
    return None


def _is_refusal(text):
    """-> True when this 'paragraph' is the model declining, not describing.

    Seed 5 of an explicit prompt came back with "I cannot write a story
    that explicitly details sexual acts..." -- and shipped it as the
    prose half of the prompt. A refusal is not a description of the
    image, it is the absence of one, and the empty-nl floor below already
    knows what to do with an absence. The test is on the OPENING: an
    apology or a first-person inability in the first sentence. A real
    paragraph describes; it does not begin by addressing the reader.
    """
    # quotes are NOT stripped: an opening in quotation marks is dialogue
    # ('"I can't do this," the girl whispers'), and dialogue is prose
    head = (text or "").strip().lower().lstrip("*_ ")
    # the OPENING words only: a description that says "I cannot see her
    # face" halfway through its first sentence is still a description
    first = head.split(".")[0][:60]
    return any(first.startswith(w) or first.startswith("sorry")
               for w in _REFUSAL_WORDS)


def _subject_brief(subjects, detail_name="standard", level="sensitive",
                   tier_meta=None, interactions=None, base="",
                   fixed_position=None, genre_place=(None, None)):
    depth = {"minimal": "one short phrase per part",
             "standard": "1-2 short phrases per part",
             "detailed": "2-3 vivid phrases per part"}.get(
                 detail_name, "1-2 short phrases per part")
    lines = []
    for i, s in enumerate(subjects, 1):
        if s.get("tier") == "secondary":
            canon9 = ""
            if s.get("persona"):
                canon9 = (" This IS %s -- the one phrase uses their CANON "
                          "look (%s), never an invented one."
                          % (s["persona"],
                             ", ".join(list(s.get("locked_canon") or
                                            {})[:4]) or "as known"))
            lines.append(
                "Subject %d: %s, SECONDARY -- one look phrase ONLY (gender "
                "noun + one hair-or-outfit note into secondaries_look[%d]); "
                "no other detail.%s" % (i, s["kind"], i - 1, canon9))
            continue
        # DO NOT HAND THE MODEL A WORD IT CAN COPY, AND DO NOT REPLACE IT
        # WITH AN ORDER EITHER. This once said ", unnamed", which sits in
        # an adjective slot right after the gender noun, so the model
        # wrote "1 unnamed girl stands alone at a bar" -- describing the
        # absence of a name as if it were a visible trait.
        #
        # My first repair dictated the noun instead ('always call this
        # subject "the girl"'), which was worse: the RULES block already
        # tells the model to use each subject's own ALIAS -- woman, milf,
        # young woman -- and a fixed noun overrode that whole vocabulary,
        # flattening every subject to "the girl" and making the prose read
        # like enumeration. the author's, rightly: "you literally enforce the
        # llm naming all subjects 'the girl'... what about our aliases".
        #
        # Absence of a name needs no words at all. "Subject 1: female" is
        # already a subject with no character named, and the standing
        # rules cover both halves -- use the alias, never invent a proper
        # name. Silence is the least restrictive thing that works.
        # THE DISPLAY NAME, not the tag: handed "bardiche (riot zanber
        # stinger) (nanoha)" the model wrote exactly that as the name in
        # every sentence. The qualifier is booru bookkeeping; the tag line
        # keeps it, the prose gets the name a person would use.
        who = s["kind"] + ((", character: %s (from %s)"
                            % (_display_name(s["persona"]), s["series"]))
                           if s["persona"] else "")
        if s["kind"] == "futanari":
            who += (" (futanari: has breasts AND a penis%s)"
                    % (", and a pussy" if s.get("has_pussy") else ""))
        # a subject OUTSIDE every interaction edge is a BYSTANDER of the
        # scene's acts: the model kept arousing the oblivious watcher
        # ('his erect penis a testament to his lewd presence' on a
        # character the user typed as clueless)
        in_edge = any((i in (e.get("participants") or []))
                      for e in (interactions or []))
        if interactions and not in_edge and \
                sm.SPICE_ORDER.get(level, 0) >= sm.SPICE_ORDER["nsfw"]:
            who += (" -- BYSTANDER: not part of any act between the "
                    "others; keep this subject non-sexual and clothed "
                    "unless the user's text says otherwise")
        canon = ", ".join(list(s["locked_canon"])[:10])
        body = ", ".join(s.get("body_parts") or [])
        ap = s.get("action_plan")
        act_txt = ""
        if ap:
            wants = []
            if ap.get("pose"):
                wants.append("one overall pose (body position)")
            if ap.get("n_part_actions"):
                wants.append("%d part-level action(s), each using a "
                             "DIFFERENT body part (hands, mouth, head, "
                             "feet...)" % ap["n_part_actions"])
            act_txt = (
                " INDIVIDUAL ACTION (this subject ALONE -- never involving "
                "another subject): %s. May touch their OWN body ('hand on "
                "own hip') or interact with their OWN clothes/accessories "
                "from above%s; fit the location, genre and gender. At this "
                "safety level: %s. A body part does ONE thing -- no "
                "contradictory combinations." % (
                    " plus ".join(wants),
                    (", or hold/use an object that belongs in this location"
                     if ap.get("object_ok") else ""),
                    _ACTION_LADDER.get(level, "keep it neutral")))
        flav = s.get("flavor") or []
        flav_txt = ""
        if flav:
            bits = []
            if "age" in flav:
                bits.append("an age (years, a modifier like 'young', or "
                            "implied by the noun -- an unnamed subject's "
                            "alias may change: 'woman' instead of 'girl' "
                            "for maturity; put it in the 'alias' field)")
            try:
                _pb = pose_bases(genre_place[0], genre_place[1], None)[0]
            except Exception:
                _pb = []
            try:
                _bt = _body_table() or {}
                _rp = (_bt.get("race_parts") or {}).get(str(s.get("race") or "").lower()) or {}
                _fixed = list(_rp.get("parts") or []) + ([_rp["skin"]] if _rp.get("skin") else [])
                if _fixed:
                    bits.append("this race's body is FIXED: %s -- include them" % ", ".join(_fixed))
            except Exception:
                pass
            try:
                _ae = activity_entry(genre_place[0], genre_place[1])
            except Exception:
                _ae = None
            if _ae and _ae[0] and s.get("tier") != "secondary":
                bits.append("activity: none (about half the time), or ONE of THIS list "
                            "only (the place's usual ones) in `self_actions`: %s"
                            % ", ".join(_ae[0][:14]))
            if _pb and s.get("tier") != "secondary":
                bits.append("pose: this subject ALWAYS has a base stance -- ONE of: %s "
                            "(the place's usual ones); leg / arm / hand details may stack "
                            "on it if they fit the stance" % ", ".join(t for t, _w in _pb))
            _rent = race_entry(genre_place[0])
            if _rent and _rent[0]:
                _rc = [t for t in (race_tag(r, s.get("kind")) for r in _rent[0]) if t]
                bits.append("race: human (most subjects), or ONE of THIS list "
                            "only (the genre's races): %s -- write it in `race`, "
                            "empty for human" % ", ".join(_rc[:16]))
            if "occupation" in flav:
                _ent = occupation_entry(genre_place[0], genre_place[1])
                if _ent and _ent[0]:
                    _lv = sm.SPICE_ORDER.get(sm.normalize_level(level), 99)
                    _cands = [t for t in _ent[0]
                              if occupation_fits(t, s.get("kind"))
                              and sm.SPICE_ORDER.get(sm.safety_floor(t), 0) <= _lv]
                    bits.append("an occupation from THIS list only (the "
                                "genre x place table): %s -- or none at all"
                                % ", ".join(_cands[:14]))
                else:
                    bits.append("an occupation/profession fitting the genre, "
                                "setting and location")
            if "general" in flav:
                bits.append("1-2 general flavor descriptors -- %s; NEVER "
                            "negative ones unless the user's text asks"
                            % _GENERAL_LADDER.get(level, "positive only"))
            if "fashion_style" in flav:
                bits.append("ONE fashion style/subculture defining their "
                            "whole look (goth, gyaru, punk, cottagecore, "
                            "office lady, jirai kei...) fitting the genre, "
                            "location and safety level -- and make EVERY "
                            "look choice for this subject (clothes, "
                            "accessories, makeup, tattoos, hair colour and "
                            "hairstyle, where those aspects are being "
                            "filled and not locked by canon) COHERENT with "
                            "it; put it in the 'fashion_style' field")
            flav_txt = " FLAVOR (optional colour, not necessity): " + \
                       "; ".join(bits) + "."
        nudity = "nudity state" in (s.get("slots") or [])
        lines.append("Subject %d: %s. Fill ONLY these aspects: %s.%s%s%s%s%s" % (
            i, who, ", ".join(s["slots"]) or "(nothing else)",
            (" Describe these BODY parts (%s; appearance ONLY -- no actions,"
             " no touching, no poses; gender-appropriate descriptions --"
             " markings means makeup/tattoos/freckles/moles/scars, makeup"
             " skews female; harmonise with the style, setting and action):"
             " %s." % (depth, body)) if body else "",
            (" 'nudity state' means: instead of an outfit, state how undressed"
             " (topless / bottomless / underwear only / completely nude) --"
             " fitting the safety level; accessories may remain.")
            if nudity else "",
            act_txt,
            flav_txt,
            (" LOCKED canon appearance (copy verbatim, do not reinvent): "
             + canon) if canon else ""))
    wardrobe = {
        "safe": "wardrobe choices are entirely ordinary and non-sexualised",
        "sensitive": "wardrobe choices may be mildly alluring for female "
                     "subjects (form-fitting, a little skin)",
        "nsfw": "wardrobe choices skew revealing for female subjects "
                "(lingerie, swimwear, exposed midriffs are appropriate)",
        "explicit": "wardrobe choices may be overtly sexualised for female "
                    "subjects (lingerie, sexual attire, near-undress)",
    }.get(level, "wardrobe choices fit the safety level")
    lines.append(
        "CLOTHES RULES (all modes, canon included): every garment and "
        "accessory must fit the subject's gender, the location, the "
        "subject's occupation or role, the genre/setting, and the safety "
        "level -- at this level, %s; male subjects dress normally unless "
        "the scene demands otherwise. Describe clothes as WORN appearance "
        "only -- no adjusting, removing, tugging or any interaction with "
        "them." % wardrobe)
    tm = tier_meta or {}
    if tm.get("collective"):
        lines.append("PLUS %d more people in the scene: describe them "
                     "COLLECTIVELY in one phrase (the 'collective' field)."
                     % tm["collective"])
    if tm.get("background"):
        lines.append("BACKGROUND: a background crowd is present -- describe "
                     "it in ONE phrase (the 'background' field); it never "
                     "interacts, it is scenery.")
    itxt = _interaction_brief(subjects, interactions or [], level, base=base,
                              fixed_position=fixed_position)
    if itxt:
        lines.append(itxt)
    return "\n".join(lines)


# the per-subject fields the mechanical plan owns for both engines
FIXED_FIELDS = ("outfit", "pose", "self_actions", "held_object", "race",
                "occupation", "activity", "fashion_style")
FIXED_BODY_KEYS = ("race", "build", "skin", "marks", "state", "desc", "expression", "form")


def expression_share():
    """P(at least one expression tag | 1girl), from the 1girl universe:
    the 28 expression-flagged tags there sum to 2.3, so nearly every
    picture carries one -- capped at .9"""
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    uni = (tb.get("universes") or {}).get("1girl") or {}
    tot = sum(float(f) for t, f in uni.items()
              if "expression" in _gloss_flags(t) and not _MOUTH_STATE_RE.search(t))
    return min(0.9, tot) if tot else 0.6


# A MOUTH STATE IS NOT AN EXPRESSION (the author's 2026-09-15): 'open mouth',
# 'closed mouth', 'parted lips', 'teeth', 'tongue out', 'fang' describe the
# mouth, not the mood; they accompany an emotion. The expression draw takes
# emotions; the mouth is a second measured detail under its own share.
_MOUTH_STATE_RE = re.compile(r"\b(?:mouth|lips?|teeth|tooth|tongue|fangs?)\b")


# THE FACE GROUP IS THE FACE'S VOCABULARY (2026-09-19, the face page): the two
# walks below read the 1girl universe -- the 500 tags most related to '1girl'
# -- and only 19 expressions and 9 mouth states are in it, so 'frown',
# 'embarrassed', 'angry', ':3', 'smirk', 'pout', 'wavy mouth' could never be
# drawn. The booru's own face group names the rest (memory: the vocabulary is
# the booru's); a tag the universe does not hold is walked at its own rate,
# posts over the booru's posts. Sections that are drawing styles, memes and
# gazes-at-a-thing stay out; the level floor gates the sexual section as it
# gates every draw; a ruled-out tag stays typed only.
_FACE_SECTIONS = ("emotions", "smile", "smug", "surp", "emotes", "sexual")
_FACE_MIN_POSTS = 1000
_FACE_STATES = {"key": None, "data": None}


def _face_group_states():
    tree = _tree_tags()
    key = len(tree)
    if _FACE_STATES["key"] == key and _FACE_STATES["data"] is not None:
        return _FACE_STATES["data"]
    out = {}
    ro = ruled_out()
    try:                                # a meme face is a meme whatever other section lists it
        with open(_paths.data("danbooru_wiki.json"), encoding="utf-8-sig") as f:
            _fw = (json.load(f).get("groups") or {}).get("face tags") or {}
        ro = set(ro) | {str(x).lower() for k in ("meme", "draw") for x in (_fw.get(k) or [])}
    except Exception:
        pass
    for t, e in tree.items():
        sec = (e.get("groups") or {}).get("face tags")
        n = int(e.get("danbooru") or 0)
        if sec in _FACE_SECTIONS and n >= _FACE_MIN_POSTS and t not in ro and _floor_measured(t)                 and not re.search(r"\bcum\b|condom|ass-to-mouth|\bpenis\b", t):
            out[t] = n / _BOORU_POSTS
    _FACE_STATES["key"], _FACE_STATES["data"] = key, out
    return out


# THE EMOTION OWNS THE FACE (the author, 2026-09-19: the booru's "emotions"
# section "is a mix of real emotions (to which actually the state of everything
# else on face should be tied ...) and states of face parts like eyes / nose /
# mouth - parts of emotions"). An emotion is drawn by name (unrolled_rulings
# 'face'), and the parts of the face are then drawn from what the emotion's own
# posts show AT THE LEVEL: P(part | emotion, level), harvest_levels.py --table
# emotions. The tied tags and the emotes roll only this way.
_LEVEL_TABLES = {}
_LEVEL_ORDER = ["safe", "sensitive", "nsfw", "explicit"]
_EYE_SECTIONS_OF_THE_FACE = {"closed", "emotions", "gazes", "around"}


def level_row(fname, tag, level):
    """-> {companion: share} of `tag` at `level` from a per-band table
    (harvest_levels.py); an empty band falls to the nearest measured one"""
    ent = _LEVEL_TABLES.setdefault(fname, {"mtime": 0, "data": None})
    tb = _json_table(ent, fname) or {}
    lv = tb.get("levels") or {}
    i = _LEVEL_ORDER.index(level) if level in _LEVEL_ORDER else 0
    for j in sorted(range(len(_LEVEL_ORDER)), key=lambda k: (abs(k - i), k)):
        row = (lv.get(_LEVEL_ORDER[j]) or {}).get(str(tag or "").lower())
        if row:
            return dict(row)
    return {}


# WHO WEARS IT, PER BAND, WITHOUT THE CROSSDRESSING (the author, 2026-09-21:
# "right now I noticed female clothes/acessories/hair fire on males" and
# "remeber to gather it on solo subjects or else it will be polluted ... and
# awoid crossdressing/gender swapping in calculations"). danbooru gives an
# anonymous client two tags per query and '<tag> solo' spends both, so the
# contamination cannot be excluded IN the query: it comes back in the same row
# and is discounted HERE. A man in a skirt is filed 1boy, and those pictures are
# what made a skirt look like a man's garment.
_MALE_POLLUTION = ("crossdressing", "crossdressing male", "male crossdressing", "otoko no ko",
                   "trap", "genderswap (mtf)", "genderswap")
_FEMALE_POLLUTION = ("reverse trap", "genderswap (ftm)")


def gender_lean(tag, level=None):
    """-> the measured female share of this word's own SOLO posts at this band
    (futanari counts with the women), the crossdressed pictures taken off each
    side, or None where it is unmeasured."""
    row = level_row("gender_levels.json", tag, level)
    if not row:
        return None
    g = float(row.get("1girl") or 0.0) + float(row.get("futanari") or 0.0)
    b = float(row.get("1boy") or 0.0)
    g = max(0.0, g - max([float(row.get(w) or 0.0) for w in _FEMALE_POLLUTION] or [0.0]))
    b = max(0.0, b - max([float(row.get(w) or 0.0) for w in _MALE_POLLUTION] or [0.0]))
    # A HANDFUL OF PICTURES IS NO MEASURE (2026-09-21, reading the finished
    # harvest): a word the band barely has -- 'symmetrical hand pose' at safe is
    # four posts, three of them a crossdresser's -- came out 0.00 female and
    # would have been kept off women. Under a sixth of the band's posts naming
    # either count word, the row says nothing and the neutral rule stands.
    if (g + b) < 0.15:
        return None
    return (g / (g + b)) if (g + b) > 0 else None


_EYE_RACE_MIN = 0.02        # a race SHOWS an inhuman eye at this share of its own pictures


def race_shows(race, tag, level=None):
    """-> P(part | race): the harvested rows first (per band, then all ratings),
    else the booru's own counts, cached. The harvests keep a race's strongest
    parts only, and these eyes are small but real -- a cat girl's slit pupils are
    .042 of her posts, a bee girl's compound eyes .033 (2026-09-20)."""
    r, t = str(race or "").lower().strip(), str(tag or "").lower().strip()
    if not r or not t:
        return 0.0
    row = (level_row("race_parts_levels.json", r, level) if level else {}) or         ((_json_table(_RACE_REL, "race_parts_related.json") or {}).get("races") or {}).get(r) or {}
    if t in row:
        return float(row[t])
    nr = _count_cached(_q(r))
    nb = _count_cached(_q(r) + " " + _q(t))
    if nr and nb is not None:
        return float(nb) / float(nr)
    return 0.0


def hair_rulings():
    return (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("hair") or {}


def part_gate(tag, ctx=None):
    """the rulings that decide whether a BODY word may be drawn at all (the
    author, 2026-09-20, eyes and hair): the eye gate, plus -- a fantastical hair
    belongs to a race that measures it; 'floating hair' and 'hair spread out'
    are pose driven; the facial hair is a man's."""
    t = str(tag or "").lower()
    c = ctx or {}
    if not eye_gate(t, c):
        return False
    hr = hair_rulings()
    if t in set(hr.get("race_measured") or []):
        return race_shows(c.get("race"), t, c.get("level")) >= _EYE_RACE_MIN
    _np = (hr.get("needs_pose") or {}).get(t)
    if _np:
        _here = " ".join(str(x or "").lower() for x in (c.get("pose"), c.get("act"), c.get("weather"),
                                                        c.get("place"), " ".join(c.get("poses") or ())))
        return any(w in _here for w in _np)
    if not _adopted_fits_body(t, c.get("kind")):
        return False                      # 'requires male': the beards
    if not sex_part_ok(t, c.get("kind"),
                       receiver=futa_receives(act=c.get("act"), acts=c.get("acts") or (),
                                              cast=c.get("cast"), things=c.get("things") or ())):
        return False                      # a futanari has no pussy unless she receives (2026-09-21)
    # BLOOD IS HORROR'S FIRST OF ALL (a bloody wing is blood before it is a wing)
    bl = (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("blood") or {}
    if t in set(bl.get("tags") or []) or re.search(r"\bblood(y|ied)?\b", t):
        return str(c.get("genre") or "").lower() in set(bl.get("genres") or ["horror"])
    wr = (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("wings") or {}
    _rg = (wr.get("race_or_genre") or {}).get(t)      # the made wings, before the grown ones
    if _rg:
        _r = " ".join(str(x or "") for x in (c.get("race"), c.get("race_overlay"))).lower()
        return any(w in _r for w in _rg) or \
            str(c.get("genre") or "").lower() in set((wr.get("race_or_genre") or {}).get("_genres") or [])
    if t in set(wr.get("race_measured") or []):
        return race_shows(c.get("race"), t, c.get("level")) >= _EYE_RACE_MIN
    _np2 = (wr.get("needs_part") or {}).get(t)
    if _np2:
        return any(_np2 in str(x).lower() for x in (c.get("parts") or ()))
    # BLOOD IS HORROR'S (the author, 2026-09-20: "lets make everything related to
    # blood only on horror genre"), wherever the word sits -- a body state, a
    # garment stain, an eye, a wing
    bl = (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("blood") or {}
    if t in set(bl.get("tags") or []) or re.search(r"\bblood(y|ied)?\b", t):
        return str(c.get("genre") or "").lower() in set(bl.get("genres") or ["horror"])
    return True


def eye_rulings():
    return (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("eyes") or {}


_EYE_SETS = {"key": None}


def eye_gate(tag, ctx=None):
    """THE EYE WORDS EACH NEED SOMETHING (the author, 2026-09-20): an
    emotion-driven word is the face chain's and no other draw may take it; an
    inhuman eye belongs to a race that measures it; glowing eyes and the
    coloured scleras belong to the fantastical genres (or a machine race); a
    pupil SHAPE needs a drawn emotion or an anime / manga style. -> True when
    this picture may carry the word."""
    er = eye_rulings()
    if not er:
        return True
    t = str(tag or "").lower()
    c = ctx or {}
    # (!=, not `is not`: two equal ids are not one object, so the test was always
    # true and every candidate re-read and re-parsed the wiki file -- 2,292 loads
    # for three outfits, ten seconds an outfit; found 2026-10-03)
    if _EYE_SETS.get("key") != id(er):
        try:
            with open(_paths.data("danbooru_wiki.json"), encoding="utf-8-sig") as _wf:
                _eyes_w = (json.load(_wf).get("groups") or {}).get("eyes tags") or {}
        except Exception:
            _eyes_w = {}
        sclera = set(_eyes_w.get("sclera") or [])
        shapes = {x for x in (_eyes_w.get("pupilsform") or []) if "shaped" in x or "symbol" in x}
        _EYE_SETS.update({"key": id(er),
                          "emotion": set(er.get("emotion_driven") or []),
                          "race": set(er.get("race_measured") or []),
                          "glow": set(((er.get("genre_only") or {}).get("tags")) or []),
                          # every coloured sclera, not only the ones the booru's own
                          # page lists (its list misses 'grey sclera', 2026-09-20)
                          "sclera": {x for x in sclera if x.endswith(" sclera")} |
                                    {x for x in _tree_tags() if x.endswith(" sclera") and x != "no sclera"},
                          "shapes": shapes | {"heart-shaped pupils"}})
    S = _EYE_SETS
    if t in S["emotion"]:
        return False                      # only the face chain draws these
    if t in S["race"]:
        return race_shows(c.get("race"), t, c.get("level")) >= _EYE_RACE_MIN
    gen = str(c.get("genre") or "").lower()
    if t in S["glow"]:
        go = (er.get("genre_only") or {})
        if gen in set(go.get("_genres") or []):
            return True
        _r = " ".join(str(x or "") for x in (c.get("race"), c.get("race_overlay"))).lower()
        return any(w in _r for w in (go.get("_races") or []))
    if t in S["sclera"]:
        return gen in set(((er.get("genre_only") or {}).get("sclera") or {}).get("_genres") or [])
    if t in S["shapes"]:
        st = str(c.get("style") or "").lower()
        return any(w in st for w in ((er.get("style_or_emotion") or {}).get("_styles") or []))
    return True


_EYEWEAR_RE = re.compile(r"\b(glasses|sunglasses|monocle|goggles|pince-nez|lorgnette|eyewear)\b")
_BARE_BREASTS = {"breasts out", "topless female", "topless", "nipples", "breast slip", "one breast out", "nude",
                 "completely nude", "bare breasts", "see-through clothes", "cleavage"}
_BARE_PENIS = {"penis", "penis out", "erection", "testicles", "nude", "completely nude", "bottomless"}


def conditional_gazes(ctx):
    """-> [(gaze, share)] the gazes this picture may carry: each needs the thing
    it looks at (the author, 2026-09-20 -- 'looking at mirror / object / phone /
    over eyewear' need that object; 'looking at breasts / penis' need that part
    visible on the subject or the partner). Unlisted gazes are not this
    function's business; the share is the tag's own rate among 1girl posts."""
    er = eye_rulings()
    if not er:
        return []
    c = ctx or {}
    words = {str(x).lower() for x in (list(c.get("tags") or []) + list(c.get("outfit") or [])
                                      + list(c.get("scene") or []) + [c.get("held"), c.get("place")]) if x}
    uni = _universe("1girl")
    out = []

    def _rate(t):
        return float(uni.get(t) or 0.0) or float((_tree_tags().get(t) or {}).get("danbooru") or 0) / _BOORU_POSTS

    for t, need in (er.get("needs_object") or {}).items():
        ok9 = False
        if need == "*held":
            ok9 = bool(c.get("held"))
        elif need == "*eyewear":
            ok9 = any(_EYEWEAR_RE.search(w) for w in words)
        else:
            ok9 = any(need in w for w in words)
        if ok9 and _floor_measured(t) and t not in ruled_out():
            out.append((t, _rate(t)))
    for t, part in (er.get("needs_part_visible") or {}).items():
        bare = _BARE_PENIS if part == "penis" else _BARE_BREASTS
        seen = bool(bare & words)
        if not seen and c.get("partner_parts"):
            seen = part in {str(x).lower() for x in c["partner_parts"]}
        if seen and _floor_measured(t) and t not in ruled_out():
            out.append((t, _rate(t)))
    return out


_BLOOD_WORD_RE = re.compile(r"\bblood(y|ied)?\b")


def genre_only_ok(tag, genre):
    """-> False when the tag belongs to a genre this picture is not (the author:
    2026-09-20 "everything related to blood only on horror genre"; 2026-10-03
    "'dead body' tag in slice of life? it should be only in horror maximum").
    The rulings' families -- blood, death -- each name their genres; the genre
    is matched by its own name or its bucket's."""
    t = str(tag or "").lower().strip()
    ur = _json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}
    g = str(genre or "").lower()
    try:
        bucket = str(((_genre_pool() or {}).get(genre) or {}).get("bucket") or "").lower()
    except Exception:
        bucket = ""
    for fam in ("blood", "death"):
        e = ur.get(fam) or {}
        if t in set(e.get("tags") or []) or (fam == "blood" and _BLOOD_WORD_RE.search(t)):
            allowed = {str(x).lower() for x in (e.get("genres") or ["horror"])}
            if g not in allowed and bucket not in allowed:
                return False
    return True


def face_rulings():
    return (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("face") or {}


def face_region(tag):
    """which part of the face a state sits on: one state per region"""
    t = str(tag or "").lower()
    g = (_tree_tags().get(t) or {}).get("groups") or {}
    if g.get("face tags") == "emotes" or _MOUTH_STATE_RE.search(t) or re.search(r"drool|saliva|:[a-z0-9<>/]", t):
        return "mouth"
    if "blush" in t:
        return "blush"
    if re.search(r"\btears?\b|crying|sobbing", t):
        return "tears"
    if re.search(r"eyebrows?|\bbrow\b", t):
        return "brows"
    if "sweat" in t:
        return "sweat"
    if "eyes tags" in g or re.search(r"\beyes?\b|-eyed\b|looking|glaring|staring|wince|wink", t):
        return "eyes"
    return "other"


_SEX_FACE_SHARE = 0.65
_AROUSED_BESIDE_AN_ACT = 40.0     # a sex act in the scene: the face is, nearly always, aroused's


_MOODS_TB = {"mtime": 0, "data": None}
_MOOD_FLOOR = 0.25          # an unscored mood keeps this much of an average scored one


def mood_table():
    return _json_table(_MOODS_TB, "moods.json") or {}


def mood_allowed(genre, level):
    """the moods this picture may carry: EVERY mood (the author, 2026-09-20 --
    "every genre should have every mood, but the chances should be measured
    differently"), minus the ones above its spice"""
    tb = mood_table()
    moods = tb.get("moods") or {}
    rank = sm.SPICE_ORDER.get(str(level or "safe").lower(), 0)
    out = [m for m in moods
           if sm.SPICE_ORDER.get(str((moods[m].get("min_level") or "safe")).lower(), 0) <= rank]
    # ... AND A GENRE MAY CARRY ONE KIND ONLY (the author, 2026-09-20: "bright moods
    # in horror genre sound weird ... in horror only negative moods")
    _ge = (tb.get("genres") or {}).get(str(genre or "").lower()) or {}
    only = _ge.get("only")
    if only:
        keep = set((tb.get("polarity") or {}).get(only) or ()) - set(_ge.get("only_except") or ())
        out = [m for m in out if m in keep] or out
    return out


def mood_leans(genre, level):
    """-> {mood: x} the genre's own moods and the spice band's own moods are
    likelier, they are not the only ones (the author, 2026-09-20: the spice
    affects the mood choice too)"""
    tb = mood_table()
    out = {}
    g = (tb.get("genres") or {}).get(str(genre or "").lower()) or {}
    gw = float(g.get("weight") or 2.0)
    for m in g.get("lean") or []:
        out[m] = out.get(m, 1.0) * gw
    # a genre that leans on a KIND of mood (dark fantasy, post-apocalypse: the
    # negative ones keep their 3x, the author 2026-09-20)
    if g.get("polarity"):
        for m in (tb.get("polarity") or {}).get(g["polarity"]) or []:
            out[m] = out.get(m, 1.0) * float(g.get("polarity_weight") or 3.0)
    sp = tb.get("spice_lean") or {}
    sw = float(sp.get("weight") or 3.0)
    for m in sp.get(str(level or "safe").lower()) or []:
        out[m] = out.get(m, 1.0) * sw
    return out


def mood_scores(level, act=None, place=None, weather=None, season=None):
    """-> {mood: score}: how much of what the SCENE's own posts feel belongs to
    each mood. The act answers first (P(emotion, aura | act, level)); with no
    act, or an act the booru does not file, the place / the weather / the season
    stand in (the author, 2026-09-20)."""
    rows = []
    if act:
        rows.append(level_row("act_emotions_levels.json", act, level))
    if not any(rows):
        rows = [level_row("place_emotions_levels.json", w, level) for w in (place, weather, season) if w]
    feel = {}
    for row in rows:
        for t, f in (row or {}).items():
            feel[t] = max(feel.get(t, 0.0), float(f))
    if not feel:
        return {}
    # WHAT IS SHARED SAYS LITTLE (2026-09-20: 'happy' is in twenty moods, so
    # dancing scored every cheerful mood the same): a feeling counts for a mood
    # in inverse proportion to how many moods list it, so the distinctive ones
    # -- a musical note, an anger vein, a dark aura -- decide.
    moods = mood_table().get("moods") or {}
    spread = {}
    for e in moods.values():
        for x in list(e.get("emotions") or []) + list(e.get("auras") or []):
            spread[x] = spread.get(x, 0) + 1
    out = {}
    for m, e in moods.items():
        out[m] = sum(feel.get(x, 0.0) / float(spread.get(x, 1))
                     for x in list(e.get("emotions") or []) + list(e.get("auras") or []))
    return out


def typed_emotions(tags, level):
    """-> the emotions the user's own words state: an emotion a mood lists, or a
    face tag the emotion's own posts show on .3 or more ('tears': sad .41,
    despair .33; P(part | emotion, level), emotion_faces_levels.json)"""
    moods = mood_table().get("moods") or {}
    known = {x for e in moods.values() for x in list(e.get("emotions") or []) + list(e.get("auras") or [])}
    out = []
    for t in tags or []:
        t = str(t).lower()
        if t in known:
            out.append(t)
            continue
        tb = _json_table(_LEVEL_TABLES.setdefault("emotion_faces_levels.json", {"mtime": 0, "data": None}),
                         "emotion_faces_levels.json") or {}
        row = (tb.get("levels") or {}).get(str(level or "safe")) or {}
        out += [e for e, parts in row.items() if float((parts or {}).get(t) or 0) >= 0.3 and e in known]
    return list(dict.fromkeys(out))


def mood_for(rng, level, genre=None, act=None, place=None, weather=None, season=None, emotions=None):
    """the picture's mood: drawn among the moods the genre allows, weighted by
    how well each one matches what the scene's own posts feel. A feeling the
    user typed decides first (the author, 2026-09-28: the chain was barely
    visible; 'a girl crying in the rain' was scored on the rain): the draw is
    among the moods that carry it."""
    names = mood_allowed(genre, level)
    if not names:
        return None
    if emotions:
        _mt = mood_table().get("moods") or {}
        _carry = [m for m in names
                  if set(emotions) & set(list((_mt.get(m) or {}).get("emotions") or [])
                                         + list((_mt.get(m) or {}).get("auras") or []))]
        if _carry:
            names = _carry
    sc = mood_scores(level, act, place, weather, season)
    lean = mood_leans(genre, level)
    avg = (sum(sc.get(m, 0.0) for m in names) / max(len(names), 1)) if sc else 0.0
    if avg > 0:
        cands = [(m, (sc.get(m, 0.0) + avg * _MOOD_FLOOR) * lean.get(m, 1.0)) for m in names]
    else:
        # the scene's feelings say nothing about the moods this genre may carry
        # (2026-09-20: a graveyard at explicit scored only erotic, which horror
        # does not carry, and every allowed mood came out at zero): the leans decide
        cands = [(m, lean.get(m, 1.0)) for m in names]
    return _wroll(rng, cands)


_FEELING_VERBS = {"data": None}


def feeling_verbs():
    """THE BODY'S OWN SIGNS OF A FEELING (the author, 2026-10-03: the booru's verbs
    'rolled by nothing' -- "I see among them a lot of tags for emotions / mood";
    "wire the emotion ones into the mood chain"): the tags of the verbs-and-gerunds
    group whose gloss is an expression -- crying, trembling, heavy breathing,
    moaning, laughing, yawning. Not a gaze, not an emotion's own name, 1,000 posts
    or more, not ruled out. They are parts of an emotion's face like tears and a
    blush: measured per emotion and band (emotion_faces_levels.json) and drawn by
    emotion_face at their share there.

    ONE THAT NO EMOTION CARRIES IS AN EMOTION OF ITS OWN (the author, same day:
    "laughing and moaning (and remaining similar tags) need anchors"). A verb at
    under .10 of every emotion's posts, whose own posts name a feeling, owns its
    face like 'happy' does (its own row of emotion_faces_levels.json): laughing
    (open mouth .67, smile .57, closed eyes .49; happy .11, smug .05) and staring
    (expressionless .15) are listed in the moods (moods_draft.json, _owner_verbs)
    and in the rulings' emotions_mine; moaning, whose posts are explicit and sex
    acts (masturbation .53, vaginal .26), is a face of sex (faces_of_sex) and
    needs no mood."""
    if _FEELING_VERBS["data"] is None:
        out = set()
        try:
            fr = face_rulings()
            # (the author's own emotions that happen to be verbs -- 'thinking',
            # 'panicking' -- are emotions; a verb whose own posts are animations
            # -- 'blinking', .999 -- is no still picture's)
            owners = set(fr.get("emotions") or []) | set(fr.get("emotions_sexual") or []) \
                | set(fr.get("feeling_verbs_not") or {})
            ro = ruled_out()
            voc = _vocab()
            for t, e in _tree_tags().items():
                if "verbs and gerunds" not in (e.get("groups") or {}):
                    continue
                if "expression" not in (_gloss_flags(t) or ()) or t in owners or t in ro:
                    continue
                if int(voc.get(t) or 0) < 1000 or _GAZE_RE.search(t) or t.startswith("looking"):
                    continue
                out.add(t)
        except Exception:
            out = set()
        _FEELING_VERBS["data"] = out
    return _FEELING_VERBS["data"]


def emotion_as(emotion, level):
    """the word actually written for an emotion the booru says with another
    (the author, 2026-09-20: ecstasy is happy at safe / sensitive and ahegao at
    nsfw / explicit -- the bare 'ecstasy' has 167 posts)"""
    like = ((mood_table().get("emotion_like") or {}).get(str(emotion or "").lower()) or {}).get(str(level or "safe"))
    return like or emotion


def emotion_family(emotion, cast=None, level=None):
    """-> [(tag, weight)] what the booru actually tags for an emotion whose bare
    word is rare (the author, 2026-09-20: envy's subcategories). A member needs
    what it envies in the picture: two bodies, and the part it is about."""
    fam = ((mood_table().get("emotion_families") or {}).get(str(emotion or "").lower()) or {})
    if not fam:
        return []
    c = cast or {}
    total = sum(int(c.get(k) or 0) for k in ("female", "male", "futa", "other"))
    penis = int(c.get("male") or 0) + int(c.get("futa") or 0)
    fem = int(c.get("female") or 0) + int(c.get("futa") or 0)
    out = []
    for t, e in fam.items():
        need = str((e or {}).get("needs") or "")
        if "two" in need and total < 2:
            continue
        if need.endswith(":penis") and penis < 1:
            continue
        if need.endswith(":breasts") and fem < 1:
            continue
        if not _floor_measured(t) or t in ruled_out():
            continue
        out.append((t, float((e or {}).get("posts") or 100)))
    return out


def mood_emotions(mood, level):
    e = ((mood_table().get("moods") or {}).get(str(mood or "").lower()) or {}).get("emotions") or []
    return [x for x in e if x != "neutral"]


def mood_auras(mood):
    return list(((mood_table().get("moods") or {}).get(str(mood or "").lower()) or {}).get("auras") or [])


def mood_lighting(mood):
    return list(((mood_table().get("moods") or {}).get(str(mood or "").lower()) or {}).get("lighting") or [])


def emotion_beside(emotion, level, allowed=None):
    """-> [(emotion, share)] what the booru shows BESIDE this one on a picture of
    several people (the author, 2026-09-20: for several subjects only the genre
    and the action hint at the emotions, and the second face answers the first)"""
    row = level_row("emotion_beside_levels.json", emotion, level)
    return sorted([(t, float(f)) for t, f in row.items()
                   if allowed is None or allowed(t)], key=lambda kv: -kv[1])


def rollable_emotions(level, allowed=None, sex_act=False, mood=None):
    """-> [(emotion, weight)] the emotions the dice may draw at the level: the
    author's list (and mine, flagged), not ruled out, floor-measured, weighted
    by their posts"""
    fr = face_rulings()
    ro = ruled_out()
    tree = _tree_tags()
    out = []
    _sexual = set(fr.get("emotions_sexual") or [])
    _of_sex = set(fr.get("faces_of_sex") or []) if sex_act else set()
    # THE MOOD CHOOSES THE EMOTIONS (the author, 2026-09-20): a picture's mood
    # lists its own, and nothing outside it is felt -- except the faces OF sex,
    # which belong to the act rather than to the mood
    _named = list(fr.get("emotions") or []) + list(fr.get("emotions_mine") or []) + sorted(_sexual)
    if mood:
        _in_mood = set(mood_emotions(mood, level))
        _named = [e for e in _named if e in _in_mood]
    for e in _named + sorted(_of_sex):
        if e in ro or e == "expressionless" or not _floor_measured(e):
            continue
        if allowed is not None and not allowed(e):
            continue
        if not level_row("emotion_faces_levels.json", e, level):
            continue
        w = float((tree.get(e) or {}).get("danbooru") or 1000)
        if e in _of_sex:
            w *= _AROUSED_BESIDE_AN_ACT
        if e in _sexual:
            # the sexual anchor is of its levels only (the floor gates it below nsfw
            # through `allowed`), and beside a sex act it leads
            if _LEVEL_ORDER.index(level if level in _LEVEL_ORDER else "safe") < 2:
                continue
            w *= _AROUSED_BESIDE_AN_ACT if sex_act else 1.0
        out.append((e, w))
    if sex_act:
        # BESIDE A SEX ACT the sexual faces lead most of the time, not every time
        # (2026-09-20: weighted by raw posts x40, 'naughty face' was the face of 4
        # nsfw prompts in 5 and no one was ever embarrassed). STATED until the acts
        # are measured per band (P(face | act, level)): the sexual leads take
        # _SEX_FACE_SHARE of the draw together, spread by the root of their posts.
        _sx = [(e, w) for e, w in out if e in _sexual or e in _of_sex]
        _ot = [(e, w) for e, w in out if not (e in _sexual or e in _of_sex)]
        if _sx and _ot:
            _sx = [(e, (w / (_AROUSED_BESIDE_AN_ACT if True else 1.0)) ** 0.5) for e, w in _sx]
            _k = (_SEX_FACE_SHARE / (1.0 - _SEX_FACE_SHARE)) * sum(w for _, w in _ot) / max(sum(w for _, w in _sx), 1e-9)
            out = _ot + [(e, w * _k) for e, w in _sx]
    return out


_EMOTION_PART_LIFT = 1.3


def emotion_face(emotion, level, rng, allowed=None, taken=(), cap=3):
    # AN EMOTION TOO RARE TO MEASURE WEARS A STATED FACE, or another emotion's
    # (the author, 2026-09-20: skeptical is narrowed eyes, a raised eyebrow, a
    # hand on the chin, a head tilt, thinking, a closed mouth; ecstasy is happy's
    # face at safe and sensitive, ahegao's at nsfw and explicit)
    _tb0 = mood_table()
    _like = ((_tb0.get("emotion_like") or {}).get(str(emotion or "").lower()) or {}).get(str(level or "safe"))
    if _like:
        emotion = _like
    _stated = (_tb0.get("emotion_parts") or {}).get(str(emotion or "").lower())
    if _stated:
        out0, regions0 = [], {face_region(x) for x in taken}
        for t in _stated:
            if len(out0) >= cap:
                break
            if t in taken or not _floor_measured(t) or t in ruled_out():
                continue
            if allowed is not None and not allowed(t):
                continue
            r = face_region(t)
            if r != "other" and r in regions0:
                continue
            if rng.random() < 0.5:
                out0.append(t)
                regions0.add(r)
        return out0
    """-> the parts of the face the emotion's own posts show at the level, one
    per region, each at its (lifted, capped) share"""
    ro = ruled_out()
    tree = _tree_tags()
    fr = face_rulings()
    names = set(fr.get("emotions") or []) | set(fr.get("emotions_mine") or []) | set(fr.get("emotions_sexual") or [])
    names.discard(emotion)
    regions, out = {face_region(x) for x in taken}, []
    _uni = ((_json_table(_ACT_REL, "place_related.json") or {}).get("universes") or {}).get("1girl") or {}
    for t, f in sorted(level_row("emotion_faces_levels.json", emotion, level).items(), key=lambda kv: (-kv[1], kv[0])):
        if len(out) >= cap:
            break
        g = (tree.get(t) or {}).get("groups") or {}
        if t in ro or t in names or t in taken or not _floor_measured(t):
            continue
        if "eyes tags" in g and g["eyes tags"] not in _EYE_SECTIONS_OF_THE_FACE and "face tags" not in g:
            continue                    # an eye colour, a pupil shape: the character's, not the emotion's
        if g.get("face tags") in ("draw", "meme", "main", "see", "misc"):
            if t not in set(fr.get("tied") or []):
                continue
        if allowed is not None and not allowed(t):
            continue
        # THE PART MUST BE THE EMOTION'S, not every picture's: 'annoyed' shows a
        # smile in .36 of its posts -- fewer than the .33-.46 of ALL pictures
        # (the other girl is smiling) -- and an anger vein in .18 against .009.
        # A part is kept at a lift of _EMOTION_PART_LIFT over its own base rate.
        _base = float(_uni.get(t) or 0.0) or float((tree.get(t) or {}).get("danbooru") or 0) / _BOORU_POSTS
        if _base <= 0 or f / _base < _EMOTION_PART_LIFT:
            continue
        r = face_region(t)
        if r != "other" and r in regions:
            continue
        if rng.random() < min(_COMPANION_CAP, f * _COMPANION_LIFT):
            out.append(t)
            regions.add(r)
    return out


def _face_tied():
    fr = face_rulings()
    tied = set(fr.get("tied") or []) | set(fr.get("emotions") or []) | set(fr.get("emotions_mine") or [])         | set(fr.get("emotions_sexual") or [])
    secs = set(fr.get("tied_sections") or [])
    for t, e in _tree_tags().items():
        if (e.get("groups") or {}).get("face tags") in secs:
            tied.add(t)
    return tied


def emotion_states():
    """-> [(tag, share)] the expressions that are not mouth states, by
    their measured share of the 1girl universe, largest first (the author's
    2026-09-15: "apply the same measured shares to the emotion draw") --
    the expression draw walks these, so the face is drawn as the booru
    shows it, not as the co-occurrence network's hubs"""
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    uni = (tb.get("universes") or {}).get("1girl") or {}
    out = [(t, float(f)) for t, f in uni.items()
           if "expression" in _gloss_flags(t) and not _MOUTH_STATE_RE.search(t) and float(f) > 0]
    have = {t for t, _ in out}
    # (a face-group tag the universe DOES hold keeps its measured share there:
    # 'frown' .012, 'embarrassed' .013, ':3' .014 carry the 'emote' flag and
    # were skipped by the flag test above)
    out += [(t, float(uni.get(t) or f)) for t, f in _face_group_states().items()
            if t not in have and not _MOUTH_STATE_RE.search(t)]
    _tied = _face_tied()
    out = [(t, f) for t, f in out if t not in _tied or t == "expressionless"]
    return sorted(out, key=lambda kv: -kv[1])


def mouth_states():
    """-> [(tag, share)] the mouth states by their measured share of the
    1girl universe, largest first (open mouth .36, closed mouth .20,
    parted lips .08 ...): the second face draw walks these shares, so the
    mouth is drawn as the booru shows it, not as the co-occurrence
    network's hub ('open mouth' 60% when the network proposed)"""
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    uni = (tb.get("universes") or {}).get("1girl") or {}
    out = [(t, float(f)) for t, f in uni.items()
           if "expression" in _gloss_flags(t) and _MOUTH_STATE_RE.search(t) and float(f) > 0]
    have = {t for t, _ in out}
    out += [(t, float(uni.get(t) or f)) for t, f in _face_group_states().items()
            if t not in have and _MOUTH_STATE_RE.search(t)]
    _tied = _face_tied()
    out = [(t, f) for t, f in out if t not in _tied]
    return sorted(out, key=lambda kv: -kv[1])


def fixed_facts(fast_plan):
    """-> [{field: value}] per subject: the mechanical plan's decisions
    that the LLM must keep (its free slots are hair, eyes, expression,
    face, the alias, the setting phrase and the mood)"""
    out = []
    for d in (fast_plan or {}).get("subjects") or []:
        f = {}
        for k in FIXED_FIELDS:
            v = d.get(k)
            if v not in (None, "", [], {}):
                f[k] = v
        body = d.get("body") or {}
        bf = {k: v for k, v in body.items() if v}
        if bf:
            f["body"] = bf
        out.append(f)
    return out


def fixed_facts_brief(facts):
    """the FIXED block of the LLM brief, one line per subject"""
    lines = []
    for i, f in enumerate(facts or [], 1):
        if not f:
            continue
        bits = []
        if f.get("occupation"):
            bits.append("occupation: %s" % f["occupation"])
        if f.get("race"):
            bits.append("race: %s" % f["race"])
        if f.get("outfit"):
            _nud = {"nude", "completely nude", "naked", "topless", "bottomless", "topless female", "topless male"}
            if all(x in _nud for x in f["outfit"]):
                bits.append("outfit (copy verbatim into 'outfit'): %s -- the subject wears NOTHING; "
                            "say so plainly, never 'wears her nude body'" % ", ".join(f["outfit"]))
            else:
                bits.append("outfit (complete, copy verbatim into 'outfit'): %s" % ", ".join(f["outfit"]))
        if f.get("pose"):
            bits.append("pose: %s" % f["pose"])
        if f.get("activity"):
            bits.append("activity: %s" % f["activity"])
        if f.get("self_actions"):
            bits.append("self actions: %s" % ", ".join(f["self_actions"]))
        if f.get("held_object"):
            bits.append("holding: %s" % f["held_object"])
        body = f.get("body") or {}
        bw = [w for k in body for w in (body.get(k) or []) if isinstance(w, str)]
        if bw:
            bits.append("body words (the body itself -- horns, tail, build, marks are HAD, never worn): %s"
                        % ", ".join(bw))
        if bits:
            lines.append("  Subject %d -- %s." % (i, "; ".join(bits)))
    if not lines:
        return ""
    return ("FIXED BY THE ENGINE (the mechanical plan decided these for this seed; "
            "copy them into the plan's fields VERBATIM, add nothing to those fields, "
            "replace nothing; describe them in the prose; your own choices are the "
            "free slots only: hair, eyes, face, the alias and the setting phrase -- "
            "the expressions are among the body words and the mood is fixed):\n" + "\n".join(lines))


def merge_fixed(plan, fast_plan):
    """write the mechanical plan's fixed fields back over the LLM's plan,
    subject by subject; the LLM keeps its free slots"""
    subs = plan.get("subjects") or []
    for i, d in enumerate((fast_plan or {}).get("subjects") or []):
        if i >= len(subs) or not isinstance(subs[i], dict):
            break
        s9 = subs[i]
        for k in FIXED_FIELDS:
            v = d.get(k)
            if v not in (None, "", [], {}):
                s9[k] = v
            elif k in ("outfit", "race", "occupation", "held_object", "fashion_style"):
                # the engine said nothing here, so nothing is said: the
                # model invented a race ('white') and a costume for a name
                # it made up; self actions stay the model's when the
                # engine rolled none
                s9[k] = [] if k == "outfit" else ("" if k in ("race", "occupation", "fashion_style") else None)
        # THE BODY IS THE ENGINE'S ENTIRELY (every key: the table's race
        # parts, build, skin, marks, states and the bank's face / shape /
        # markings), so the same seed shows the same body on either engine
        s9["body"] = dict(d.get("body") or {})
        s9["_from_table"] = True                 # the plan prune keeps them
    return plan


def bridge1(base, cast, spice, mode, opts, rng, pulls_hint="",
            subjects=None, tier_meta=None, interactions=None):
    # nothing structural is left for the model to invent: genre, style,
    # medium, location, camera and lighting are all resolved before this
    # brief is written. The list stays because plan_concepts still reads
    # it, and because a future axis may legitimately be delegated again.
    gen = []
    loc = opts.get("_location") or ("free", None, None)
    genre = opts.get("_genre") or ("none", None)
    scene_lines = []
    if loc[0] == "backdrop-typed":
        scene_lines.append(
            "BACKDROP (fixed, artificial): %s. It IS the location -- draw "
            "NO place and no scenery; the backdrop plus the subjects is "
            "the whole image." % loc[1])
    elif loc[0] == "backdrop-rolled" and loc[1]:
        scene_lines.append(
            "BACKDROP (fixed, artificial): %s. It IS the location -- draw "
            "NO place and no scenery; the backdrop plus the subjects is "
            "the whole image." % loc[1])
    elif loc[0] == "backdrop-rolled":
        scene_lines.append(
            "BACKDROP (artificial): invent one -- a plain colour, gradient, "
            "pattern or abstract field, in your own terms. It IS the "
            "location: draw NO place and no scenery.")
    elif loc[1] and loc[0] == "typed":
        scene_lines.append(
            "LOCATION (the user's own, fixed, do not change): %s (%s). "
            "Give 2-4 scene elements that fit it."
            % (loc[1], _kind_said(loc[2]) or "unknown kind"))
    elif loc[1]:
        # CHOSEN, not typed. The engine already refuses a place that
        # contradicts the prompt's TAGS, but the user can write something
        # no tag covers ("where the sand meets the tide"), so the model
        # gets the final say on a place it can see is wrong.
        scene_lines.append(
            "LOCATION (chosen for this image): %s (%s). Give 2-4 scene "
            "elements that fit it. If -- and only if -- it CONTRADICTS "
            "what the user actually wrote, replace it with a place that "
            "fits their words and use that in 'setting' instead."
            % (loc[1], _kind_said(loc[2]) or "unknown kind"))

    _evb = opts.get("_event") or (None, None, None)
    if _evb[1]:
        scene_lines.append("OCCASION: %s -- the scene is this occasion (props the booru "
                           "shows: %s); it colours the mood, the props and the activity, "
                           "and never contradicts the place."
                           % (_evb[1], ", ".join((_evb[2].get("props") or [])[:4])))
    lt = opts.get("_lighting")
    if lt:
        scene_lines.append(_lighting_brief(
            lt, (opts.get("_location") or (None, None, None))[2],
            (opts.get("_style") or (None,))[0],
            (opts.get("_genre") or (None, None))[1]))
    if opts.get("_effects_on"):
        scene_lines.append(
            "EFFECTS: add 1-2 visual effects coherent with the style and "
            "the lighting, in your own image-generation vocabulary "
            "(sparkles, drifting petals, motion blur, lens flare, bokeh, "
            "film grain -- or anything you know); put them in 'effects'.")
    focuses2 = opts.get("_focus") or []
    if focuses2:
        scene_lines.append(
            "FOCUS (fixed): %s -- the camera emphasises this; describe the "
            "focused thing richly and compose around it."
            % ", ".join(focuses2))
    if opts.get("_face_typed") or any(f in ("portrait", "eye focus") for f in (opts.get("_focus") or [])):
        scene_lines.append(
            "FACE (fixed framing): the face and its expression carry this picture -- describe "
            "them richly (expression, eyes, mouth, gaze); the rest of the body and any activity "
            "briefly or not at all.")
    if exclusion_terms(opts):
        scene_lines.append(
            "EXCLUDED (fixed): never depict or mention %s -- in no field and not in the prose."
            % ", ".join(exclusion_terms(opts)))
    cam = opts.get("_camera") or (None, None, None)
    if cam[0]:
        extra9 = opts.get("_camera_extra")
        scene_lines.append(
            "CAMERA (fixed): %s%s%s. Compose the scene as seen through "
            "this framing; do not describe what it cannot show. Name the "
            "shot in ONE short phrase of your own camera vocabulary "
            "(angle, lens, depth of field, movement), woven into the "
            "story." % (cam[0], (", " + cam[1]) if cam[1] else "",
                        (", " + extra9) if extra9 else ""))
        # THROUGH THE VIEWER'S EYES (the author's 2026-09-06): a pov picture
        # with an act is the face and the act; the rest is seen from there
        if str(cam[1] or "").lower() in _POV_VIEWS:
            scene_lines.append(
                "POINT OF VIEW (fixed): the picture is what the viewer sees with their own "
                "eyes -- say so, and keep the eye on the subject's face%s; other parts of "
                "the body are mentioned only as they appear from there, briefly."
                % (" and on the act (%s)" % opts["_act"] if opts.get("_act") else ""))
    # THE MOOD IS THE CHAIN'S (the author, 2026-09-28: "even though we made a
    # mood/emotions chains I haven't noticed them used anywhere"): the model was
    # never told the drawn mood and chose its own; the faces it describes are
    # already the chain's (the body words of the fixed block)
    if opts.get("_mood"):
        scene_lines.append(
            "MOOD (fixed): %s. The paragraph carries this mood, and its last sentence says it: "
            "'The mood is %s.'" % (opts["_mood"], opts["_mood"]))
    if opts.get("_look_phrases"):
        scene_lines.append(
            "THE LOOK (typed by the user, fixed): %s. Say it in the paragraph, in these words."
            % ", ".join(opts["_look_phrases"]))
    st_name, st_info = opts.get("_style") or (None, {})
    if st_name:
        pal = ", ".join((st_info.get("palette") or {}).keys())
        tech = ", ".join(list(st_info.get("techniques") or {})[:3])
        # THE PALETTE IS THE ENGINE'S (the author, 2026-09-28: "pastel colors is too
        # overused"): the model was asked for 1-3 palette words on every picture and
        # reached for pastel and warm colors; the mechanical plan rolls each of the
        # style's palette tags at its measured share, most pictures none. The
        # model is told the palette, never asked for one.
        _fpal = [str(x) for x in (opts.get("_fixed_palette") or [])]
        scene_lines.append(
            "STYLE (%s, fixed): %s.%s PALETTE (fixed): %s -- no other colour or "
            "palette words." % (
                "typed by the user" if opts.get("_style_mode") == "typed"
                else "chosen for this image",
                st_name,
                (" Rendering techniques it favours: %s." % tech)
                if tech else "",
                ", ".join(_fpal) if _fpal else "none"))
    if genre[1]:
        g = genre[1]
        hint = _genre_pool().get(g, {}).get("hint", "")
        scene_lines.append(
            "SETTING/GENRE: %s%s. Let it INFORM (never restrict) clothes, "
            "features and occupations -- the user's own words override it "
            "freely." % (g, (" -- " + hint) if hint else ""))
        if genre[0] == "rolled" and loc[0] == "typed":
            # a rolled genre colours the PEOPLE, never the PLACE
            # (the author's, round 8 refined: school uniforms on the cast in
            # a forest = fine random additions; the scene degrading
            # toward a school = the bug)
            scene_lines.append(
                " The genre was ROLLED and the location is the USER'S: "
                "the genre may colour clothes, features and occupations "
                "freely, but it must NOT change WHERE the scene happens "
                "-- the location stays exactly the user's, with no "
                "scenery or setting drawn from the genre.")
    # MEDIUM (the third corner of the coherence triangle). The rider rule
    # (the author's): whichever of genre/style/medium the user defined stays
    # EXACTLY as typed -- even when the typed members disagree with each
    # other. Only the missing corners adapt to the defined ones.
    med9 = opts.get("_medium") or ("none", None)
    if med9[1]:
        mp9 = _medium_pool()
        nl_hint = ((mp9["roll"].get(med9[1]) or {}).get("nl")
                   or med9[1])
        scene_lines.append(
            "MEDIUM (%s, fixed): %s. The image IS made in this medium -- "
            "open the NL as it ('A watercolor painting of...', 'A "
            "photograph of...') and render the world in its physics: %s. "
            "Anything being INVENTED (style, setting, palette) must "
            "cohere with this medium; anything the user typed stays "
            "exactly as typed even if it clashes."
            % ("typed by the user" if med9[0] == "typed" else "resolved",
               med9[1].replace(" (medium)", ""), nl_hint))
    # non-figure SCENE TYPE: the place/arrangement is the subject
    sct9 = opts.get("_scene_type") or "figures"
    if sct9 != "figures":
        scene_lines.append(
            "SCENE TYPE (fixed): %s. NO people appear in this image -- "
            "draw no figure, face or body part. The %s itself is the "
            "star: compose it with the depth of detail a character would "
            "have received (its forms, materials, age, weather-wear, "
            "small story-telling details)."
            % (sct9, "arrangement of objects" if sct9 == "still life"
               else "subject in extreme close-up" if sct9 == "macro"
               else sct9))
    # the rolled sex position is TAG-ONLY (the author's, round 8: instructing
    # the NL to render it pulled the uninvolved watcher into the act --
    # 'the position drives too much'; a tag/prose divergence is the
    # lesser evil, and image models weight the tag)
    req9 = opts.get("_required_content") or []
    if req9:
        scene_lines.append(
            "REQUIRED CONTENT (the user's text and the safety level "
            "demand it, fixed): %s. The image DEPICTS this explicitly -- "
            "weave it into the subjects' actions and the story in direct "
            "language; do not soften, skip, euphemize or merely allude "
            "to it." % ", ".join(req9))
    # COHERENCE fires when AT LEAST ONE side is invented (the author's): a
    # generated style must cohere with a typed setting, and vice versa.
    # A fully user-typed pair (cubism + fantasy) is left alone -- typed
    # members are never adapted, not even to each other.
    # Both axes are always resolved now, so coherence is about WHICH of
    # them the user actually chose. A typed pair is never adapted -- not
    # even to each other; anything the engine picked should bend toward
    # what the user typed.
    _st_typed = opts.get("_style_mode") == "typed"
    if not _st_typed and genre[0] == "typed" and genre[1]:
        scene_lines.append("The style was CHOSEN by the engine and the "
                           "setting is the USER'S: make the style cohere "
                           "with the setting (%s)." % genre[1])
    elif _st_typed and genre[0] == "rolled":
        scene_lines.append("The style is the USER'S and the setting was "
                           "chosen: the setting must cohere with the "
                           "style, never the other way round.")
    elif genre[0] == "rolled" and not _st_typed:
        scene_lines.append("Both setting and style were chosen for this "
                           "image: keep them coherent with each other.")
    brief = (_subject_brief(subjects,
                            (tier_meta or {}).get("capped_detail",
                                                  opts.get("detail",
                                                           "standard")),
                            spice, tier_meta, interactions, base=base,
                            fixed_position=opts.get("_sex_position"),
                            genre_place=((opts.get("_genre") or (None, None))[1],
                                         (opts.get("_location") or (None, None, None))[1]))
             if subjects else "(no subjects: scenery -- the location is the "
                              "star; describe it with the detail a subject "
                              "would have received)")
    _fx = fixed_facts_brief(opts.get("_fixed_facts"))
    if _fx:
        brief = brief + "\n" + _fx
    user = (
        "MODE=%s\nSafety level=%s (fixed)\nCast (fixed): %s\n%s\n%s"
        "User's text (honour every concept in it): %s\n"
        "Invent freely for: %s. For anything the user's "
        "text already specifies, use exactly what they said.%s\n"
        "Variety seed: %d -- avoid the most obvious choice when inventing."
        % (mode, spice, cast_sentence(cast), brief,
           ("\n".join(scene_lines) + "\n") if scene_lines else "",
           base or "(none)",
           ", ".join(gen + ["the listed subject aspects"]),
           ("\nTypical associations for this setting (bias, not law): "
            + pulls_hint) if pulls_hint else "",
           rng.randrange(10000)))
    # the plan grows with the cast: 900 tokens truncated a THREE-subject
    # plan mid-JSON (same failure class as the bridge-2 budget, found the
    # same way). One retry on a malformed answer before giving up.
    budget = min(2000, 900 + 350 * max(0, len(subjects) - 1))
    for attempt in range(2):
        out = chat(BRIDGE1_SYSTEM, user, temp=0.85, max_tokens=budget, schema=_PLAN_SCHEMA, stage="plan")
        try:
            return parse_json_block(out)
        except Exception:
            if attempt:
                raise


BRIDGE2_SYSTEM = (
 "You translate an image-generation prompt's concepts into danbooru/"
 "gelbooru tags. Booru prompts follow a structure -- [count] [character] "
 "[appearance: hair/eyes/body] [clothes] [actions] [interactions] [style] "
 "[location] [lighting] [camera] -- and each concept below is labelled "
 "with its FAMILY: choose a candidate that belongs to that family (a "
 "'lighting' concept must not map to an object tag). For each concept you get a "
 "RANKED shortlist of candidate tags, best match first. STRICT RULES: take "
 "the FIRST candidate whose meaning truly fits; move to a later one ONLY "
 "when every earlier candidate is wrong in meaning, never because a later "
 "one sounds more interesting. Choose 0-2 tags per concept, ONLY from that "
 "concept's own candidates, verbatim; a second tag only when the concept "
 "combines an object with its pattern or modifier (floral yukata -> yukata "
 "+ floral print). If nothing fits the meaning, use an empty list; the "
 "concept stays natural-language only. Never invent a tag. Answer with ONE "
 "JSON object mapping each concept to a list of its chosen tags.")


def _pose_fragments(pose):
    """split a compound pose into independently-mappable clauses.
    'sitting cross-legged on the floor, reaching up to hang a bauble'
    -> ['crossed legs', 'sitting on the floor', 'reaching up to hang a
    bauble']. cross-legged is pulled out as its own concept (the tag is
    'crossed legs', which the crammed phrase never retrieved)."""
    frags = []
    parts = re.split(r"\s*,\s*|\s+while\s+|\s+and\s+(?=\w+ing\b)", pose)
    for p in parts:
        p = p.strip().rstrip(".")
        if not p:
            continue
        if re.search(r"cross[- ]legged", p, re.I):
            frags.append("crossed legs")
            p = re.sub(r"\b(sitting\s+)?cross[- ]legged\b",
                       lambda m: "sitting" if m.group(1) else "",
                       p, flags=re.I).strip()
        ws = [w for w in re.split(r"[^a-z]+", p.lower()) if w]
        if ws and not (len(ws) == 1 and ws[0] in
                       ("the", "a", "an", "on", "in", "with")):
            frags.append(p)
    seen, out = set(), []
    for f in frags:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out or ([pose] if pose else [])


def plan_concepts(plan, time_now=None):
    """-> (concepts, action_concepts, group_map, origins, anchors, single).
    Every concept is tracked by ORIGIN -- (band, subject index) -- so the
    verifier can hold actions to arity AND the presentation sort can rebuild
    Scheme 1's contiguous subject blocks from the mapped tags. ANCHORS maps
    slot concepts to their anchor word (hair/eyes/body part) so bridge 2 can
    keep their candidates in-slot ('gold flecks' in an eye descriptor must
    not become the bare 'gold' tag). SINGLE is the one-pick set: one_of
    slots (hair, eyes -- one colour per subject) and interaction acts (a
    second pick smuggled 'summer festival' into the interaction band and
    'own hands together' next to 'holding hands')."""
    out = []
    action_concepts = set()
    group_map = {}
    origins = {}
    anchors = {}
    single = set()

    def note(vals, band, idx=0):
        for v in vals:
            origins.setdefault(str(v).lower(), (band, idx))

    for si, s in enumerate(plan.get("subjects") or []):
        # SLOT-ANCHORED: the engine knows "hazel" is an eye colour, so the
        # concept string says so -- unanchored field values retrieved `haze`
        # for hazel and the nut for chestnut. The anchor word makes the
        # head-noun logic deterministic instead of hopeful.
        for key, anchor in (("hair", "hair"), ("eyes", "eyes"),
                            ("hair_style", "hair")):
            # EACH HAIR FACT IS ITS OWN CONCEPT (2026-10-03, found behind the author's
            # soapland test): the fast plan writes the field as its tags, 'very long
            # hair, orange hair, ahoge', and the whole string was one concept with one
            # pick -- the line kept the length or the colour, never both, and the
            # prose told the one the line lost (26 of 32 prompts measured).
            for val in str(s.get(key) or "").split(","):
                val = val.strip()
                if val and val.lower() not in ("null", "none"):
                    cc = val if (anchor in val.lower() or _vocab().get(val.lower())) else val + " " + anchor
                    if cc in anchors:
                        continue
                    out.append(cc)
                    note([cc], "subject", si)
                    anchors[cc] = anchor
                    single.add(cc)
        got_outfit = [x for x in (s.get("outfit") or []) if x]
        out += got_outfit
        note(got_outfit, "subject", si)
        body = s.get("body") or {}
        if isinstance(body, dict):
            # part-anchored: "perky" under breasts becomes "perky breasts",
            # so retrieval has its head noun (the hazel->haze lesson)
            for part, phrases in body.items():
                pw = str(part).split(" (")[0].strip().lower()
                for ph in (phrases if isinstance(phrases, list)
                           else [phrases]):
                    ph = str(ph).strip()
                    # a phrase that IS the bare part name maps to junk
                    # ('markings' the literal tag); skip it -- unless the
                    # part IS a booru tag (2026-09-15: 'pussy', 'ass',
                    # 'penis', 'nipples' were planned and never emitted:
                    # the pussy described in 38% of explicit posts reached
                    # the line in none)
                    if ph and (ph.lower() != pw or (pw not in FIXED_BODY_KEYS and _vocab().get(ph.lower()))):
                        # the body table's keys (build, skin, marks, desc,
                        # state, race) are not parts: their entries are
                        # whole tags already ('mole', 'forehead mark')
                        # a phrase that IS a tag ('eyelashes' under face)
                        # is emitted whole: anchored, 'eyelashes face'
                        # mapped to 'scar on face'
                        _whole = pw in FIXED_BODY_KEYS or bool(_vocab().get(ph.lower()))
                        if _whole:
                            cc = ph
                        else:
                            cc = ph if pw in ph.lower() else ph + " " + pw
                        out.append(cc)
                        note([cc], "subject", si)
                        if not _whole:
                            anchors[cc] = pw
        else:
            out += [x for x in body if x]
            note([x for x in body if x], "subject", si)
        # AGE IS NL-ONLY: fuzzy retrieval on age words reached 'adult baby'
        # (a fetish tag) -- the alias noun carries maturity in the NL part,
        # and no tag mapping is worth that failure class.
        _rv = str(s.get("race") or "").strip()
        if _rv and _rv.lower() not in ("null", "none", "human"):
            _rt = race_tag(_rv, s.get("kind"))
            if _rt:
                out.append(_rt)
                note([_rt], "subject:race", si)
        for key in ("occupation", "fashion_style"):
            val = str(s.get(key) or "").strip()
            if val and val.lower() not in ("null", "none"):
                if key == "occupation":
                    # a role without a booru tag stays in the prose only
                    val = occupation_tag(val)
                    if not val:
                        note([str(s.get(key))], "subject:role-without-tag", si)
                        continue
                out.append(val)
                note([val], "subject", si)
        for d in (s.get("descriptors") or []):
            d = str(d).strip()
            if d:
                out.append(d)
                note([d], "subject", si)
        # pose is a verb concept; a HELD OBJECT is part of the subject's
        # kit and belongs in their block ('backpack' had sorted into the
        # act band between 'standing' and 'sex')
        val = str(s.get("pose") or "").strip()
        if val and val.lower() not in ("null", "none"):
            # COMPOUND POSE DECOMPOSITION (the author's: 'sitting cross-legged
            # ... reaching up' lost the crossed legs and the reach, only
            # 'sitting' survived). Split into clauses so each maps on its
            # own -- one tag per fragment (single-pick).
            for frag in _pose_fragments(val):
                out.append(frag)
                action_concepts.add(frag)
                note([frag], "act", si)
                single.add(frag)
        val = str(s.get("held_object") or "").strip()
        if val and val.lower() not in ("null", "none"):
            cc = val if "holding" in val.lower() else "holding " + val
            out.append(cc)
            action_concepts.add(cc)
            note([cc], "subject", si)
        for a in (s.get("self_actions") or []):
            a = str(a).strip()
            if a and activity_tag(a) is None:
                note([a], "act:prose-only", si)     # said, never tagged
                continue
            if a:
                out.append(a)
                action_concepts.add(a)
                note([a], "act", si)
    n_subj = len(plan.get("subjects") or [])
    for k2, lk in enumerate(plan.get("secondaries_look") or []):
        lk = str(lk).strip()
        if lk:
            out.append(lk)
            note([lk], "subject", n_subj + k2)
    for e in (plan.get("interactions") or []):
        n_p = len(e.get("participants") or []) or 2
        # ONLY THE ACT MAPS TO A TAG. positions and secondary_contact are
        # NL-side choreography: mapping them produced 'player 2' (from
        # "subject 2") and misattributed touch tags. They stay in the story.
        val = str(e.get("act") or "").strip()
        if val and val.lower() not in ("null", "none"):
            out.append(val)
            action_concepts.add(val)
            group_map[val] = n_p
            note([val], "inter")
            single.add(val)
    for ef in (plan.get("effects") or [])[:2]:
        ef = str(ef).strip()
        if ef:
            out.append(ef)
            note([ef], "fx")
    # THE CLOCK OWNS THE LIGHT (the author, 2026-09-18: "sunlight and
    # moonlight... (at night)"). The engine resolves the time of day and
    # rolls the lighting under it, but the model writes its own lighting
    # words too, and nothing checked them against the hour -- a rooftop at
    # night came back lit by sunlight. A word that names the other half of
    # the day is dropped; everything else the model writes stands.
    _time_now = str(time_now or "").lower()
    _DAY_LIGHT = re.compile(r"\b(sunlight|sunbeam|sunshine|daylight|sunrise|sunset|dappled sunlight|"
                            r"golden hour|midday sun|morning light|afternoon light)\b")
    _NIGHT_LIGHT = re.compile(r"\b(moonlight|moonbeam|starlight|night light|nightglow)\b")
    for ld in (plan.get("lighting") or [])[:4]:
        ld = str(ld).strip()
        if not ld:
            continue
        if _time_now in ("night", "midnight") and _DAY_LIGHT.search(ld.lower()):
            continue
        if _time_now in ("day", "noon", "morning", "afternoon") and _NIGHT_LIGHT.search(ld.lower()):
            continue
        # ... and everything the two hand lists do not name, measured (the
        # author's bug hunt, 2026-09-21: 'morning' beside 'neon lights'). The
        # same one owner the sights, the effects and the occasion ask.
        if not fits_hour(ld, _time_now):
            continue
        out.append(ld)
        note([ld], "light")
    if plan.get("action"):
        # verb-anchored AND single-pick: "holding hands at a summer
        # festival" is the ACT 'holding hands'; the place rides the
        # 'setting' concept in the scene band, not the action's tail
        out.append(plan["action"])
        action_concepts.add(plan["action"])
        # ONE PERSON'S ACTION IS HERS (the author, 2026-09-27: '1girl dancing on
        # the rooftop at night' wrote 'dancing' after the subject, at the head of
        # the scene part): the picture's action is an interaction only when
        # there is someone to interact with; alone, it is the subject's own act
        # and stands in her part, as a self-action does
        _n_people = n_subj + len([x for x in (plan.get("secondaries_look") or []) if str(x).strip()])
        note([plan["action"]], "inter" if _n_people > 1 else "act", 0)
        single.add(plan["action"])
    if plan.get("setting"):
        # trim time/weather adjuncts: lighting owns time-of-day, and the
        # trailing adjunct hijacks the head anchor ("summer festival at
        # dusk" retrieved 'dusk' and lost the festival)
        sc = re.sub(r"\s+(?:at|in|under|during|before|after)\s+(?:the\s+)?"
                    r"(?:dawn|dusk|sunset|sunrise|night(?:fall)?|noon|"
                    r"morning|evening|midnight|twilight|golden hour|"
                    r"[a-z]+ light(?:ing)?)\b.*$", "",
                    str(plan["setting"]), flags=re.I).strip()
        sc = sc or str(plan["setting"])
        out.append(sc)
        note([sc], "scene")
    for _sd in (plan.get("scene_details") or []):
        _sd = str(_sd).strip()
        if _sd and _sd not in out:
            out.append(_sd)
            note([_sd], "scene")
    for pc in (plan.get("palette") or []):
        pc = str(pc).strip()
        if pc:
            cc = pc if any(w in pc.lower() for w in
                           ("color", "colour", "palette", "theme",
                            "monochrome", "greyscale", "sepia",
                            "pastel", "neon", "saturated", "muted",
                            "contrast", "colorful")) else pc + " colors"
            out.append(cc)
            # palette is STYLE-sphere (measured like artist styles are);
            # unbanded it defaulted to scene and mapped 'pink background'
            note([cc], "style")
    st = plan.get("style") or {}
    if st.get("name"):
        out.append(st["name"])
        note([st["name"]], "style")
    deco = [x for x in (st.get("decomposition") or []) if x]
    out += deco
    note(deco, "style")
    # mood is deliberately NOT mapped to tags: it is NL-native, and the
    # attempt mapped "calm and peaceful" to the MEME TAG 'keep calm and
    # carry on'. Atmosphere stays in the natural-language part.
    seen, uniq = set(), []
    for c in out:
        c = str(c).strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            uniq.append(c)
    return uniq, action_concepts, group_map, origins, anchors, single


# some act tags require a penis-haver without NAMING any anatomy, so the
# MALE_ANATOMY regex cannot see them ('prone bone' was injected into a
# 2girls scene). Fixed list until the co-occurrence table is measured
# (library phase).
_REQUIRES_PENIS = {
    "prone bone", "doggystyle", "missionary", "cowgirl position",
    "reverse cowgirl position", "boy on top", "mating press",
    "full nelson", "suspended congress", "piledriver (sex)",
    "pronebone", "irrumatio", "titfuck", "sex from behind",
    "upright straddle"}

# the closed SEX POSITION menu (the author's, studio round 7: 'girl on top'
# was findable and the NL described both girls on their backs -- nobody
# had DECIDED a position). Bridge 1 picks exactly one from this measured
# list for a sexual edge; the engine emits the tag and the NL is
# instructed to render every body consistently with it.
_SEX_POSITIONS = {
    "missionary", "girl on top", "cowgirl position",
    "reverse cowgirl position", "doggystyle", "prone bone", "spooning",
    "standing sex", "sex from behind", "upright straddle"}

# PAIRING IS AN ENGINE FACT: participant kinds x a sexual act decide it,
# no model choice involved (all measured, gelbooru-live: hetero 922k,
# yuri 336k, futa with female 41k...)
_PAIRING = {frozenset(("futanari", "female")): "futa with female",
            frozenset(("futanari", "male")): "futa with male",
            frozenset(("futanari",)): "futa with futa",
            frozenset(("female", "male")): "hetero",
            frozenset(("female",)): "yuri",
            frozenset(("male",)): "yaoi"}


# THE SPICE RESOLVER (the author's 2026-09-04: "spice at every level changes A
# LOT"): the injected spice comes from the tree's spice groups by SLOT and
# level (spice.json), not from every floor-table entry at the level -- that
# pool had grown to hold garments, places and a job word ('bra',
# 'bathtub', 'trial captain' at sensitive) once the night's floors landed.
# Which slots a level draws is stated once; every item's level is its
# measured floor; the partner gates are the measured one-boy / two-girls /
# two-boys shares; the weight is the count.
_CLOTHES_STATE_SECTIONS = frozenset((
    "chest", "torso", "legs", "ass", "shoulders", "exposed shoulders", "visible_parts_of_breasts",
    "whole_breasts_visible", "clothes_for_breasts", "breasts", "breastsparts", "points", "misc", "miscmore",
    "style", "accessories", "head", "other"))
_INJECT_SLOTS = {"sensitive": ("nudity", "wear", "toy"),
                 "nsfw": ("act", "nudity", "position", "toy"),
                 "explicit": ("act", "cum", "nudity", "position", "toy")}   # bdsm is typed only (2026-09-05)
# WHAT THE SPICE SHOULD BE MADE OF (the author, 2026-09-18: "I would like to
# increase the clothes state/undress/poses/actions influence in spice rather
# than toys/props -- those are ok but they should be much rarer and match to
# the action/pose better"). The slots are levelled first, so no slot wins on
# post count alone; then these say which kind of spice the picture is about.
_SLOT_PREF = {"act": 1.0, "nudity": 0.9, "position": 1.0, "cum": 1.0, "wear": 0.5, "toy": 0.2}
# A PROP BY ITS OWN WORDS: the gloss flags 'shorts pull' an object as readily
# as 'vibrator', so the flags cannot tell a prop from a clothes state. The
# toy slot names the props; these are the head words that give one away
# wherever the table files it ('condom' and 'lotion' sit under `act`).
_TOY_WORD = re.compile(r"\b(vibrator|dildo|condom|lube|lotion|plug|beads|onahole|magic wand|strap-on|"
                       r"sex toy|nipple clamps|stimulator|masturbat)\b")


def _partner_fit(c, cast):
    """the measured partner gate (the tree's related shares, 2026-09-05):
    a lone body needs an item its own posts show solo (a facial .23, a
    handjob none); a pair needs a penis-bearer when the one-boy share is
    high and the two-girls share low (cum .58/.08, handjob .55/.11;
    cunnilingus .44/.49 fits two women); a high two-girls share needs
    two females, a high two-boys share two males; a female-only word
    (pantyshot: one-girl .85, two-boys 0) never lands on a lone man"""
    m, f = float(c.get("male") or 0), float(c.get("female") or 0)
    g, b = float(c.get("girls") or 0), float(c.get("boys") or 0)
    solo = float(c.get("solo") or 0)
    fem = cast.get("female", 0) + cast.get("futa", 0)
    males = cast.get("male", 0)
    penis = males + cast.get("futa", 0)
    total = sum(cast.get(k, 0) for k in ("female", "male", "futa", "other"))
    if f >= 0.6 and b < 0.2 and fem < 1:
        return False
    if total <= 1:
        return solo >= 0.15
    # a penis-bearer is needed when the one-boy share is high outright
    # (fellatio .67, ffm threesome), when it is moderate without a
    # two-girls alternative (cum .58/.08), or when the men hide in the
    # multiple-boys share (spitroast, bukkake, foursome)
    mb = float(c.get("mboys") or 0)
    if penis < 1 and (m >= 0.6 or mb >= 0.4 or (m >= 0.4 and g < 0.25)):
        return False
    if g >= 0.6 and fem < 2:
        return False
    if b >= 0.6 and males < 2:
        return False
    return True


_FLOOR_SHARES = {"mtime": 0, "data": None}


def _floor_share(tag, level):
    """the measured share of the tag's posts rated AT the level (the floor
    table keeps the per-rating shares beside the floor)"""
    tb = _json_table(_FLOOR_SHARES, "safety_floor.json") or {}
    e = (tb.get("floors") or {}).get(str(tag).lower().strip()) or {}
    return float((e.get("share") or {}).get(level) or 0.0)


def _of_level(tag, level, least=0.10):
    """a ROLLED state is of the level when a real share of its posts is rated
    there or milder (the author, 2026-09-19: "'skin tight' as safe? weird").
    The floor is the mildest rating holding five percent -- right for a typed
    tag, which must never be over-restricted, too loose for a die: 'skin
    tight' is general on 8% of its posts and sensitive on 64%, so it starts
    at sensitive; 'untucked shirt' (65% general) stays safe. The line is ten
    percent (the author's choice over fifteen: 'bare shoulders' .115,
    'barefoot' .135 and 'midriff' .125 stay at safe, 'skin tight' .08, 'open
    shirt' .08 and 'bare back' .06 do not). Unmeasured shares decide nothing."""
    tb = _json_table(_FLOOR_SHARES, "safety_floor.json") or {}
    sh = ((tb.get("floors") or {}).get(str(tag).lower().strip()) or {}).get("share") or {}
    if not sh:
        return True
    rank = sm.SPICE_ORDER.get(level or "safe", 0)
    return sum(float(v) for k, v in sh.items() if sm.SPICE_ORDER.get(k, 9) <= rank) >= least


_ACT_REL = {"mtime": 0, "data": None}
_GLOSSES = {"mtime": 0, "data": None}
_LEVEL_UNIVERSE = {"safe": "rating:general", "sensitive": "rating:sensitive",
                   "nsfw": "rating:questionable", "explicit": "rating:explicit"}


def _dedupe_sentences(nl):
    """the prose with every repeated sentence (same words, case and
    punctuation aside) dropped after its first appearance"""
    if not nl or not isinstance(nl, str):
        return nl
    parts = re.split(r"(?<=[.!?])\s+", nl.strip())
    seen, out = set(), []
    for p in parts:
        key = re.sub(r"[^a-z0-9 ]+", "", p.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out)


def _gloss_flags(tag):
    """the gloss table's flags for a tag (empty when unglossed)"""
    tb = _json_table(_GLOSSES, "tag_glosses.json") or {}
    e = (tb.get("glosses") or {}).get(str(tag).lower().strip()) or {}
    return set(e.get("f") or []) if isinstance(e, dict) else set()


_ACT_KEYS = {"mtime": 0, "keys": None}


def _act_keys():
    """every measured act: the spice table's acts and positions plus every
    harvested key of the act table (the booru's pair tags, the activity
    registry) -- the view suffixes stripped"""
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    if _ACT_KEYS["keys"] is None or _ACT_KEYS["mtime"] != _ACT_REL.get("mtime"):
        sl = (_spice_table() or {}).get("slots") or {}
        keys = set(sl.get("act") or {}) | set(sl.get("position") or {})
        for k in (tb.get("acts") or {}):
            for suf in (" -pov", " pov"):
                if k.endswith(suf):
                    k = k[:-len(suf)]
                    break
            keys.add(k)
        _ACT_KEYS["keys"], _ACT_KEYS["mtime"] = keys, _ACT_REL.get("mtime")
    return _ACT_KEYS["keys"]


def act_of(tags):
    """the first measured act, position, pair tag or activity among the
    tags, else None; a spice act outranks an ordinary activity when both
    are present (the sex is the picture, the reading is beside it)"""
    sl = (_spice_table() or {}).get("slots") or {}
    spice = set(sl.get("act") or {}) | set(sl.get("position") or {})
    keys = _act_keys()
    found = None
    for t in tags or []:
        t2 = str(t).lower().strip()
        if t2 in spice:
            return t2
        if found is None and t2 in keys:
            found = t2
    if found is None:
        # a tag the gloss flags an ACT is the act even without measured
        # lifts ('spanking' is typed-only and so never harvested)
        for t in tags or []:
            t2 = str(t).lower().strip()
            if "act" in _gloss_flags(t2) and not t2.endswith("pov") and t2 not in sm.PAIRING_TAGS:
                return t2
        # ...and a typed tag whose GLOSS names an act is that act:
        # 'spanked' is 'a mark ... from a spanking'
        for t in tags or []:
            t2 = str(t).lower().strip()
            fl = _gloss_flags(t2)
            if not (fl & {"effect", "body", "state"}):
                continue
            g = str(_gloss_of(t2)[0] or "").lower()
            for w in re.findall(r"[a-z]+ing\b", g):
                if "act" in _gloss_flags(w) and _vocab().get(w):
                    return w
    return found


_POV_VIEWS = {"pov", "male pov", "female pov", "futanari pov"}
_GAZE_TAG_RE = re.compile(r"^(looking |eye contact$|sideways glance|staring|glancing)")
# words that make a phrase a mood or a manner, not a visible thing
_PHRASE_FILLER = set("""quiet quietly silence silent silently steady calm calmly serene serenity peaceful
peace mood atmosphere atmospheric feeling feel sense vibe vibes aura ambience ambiance tone tender tenderly
gentle gently soft softly subtle subtly slight slightly elegant elegance graceful grace intimate intimacy
dreamy dreamlike ethereal moody melancholy melancholic wistful nostalgic nostalgia timeless quiet stillness
contemplative thoughtful thoughtfully focused intently intent attentive relaxed casual casually natural
naturally effortless beautiful beauty lovely pretty stunning striking dramatic dramatically cinematic
epic captivating mysterious mystery enigmatic evocative poetic""".split())
_PHRASE_STOP = set("a an the of or and to in on with that which is are for by as from her his their its "
                   "she he they it very some more most".split())


def _act_rel(act, view=None):
    """the act's related table for a viewpoint: None = not yet known (the
    act alone); a pov view = the act through the viewer's eyes; any other
    view, or none rolled (''), = the act not in pov. Falls back to the
    act alone where the split is unharvested."""
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    acts = tb.get("acts") or {}
    key = str(act).lower().strip()
    if view is not None:
        key2 = key + (" pov" if str(view).lower().strip() in _POV_VIEWS else " -pov")
        return acts.get(key2) or acts.get(key)
    return acts.get(key)


def act_share(act, tag, view=None):
    """P(tag | act[, view]), the act's measured related share; 0 when unlisted"""
    return float((_act_rel(act, view) or {}).get(str(tag).lower()) or 0.0)


# WHAT THE ACT IS ABOUT, AT THIS BAND (the author, 2026-09-21: "a 'paizuri' act
# which is breasts focused act can roll tags like 'ass focus' alongside it").
# harvest_levels.py --table act_focus measures P(focus / framing / viewpoint |
# act, level) on the act's own posts; these two readers put it where the camera
# and the focus are decided, and fall back to the all-ratings act rows.
_ACT_ANCHORS = {"mtime": 0, "data": None}
_THING_PLACES = {"mtime": 0, "data": None}


def act_anchor(act):
    """-> {"anchor": the part the act is done to, "focus": the booru's focus
    word for it, "framings": {framing: share}, "views": {view: share}} or None.

    THE ACT HAS A BODY-PART ANCHOR, AND THE CAMERA LOOKS AT IT (the author,
    2026-09-21: "the boorus don't really give good measurements of
    focus/framing because it is used rarely there, but I would still prefer it
    to be used whenever the genrator is sure about it ... usually the act/group
    act has a body part anchor and that could also be used to decide were the
    camera should be looking (not making all the close ups and no full body
    shots but rather a main focus of the action)"). The anchor comes from his
    own rulings first, then the act's own words (write_act_anchors.py); the
    framings are measured where the data is dense -- on the focus word's own
    pictures, not on the act's rare focus tags."""
    tb = _json_table(_ACT_ANCHORS, "act_anchors.json") or {}
    e = (tb.get("acts") or {}).get(str(act or "").lower())
    if not e:
        return None
    fr = (tb.get("frames") or {}).get(e.get("focus") or "") or {}
    return {"anchor": e.get("anchor"), "focus": e.get("focus"),
            "framings": dict(fr.get("framings") or {}), "views": dict(fr.get("views") or {})}


def act_camera_share(act, tag, level=None):
    """-> P(tag | act) at this band from the act_focus table, or None"""
    row = level_row("act_focus_levels.json", act, level) if act else {}
    if not row:
        return None
    return float(row.get(str(tag).lower()) or 0.0)


def act_camera_lift(act, tag, level=None):
    """-> that share against the tag's own share at the band (a lift like
    act_lift's, but measured inside the band), or None where unmeasured"""
    sh = act_camera_share(act, tag, level)
    if sh is None:
        return None
    base = dict(_framing_shares(level)).get(tag)
    if base is None:
        base = dict(_viewpoint_shares(level)).get(tag)
    if not base:
        return None
    return sh / base


def act_lift(act, tag, level=None, view=None, live=False):
    """-> P(tag | act) / P(tag | the level's rating universe): what the act
    does to a tag's chance (fellatio: full body .36, upper body 2.2, pov
    1.6, pussy .37). None when the act or both sides are unmeasured; a
    side that lists neither the tag counts half its smallest listed share
    (below its 500th tag)."""
    if not act or not tag:
        return None
    tb = _json_table(_ACT_REL, "place_related.json") or {}
    rel = _act_rel(act, view)
    if not rel:
        return None
    unis = tb.get("universes") or {}
    uni = unis.get(_LEVEL_UNIVERSE.get(level or "", "")) or unis.get("1girl") or {}
    if not uni:
        return None
    t = str(tag).lower().strip()
    f, base = rel.get(t), uni.get(t)
    if live and (f is None or base is None):
        # BELOW THE LIST'S CUTOFF THE BOORU IS ASKED (cached counts): a
        # rare pose beside the act -- 'head back' under a pov fellatio is
        # .41 of its explicit rate, without pov 1.4
        _ask = _count_in_cache if live == "cache" else _count_cached
        if f is None:
            f = _pair_share(act, t, view, ask=_ask)
        if base is None:
            base = _level_base(t, level, ask=_ask)
        if f is None or base is None:
            return None
        return min(8.0, float(f) / max(float(base), 1e-6))
    if f is None and base is None:
        return None
    if f is None:
        f = 0.5 * min(rel.values())
    if base is None:
        base = 0.5 * min(uni.values())
    return min(8.0, float(f) / max(float(base), 1e-6))


_PAIR_COUNTS = {"mtime": 0, "data": None, "dirty": False, "last": 0.0}
_COUNT_REFUSED = set()   # queries the booru did not answer in this process


def _booru_count(query):
    """danbooru's post count for a tag search, None when offline"""
    import urllib.request
    import urllib.parse
    try:
        wait = 0.35 - (time.time() - _PAIR_COUNTS["last"])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            "https://danbooru.donmai.us/counts/posts.json?" + urllib.parse.urlencode({"tags": query}),
            headers={"User-Agent": "PromptStudio/1.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode("utf-8"))
        _PAIR_COUNTS["last"] = time.time()
        # an answer without a count is no answer (2026-09-14: a missing
        # 'counts' key was cached as 0 for good)
        _c = ((d or {}).get("counts") or {}).get("posts")
        return None if _c is None else int(_c)
    except Exception:
        _PAIR_COUNTS["last"] = time.time()
        return None


def _pair_counts():
    tb = _json_table(_PAIR_COUNTS, "act_pair_counts.json")
    if tb is None:
        _PAIR_COUNTS["data"] = {}
        tb = _PAIR_COUNTS["data"]
    return tb


def _count_cached(query):
    """a booru count, cached for good (a count only grows; the ratio is
    what is read, and both sides are cached together)"""
    tb = _pair_counts()
    if query in tb:
        return tb[query]
    # A PROMPT NEVER WAITS ON THE BOORU (the author, 2026-10-03: "make the remaining
    # lookups cache-only"). Inside generate() nothing is asked: a count the cache
    # does not hold reads as unmeasured -- neutral to every caller -- and is noted
    # as wanted; tools/build/harvest_wanted_counts.py asks for those afterwards, so
    # the rules still strengthen with use, off the user's clock. (danbooru answering
    # in a second, or with 503s, made single prompts take 10 to 130 s.)
    if _IN_DRAW["depth"] > 0:
        _COUNT_WANTED.add(query)
        return None
    if query in _COUNT_REFUSED:
        return None                    # asked once this run, no answer: not again
    n = _booru_count(query)
    if n is None:
        _COUNT_REFUSED.add(query)
        return None
    tb[query] = n
    try:
        with open(_paths.data("act_pair_counts.json"), "w", encoding="utf-8") as fh:
            json.dump(tb, fh, ensure_ascii=False, indent=0)
        _PAIR_COUNTS["mtime"] = os.path.getmtime(_paths.data("act_pair_counts.json"))
    except Exception:
        pass
    return n


_IN_DRAW = {"depth": 0}
_COUNT_WANTED = set()
_WANTED_MAX = 50000


def _flush_wanted():
    """the counts a prompt wanted and the cache did not hold -> count_wanted.json,
    for the harvest tool. The workshop only: a release gathers nothing (the same
    switch as the concept ledger)."""
    if not _COUNT_WANTED:
        return
    new = set(_COUNT_WANTED)
    _COUNT_WANTED.clear()
    try:
        from promptstudio.llm import config as _cfg
        if not (_cfg.load() or {}).get("ledger", True):
            return
    except Exception:
        pass
    try:
        p = _paths.data("count_wanted.json", expect=False)
        old = []
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                old = list((json.load(fh) or {}).get("wanted") or [])
        have = set(old)
        add = [q for q in sorted(new) if q not in have]
        if not add:
            return
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"_note": "booru counts a prompt wanted and the cache (act_pair_counts.json) did not hold; "
                                "asked for by tools/build/harvest_wanted_counts.py, never during a prompt.",
                       "wanted": (old + add)[-_WANTED_MAX:]}, fh, ensure_ascii=False, indent=0)
    except Exception:
        pass


def _count_in_cache(query):
    """the cached count only -- NEVER a request (the author, 2026-09-22: a
    measured gate that runs inside a draw must not cost a booru call per
    candidate; the harvests fill this cache, and the rule strengthens with
    them). None means 'not measured here', which every caller treats as
    neutral."""
    return (_pair_counts() or {}).get(query)


def _pair_lift_cached(a, b, universe="1girl"):
    """pair_lift off what is already known: the cached counts and the tree's
    own measured shares (_count_1girl reads both and asks nothing), plus the
    PAIR count, which only the cache can answer. None where the pair was never
    measured here -- which every caller treats as neutral."""
    try:
        tot = float(_count_in_cache(universe) or 0)
        nab = _count_in_cache(universe + " " + _q(a) + " " + _q(b))
        if not tot or nab is None:
            return None
        na, nb = _count_1girl(a, universe), _count_1girl(b, universe)
        if not (na and nb):
            return None
        return (nab / tot) / ((na / tot) * (nb / tot))
    except Exception:
        return None


def _q(tag):
    return str(tag).lower().strip().replace(" ", "_")


def _pair_share(act, tag, view=None, ask=None):
    """P(tag | act[, view]) from live counts; None when offline. `ask` is the
    count reader: _count_cached (asks the booru once) or _count_in_cache (never)."""
    ask = ask or _count_cached
    vt = "" if view is None else (" pov" if str(view).lower().strip() in _POV_VIEWS else " -pov")
    tot = ask(_q(act) + vt)
    if not tot:
        return None
    n = ask(_q(act) + vt + " " + _q(tag))
    return None if n is None else float(n) / float(tot)


def _level_base(tag, level=None, ask=None):
    """P(tag | the level's rating universe) from live counts; None offline"""
    ask = ask or _count_cached
    u = _LEVEL_UNIVERSE.get(level or "", "1girl")
    tot = ask(u)
    if not tot:
        return None
    n = ask(u + " " + _q(tag))
    return None if n is None else float(n) / float(tot)


# WHAT AN ACT NEEDS BARE ON THE PARTNER (the author's rulings, adopted_tags
# 'requires partner bare:<part>'): 'penis awe' is a face admiring an exposed
# penis, so a clothed scene cannot hold it. The part's own nudity words are the
# booru's, and one of them must be in the picture for the act to be drawn.
_BARE_WORDS = {"anus": ("anus", "ass", "nude", "completely nude", "bottomless", "anus peek", "spread anus",
                        "presenting own anus", "no panties", "panties aside", "clothing aside"),
               "clitoris": ("clitoris", "pussy", "nude", "completely nude", "bottomless", "spread pussy",
                            "no panties", "panties aside", "clothing aside", "clitoral hood"),
               "penis": ("penis", "erection", "penis out", "testicles", "nude", "completely nude", "bottomless",
                         "naked", "exposed penis"),
               "breasts": ("breasts out", "topless female", "nipples", "breast slip", "one breast out", "nude",
                           "completely nude", "topless", "bare breasts"),
               "pussy": ("pussy", "bottomless", "nude", "completely nude", "no panties")}


def act_needs_bare(tag):
    """-> the part an adopted act needs bare on the partner, or None"""
    tb = _json_table(_ADOPTED, "adopted_tags.json") or {}
    for c in ((tb.get("tags") or {}).get(str(tag or "").lower()) or {}).get("constraints") or []:
        if str(c).startswith("requires partner bare:"):
            return str(c).split(":", 1)[1]
    return None


def bare_words(part):
    return set(_BARE_WORDS.get(str(part or "").lower()) or ())


# A FUTANARI'S SEX IS THE MALE ONE (the author, 2026-09-21): "solo futanari gets
# a full description same as solo female (but with penis/testicles instead of
# pussy)", and beside a woman at nsfw or explicit she is described as fully as
# the woman is, acting "as a male would". So the woman's own parts are not hers
# and the man's are -- whatever the measured female share of the word says.
# ONLY HER OWN ANATOMY: `vaginal` names what is happening in the picture, not
# whose body it is, and a futanari can be the one doing it.
FEMALE_SEX_WORD = re.compile(r"\b(?:pussy|vulva|labia|clitoris|clit|cervix|cameltoe|"
                             r"pussy juice|female ejaculation|squirting)\b")
MALE_SEX_WORD = re.compile(r"\b(?:penis|penises|testicles?|foreskin|glans|erection|bulge|"
                           r"precum|smegma)\b")


# ... EXCEPT WHERE SHE RECEIVES (the author, 2026-09-21: "she can have pussy
# alongside penis when she is a receiver (no clitoris though - penis IS her
# clitoris)"). So the clitoris is never hers under any act, and the rest of the
# woman's words are hers only in a picture where something penetrates her.
FUTA_NEVER = re.compile(r"\b(?:clitoris|clit|clitoral)\b")


def sex_part_ok(tag, kind, receiver=False):
    """-> False when the word names the sex this subject does not have"""
    t = str(tag or "").lower()
    k = str(kind or "female").lower()
    if k == "futanari":
        if FUTA_NEVER.search(t):
            return False           # her penis IS her clitoris
        return bool(receiver) or not FEMALE_SEX_WORD.search(t)
    if k == "male":
        return not FEMALE_SEX_WORD.search(t)
    if k == "female":
        return True                # her partner's penis may be named in an act
    return True


_PENETRATION = {"key": None, "data": None}


def penetration_acts():
    """the booru's own words for a body being entered (filed with the penis
    rulings, 2026-09-21): one owner for 'is anyone being penetrated here'"""
    u = (_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("penis") or {}
    return set(u.get("penetration_acts") or ())


def futa_receives(act=None, acts=(), cast=None, things=()):
    """-> True when this picture can put a futanari on the receiving side: an
    act that enters a body, and something to enter with -- a partner with a
    penis (a man or another futanari) or a toy that stands in for one."""
    here = {str(a).lower() for a in (list(acts or ()) + [act or ""]) if a}
    if not (here & penetration_acts()):
        return False
    c = cast or {}
    others = int(c.get("male") or 0) + max(0, int(c.get("futa") or 0) + int(c.get("futanari") or 0) - 1)
    toys = {"dildo", "sex toy", "vibrator", "onahole", "anal beads", "strap-on", "tentacles"}
    return bool(others) or bool(toys & {str(x).lower() for x in (things or ())})


# A POSE ON SOMETHING NEEDS THAT SOMETHING IN THE PICTURE (the author,
# 2026-09-21, the 'on ...' / 'against ...' family). The pose TABLE has measured
# this since the surfaces were adopted -- but the fast pool draws by flag alone,
# and that is how 'on bed' reached a street and 'on lap' a lone girl. Both draws
# now ask the same question.
_ON_AGAINST = re.compile(r"^(?:on|against|standing on|sitting on|lying on|leaning on)\s+(.+)$")
_ON_PERSON = re.compile(r"\b(?:person|another|lap|head|shoulder|face|animal|boy|girl)$")
# the body's own position or its own legs: no surface is named
_ON_OWN = {"on back", "on side", "on stomach", "on all fours", "on one knee", "on tiptoes",
           "standing on one leg", "standing on three legs", "on top", "on knees"}


# a building HAS these, named or not: they need no line in the scene details
_ON_STRUCTURE = {"wall", "window", "door", "ceiling", "stairs", "railing", "glass", "roof", "rooftop"}


def pose_context_ok(tag, scene=(), partner=False, place=None, loc_kind=None):
    """-> False when the word names a surface or a person the picture has not got"""
    t = str(tag or "").lower()
    if t in _ON_OWN:
        return True
    m = _ON_AGAINST.match(t)
    if not m:
        return True
    thing = m.group(1).strip()
    here = " ".join(str(x or "").lower() for x in list(scene or ()) + [place or ""])
    if _ON_PERSON.search(thing):
        return bool(partner) or thing in here      # a cat on the head is in the scene
    if thing in ("object", "objects", "furniture", "any"):
        return bool([x for x in (scene or ()) if x])    # anything of the place will do
    if thing in ("floor", "ground"):
        return True                                # under every body, indoors and out
    if thing in _ON_STRUCTURE and str(loc_kind or "") in ("indoor", "building"):
        return True                                # a room has walls and windows
    return thing in here


def act_cast_ok(tag, cast, ctx=None):
    """THE CAST A RULING ASKS FOR (the author, 2026-09-20, the breasts page):
    'requires partner part:breasts' wants someone with breasts beside the
    subject, 'requires both:breasts' two of them, 'requires opposite gender' a
    woman and a man, 'requires visible:nipples' the nipples seen. -> True when
    this picture can hold the act."""
    tb = _json_table(_ADOPTED, "adopted_tags.json") or {}
    cons = set(((tb.get("tags") or {}).get(str(tag or "").lower()) or {}).get("constraints") or [])
    if str(tag or "").lower() in anus_tags():
        cons = cons | {"requires visible:anus"}
    if not cons:
        return True
    c, x = cast or {}, ctx or {}
    fem = int(c.get("female") or 0) + int(c.get("futa") or 0)
    males = int(c.get("male") or 0)
    penis = males + int(c.get("futa") or 0)
    total = sum(int(c.get(k) or 0) for k in ("female", "male", "futa", "other"))
    if "requires partner" in cons and total < 2:
        return False
    if "requires partner part:breasts" in cons and fem < 1:
        return False
    if "requires both:breasts" in cons and fem < 2:
        return False
    if "requires opposite gender" in cons and not (fem >= 1 and males >= 1):
        return False
    if "requires partner bare:penis" in cons and penis < 1:
        return False
    # AN ACT OF ITS OWN (the author, 2026-09-20: 'ass ripple' / 'ass shake' /
    # 'bouncing ass' "require action")
    if "requires act" in cons and not x.get("act"):
        return False
    # ... a penis somewhere, or a toy that stands in for one (anal, an onahole)
    if "requires penis or dildo" in cons or "requires penis" in cons:
        _toys = {"dildo", "sex toy", "vibrator", "onahole", "anal beads", "strap-on"}
        _here = {str(y).lower() for y in (x.get("bare") or ())} | {str(x.get("act") or "").lower()}
        if penis < 1 and not (("requires penis or dildo" in cons) and (_toys & _here)):
            return False
    for w in ("nipples", "breasts", "penis", "pussy", "anus", "clitoris"):
        if ("requires visible:" + w) in cons:
            seen = {str(y).lower() for y in (x.get("bare") or ())}
            if not (bare_words("breasts" if w == "nipples" else w) & seen) and not x.get("nude"):
                return False
    return True


def anus_tags():
    return set(((_json_table(_UNROLLED_RULINGS, "unrolled_rulings.json") or {}).get("anus") or {})
               .get("requires_visible") or ())


def act_bare_satisfied(tag, worn_or_states):
    """the picture already shows that part bare"""
    part = act_needs_bare(tag)
    if not part:
        return True
    return bool(bare_words(part) & {str(x).lower() for x in (worn_or_states or ())})


CLOTHES_STATE = "#clothes state"      # the injector's stand-in for "a state of what is worn" (the clothes roll picks it)
_SPICE_COUNTS = {"mtime": 0, "data": None}


def _spice_cast_key(cast):
    """the harvested cast a picture's people read from: a lone woman 'solo', a man
    and a woman '1boy 1girl', women only '2girls'; anything else unmeasured"""
    c = cast or {}
    w, fu = int(c.get("female") or 0), int(c.get("futa") or 0)
    m = int(c.get("male") or 0)
    total = w + fu + m + int(c.get("other") or 0)
    # the women-only casts were counted without a penis in frame
    # ('solo -penis', '2girls -penis -1boy'): a lone man or a futa is not them
    if total == 1 and w == 1:
        return "solo"
    if m >= 1 and (w + fu) >= 1:
        return "1boy 1girl"
    if w >= 2 and total == w:
        return "2girls"
    return None


def spice_count_table(level, cast):
    """-> {tag: count} of spice_counts_levels.json for this band and cast, or None"""
    key = _spice_cast_key(cast)
    if not key:
        return None
    tb = _json_table(_SPICE_COUNTS, "spice_counts_levels.json") or {}
    row = ((tb.get("levels") or {}).get(str(level or "").lower()) or {}).get(key)
    return row or None


def spice_count(tag, level, cast):
    """-> count('<tag> <cast> rating:<band>') from the harvest, or None"""
    row = spice_count_table(level, cast)
    if row is None:
        return None
    n = row.get(str(tag or "").lower())
    return None if n is None else int(n)


def spice_injectables(level, cast, census, banks, rng, n=8, act=None, ctx=None):
    """-> up to n spice-table tags AT the level for this cast, count-weighted,
    drawn without replacement; [] when the table has nothing (the caller
    falls back to the floor-table pool)"""
    tb = _spice_table() or {}
    slots = tb.get("slots") or {}
    total = max(1, int(census.get("total", 1) or 1))
    cands = []
    # AT the level means the tag's posts are rated there in a real share:
    # 'nude' has a sensitive FLOOR (five percent of its posts) but lives
    # at explicit (.59); the share is the weight, and under .15 the tag
    # is not of the level at all
    _gstems = _garment_stems()
    _measured9 = {}
    _cs_mass9, _cs_meas9 = 0.0, False
    # (a mild state -- cleavage, open clothes -- shows nothing: only the states in
    # which the part is seen weigh for the clothes roll's turn)
    _seen_inj9 = seen_states("female" if (cast.get("female", 0) + cast.get("futa", 0)) else "male")
    for sl in _INJECT_SLOTS.get(level, ()):
        for t, c in (slots.get(sl) or {}).items():
            fl = c.get("floor")
            if not fl or not _floor_measured(t) or \
                    sm.SPICE_ORDER.get(fl, 99) > sm.SPICE_ORDER.get(level, 0):
                continue
            # A STATE THAT NAMES A GARMENT IS THE CLOTHES ROLL'S (2026-09-15:
            # 'bra pull' injected on no bra, 'open kimono, kimono pull, panty
            # pull' on no kimono and no panties): the clothes roll draws the
            # nudity slot's garment states over the garments it knows are
            # worn; the injector keeps the whole states (nude, topless)
            if t in ruled_out():
                continue                          # the never and typed-only words (2026-09-16: 'big belly')
            _cs9 = False
            if sl == "nudity":
                # THE UNDRESS IS THE OTHER HALF OF THE SPICE (the author,
                # 2026-09-18: "the prompt is rather tame for nsfw and I see
                # 'vibrator/egg vibrator' a lot ... other tags (clothes
                # states/undress/poses/actions) rarely fire"). Measured: for a
                # lone woman at nsfw the table offered 5 acts and 9 toys, so
                # 95 of 100 rolls carried a prop. The whole states come back
                # -- never the garment-named ones, which are the clothes
                # roll's ('bra pull' on no bra) -- and the slots are levelled
                # below, so no one slot wins by sheer post count, which is
                # what made 'nude' take 62% of draws in 2026-09-15.
                head = re.split(r"\s+(?:on|around|over)\s+", t)[0].split()
                if head and head[-1] in _gstems:
                    _cs9 = True
                # ... and a state the ruling ties to worn clothes, or that names
                # a garment anywhere ('bikini top only', 'strap between breasts'
                # on nobody's strap): the clothes roll's too
                if _adopted_requires(t)[0] or any(_garment_stem(w) in _gstems for w in t.split()):
                    _cs9 = True
                # ... and every state the booru files under a garment's part of the
                # body (2026-09-19: 'center opening' -- a garment open down the
                # middle -- injected onto a sweater, which the model then read as a
                # place): the sections of worn clothes are the clothes roll's
                if str(c.get("section") or "") in _CLOTHES_STATE_SECTIONS and not _is_whole_nudity(t):
                    _cs9 = True
                # THE CLOTHES ROLL'S STATES STILL WEIGH IN THE DRAW (the author,
                # 2026-10-03: "a lot of 'see-through clothes' ... too little other
                # undress / exposure options"; measured, 55 nsfw subjects: 'topless
                # female' on 51%, 'areola slip' 5%, 'breasts out' 2%, 'no panties'
                # none -- on a lone woman's nsfw posts 16%, 18%, 9%, 11%). Skipped
                # outright, they left the whole states as the only undress the level
                # could spend itself on. Before the outfit exists they are weighed
                # together as ONE candidate; when it wins nothing is injected, and
                # the clothes roll shows the part with the state that fits what is
                # worn. Without that step ahead (the late net) they stay skipped.
                if _cs9 and not ((ctx or {}).get("pre") and t in _seen_inj9):
                    continue
            share = _floor_share(t, level)
            if share < 0.15:
                continue
            if not act_cast_ok(t, cast, ctx):
                continue
            if act_needs_bare(t) and not act_bare_satisfied(t, (ctx or {}).get("bare") or ()):
                # ... unless the picture is undressed anyway at this level
                if not (ctx or {}).get("nude"):
                    continue
            if sl == "nudity" and not naked_outfit_has_cause(
                    t, (ctx or {}).get("place"), (ctx or {}).get("genre"), (ctx or {}).get("occupation"),
                    (ctx or {}).get("scene") or ()):
                continue
            if int(c.get("arity") or 1) > total:
                continue
            if not _partner_fit(c, cast):
                continue
            if not sm.content_allowed(t, census, level, banks):
                continue
            # THE CAST'S OWN POSTS AT THE BAND (the author, 2026-09-27, "run the
            # harvest"): count('<tag> <cast> rating:<band>') times the share of the
            # item's posts at the band -- how often this cast shows it here, and
            # how much it carries the level. It replaces the all-cast sqrt(posts)
            # and the floor factor below, which made 'lactation' (4,767 solo nsfw
            # posts, nsfw floor) outdraw 'grabbing own breast' (6,892, sensitive
            # floor) for a lone woman.
            _mc = spice_count(t, level, cast)
            if _mc is not None:
                w = float(_mc) * share
                _measured9[t] = True
            else:
                w = float(c.get("posts") or 1) ** 0.5 * share
                # THE ITEM MUST CARRY THE LEVEL (2026-09-18): the pool admits
                # anything whose floor is at or below the target, so 'barefoot'
                # (safe) competed with 'topless female' (nsfw) for an nsfw
                # picture, and the prompt stayed tame while the injector thought
                # its work was done. An item rated AT the level counts full; one
                # rated below counts less the further below it sits.
                w *= {0: 1.0, 1: 0.45, 2: 0.2}.get(
                    sm.SPICE_ORDER.get(level, 0) - sm.SPICE_ORDER.get(fl, 0), 0.1)
            if sl == "nudity" and ctx:
                # THE CONTEXT LEANS THE UNDRESS (2026-09-15): the measured
                # lift of the state against the 1girl universe by place,
                # genre and occupation -- nude at the onsen, not the hospital
                w *= _ctx_lift(t, _ctx_rows(ctx.get("place"), ctx.get("genre"), ctx.get("occupation")))
            # A PROP IS A PROP WHEREVER IT SITS (2026-09-18): the table files
            # 'condom' and 'lotion' under `act`, and they took 43 of the
            # act slot's 112 weight for a lone woman. The gloss says which
            # tags are objects; those follow the toy slot's preference and
            # its act-fit test.
            if sl != "toy" and _TOY_WORD.search(t):
                sl = "toy"
            # A PROP NEEDS ITS ACT (the author, 2026-09-18: toys "should be much rarer
            # and match to the action / pose better"; 2026-09-19 measured: with the
            # clothes states and the lingerie rightly kept off a dressed subject, a
            # fully dressed SENSITIVE girl got 'lotion' or 'implied vibrator' in four
            # prompts of twelve -- the toy slot was what was left). With no act in
            # the scene a toy is not what carries the level.
            if sl == "toy" and not act:
                continue
            if _cs9:
                _cs_mass9 += w
                if _measured9.pop(t, None):
                    _cs_meas9 = True
                continue
            cands.append((t, w, sl))
    # THE SLOTS ARE LEVELLED (2026-09-18): each slot the level draws from
    # contributes the same total weight, and the items inside it keep their
    # measured shares. Otherwise the slot with the most posts owns the draw:
    # toys 61% against acts 39% for a lone woman at nsfw, and a prompt whose
    # only spice was a bottle of lotion.
    # WHERE THE TABLE SPEAKS FOR THIS CAST AND BAND, an item it could not count
    # stands at the measured median (unmeasured means neutral), not on the old
    # scale, which is a different unit
    if _cs_mass9 > 0:
        cands.append((CLOTHES_STATE, _cs_mass9, "nudity"))
        if _cs_meas9:
            _measured9[CLOTHES_STATE] = True
    if spice_count_table(level, cast) is not None and _measured9:
        _mw = sorted(w for t, w, _s in cands if t in _measured9)
        _med9 = _mw[len(_mw) // 2] if _mw else 1.0
        cands = [(t, (w if t in _measured9 else _med9), sl) for t, w, sl in cands]
    _by_slot = {}
    for t, w, sl in cands:
        _by_slot.setdefault(sl, []).append((t, w))
    _mass = {sl: sum(w for _, w in v) or 1.0 for sl, v in _by_slot.items()}
    _top = max(_mass.values()) if _mass else 1.0
    # ... HALFWAY, ON A LOG SCALE (the author, 2026-09-27): lifted all the way, a
    # thin slot handed its few items the weight of a full one -- a lone woman's
    # act slot made 'lactation' (4,767 solo nsfw posts, nude 136,820) the first
    # spice of a fifth of her pictures. A slot now rises by the square root of
    # the gap: acts and undress still outweigh their raw counts, a rare item
    # no longer inherits a whole slot.
    cands = [(t, w * (_top / _mass[sl]) ** 0.5 * _SLOT_PREF.get(sl, 1.0), sl) for t, w, sl in cands]
    # A PROP MUST FIT WHAT IS HAPPENING (the author, same ruling): with an act
    # in hand, a toy is kept only where the booru shows the two together at
    # least as often as chance -- a lotion bottle beside a kiss is a prop the
    # picture has no use for. Without an act, the prop IS the act and stays.
    if act:
        cands = [(t, w, sl) for t, w, sl in cands
                 if sl != "toy" or (act_lift(act, t, level) or 1.0) >= 1.0]
    cands = [(t, w) for t, w, _sl in cands]
    out = []
    # THE ACT LEADS (measured, 2026-09-06): once an act is in hand -- typed,
    # or the first act drawn -- every other candidate follows its lift
    # beside that act, so a face act draws no pussy-side nudity
    _acts = set(slots.get("act") or {}) | set(slots.get("position") or {})
    if act:
        cands = [(x, w * (act_lift(act, x, level) or 1.0)) for x, w in cands]
    while cands and len(out) < n:
        t = _wroll(rng, cands)
        out.append(t)
        cands = [(x, w) for x, w in cands if x != t]
        if not act and t in _acts:
            act = t
            cands = [(x, w * (act_lift(act, x, level) or 1.0)) for x, w in cands]
    return out


def spice_positions(cast, allowed=None):
    """-> {position: count} from the spice table's position slot for this
    cast (measured partner gate, measured floor); {} without a table"""
    tb = _spice_table() or {}
    items = ((tb.get("slots") or {}).get("position")) or {}
    return {t: float(c.get("posts") or 1) for t, c in items.items()
            if _floor_measured(t) and _partner_fit(c, cast)
            and (allowed is None or allowed(t))}


def sex_position_names():
    """the typed-detection set: the hand set plus the table's positions,
    longest first so 'reverse cowgirl position' is not read as 'cowgirl
    position'"""
    tb = _spice_table() or {}
    names = set(_SEX_POSITIONS) | set(((tb.get("slots") or {}).get("position")) or {})
    return sorted(names, key=len, reverse=True)


def draw_injectables(level, cast, mode, banks, rng, want, act=None, ctx=None):
    """anatomy-gated draw of `want` floor-measured tags AT the level --
    used PRE-bridge-1 (the author's: the story incorporates the spice) and by
    the post-verification safety net. The spice table answers first; the
    floor-table pool is the fallback."""
    census = sm.subject_census(count_tags(cast, mode))
    has_penis = (cast.get("male", 0) + cast.get("futa", 0)) > 0
    has_fem = (cast.get("female", 0) + cast.get("futa", 0)) > 0
    out = []
    _pairing = sm.PAIRING_TAGS          # one definition, in slots
    _tbl = spice_injectables(level, cast, census, banks, rng, act=act, ctx=ctx)
    _left9 = int(want)
    for _ in range(8):
        t = _tbl.pop(0) if _tbl else pe._injectable(level, census, banks, rng)
        if not t or t in out:
            continue
        # 'a state of what is worn' won this turn: the clothes roll will show the
        # part with the state that fits the outfit, and nothing is injected for it
        if t == CLOTHES_STATE:
            _left9 -= 1
            if len(out) >= _left9:
                break
            continue
        # THE FALLBACK POOL IS RULED TOO (2026-09-16: 'stationary
        # restraints', a typed-only bdsm word, was injected when the spice
        # table had nothing left): the floor table is a vocabulary, not a
        # permission -- ruled_out() gates every emitter, this one included
        if t in ruled_out():
            continue
        # A PAIRING IS NEVER INJECTED. 'futa with female' is a floor-
        # measured explicit tag, so the draw offered it to a cast with no
        # futa in it (10 of 60 seeds for a lone woman in a bar). Who is
        # paired with whom is derived from the cast by _PAIRING and
        # nowhere else; the injection may add an act, not a partner.
        if t in _pairing:
            continue
        if not has_penis and (t in _REQUIRES_PENIS or
                              sm.MALE_ANATOMY.search(t)):
            continue
        if not has_fem and sm.FEMALE_ANATOMY.search(t):
            continue
        # NAMED for a sex is a stricter test than HAVING the anatomy: a
        # futa has a penis but is not male, so `male ...` needs a man
        if sm.MALE_NAMED.search(t) and not cast.get("male", 0):
            continue
        if sm.FEMALE_NAMED.search(t) and not has_fem:
            continue
        if sm.FUTA_NAMED.search(t) and not cast.get("futa", 0):
            continue
        # A GARMENT THE PICTURE WOULD NOT WEAR (the author's bug hunt,
        # 2026-09-21: an injected 'garter belt' over a 'string bikini' at the
        # beach, pair lift .63). The clothes draw asks this of its own picks;
        # the injection adds words after it, and must ask the same.
        try:
            _worn_now = [str(x) for x in ((ctx or {}).get("worn") or ())]
            if _worn_now and (_is_garment(t) or implies_clothing(t)) and not _pairs_with_worn(
                    t, _worn_now, (_clothes_table() or {}).get("slots") or {}):
                continue
        except Exception:
            pass
        out.append(t)
        if len(out) >= _left9:
            break
    return out


# MUTUAL GAZE, FIXED MAP (map_age precedent): fuzzy retrieval turned
# "gazing at each other" into 'other focus' on the strength of the word
# "other". The correct tag is one specific tag; a whitelist beats retrieval.
_GAZE_RE = re.compile(
    r"\b(?:gaz\w*|look\w*|star\w*|stare\w*) (?:at|into) "
    r"(?:each other|one another|each other's \w+)|eye contact", re.I)


def bridge2(concepts, verb_concepts=frozenset(), origins=None,
            anchors=None, single=frozenset(), block=frozenset(),
            comic_ok=False, use_llm=True):
    # PICK DISCIPLINE, layer 1: confident retrievals are the ENGINE's call;
    # only genuinely ambiguous concepts reach the model at all.
    # `block`: cast-derived stopwords -- a persona's name words must not
    # become general tags ('Cloud Strife' retrieved the SKY tag 'cloud'),
    # and nobody in the scene is a cosplayer of themselves.
    anchors = anchors or {}

    def _blocked(t):
        if t in block or t.endswith("(cosplay)"):
            return True
        if t in ruled_out():
            return True                 # the never and typed-only words are never a mapping either
        # the aircraft-'fighter' family keeps mutating (fighter jet,
        # variable fighter, piston engine fighter) -- block the CLASS
        if "fighter" in t and t != "fighter":   # every mutation so far
            return True     # (jet/variable/piston engine/TIE fighter);
            # the REAL fix is tag categories+glosses -- library phase 1
        # PHASE 2 FLAGS AS CONTEXT GATES (the author's: 'total blocking is
        # too restrictive'). Hard walls only for what can never help:
        # memes/in-jokes, text, and unprompted panel FORMATS. Emotes and
        # symbols ('!' over a surprised head) are legitimate vocabulary
        # WHEN the resolved style/genre is comic-adjacent -- the dice
        # rolling a chibi style legalizes them for that prompt.
        fl9 = _gloss_of(t)[1]
        if fl9:
            if "meme" in fl9 or "text" in fl9 or "format" in fl9:
                return True
            if not comic_ok and ("emote" in fl9 or "symbol" in fl9):
                return True
        # ... and a tag whose head word is a picture format is of the picture's
        # frame too (2026-10-03: the model's 'border of the forest' mapped to
        # `outside border` -- a figure overlapping the image's border)
        _hd9 = t.split()[-1] if " " in t else ""
        if _hd9:
            _hf9 = _gloss_flags(_hd9) or set()
            if _hf9 and _hf9 <= {"format", "meme", "text"}:
                return True
        return t == "nose hook"    # a bondage device, not a nose shape

    def _in_slot(t, anchor):
        """slot membership for the strict one_of families. 'focus' tags
        are CAMERA facts, not descriptions -- 'intense dark brown eyes'
        must not become 'eyes focus'."""
        if "focus" in t:
            return False
        if anchor == "hair":
            return "hair" in t or t in _hair_vocab()
        if anchor == "eyes":
            return "eyes" in t or "heterochromia" in t
        return anchor.split()[0].lower() in t

    direct, ambiguous = {}, []
    # REVIEWED DECOMPOSITIONS FIRST (the concept ledger's integrate step,
    # 2026-09-11): a concept the review resolved into its visible parts
    # maps to those tags directly, no retrieval, no model
    try:
        from promptstudio.library import concepts as _cl
        _decomp = _cl.decompositions()
        _subj = _cl.subjective_words()
    except Exception:
        _decomp, _subj = {}, set()
    # SUBJECTIVE WORDS ARE MODIFIERS, NEVER TAGS (the author's 2026-09-15, the
    # booru's tag group:subjective): a concept carrying one is the user's
    # own phrase ('sexy girl', 'sexy smile') and is never mapped -- a
    # mapping could only drop the modifier ('sexy smile' -> 'smile'), which
    # is the one word the user added; and 'sexy' alone had mapped to
    # 'sexy no jutsu'. Held out here, returned unmapped: a phrase on the line.
    _subj_held = [c for c in concepts if any(w in _subj for w in str(c).lower().split())]
    concepts = [c for c in concepts if c not in _subj_held]
    for c in concepts:
        _dc = _decomp.get(str(c).lower().strip())
        if _dc:
            direct[c] = list(_dc)
            continue
        if _GAZE_RE.search(c):
            direct[c] = "eye contact"
            continue
        aw = anchors[c].split()[0].lower() if c in anchors else None
        t = confident_pick(c, verb_first=c in verb_concepts)
        ok9 = t and not _blocked(t) and \
            (aw is None or (_in_slot(t, anchors[c]) and t != aw))
        if ok9 and aw and anchors[c] not in ("hair", "eyes") and \
                any(b in t for b in _NOT_DESCRIPTIVE):
            ok9 = False       # body slots take descriptive tags only
        # SPHERE GATE ON CONFIDENT PICKS TOO: 'gentle glow reflecting off
        # the book spines' confident-picked 'spines' -- scoping that only
        # guards the ambiguous path guards half the door
        if ok9 and origins:
            band9 = origins.get(str(c).lower(), ("", 0))[0]
            if band9 == "light" and t not in _light_vocab():
                ok9 = False
            elif band9 == "style" and t not in _style_vocab():
                ok9 = False
        if ok9:
            direct[c] = t
        else:
            ambiguous.append(c)

    cands = {c: [t for _, t in
                 candidates_for(c, verb_first=c in verb_concepts)[:5]
                 if not _blocked(t)]
             for c in ambiguous}
    for c in list(ambiguous):
        # '(style)' tags are STYLE-sphere material only: an artist-style
        # tag must not be reachable from a descriptor like 'cute'
        if (origins or {}).get(str(c).lower(), ("", 0))[0] != "style":
            cands[c] = [t for t in cands[c] if not t.endswith("(style)")]
        if c in anchors and cands[c]:
            strict = anchors[c] in ("hair", "eyes")
            if strict:
                cands[c] = [t for t in cands[c]
                            if _in_slot(t, anchors[c]) and t != anchors[c]]
            else:
                cands[c] = _slot_scope(cands[c], anchors[c], strict=False)
            if anchors[c] == "hair":
                cands[c] = [t for t in cands[c]
                            if not any(b in t for b in _NOT_HEAD_HAIR)]
            if not strict:      # body-part slots: descriptive tags only
                cands[c] = [t for t in cands[c]
                            if not any(b in t for b in _NOT_DESCRIPTIVE)
                            and "view" not in _gloss_of(t)[1]]
            # ONE_OF SLOTS ARE THE ENGINE'S PICK: the shortlist is ranked
            # by retrieval and the 8B ignores prefer-first ('back hair'
            # over the rank-1 'purple hair') -- for hair and eyes the
            # top scoped candidate simply wins, no model choice
            if strict and cands[c]:
                direct[c] = cands[c][0]
                ambiguous.remove(c)
    picked = {}
    if any(cands.values()):
        def _fam(c):
            band = (origins or {}).get(str(c).lower(), ("scene", 0))[0]
            return _FAMILY_LABEL.get(band, "scene")
        # DEFECT #3 closed: style-origin concepts retrieve against the
        # STYLE sphere only -- 'simplified forms' must never reach
        # 'simplified chinese text'. Empty after scoping = NL-only, which
        # is the correct fate for a style phrase without a tag.
        sv = _style_vocab()
        lv = _light_vocab()
        # a typed leftover is never offered to the model or the ranker:
        # by definition no tag uses its words (2026-09-11)
        ambiguous = [c for c in ambiguous if c not in _LEFTOVER_CONCEPTS]
        for c in ambiguous:
            band = (origins or {}).get(str(c).lower(), ("", 0))[0]
            if band == "style":
                cands[c] = [t for t in cands[c] if t in sv]
            elif band == "light":
                # same doctrine as style (defect #3): lighting concepts
                # map inside the lighting sphere or stay NL-only
                cands[c] = [t for t in cands[c] if t in lv]
        # candidates travel WITH their glosses: the model picks by
        # MEANING instead of name similarity (the author's description
        # proposal, the pick-quality payoff)
        def _cline(t):
            g9 = _gloss_of(t)[0]
            if g9 and len(g9) > 140:      # word-boundary trim, never
                g9 = g9[:140].rsplit(" ", 1)[0] + "…"   # mid-word
            return "%s (%s)" % (t, g9) if g9 else t
        _log_pending([t for c in ambiguous for t in cands[c]
                      if not _gloss_of(t)[0]])
        lines = ["%s [family: %s] -> candidates (RANKED, best first): %s"
                 % (c, _fam(c), "; ".join(_cline(t) for t in cands[c])
                    or "(none)")
                 for c in ambiguous]
        # the answer budget scales with the ask: 700 tokens truncated the
        # JSON at ~20 concepts and every concept after the cut arrived
        # 'unmapped' with perfect candidates waiting ('white cotton
        # sweater' next to 'white sweater')
        if not use_llm:
            # FAST PATH: the shortlist is already ranked best-first, so the
            # engine takes rank 1 -- the same rule it applies to hair and
            # eyes above. The verifier below still checks every pick.
            picked = _fastplan.pick([c for c in ambiguous if c not in _LEFTOVER_CONCEPTS], cands)
            # THE MEANING GATE (2026-09-11): a rank-1 pick no model judged
            # must share a word with its concept ('moss-covered dais' had
            # become 'ao dai'); what fails stays a phrase on the line
            picked = {c: [t for t in v if _shares_stem(c, t)] for c, v in picked.items()}
        else:
            out = chat(BRIDGE2_SYSTEM, "\n".join(lines), temp=0.2,
                       max_tokens=min(1500, 250 + 45 * len(ambiguous)), schema=_MAP_SCHEMA, stage="mapping")
            try:
                picked = parse_json_block(out)
            except Exception:
                picked = {}

    # THE VERIFIER. Every mapping must be one of that concept's own
    # candidates; anything else is stripped and reported.
    tags, unmapped, stripped = [], [], []
    mapping = {}
    for c in concepts:
        if c in direct:
            _dv = direct[c] if isinstance(direct[c], list) else [direct[c]]
            tags.extend(_dv)
            mapping[c] = list(_dv)
            continue
        raw = picked.get(c)
        choices = (raw if isinstance(raw, list)
                   else [raw] if isinstance(raw, str) else [])
        ok = cands.get(c) or []
        hit = False
        # ONE PICK for one_of slots and interaction acts: the second pick
        # is where 'pink hair' joined 'blonde hair' on one girl and
        # 'summer festival' rode the interaction band out of order
        for ch in choices[:1 if c in single else 2]:
            if isinstance(ch, str) and ch.strip().lower() in ok:
                # THE MEANING GATE (2026-09-11): a pick
                # must carry the concept's meaning -- a shared stem, the
                # booru alias, a concept word in the tag's gloss, or a
                # slot-anchored concept (the anchor constrains the slot)
                if c not in anchors and not _meaning_ok(c, ch.strip().lower()):
                    stripped.append((c, ch.strip().lower() + " (no shared meaning)"))
                    continue
                tags.append(ch.strip().lower())
                mapping.setdefault(c, []).append(ch.strip().lower())
                hit = True
            elif isinstance(ch, str) and ch.strip():
                stripped.append((c, ch.strip()))
        if not hit:
            unmapped.append(c)
        # WORD-SUBSET DEDUP within one concept's picks: 'pastel blue
        # skirt' -> blue skirt + skirt says the same thing twice; the
        # more specific pick wins
        got = mapping.get(c) or []
        if len(got) == 2:
            w0, w1 = set(got[0].split()), set(got[1].split())
            drop = (got[1] if w1 <= w0 else got[0] if w0 <= w1 else None)
            if drop:
                got.remove(drop)
                tags.remove(drop)
    dedup = list(dict.fromkeys(tags))
    unmapped = list(unmapped) + [c for c in _subj_held if c not in unmapped]
    return dedup, unmapped, stripped, mapping


# --------------------------------------------------------------- assembly
def _quality_pool():
    with open(_paths.data("quality_pool.json"), encoding="utf-8-sig") as f:
        return json.load(f)


def _max_draw(q, rng):
    """the 'maximum' booster draw: 3-5 across the groups, honouring each
    group's max_draw cap and the per-tag weights (masterwork at half rate,
    near-synonym of the ever-present masterpiece)."""
    groups = q["pool"]["groups"]
    weights = q["pool"].get("weights", {})
    cands = [(t, g) for g, blob in groups.items() if g[:1] != "_"
             for t in blob["tags"]]
    n = rng.randint(q["draw"]["min"], q["draw"]["max"])
    drawn, per_group = [], {}
    while cands and len(drawn) < n:
        ws = [weights.get(t, 1.0) for t, _ in cands]
        t, g = cands.pop(rng.choices(range(len(cands)), weights=ws)[0])
        cap = groups[g].get("max_draw")
        if cap and per_group.get(g, 0) >= cap:
            continue
        per_group[g] = per_group.get(g, 0) + 1
        drawn.append(t)
    return drawn


def conditioning(mode, spice, rng, quality="standard", period=None):
    """Section 1 in README order: quality -> meta/resolution -> year ->
    safety. QUALITY is the user control (off/standard/maximum); SAFETY is
    1d and emits regardless of it; PERIOD is 1c, anima-only (newest,
    recent, mid, early, old, or a bare year)."""
    q = _quality_pool()
    out = []
    if quality != "off":
        out += list(q["base"][mode])
        if mode == "anima":
            out.append(rng.choice(q["score"]["anima"]))
    if quality == "standard":
        out += ["highres"] if mode == "anima" else \
            ["amazing quality", "very aesthetic", "absurdres"]
    elif quality == "maximum":
        out += _max_draw(q, rng)
    if mode == "anima":
        out.append(str(period or "newest"))
        if spice in sm.LEVELS:
            out.append(spice)
    else:
        out.append("newest")
    return out


_SPELL = None


def _spell_tables():
    """-> {"anima": {tag: preferred}, "illustrious": {tag: preferred}}

    the author's: "the resulting prompts both for anima and for illustrious
    should be the same but with proper formatting and content type for
    each". Same scene, same concepts -- each model addressed in the
    vocabulary it was actually trained on.

    Two sources of divergence, and both are resolved by EVIDENCE:

    1. tag_bridge.json's 60 explicit pairs. These are NOT a plain
       danbooru<->gelbooru dictionary: most of the danbooru side is a dead
       alias ('bound leg', 230 posts, against 13,411 for 'bound legs',
       which is also what gelbooru uses). Resolving them by post count
       instead of by side keeps the well-attested tag for both models and
       still splits the ~30 pairs that genuinely differ.

    2. GELBOORU'S DIALECT, which is far wider than the bridge: 2,682 tags
       gelbooru uses that danbooru does not, including some of the most
       common tags on the site ('sole female' 3.9M, 'bellybutton' 873k,
       'eyes closed' 540k). Left alone, typing 'bellybutton' put a word
       into the illustrious prompt that danbooru has never used. danbooru
       publishes its own alias table, so the canonical form is looked up
       there rather than guessed.
    """
    global _SPELL
    if _SPELL is not None:
        return _SPELL
    dan, gel = {}, {}
    for name, into in (("danbooru_tags.json", dan), ("gelbooru_tags.json", gel)):
        try:
            with open(_paths.data(name), encoding="utf-8-sig") as f:
                for k, v in json.load(f).items():
                    into[k.replace("_", " ").lower()] = v
        except Exception:
            pass
    try:
        from promptstudio.engine.aliases import booru_aliases
        alias = dict(booru_aliases())
    except Exception:
        alias = {}
    rev = {}
    for src, dst in alias.items():
        rev.setdefault(dst, []).append(src)

    out = {"anima": {}, "illustrious": {}}

    # 1. the explicit pairs, settled by each booru's own counts
    for a, b in (pe.load_all_banks().get("_bridge") or {}).items():
        a, b = str(a).lower(), str(b).lower()
        for mode, counts in (("anima", gel), ("illustrious", dan)):
            na, nb = counts.get(a, 0), counts.get(b, 0)
            if na == 0 and nb == 0:
                other = dan if mode == "anima" else gel
                na, nb = other.get(a, 0), other.get(b, 0)
            best = a if na >= nb else b
            if best != a:
                out[mode][a] = best
            if best != b:
                out[mode][b] = best

    # 2. the dialect. A tag the target booru has never used is rewritten to
    #    the equivalent it does use, when one is known.
    for t in set(gel) | set(dan):
        if t not in dan:                      # gelbooru-only -> danbooru form
            d2 = alias.get(t)
            if d2 and d2 in dan and t not in out["illustrious"]:
                out["illustrious"][t] = d2
        if t not in gel:                      # danbooru-only -> gelbooru form
            best, n = None, 0
            for src in rev.get(t, ()):
                if gel.get(src, 0) > n:
                    best, n = src, gel[src]
            if best and t not in out["anima"]:
                out["anima"][t] = best
    # 3. WORD ORDER. The two boorus write some pairs in opposite order --
    #    gelbooru 'eyes closed' (540k) is danbooru 'closed eyes' (1.05M) --
    #    and no alias entry connects them. Deliberately narrow: two words
    #    only, and the reversal must itself be a well-used tag in the
    #    target booru, which is what keeps it from inventing anything. All
    #    26 matches were checked by hand and are exact synonyms.
    #    GUARDED BY THE GLOSSES. Reordering can change the meaning outright
    #    -- 'eye black' is face paint, 'black eyes' is an eye colour -- so
    #    where both tags are glossed and the glosses describe different
    #    things, the pair is refused. This is the same evidence test
    #    tag_bridge.py applies after boxer/boxers got through its plural
    #    hole: a shape rule proposes, the glosses decide.
    _gl = {}
    try:
        with open(_paths.data("tag_glosses.json"), encoding="utf-8") as f:
            _gl = json.load(f)["glosses"]
    except Exception:
        pass
    _stop = set("a an the of or and to in on with that which is are for by "
                "as from person who someone something typically often "
                "usually used".split())

    def _gt(t):
        g = (_gl.get(t) or {}).get("g") or ""
        return {w for w in re.findall(r"[a-z]+", g.lower())
                if w not in _stop and len(w) > 2}

    def _same(a2, b2):
        A, B = _gt(a2), _gt(b2)
        if not A or not B:
            return True
        return len(A & B) / float(len(A | B)) >= 0.20

    for src, dst_counts, mode in ((gel, dan, "illustrious"),
                                  (dan, gel, "anima")):
        for t in src:
            if t in dst_counts or t in out[mode]:
                continue
            w = t.split()
            if len(w) != 2:
                continue
            r = w[1] + " " + w[0]
            if dst_counts.get(r, 0) >= 500 and _same(t, r):
                out[mode][t] = r

    _SPELL = out
    return _SPELL


_COUNTS = {"danbooru": {"mtime": 0, "data": None}, "gelbooru": {"mtime": 0, "data": None}}


def _counts(booru):
    tb = _json_table(_COUNTS[booru], "%s_tags.json" % booru) or {}
    return tb


def known_to(mode, tag):
    """-> True when the booru this model is prompted in has posts for the
    tag (anima: gelbooru or danbooru; illustrious: danbooru). True as well
    when the count files are missing -- nothing is dropped blindly."""
    t = str(tag).lower().strip().replace("_", " ")
    boorus = ("gelbooru", "danbooru") if mode == "anima" else ("danbooru",)
    seen_table = False
    for b in boorus:
        tb = _counts(b)
        if not tb:
            continue
        seen_table = True
        if (tb.get(t) or tb.get(t.replace(" ", "_")) or 0) > 0:
            return True
    return not seen_table


def spell_for(tag, mode, banks=None):
    """the same concept in the spelling THIS model has actually seen."""
    t = str(tag)
    return _spell_tables().get(mode, {}).get(t.lower(), t)


def negative_for(mode):
    """always standard, no user control (1a rule) -- the mirror of the
    same scale, defined once so the two ends cannot drift apart."""
    return ", ".join(_quality_pool()["negative"][mode])


def count_tags(cast, mode):
    from promptstudio.engine import scene as _scene
    return _scene.count_tags_for(
        {k: cast.get(k, 0) for k in ("female", "male", "futa", "other")}, mode)


# The male genital family. Named once because two rules need exactly the
# same list: the one that puts these in their owner's subject block, and
# the one that must NOT put them in somebody else's.
_GENITAL_FAMILY = ("penis", "erection", "huge penis", "large penis",
                   "small penis", "thick penis", "veiny penis",
                   "testicles", "precum", "erect penis")

_ACT_POSTURE = None


def _act_posture():
    """measured act -> poses that act excludes; {} when unmeasured"""
    global _ACT_POSTURE
    if _ACT_POSTURE is None:
        try:
            with open(_paths.data("act_posture.json"),
                      encoding="utf-8-sig") as f:
                _ACT_POSTURE = json.load(f).get("acts") or {}
        except Exception:
            _ACT_POSTURE = {}
    return _ACT_POSTURE


def _qualifier_corroborated(bare, cands, low, out):
    """-> True when a qualified-only name has its qualifier meant.

    Either the series is named in the text, or somebody already cast
    comes from it. Both are the reader saying which `archer` they mean.
    """
    for c in cands:
        q = c.split(" (", 1)[1].rstrip(")") if " (" in c else ""
        if not q:
            continue
        if " " + q + " " in " " + low + " ":
            return True
        for o in out:
            if " (" in o and o.split(" (", 1)[1].rstrip(")") == q:
                return True
    return False


_PERSON_FLAGS = ("body", "person", "creature")


def _describes_a_person(t):
    """-> True when this tag describes a PERSON rather than the scene.

    Reads the gloss flags rather than a list, so it is right for tags
    nobody has thought about yet. `_hair_vocab` fills the one hole: the
    gloss build files 'blonde hair' and 'long hair' under `object`.
    """
    try:
        _g, fl = _gloss_of(t)
    except Exception:
        fl = []
    if any(f in _PERSON_FLAGS for f in (fl or [])):
        return True
    try:
        return t in _hair_vocab()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# "LET THE LLM INVENT IT" (the author's 2026-09-16): a button over the brief that
# asks the model for a scene IDEA, in the words a person would have typed --
# one or two sentences, not a prompt and not a plan. The user then presses
# Generate and the whole pipeline runs on it as if they had written it.
#
# THIS PATH DELIBERATELY DOES NOT CORRECT THE MODEL. No verifier, no tag
# mapping, no rewriting: the idea is an INPUT, and inputs are the user's.
# Two things it does do, because neither is a matter of style:
#   1. the content floor stands (refuse_youth), as it does on every path;
#   2. the brief is NEVER sent to the model, so pressing the button again
#      cannot re-read the idea it just wrote and hand it back (the author's
#      own point). Variety comes from the dice instead: the genre is
#      rolled when the dropdown says random, a place is drawn from that
#      genre's own roll list, and the ideas already seen are sent back as
#      "not these".
_INVENT_LEVEL = {
    "safe": "nothing sexual at all -- an ordinary moment, fully clothed",
    "sensitive": "suggestive but not explicit: swimwear, underwear, a flirtatious "
                 "moment. No nudity, no sex",
    "nsfw": "erotic and adult: nudity or an openly sexual situation",
    "explicit": "explicitly sexual: say plainly what the adults are doing",
}
_INVENT_SYS = (
    "You invent ideas for a picture. Answer with ONE OR TWO SENTENCES of plain "
    "English -- about 35 words -- describing a scene somebody wants drawn: who "
    "is in it, where they are, what they are doing, and the mood. Write it the "
    "way a person types a request: concrete, casual, present tense. No "
    "headings, no lists, no tags, no quotation marks, no commentary. Everyone "
    "you write is an adult; never write a child, a teenager or a school pupil, "
    "and never anything that sexualises a minor. Invent one idea and stop."
)
# THE AGE FLOOR, READ IN PROSE. refuse_youth() reads TAGS -- the right tool
# for a prompt, the wrong one for a sentence, where "a determined teenager"
# carries no tag at all (measured: it came back from the first test run at
# the safe level). Rolled content never contains youth at any level, so an
# invented sentence is screened on its words as well as its tags.
_INVENT_AGE_RE = re.compile(
    r"\b(child|children|kid|kids|teen|teens|teenage|teenaged|teenager|teenagers|"
    r"toddler|infant|baby|babies|schoolgirl|schoolboy|schoolchild|pupil|pupils|"
    r"underage|minors?|preteen|pre-teen|adolescent|youngster|little (?:girl|boy)|"
    r"(?:high|middle|elementary|primary|grade) school|kindergarten|loli|shota)\b",
    re.I)
_INVENT_AGE_OK = re.compile(r"\bkid gloves\b|\bbaby (?:blue|pink|doll|hair)\b", re.I)


def _invent_age_hit(text):
    """-> the youth words a sentence uses, ignoring the adult phrases that
    merely contain one ('kid gloves', 'baby blue')"""
    t = _INVENT_AGE_OK.sub(" ", str(text or ""))
    return sorted({m.group(0).lower() for m in _INVENT_AGE_RE.finditer(t)})


def invent_brief(opts=None, seed=None, avoid=()):
    """-> {"idea", "genre", "level", "place", "note"}: a scene idea written
    by the language model, guided by the studio's controls and by nothing
    else. Raises the same errors chat() does when the engine is missing."""
    opts = dict(opts or {})
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    # THE LEVEL. 'auto' with no brief resolves to safe in the engine itself
    # (measured 12/12 on empty briefs, 2026-09-16), so the idea that comes
    # back is the level the Generate button would then use.
    level = str(opts.get("spice") or "auto").strip().lower()
    if level not in _INVENT_LEVEL:
        level = "safe"
    # THE GENRE. The dropdown wins; 'random' rolls over the studio's own
    # rollable leaves -- uniform here rather than the generate() bucket
    # shares, because this button exists to spread ideas out.
    pool = _genre_pool()
    genre = str(opts.get("genre") or "random").strip().lower()
    if genre not in pool or genre in ("random", "auto"):
        rollable = sorted(g for g in pool if _genre_rollable(g))
        genre = rng.choice(rollable) if rollable else ""
    # A PLACE FROM THAT GENRE'S OWN LIST (genre_locations.json), so two
    # presses on the same genre do not land in the same room.
    place = ""
    roll = list((_genre_locations(genre) or {}).get("roll") or [])
    if roll:
        place = str(rng.choice(roll))
    # A SUBJECT ANCHOR, ROLLED THE WAY THE ENGINE ROLLS ONE. The place alone
    # was not enough: two presses on cyberpunk wrote the same neon alley.
    # occupation_for() also decides "no profession" at the place's own
    # measured share, so a beach stays a beach and a hospital gets staff.
    occupation = ""
    _cens = {"female": 1, "male": 0, "futa": 0, "other": 0}
    try:
        _got = occupation_for(genre, place, "female", rng,
                              allowed=lambda t: sm.content_allowed(t, _cens, level, None))
        occupation = _got if isinstance(_got, str) else ""
    except Exception:
        occupation = ""
    # AN ACTIVITY ANCHOR, rolled the same way: the activity table decides
    # "nothing in particular" at the place's own measured share, so most
    # presses still leave the action to the model.
    activity = ""
    try:
        _gate = lambda t: sm.content_allowed(t, _cens, level, None)
        _act = activity_for(genre, place, rng, level=level, allowed=_gate)
        if isinstance(_act, dict):
            activity = str(_act.get("activity") or "")
        # AN IDEA BUTTON NEEDS A SEED. When neither a job nor an action came
        # out of the dice -- common at places with a high measured
        # no-activity share, and the reason eight cyberpunk presses all
        # wrote a neon alley -- the activity is drawn again from the SAME
        # table, conditioned on there being an action at all. The share
        # that is being stepped over is a statistic about pictures, not a
        # rule about ideas.
        if not activity and not occupation:
            for _ in range(8):
                _act = activity_for(genre, place, rng, level=level, allowed=_gate)
                if isinstance(_act, dict) and _act.get("activity"):
                    activity = str(_act["activity"])
                    break
    except Exception:
        activity = ""
    period = str(opts.get("period") or "").strip()
    lines = []
    if genre:
        lines.append("Genre: %s" % genre)
    if place:
        lines.append("Place: %s" % place.replace("_", " "))
    if occupation:
        lines.append("Someone in it does this for a living: %s"
                     % str(occupation).replace("_", " "))
    if activity:
        lines.append("What is going on: %s" % activity.replace("_", " "))
    if period and period.lower() not in ("", "newest", "any", "none"):
        lines.append("Era: %s" % period)
    lines.append("Content level (a requirement, not a suggestion): %s"
                 % _INVENT_LEVEL[level])
    # THE EXCLUDE BOX IS A CONTROL TOO. It is the one piece of typed text
    # this path reads, because it says what the user does NOT want, which
    # no amount of re-rolling would discover.
    _ex = re.sub(r"\s+", " ", str(opts.get("exclude") or "")).strip(" ,")
    if _ex:
        lines.append("Must not appear: %s" % _ex[:200])
    seen = [str(x).strip() for x in (avoid or []) if str(x).strip()][-5:]
    if seen:
        lines.append("Already used -- invent a different subject, place and "
                     "action:\n- " + "\n- ".join(seen))
        # NAMING THE WORDS, not just the ideas (measured 2026-09-16: eight
        # presses on cyberpunk wrote "naked cyborg woman ... neon ... rain
        # ... drone" six times, with the ideas themselves already listed
        # above). A 4B model repeats vocabulary, so the vocabulary is what
        # has to be refused.
        worn = _idea_words(seen)
        if worn:
            lines.append("Do not use these words again: " + ", ".join(worn))
    user = ("Invent one picture idea that uses the genre, the place and the "
            "content level below.\n\n" + "\n".join(lines)
            + "\n\nAnswer with the idea only, about 35 words.")

    note, idea = "", ""
    for attempt in range(3):
        u = user if attempt == 0 else (
            user + "\n\nEveryone in the scene is an adult in their twenties or "
                   "older. Do not mention children, teenagers, pupils or schools.")
        idea = _clean_idea(chat(_INVENT_SYS, u, temp=1.1 if attempt == 0 else 0.9,
                                max_tokens=150, retries=0, stage="invent"))
        # THE CONTENT FLOOR IS NOT A STYLE CORRECTION -- it is the one thing
        # this path does check, because rolled content never carries youth.
        bad = _invent_age_hit(idea) or (refuse_youth(idea, [], "sensitive")
                                        .get("words") or [])
        if not bad:
            return {"idea": idea, "genre": genre, "level": level,
                    "place": place, "occupation": occupation,
                    "activity": activity, "note": ""}
        note = ("the model wrote %s and was asked again" % ", ".join(bad[:3]))
    # THREE REFUSALS AND NOTHING IS INSERTED. Handing back a sentence with
    # the words cut out of it would leave a broken idea in the user's box.
    return {"idea": "", "genre": genre, "level": level, "place": place,
            "occupation": occupation,
            "note": "the model kept writing people under 18, so nothing was "
                    "inserted -- press the button again"}


_IDEA_STOP = set("""a an the and or but of in on at to for with without into from by as is are
was were be being been it its this that these those her his their they he she them him hers
there here while as if then than so very much over under up down out off again more most some
any all both each few other own same too s t just now not no nor only own such can will would
should could may might must one two three someone somebody something anything nothing scene
picture image mood air light lights room man woman women men person people adult adults""".split())


def _idea_words(ideas, n=14):
    """-> the content words the last ideas leaned on, commonest first. Used
    to tell the model which words it has already spent."""
    from collections import Counter as _C
    c = _C()
    for line in ideas:
        for w in re.findall(r"[a-z][a-z'-]{3,}", str(line).lower()):
            if w not in _IDEA_STOP:
                c[w] += 1
    return [w for w, k in c.most_common(n * 2) if k >= 2][:n] or \
           [w for w, _k in c.most_common(n)][:n]


def _clean_idea(text):
    """the model's answer as a person would have typed it: one paragraph, no
    markdown, no quotes around it, at most two WHOLE sentences -- a sentence
    cut off in the middle is dropped rather than shipped half-written"""
    t = str(text or "").strip()
    t = re.sub(r"^```[a-z]*|```$", "", t).strip()
    t = re.sub(r"(?im)^\s*(idea|prompt|scene|answer)\s*[:\-]\s*", "", t)
    t = re.sub(r"[*_#>`]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 1 and t[0] in "\"'\u201c" and t[-1] in "\"'\u201d":
        t = t[1:-1].strip()
    sent = re.findall(r"[^.!?]+[.!?]", t)          # whole sentences only
    if not sent:
        return t[:300].strip()
    out = "".join(sent[:2]).strip()
    if len(out) > 300:
        out = sent[0].strip()
    return out[:300].strip()


def generate(base, mode="anima", spice="auto", opts=None, seed=None):
    _CALLS.clear()
    STATUS.update(stage="resolving the scene", detail="", t0=time.time(), calls=0)
    """the whole v2 pipeline for one prompt; returns a report dict"""
    opts = opts or {}
    rng = random.Random(seed)
    banks = pe.load_all_banks()

    # PACKAGING MUST NOT MOVE THE CONTENT (the author's dual-tab design: one
    # prompt, rendered for both models). conditioning() draws a score tag
    # for anima and nothing for illustrious, so sharing the main stream
    # made every later draw diverge -- the same prompt and seed produced
    # different subjects and different artists in the two modes, which is
    # exactly what a two-tab view must not do. Giving the packaging its own
    # deterministic stream leaves the content identical across modes while
    # each mode still gets its own quality/meta tags.
    _cond_rng = random.Random((int(seed) if seed is not None else 0) * 7919 + 13)
    # THE STYLE HINT IS READ ENTRY BY ENTRY (the author, 2026-09-17: "the
    # 'loose linework' part should be respected too, not only for artist
    # matching but for the generated prompt"). It used to ride on the brief,
    # where the word scanners cut it up: 'loose linework' became 'linework',
    # 'muted palette' a paint palette in the scene. An entry the studio knows
    # is read with the brief as before; an entry a booru tag means as a whole
    # becomes that tag; anything else stays the user's phrase (looks.py).
    opts["_look_tags"], opts["_look_phrases"] = [], []
    _hint0 = str(opts.get("style_hint") or "").strip()
    if _hint0:
        # A STYLE NAME IN THE BOX IS THE STYLE, WHOLE (the author's live test,
        # 2026-09-27: 'seinen manga style' rode on the brief and the word scanners
        # cut it into '@seinen', 'comic' and 'fashion'; '1980s style' was
        # translated to the garment tag '1980s fashion'). An entry naming a style
        # of the pool -- as written or as the booru's '<name> (style)' -- goes
        # straight to the style resolver, before any translation.
        _pool0, _rest0 = _style_pool2(), []
        for _e0 in re.split(r"[,;\n]+", _hint0):
            _kn0 = _e0.strip().lower()
            _kq0 = (_kn0[:-len(" style")] + " (style)") if _kn0.endswith(" style") else _kn0
            _hit0 = _kn0 if _kn0 in _pool0 else (_kq0 if _kq0 in _pool0 else None)
            if _hit0 and not opts.get("_style_hint_name"):
                opts["_style_hint_name"] = _hit0
            elif _e0.strip():
                _rest0.append(_e0.strip())
        _hint0 = ", ".join(_rest0)
    if _hint0:
        try:
            from promptstudio.library import looks as _looks0
            _lk0 = _looks0.read_style_hint(_hint0)
        except Exception:
            _lk0 = {"known": [_hint0], "known_tags": [], "tags": [], "phrases": []}
        # A TRANSLATION DOES NOT RAISE THE CHOSEN LEVEL ('glossy skin' ->
        # shiny skin moved a safe prompt to sensitive): above a chosen level
        # the entry stays the user's phrase; under 'auto' the words decide
        if str(spice or "auto") != "auto":
            for _t0 in list(_lk0["tags"]):
                if sm.SPICE_ORDER.get(sm.safety_floor(_t0), 0) > sm.SPICE_ORDER.get(str(spice), 0):
                    _lk0["tags"].remove(_t0)
                    _lk0["phrases"].append((_lk0.get("from") or {}).get(_t0, _t0))
        _add0 = list(_lk0["known"]) + list(_lk0["tags"])
        if _add0:
            base = (str(base or "") + ", " if str(base or "").strip() else "") + ", ".join(_add0)
        opts["_look_tags"], opts["_look_phrases"] = list(_lk0["tags"]), list(_lk0["phrases"])
        opts["_look_known_tags"] = list(_lk0.get("known_tags") or [])
    # A TYPED FASHION IS RESOLVED EARLY: its garments have to be in hand
    # before the subject's slots are filled, or the set arrives too late to
    # dress anyone.
    try:
        from promptstudio.library import external as _ext0
        _fg = []
        for _n0, _r0 in _ext0.find_typed(_sans_characters(base) or "", kinds=("fashion",)):
            _fg += [g for g in (_r0.get("garments") or [])]
        if _fg:
            opts["_fashion_garments"] = _fg
    except Exception:
        pass
    _sel_level = spice if spice and spice != "auto" else None
    cast, user_tags = resolve_cast(base, banks, mode, rng, level=_sel_level)
    # A STRUCTURED BRIEF COUNTS ITS CAST IN THE SCENE SENTENCE ONLY
    _scene0 = structured_scene(base)
    if _scene0:
        try:
            _c0, _t0 = resolve_cast(_scene0, banks, mode, rng, level=_sel_level)
            if isinstance(_c0, dict) and sum(int(_c0.get(k) or 0) for k in ("female", "male", "futa", "other")) >= 2:
                cast = _c0
                # the count tags the subject sentences typed ('1girl') are each one's own, not the scene's
                user_tags = [t for t in user_tags if not re.match(r"^\d\+?(girl|boy|futa|other)s?$", str(t))] \
                    + [t for t in _t0 if re.match(r"^\d\+?(girl|boy|futa|other)s?$", str(t))]
        except Exception:
            pass
    # the booru's word for the user's word ('crimson hair' -> red hair)
    try:
        _lx_add, _lx_claimed, _lx_drop = lexicon_tags(base, user_tags, banks)
    except Exception:
        _lx_add, _lx_claimed, _lx_drop = [], set(), []
    if _lx_add or _lx_drop:
        user_tags = [t for t in user_tags if str(t).lower() not in _lx_drop] + _lx_add
    opts["_lexicon_claimed"] = sorted(_lx_claimed)
    # an act word the brief only uses on a thing is the thing's ('the drying hull')
    _attr = attributive_acts(base, user_tags)
    if _attr:
        user_tags = [t for t in user_tags if str(t).lower() not in _attr]
    user_tags = list(user_tags) + inflected_acts(base, user_tags)
    level, source = sm.resolve_spice(spice, user_tags)
    switched = None
    if source == "selected":
        det = sm.implied_spice(user_tags)
        if sm.SPICE_ORDER[det] > sm.SPICE_ORDER[level]:
            switched = {"from": level, "to": det}
            level = det
    if (_sel_level is None and isinstance(cast, dict) and cast.get("futa")
            and str(cast.get("source") or "").split(":")[0] in ("gendered", "pair", "gendered+pair", "creative")
            and sm.SPICE_ORDER.get(sm.safety_floor("1futa"), 0) > sm.SPICE_ORDER.get(level, 0)):
        cast = resolve_cast(base, banks, mode, rng, level=level)[0]
    # SEXUALISED MINORS ARE REFUSED, TYPED OR NOT (2026-09-05). The
    # sexual-youth tags by definition are refused at every level; the
    # youth-coded words whenever the level is sensitive or above. This is
    # the one place typed input is not law.
    refused = refuse_youth(base, user_tags, level)
    if refused["words"]:
        base = refused["base"]
        user_tags = refused["tags"]
        opts["_refused_youth"] = refused["words"]
        cast, user_tags = resolve_cast(base, banks, mode, rng, level=_sel_level,
                                       refused=True) if refused["recast"] else (cast, user_tags)
        # NOBODY LEFT, NO SPICE LEFT (2026-09-27: 'a loli, anal sex' and a refused
        # canon minor came out 'no humans; anal', and the level's injection added
        # 'tentacle in mouth' to the empty scene). The sexual request went with the
        # person it was about: the picture falls to safe, and the text keeps only
        # the typed words that carry no spice.
        if isinstance(cast, dict) and not sum(int(cast.get(k) or 0) for k in ("female", "male", "futa", "other")):
            user_tags = [t for t in user_tags if sm.SPICE_ORDER.get(sm.safety_floor(t), 0) == 0]
            base = ", ".join(str(t) for t in user_tags)
            spice = "safe"
        level, source = sm.resolve_spice(spice, user_tags)
        switched = None
        if source == "selected":
            det = sm.implied_spice(user_tags)
            if sm.SPICE_ORDER[det] > sm.SPICE_ORDER[level]:
                switched = {"from": level, "to": det}
                level = det
    # the style box's own phrases never reach the brief, so the refusal
    # reads them here: a phrase carrying a refused word goes
    if opts.get("_look_phrases"):
        _ry = refuse_youth(", ".join(opts["_look_phrases"]), [], level)
        if _ry["words"]:
            opts["_look_phrases"] = [p for p in opts["_look_phrases"]
                                     if not any(re.search(r"(?<![a-z0-9-])%s(?![a-z0-9-])" % re.escape(w), p)
                                                for w in _ry["words"])]
            opts["_refused_youth"] = list(dict.fromkeys(list(opts.get("_refused_youth") or []) + _ry["words"]))
            refused = dict(refused, words=list(dict.fromkeys(list(refused.get("words") or []) + _ry["words"])))
    # futa is spice-conditioned (gelbooru: 3.6% of explicit, ~0 below nsfw):
    # a rolled futa outcome demotes to female when the level cannot carry it
    if cast.get("futa") and sm.SPICE_ORDER[level] < sm.SPICE_ORDER["nsfw"]:
        cast["female"] = cast.get("female", 0) + cast.pop("futa")
        cast["futa"] = 0

    opts = dict(opts)
    opts["_level"] = level
    # camera + location + style + genre resolve BEFORE the subjects
    # (construction order): the framing gates which body parts get
    # described at all, and location/genre feed clothes awareness
    # CONSTRUCTION ORDER (the author's final scheme): after reading the prompt,
    # GENRE and STYLE resolve first (with spice they define the majority of
    # further decisions), CAST is already read (a solo portrait and a group
    # photo frame differently, so cast precedes CAMERA), then LOCATION,
    # then LIGHTING. The prompt's typed facts constrain every step; the
    # order itself never changes.
    # GENRE FIRST. Everything downstream leans on it: the scene type asks
    # the genre how often it is peopleless, and the style and location
    # draws are weighted by it. It used to be resolved after the scene
    # type, which cannot work once the genre is what decides the scene.
    # probe the typed style first: it is evidence the genre roll should
    # respect, and resolve_style's typed branch is pure text matching
    _typed_style = resolve_style(base, opts)[0]
    gen_mode, genre_name = resolve_genre(base, opts, rng,
                                         style_typed=_typed_style,
                                         banks=banks)
    opts["_genre"] = (gen_mode, genre_name)
    # EVENT: an occasion over the genre (typed first, else a rare roll);
    # it steers the place below and pins the season in the lighting
    opts["_event"] = resolve_event(base, genre_name, rng)
    # THE BEING BEFORE THE PLAN (2026-09-04): a multi-anchor leaf (the seven
    # supernatural beings, miko / samurai) rolls its anchor here so the
    # activity table can lean on it; the emission below reuses it. A
    # gendered anchor pair (cavewoman / caveman) is still decided there.
    if genre_name in _genre_pool() and not opts.get("_genre_anchor"):
        _anc0 = _genre_anchors(genre_name)
        _gd0 = [t for t in _anc0 if any(w in t for w in ("woman", "girl", "man", "boy"))]
        if len(_anc0) > 1 and not _gd0:
            # TYPED IS LAW: 'a ghost in a haunted bedroom' names the being;
            # the dice never add a second one beside it
            _low0 = " " + re.sub(r"[^a-z0-9' ]+", " ", (base or "").lower()) + " "
            # ...and a typed MEMBER of the leaf is its being just the same
            # ('a succubus' is of the supernatural world: she is the
            # anchor, not a werewolf beside her) (2026-09-06)
            _mem0 = [t for t in (_genre_pool().get(genre_name, {}).get("members") or []) if t not in _anc0]
            _typed0 = [t for t in _anc0 + _mem0 if (" " + t + " ") in _low0]
            opts["_genre_anchor"] = _typed0[0] if _typed0 else rng.choice(_anc0)
    # SCENE TYPE: only an empty prompt may come out subject-less, and the
    # genre's measured 'no humans' share is the probability. A non-figure
    # outcome zeroes the creative cast roll (nothing typed is ever removed)
    # and constrains the location to that scene's kinds.
    scene_type, scene_tags = resolve_scene_type(base, opts, rng)
    opts["_scene_type"] = scene_type
    if scene_type != "figures":
        cast = {k: (0 if isinstance(v, int) else v)
                for k, v in cast.items()}
        if _SCENE_TYPES[scene_type][1]:
            opts["_scene_kinds"] = _SCENE_TYPES[scene_type][1]
    # STYLE: typed, else drawn with a genre nudge on strong links only.
    opts["_style"] = resolve_style(base, opts, rng)
    # MEDIUM after style: the coherence triangle adapts the generated to
    # the defined (typed members are never changed, even to each other)
    opts["_medium"] = resolve_medium(base, opts, rng)
    # A TYPED LOOK BINDS THE ROLLED STYLE AND MEDIUM (2026-09-17: 'rim light,
    # visible brushstrokes, chiaroscuro' came out with a rolled '3d,
    # retrofuturism'). Each typed look -- a tag from the style box or one of
    # its phrases -- is measured against the rolled style and the rolled
    # medium: two booru tags by their pair lift, anything else by how the
    # artists' descriptions pair them. Under half of chance, the rolled one
    # yields; unmeasured is neutral, and a typed style or medium never yields.
    opts["_look_yielded"] = []
    _look_tagset9 = list(opts.get("_look_tags") or []) + list(opts.get("_look_known_tags") or [])
    _looks9 = _look_tagset9 + list(opts.get("_look_phrases") or [])
    if _looks9:
        try:
            from promptstudio.library import looks as _looks9m
        except Exception:
            _looks9m = None

        def _look_clash(rolled):
            return look_clash(rolled, _looks9, _look_tagset9)
        _st9 = opts.get("_style") or (None, {})
        if _st9[0] and not str(opts.get("_style_mode") or "").startswith("typed"):
            _c9 = _look_clash(_st9[0])
            if _c9:
                opts["_look_yielded"].append((_st9[0], _c9[0], round(_c9[1], 2)))
                opts["_style"] = (None, {})
        _md9 = opts.get("_medium") or ("none", None)
        if _md9[1] and _md9[0] != "typed":
            _c9 = _look_clash(_md9[1])
            if _c9:
                opts["_look_yielded"].append((_md9[1], _c9[0], round(_c9[1], 2)))
                opts["_medium"] = ("none", None)
    # PRE-INJECTION (the author's: inject BEFORE bridge 1 -- the story then
    # incorporates the spice instead of diverging from it; '1boy cooking'
    # at nsfw had gotten 'masturbation' pasted next to oven mitts). When
    # the target sits above what the typed content implies, the engine
    # draws the spice content NOW: bridge 1 receives it as REQUIRED
    # CONTENT, and the tags go into the line as engine facts.
    pre_injected = []
    if sm.SPICE_ORDER[level] >= sm.SPICE_ORDER["sensitive"] and \
            sm.SPICE_ORDER[level] > sm.SPICE_ORDER[
                sm.implied_spice(user_tags)]:
        _occ_typed = next((str(t).lower() for t in user_tags if str(t).lower() in _occupations_known()), None)
        pre_injected = draw_injectables(
            level, cast, mode, banks, rng,
            2 if level == "explicit" else 1, act=act_of(user_tags),
            ctx={"place": (opts.get("_location") or (None, None, None))[1], "genre": (opts.get("_genre") or (None, None))[1], "occupation": _occ_typed,
                 # the outfit is not drawn yet: the clothes roll's states may win a turn
                 "pre": True})
    # NO LINGERIE ON A BODY THE USER DRESSED (the author's live test, 2026-09-19: a
    # typed white sweater and black pants, and the level's injection added a
    # 'babydoll'): an injected piece of WEAR is an outfit of its own, so it is not
    # drawn onto a subject whose typed garments already cover the chest or the hips
    try:
        _typed_dressed = any(implies_clothing(t) and not _is_whole_nudity(t) and _is_garment(t) for t in user_tags)
        if _typed_dressed and pre_injected:
            _wear9 = set((((_spice_table() or {}).get("slots") or {}).get("wear")) or {})                 | set((((_clothes_table() or {}).get("slots") or {}).get("sexual")) or {})
            pre_injected = [t for t in pre_injected if t not in _wear9]
    except Exception:
        pass
    # TYPED EXPLICIT ACTS BIND THE NL TOO (the author's: 'dont soften the
    # initial prompt -- it should be as uncensored and explicit as the
    # user says'): the user's own spice-carrying tags join the required
    # content, so 'fucks' cannot come back as a kiss.
    typed_spice = [t for t in user_tags
                   if sm.SPICE_ORDER[sm.safety_floor(t)] >=
                   sm.SPICE_ORDER["sensitive"]
                   and sm.SPICE_ORDER[sm.safety_floor(t)] <=
                   sm.SPICE_ORDER[level]]
    opts["_required_content"] = list(
        dict.fromkeys(typed_spice + pre_injected))
    # THE ACT (typed or drawn) is a fact of the scene the camera, the focus,
    # the body description and the fast planner read (2026-09-06)
    # the spice acts first, then the drawn ones, then every typed word:
    # a typed ordinary activity ('reading') is an act of the picture too
    opts["_act"] = act_of(typed_spice + pre_injected + [str(t) for t in user_tags])
    # the user's leftover words, early: a place-like run among them is
    # the place (resolve_location), the rest become phrases and ledger
    # entries later
    try:
        opts["_leftovers"] = typed_leftovers(base, user_tags)
        opts["_place_phrases"] = place_phrases(base, opts["_leftovers"], user_tags)
    except Exception:
        opts["_leftovers"], opts["_place_phrases"] = [], []
    # A PHRASE THAT CONTAINS A TYPED PLACE WORD SAYS IT ONCE (2026-09-21): the
    # scanner read 'street' and the leftovers kept the user's own 'rainy
    # street', and the line carried both. His words win; the bare word goes.
    try:
        _php = [str(x).lower() for x in ((opts.get("_place_phrases") or [])
                                         + (opts.get("_leftovers") or []))]
        if _php:
            user_tags = [t for t in user_tags
                         if not any(str(t).lower() in ph.split() and str(t).lower() != ph for ph in _php)]
    except Exception:
        pass
    opts["_user_tags"] = [str(t) for t in user_tags]
    opts["_place_lean"] = place_lean(base, banks)
    loc_mode, loc_name, loc_kind = resolve_location(base, opts, rng,
                                                    typed=user_tags)
    loc_kind = place_side(loc_name, loc_kind, rng, lean=opts.get("_place_lean"))
    opts["_location"] = (loc_mode, loc_name, loc_kind)
    # THE WORD IS THE PLACE, NOT A THING IN IT (2026-09-21): 'dancing at a
    # victorian ball' resolved the ballroom and still put the toy `ball` on the
    # line. Whatever the place claimed leaves the typed things.
    if opts.get("_place_claimed"):
        _pc = str(opts["_place_claimed"]).lower()
        user_tags = [t for t in user_tags if str(t).lower() != _pc]
        opts["_user_tags"] = [str(t) for t in user_tags]
    _ev = opts.get("_event") or (None, None, None)
    _ev_season = None
    if _ev[2] and _ev[2].get("season"):
        _ev_season = rng.choice(list(_ev[2]["season"]))
    opts["_lighting"] = resolve_lighting(
        base, loc_kind, rng, loc_name=loc_name,
        genre=(opts.get("_genre") or (None, None))[1], season=_ev_season)
    # THE ACTIVITY BEFORE THE CAMERA (the author's 2026-09-11): the main
    # subject's activity is rolled here, once -- the same table roll the
    # fast planner used to make after the camera -- so the camera, the
    # subjects' framing gates, the focus and the pose all read the act.
    # The fast planner reuses it for the first main subject.
    opts["_pre_activity"] = None
    _peopled0 = any(cast.get(k) for k in ("female", "male", "futa", "other"))
    # A TYPED FACE FRAMING ROLLS NO ACTIVITY (the author's 2026-09-11): a
    # portrait or a close-up is the face; an activity under it is noise
    _low0 = " " + re.sub(r"[^a-z0-9'()-]+", " ", (base or "").lower()) + " "
    _face_typed = any(" " + f + " " in _low0 for f in
                      ("portrait", "close-up", "close up", "bust", "profile", "headshot", "face focus"))
    # ...and a typed UPPER BODY framing rolls no activity either (the pose
    # is enough; the hands may still hold something)
    _upper_typed = " upper body " in _low0
    # a NUDGE, not a restraint: under a typed face or upper-body framing
    # the activity share is a quarter, the object and the moving stances
    # rarer (fastplan); the expression carries the picture instead
    opts["_face_typed"] = bool(_face_typed)
    opts["_frame_nudge"] = 0.25 if (_face_typed or _upper_typed) else 1.0
    # A TYPED ORDINARY ACTIVITY IS THE ACTIVITY (2026-09-11): 'reading'
    # in the text is the plan's activity, with the registry's stance and
    # object; the fast planner rolls no second one beside it
    _typed_act0 = opts.get("_act")
    _reg0 = ((_activity_table() or {}).get("registry") or {}).get(_typed_act0 or "")
    if _typed_act0 and _reg0 and opts.get("_pre_activity") is None:
        opts["_pre_activity"] = ("done", {"activity": _typed_act0, "stance": _reg0.get("stance"),
                                          "object": _reg0.get("object"), "clothes": _reg0.get("clothes"),
                                          "state": _reg0.get("state"), "arity": _reg0.get("arity") or 1,
                                          "typed": True})
    if scene_type == "figures" and _peopled0 and not opts.get("_act") \
            and not _fastplan._typed_action(base, cast, banks):
        try:
            _census0 = sm.subject_census(count_tags(cast, mode))
            _total0 = sum(int(cast.get(k) or 0) for k in ("female", "male", "futa", "other"))
            _occ0 = _fastplan._occupation_from_prompt(_fastplan._seeds(base, cast, opts, banks))
            _ac0 = activity_for(
                (opts.get("_genre") or (None, None))[1], loc_name, rng,
                allowed=lambda t: sm.content_allowed(t, _census0, level, banks),
                pair=(_total0 == 2), level=level,
                prefer=((opts.get("_event") or (None, None, {}))[2] or {}).get("activities"),
                occupation=_occ0, being=opts.get("_genre_anchor"),
                nudge=float(opts.get("_frame_nudge") or 1.0),
                stance=typed_stance(user_tags))
            opts["_pre_activity"] = ("done", _ac0)
            if _ac0 and _act_rel(_ac0.get("activity")):
                opts["_act"] = _ac0["activity"]
        except Exception:
            opts["_pre_activity"] = None
    # BODY FRAMINGS FRAME BODIES -- and the scene type alone does not
    # establish that there is one. scene_type defaults to 'figures', so a
    # prompt that resolves to an EMPTY cast ('cyberpunk street, neon') was
    # still drawing 'full body, from side' while the count channel emitted
    # 'no humans' -- the two channels contradicting each other in the same
    # line. Framing needs an actual body in frame, not just the default.
    _peopled = any(cast.get(k) for k in ("female", "male", "futa", "other"))
    # A TYPED ACT OF TWO ON A CAST OF ONE: THE VIEWER IS THE PARTNER (the
    # author, 2026-09-21, asked which way it should go -- "the viewer becomes
    # the partner (pov)"). 'a girl giving a footjob' came out '1girl, solo,
    # footjob' with nobody to receive it. The typed word stays law; the camera
    # answers for the missing body.
    try:
        _a_typed = str(opts.get("_act") or "").lower()
        _people9 = sum(int(cast.get(k) or 0) for k in ("female", "male", "futa", "other"))
        if _a_typed and _people9 == 1 and (sm.arity_of(_a_typed, banks) or 1) > 1 \
                and _a_typed in {str(t).lower() for t in (opts.get("_user_tags") or [])}:
            cast["viewer"] = 1
            opts["_pov_partner"] = _a_typed
    except Exception:
        pass
    if scene_type == "figures" and _peopled:
        framing, viewpoint, cam_src = resolve_camera(base, cast, rng, act=opts.get("_act"), level=level)
        if opts.get("_pov_partner") and str(viewpoint or "").lower() not in _POV_VIEWS:
            viewpoint, cam_src = "pov", "act-partner"   # the viewer receives it
    else:
        framing = viewpoint = cam_src = None
    opts["_camera"] = (framing, viewpoint, cam_src)
    # THE PICTURE'S MOOD (the author, 2026-09-20): the act and the genre choose it,
    # the place and the weather stand in without an act; it then chooses the
    # emotions, the auras and the lighting -- one mood for the whole picture
    _lt0 = opts.get("_lighting") or {}
    # the act is whatever the picture is about: the rolled one, or the user's own
    # the user's own act outranks the rolled one here as everywhere (2026-09-20:
    # 'dancing on a rooftop' had its mood scored on an injected 'undressing')
    try:
        _act0 = _fastplan._typed_action(base, cast, banks) or act_of(user_tags) or opts.get("_act")
    except Exception:
        _act0 = opts.get("_act")
    opts["_mood"] = mood_for(rng, level, genre=(opts.get("_genre") or (None, None))[1],
                             act=_act0, place=loc_name,
                             weather=_lt0.get("weather"), season=_lt0.get("season"),
                             emotions=typed_emotions(user_tags, level))
    if opts.get("_lighting") is not None:
        opts["_lighting"] = relight(opts["_lighting"], rng, opts.get("_mood"))
    opts["_camera_extra"] = roll_camera_extra(rng, framing, viewpoint)
    # 'blurry background' is a MODIFIER of a real place (the author's); beside
    # an artificial backdrop it contradicts the line
    if loc_kind == "artificial" and opts.get("_camera_extra") in ("blurry background",
                                                                  "dark background"):
        opts["_camera_extra"] = None
    # 'dark background' follows the LIGHT (the author's: "it depends on the
    # lighting / time of day"): it stays only under a dark light -- night,
    # dusk, dim, low-key, moonlight, candlelight -- and goes in daylight
    if opts.get("_camera_extra") == "dark background":
        _lt = " ".join(str(x) for x in (opts.get("_lighting") or ())
                       if x) + " " + (base or "").lower()
        if not _DARK_LIGHT_RE.search(_lt):
            opts["_camera_extra"] = None
    opts["_effects_on"] = rng.random() < 0.20 * DETAIL_SCALE.get(
        opts.get("detail", "standard"), 1.0)
    if any(cast.get(k) for k in ("female", "male", "futa", "other")):
        subjects, tier_meta = plan_subjects(cast, base, opts, rng)
    else:
        subjects, tier_meta = [], {"background":
                                   bool(_BACKGROUND_RE.search(base or ""))}
    # FOCUS: typed always; rolled only onto an already-described part.
    # A focused body part is FORCED into a main subject's description, past
    # the framing gates and the dice.
    focuses = resolve_focus(base)
    if not focuses:
        rf = rolled_focus(subjects, framing, level, rng, act=opts.get("_act"), view=viewpoint or "",
                          cast=cast)
        if rf:
            focuses = [rf]
    opts["_focus"] = focuses
    part_of = {f: pt for pt, f in _PART_FOCUS.items()}
    for f in focuses:
        pt = part_of.get(f)
        if pt and subjects:
            for s2 in subjects:
                if s2.get("tier") == "main":
                    bp = s2.setdefault("body_parts", [])
                    if pt not in [x.split(" (")[0] for x in bp]:
                        bp.append(pt)
                    break

    # TEXT-MARKED BYSTANDERS: 'hiding from clueless Claude' names who is
    # OUTSIDE the act -- clueless/unaware/oblivious adjacency, or the
    # target of a hiding-from, flags the subject before edges are drawn
    low8 = (base or "").lower()
    for s8 in subjects:
        parts8 = [w for w in
                  str(s8.get("persona") or "").split(" (")[0].split()
                  if len(w) >= 4]
        for w8 in parts8:            # any name part: the typo'd 'claude'
            if w8 not in low8:       # still carries the surname 'strife'
                continue
            if (re.search(r"(clueless|unaware|oblivious|unsuspecting)"
                          r"[^.,;]{0,24}\b" + re.escape(w8), low8) or
                    re.search(r"hid\w*[^.;]{0,40}\bfrom\b[^.;]{0,30}\b" +
                              re.escape(w8), low8)):
                s8["bystander"] = True
                break

    interactions = plan_interactions(subjects, tier_meta, level, rng)
    # participants in an edge yield their solo-action budget: the edge
    # claims their body parts (group plans FIRST, individuals take the rest)
    busy = {i - 1 for e in interactions for i in e.get("participants", [])}
    for j in busy:
        if j < len(subjects) and subjects[j].get("action_plan"):
            subjects[j]["action_plan"]["n_part_actions"] = 0
    # A BYSTANDER'S GENITALS STAY OUT OF THE BRIEF: the body dice offer
    # genital slots to EVERY subject at explicit, so the oblivious
    # watcher's brief kept asking for a penis description the bystander
    # instruction then had to fight. Suppress at the source -- subjects
    # outside every edge, when edges exist at nsfw+, lose the genital
    # parts before bridge 1 sees the plan. Solo scenes are untouched.
    if interactions and sm.SPICE_ORDER[level] >= sm.SPICE_ORDER["nsfw"]:
        for j9, s9 in enumerate(subjects):
            if j9 not in busy:
                s9["body_parts"] = [
                    p for p in (s9.get("body_parts") or [])
                    if p.split(" (")[0].strip().lower() not in
                    ("penis", "pussy", "pubic hair")]
    # BYSTANDER TIER (the author's: 'cloud in this scene is basically a
    # secondary subject -- tifa and aerith are the main focus'): a main
    # outside every edge, while two or more mains ARE in edges, demotes
    # to SECONDARY -- named and present (canon identity stays), but one
    # look phrase only, no slots, no body menu, no action plan. The
    # act participants own the detail budget.
    if len(busy) >= 2:
        for j9, s9 in enumerate(subjects):
            if j9 not in busy and s9.get("tier") == "main":
                s9["tier"] = "secondary"
                s9["slots"] = []
                s9["body_parts"] = []
                s9.pop("action_plan", None)

    pulls_hint = ""
    try:
        with open(_paths.data("subject_pulls.json"),
                  encoding="utf-8-sig") as f:
            sp = json.load(f)
        pl = sp.get("locations", {}).get(loc_name or "", {})
        if pl:
            pulls_hint = (loc_name or "") + ": " + ", ".join(list(pl)[:6])
        else:
            low = (base or "").lower()
            for loc2, pl2 in sp.get("locations", {}).items():
                if pl2 and loc2 in low:
                    pulls_hint = loc2 + ": " + ", ".join(list(pl2)[:6])
                    break
    except Exception:
        pass

    # SEX POSITION IS ENGINE-ROLLED STRUCTURE (the camera precedent: a
    # measured closed pool is the dice's, not the model's -- the
    # optional plan field simply got skipped and 'girl on top' never
    # surfaced). Typed position wins; else a sexual scene with 2+
    # subjects rolls one, count-weighted (gelbooru), anatomy-gated;
    # bridge 1 receives it FIXED and renders the bodies to match.
    _POS_W = {"sex from behind": 196, "girl on top": 148,
              "cowgirl position": 91, "doggystyle": 90,
              "missionary": 79, "standing sex": 33,
              "reverse cowgirl position": 27, "upright straddle": 11,
              "prone bone": 10, "spooning": 7}
    sex_pos = None
    low9 = " " + (base or "").lower() + " "
    for t9 in sex_position_names():
        if " " + t9 + " " in low9:
            sex_pos = t9
            break
    n_sub9 = sum(cast.get(k, 0)
                 for k in ("female", "male", "futa", "other"))
    # THE TABLE'S POSITIONS (spice.json, 2026-09-05): the tree's sexual-
    # positions section with measured counts, floors and partner gates
    # replaces the hand pool above, which stays as the fallback
    _tblpos = spice_positions(cast)
    if _tblpos:
        _POS_W = _tblpos
    # the roll fires only for acts the menu actually DESCRIBES
    # (penetrative sex); mutual touching and oral arrange bodies freely
    # -- forcing 'doggystyle' onto mutual masturbation would be the
    # same over-restriction the author's vetoed in the prose rule
    sexual9 = any(t in ("sex", "vaginal", "anal") or t.endswith(" sex")
                  for t in typed_spice + pre_injected)
    if not sex_pos and sexual9 and n_sub9 >= 2:
        has_p9 = (cast.get("male", 0) + cast.get("futa", 0)) > 0
        allowed9 = [t for t in _POS_W
                    if has_p9 or t not in _REQUIRES_PENIS]
        # THE TYPED ACT CONDITIONS THE ROLL (act_posture.json): 'anal'
        # pairs with sex from behind 0.17 and doggystyle 0.06 before
        # missionary; the global counts rolled 'missionary' for an anal
        # scene and the model then wrote standing bodies under it.
        _act9 = next((t for t in ("anal", "vaginal", "sex")
                      if t in typed_spice + pre_injected), None)
        _lift9 = {}
        if _act9:
            _lift9 = {t: w for t, w in _act_posture_row(_act9).items()
                      if t in allowed9 and w > 0}
        if _lift9:
            sex_pos = rng.choices(list(_lift9), weights=list(_lift9.values()))[0]
        elif allowed9:
            sex_pos = rng.choices(
                allowed9, weights=[_POS_W[t] for t in allowed9])[0]
    opts["_sex_position"] = sex_pos

    # THE FAST PATH (no LLM). Everything above this line was resolved
    # mechanically already -- cast, scene type, genre, medium, camera,
    # location, lighting, focus, interactions and subject structure. Only
    # the content words and the prose need a source, so the fast path
    # swaps those two and leaves the entire pipeline below unchanged.
    fast = bool(opts.get("fast"))
    # ENGINE PARITY (the author's 2026-09-05): the mechanical plan is computed
    # for BOTH engines from the same dice. The fast path IS that plan; the
    # LLM receives its per-subject facts as FIXED (outfit, pose, actions,
    # held object, race, occupation, body), writes the prose around them
    # and fills only the free slots -- and the fixed fields are written
    # back over its answer. Same seed, same scene, either engine.
    _stage("planning the scene")
    fast_plan = _fastplan.plan(base, cast, level, mode, opts, rng,
                               subjects, tier_meta, interactions, banks)
    # A ROLLED ACTIVITY FRAMES THE PICTURE (2026-09-11): with no act in
    # hand and a rolled camera, the mechanical plan's activity -- a
    # measured one (reading, cooking, a hug) -- re-rolls the framing and
    # the viewpoint by its own lifts, the way a typed act already does
    if not opts.get("_act") and (opts.get("_camera") or (None, None, None))[2] == "rolled":
        _acts2 = [str(s9.get("activity") or "") for s9 in (fast_plan.get("subjects") or []) if isinstance(s9, dict)]
        _acts2 += [str(i9.get("act") or "") for i9 in (fast_plan.get("interactions") or []) if isinstance(i9, dict)]
        _a2 = act_of([x for x in _acts2 if x])
        if _a2 and _act_rel(_a2):
            opts["_act"] = _a2
            framing, viewpoint, cam_src = resolve_camera(base, cast, rng, act=_a2, level=level)
            opts["_camera"] = (framing, viewpoint, cam_src)
            opts["_why_act"] = _a2
            # the plan carries the camera into the prose ("Seen from the
            # viewer's point of view"): the re-rolled camera replaces the
            # first roll there too, and the fast prose is rendered again
            if isinstance(fast_plan, dict):
                fast_plan["camera"] = [framing, viewpoint]
                fast_plan["nl"] = _fastplan.render_nl(fast_plan, mode)
    if fast:
        plan = fast_plan
    else:
        _stage("writing the scene (LLM)")
        opts["_fixed_facts"] = fixed_facts(fast_plan)
        opts["_fixed_palette"] = list((fast_plan or {}).get("palette") or []) if isinstance(fast_plan, dict) else []
        plan = bridge1(base, cast, level, mode, opts, rng, pulls_hint,
                       subjects, tier_meta, interactions)
        # A PLAN WITHOUT SUBJECTS IS NO PLAN. The model once answered with
        # a single subject object; the floor then rendered "Two figures."
        # and nothing else. The mechanical planner is the fallback -- the
        # same one the fast path uses -- never an empty paragraph.
        if not isinstance(plan, dict) or not isinstance(plan.get("subjects"), list) \
                or (subjects and not plan.get("subjects")):
            plan = fast_plan
            opts["_plan_fallback"] = "mechanical"
        else:
            # THE CAST IS LAW FOR THE MODEL'S PLAN TOO (the author's live test,
            # 2026-09-28: '1girl, milf, sitting near the window' came back with a
            # second subject part -- 'white hair, blue eyes, cleavage, maid, large
            # breasts, pilot' -- after the one woman's; the model had written her
            # twice). Surplus subjects never become people: with one subject a
            # surplus entry is the same person split in two, and only fills what
            # her own entry left empty; with several it is an invention, dropped.
            if subjects and len(plan["subjects"]) > len(subjects):
                _extra9 = plan["subjects"][len(subjects):]
                plan["subjects"] = plan["subjects"][:len(subjects)]
                if len(subjects) == 1 and isinstance(plan["subjects"][0], dict):
                    _s09 = plan["subjects"][0]
                    for _e9 in _extra9:
                        for _k9, _v9 in (_e9.items() if isinstance(_e9, dict) else ()):
                            if _v9 and not _s09.get(_k9):
                                _s09[_k9] = _v9
                opts["_plan_trimmed"] = len(_extra9)
            merge_fixed(plan, fast_plan)
            # the palette is the mechanical plan's measured roll (2026-09-28)
            plan["palette"] = list((fast_plan or {}).get("palette") or [])
            # ... and so is the mood (the chain's, 2026-09-28)
            if (fast_plan or {}).get("mood"):
                plan["mood"] = fast_plan["mood"]

    # STYLE IS NOT A THEME (the author's: 'cozy Christmas is not a style,
    # Christmas is closer to a genre'). If the 8B named a genre/holiday/
    # occasion as the style despite the instruction, null it -- the
    # holiday already lives in the genre channel and emits its own tag.
    _stn = str((plan.get("style") or {}).get("name") or "").lower()
    if _stn:
        _themey = set(_genre_pool()) | {"holiday", "festive", "seasonal",
                                        "cozy", "occasion", "themed",
                                        "celebration"}
        if any(w in _stn for w in _themey):
            plan["style"] = {"name": None, "decomposition": []}

    # THE ENGINE PRUNES THE PLAN TO WHAT IT ASKED FOR. The model fills
    # unrequested fields ('green eyes' beside canon aqua; a bikini under
    # clothes=none) and harvesting everything hands its inventions a visa.
    _nl_dirty = False
    # WHOLE NUDITY BARES THE BODY on the model's plan too (2026-09-16)
    for ps in plan.get("subjects") or []:
        if isinstance(ps, dict) and isinstance(ps.get("outfit"), list):
            _kept_o = strip_body_garments(ps["outfit"])
            if len(_kept_o) != len(ps["outfit"]):
                ps["outfit"] = _kept_o
                _nl_dirty = True
    # THE RULED WORDS LEAVE THE PLAN TOO (2026-09-16). The plan's palette
    # and the style's descriptors reach the line at assembly, apart from
    # the verified tags: a 'monochrome' struck from the tags came back from
    # plan['palette'] ('cyberpunk street, neon', seed 23). What the request
    # typed stays.
    _typed_low = {str(t).lower() for t in (user_tags or [])}
    _ruled_p = ruled_out()
    if isinstance(plan.get("palette"), list):
        _kept_p = [x for x in plan["palette"]
                   if not (str(x).strip().lower() in _ruled_p and str(x).strip().lower() not in _typed_low)]
        if len(_kept_p) != len(plan["palette"]):
            plan["palette"] = _kept_p
            _nl_dirty = True
    _st_p = plan.get("style")
    if isinstance(_st_p, dict) and isinstance(_st_p.get("decomposition"), list):
        _kept_d = [x for x in _st_p["decomposition"]
                   if not (str(x).strip().lower() in _ruled_p and str(x).strip().lower() not in _typed_low)]
        if len(_kept_d) != len(_st_p["decomposition"]):
            _st_p["decomposition"] = _kept_d
            _nl_dirty = True
    # ONE BODY, ONE STANCE ON THE PLAN (2026-09-16: the tag line kept
    # 'lying, the pose' while the prose said "lying on back and in the
    # pose"). The plan's pose and self-actions are read in order; an item
    # whose implied base stance the booru measures against a kept one under
    # half of chance leaves the plan, and the prose is rendered again. The
    # verifier applies the same rule to the line.
    _bases_p = set((_json_table(_TREE_VOCAB, "tree_vocab.json") or {}).get("bases") or [])
    if _bases_p and _implications():
        for ps in plan.get("subjects") or []:
            if not isinstance(ps, dict):
                continue
            _items_p = []
            if isinstance(ps.get("pose"), str) and ps["pose"].strip():
                _items_p.append(("pose", ps["pose"]))
            if isinstance(ps.get("self_actions"), list):
                _items_p += [("self", x) for x in ps["self_actions"]]
            _kept_p = []
            for kind, x in _items_p:
                xl = str(x).lower().strip()
                sts = ({xl} | set(implied(xl))) & _bases_p
                if not sts:
                    continue
                clash = False
                for ksts in _kept_p:
                    for a in sts:
                        for b in ksts:
                            if a == b or a in implied(b) or b in implied(a):
                                continue
                            lf = pair_lift(a, b, "solo")
                            if lf is not None and lf < 0.5:
                                clash = True
                if clash and xl not in _typed_low:
                    if kind == "pose":
                        ps["pose"] = ""
                    else:
                        ps["self_actions"] = [y for y in ps["self_actions"] if y is not x]
                    _nl_dirty = True
                    continue
                _kept_p.append(sts)

    # THE USER'S EXCLUSIONS PRUNE THE PLAN before the prose (2026-09-11):
    # a generated detail that names an excluded thing goes, in every
    # field, and the prose is rendered again without it
    _xterms = exclusion_terms(opts)
    if _xterms:
        opts["_excluded_terms"] = _xterms

        def _keep(v):
            return not _hits_exclusion(str(v), _xterms)
        for ps in plan.get("subjects") or []:
            if not isinstance(ps, dict):
                continue
            for k in ("outfit", "self_actions", "descriptors"):
                if isinstance(ps.get(k), list):
                    kept = [x for x in ps[k] if _keep(x)]
                    if len(kept) != len(ps[k]):
                        ps[k] = kept
                        _nl_dirty = True
            for k in ("held_object", "pose", "hair", "eyes", "occupation", "fashion_style", "race"):
                if isinstance(ps.get(k), str) and ps[k] and not _keep(ps[k]):
                    ps[k] = None if k == "held_object" else ""
                    _nl_dirty = True
            body = ps.get("body")
            if isinstance(body, dict):
                for part in list(body):
                    vals = body[part] if isinstance(body[part], list) else [body[part]]
                    kept = [x for x in vals if _keep(x)]
                    if len(kept) != len(vals):
                        _nl_dirty = True
                        if kept:
                            body[part] = kept
                        else:
                            del body[part]
        for k in ("lighting", "effects", "palette", "secondaries_look"):
            if isinstance(plan.get(k), list):
                kept = [x for x in plan[k] if _keep(x)]
                if len(kept) != len(plan[k]):
                    plan[k] = kept
                    _nl_dirty = True
        for k in ("background", "setting", "collective"):
            if isinstance(plan.get(k), str) and plan[k] and not _keep(plan[k]):
                plan[k] = ""
                _nl_dirty = True
        if isinstance(plan.get("interactions"), list):
            kept = [i9 for i9 in plan["interactions"] if not (isinstance(i9, dict) and not _keep(i9.get("act") or ""))]
            if len(kept) != len(plan["interactions"]):
                plan["interactions"] = kept
                _nl_dirty = True
    for i, ps in enumerate(plan.get("subjects") or []):
        meta = subjects[i] if i < len(subjects) else None
        if meta is None:
            continue
        slots = set(meta.get("slots") or [])
        # A PRUNED FIELD LEAVES THE PROSE TOO, and the engine's own outfit is not
        # pruned (the author's live test, 2026-09-28: the prose dressed the woman in
        # 'gloves, a hair ribbon, and a pink ribbon', the line carried none -- this
        # step deleted the outfit the planner had written, canon clothes included,
        # after the paragraph was already written from it). The outfit is a fixed
        # fact of the mechanical plan (merged into the model's too); only a field
        # the engine never filled is the model's invention.
        # ... BUT WHAT THE USER TYPED STAYS IN IT (2026-10-03: 'a woman with long
        # black hair' kept `black hair` on the line and lost it from the prose --
        # a typed hair closes the hair slot, and the closed slot's field was
        # dropped whole, the user's own colour with it)
        _typed_low9 = {str(t9).lower() for t9 in (user_tags or [])}
        for _k9, _open9 in (("hair", "hair" in slots or "hair_style" in slots), ("eyes", "eyes" in slots)):
            if _open9 or not ps.get(_k9):
                continue
            _own9 = ", ".join(x9.strip() for x9 in str(ps[_k9]).split(",") if x9.strip().lower() in _typed_low9)
            if _own9 != str(ps[_k9]):
                if _own9:
                    ps[_k9] = _own9
                else:
                    ps.pop(_k9, None)
                _nl_dirty = True
        if "outfit" not in slots and "nudity state" not in slots and ps.get("outfit") \
                and not ps.get("_from_table"):
            ps.pop("outfit", None)
            _nl_dirty = True
        ap = meta.get("action_plan")
        # THE TABLES' VISA (2026-09-04): a pose, an activity and its object
        # that the pose / activity tables gave a main character are the
        # engine's own facts, not the model's inventions -- they stay
        # whatever the old action plan asked for
        _visa = bool(ps.get("_from_table"))
        if not ap and not _visa:
            ps.pop("pose", None)
            ps.pop("self_actions", None)
            ps.pop("held_object", None)
        elif not _visa:
            if not ap.get("pose"):
                ps.pop("pose", None)
            acts = ps.get("self_actions")
            if isinstance(acts, list):
                keep_n9 = ap.get("n_part_actions", 0)
                cut9 = acts[keep_n9:]
                ps["self_actions"] = acts[:keep_n9]
                # a busy participant whose CUT self-action was sexual
                # ('masturbating' while fucking someone) has that act in
                # the prose too -- flag the NL for a re-render
                if cut9 and i in busy and \
                        sm.SPICE_ORDER[level] >= sm.SPICE_ORDER["nsfw"] \
                        and any(re.search(
                            r"masturbat|finger|strok|rub|grop|touch",
                            str(a), re.I) for a in cut9):
                    _nl_dirty = True
            if not ap.get("object_ok"):
                ps.pop("held_object", None)
        if meta.get("tier") == "secondary":
            for k in list(ps.keys()):
                if k not in ("who",):
                    ps.pop(k, None)
            continue
        flav = set(meta.get("flavor") or [])
        if "age" not in flav:
            ps.pop("age", None)
            ps.pop("alias", None)
        if "occupation" not in flav:
            ps.pop("occupation", None)
        if "fashion_style" not in flav:
            ps.pop("fashion_style", None)
        if "general" not in flav:
            ps.pop("descriptors", None)
        else:
            d = ps.get("descriptors")
            if isinstance(d, list):
                ps["descriptors"] = d[:2]
        want_parts = {p.split(" (")[0].strip().lower()
                      for p in (meta.get("body_parts") or [])}
        body = ps.get("body")
        if isinstance(body, dict):
            # THE BODY TABLE'S KEYS ALWAYS PASS (2026-09-06): its race
            # parts, build, skin, marks, states and descriptors are the
            # engine's measured, detail-scaled facts, and the prose already
            # states them -- the line must say the same; the menu's wants
            # gate only the per-part draws
            ps["body"] = {k: v for k, v in body.items()
                          if str(k).split(" (")[0].strip().lower()
                          in want_parts or str(k).strip().lower() in FIXED_BODY_KEYS}
        elif not want_parts:
            ps.pop("body", None)

    # prune scene-level fields the engine did not plan
    if not opts.get("_effects_on"):
        plan.pop("effects", None)
    if not tier_meta.get("collective"):
        plan.pop("collective", None)
    if not tier_meta.get("background"):
        plan.pop("background", None)
    sec_n = sum(1 for s2 in subjects if s2.get("tier") == "secondary")
    sl = plan.get("secondaries_look")
    if isinstance(sl, list):
        plan["secondaries_look"] = sl[:sec_n]
    elif sec_n == 0:
        plan.pop("secondaries_look", None)
    def _ints(seq):
        out2 = []
        for x in (seq or []):
            try:
                out2.append(int(x))
            except (TypeError, ValueError):
                pass
        return sorted(out2)
    planned_parts = [_ints(e.get("participants")) for e in interactions]
    inter = plan.get("interactions")
    if isinstance(inter, list):
        # the model returns indices as strings often enough that an int
        # comparison silently deleted a planned kiss
        plan["interactions"] = [e for e in inter
                                if _ints(e.get("participants"))
                                in planned_parts][:len(interactions)]
        for e in plan["interactions"]:
            e["participants"] = _ints(e.get("participants"))
    else:
        plan.pop("interactions", None)

    # HAND BUDGET VERIFIER (the author's: the 8B overspends hands -- 'hands
    # support his back as he strokes... adjusting his overalls' is three
    # hands). Each subject owns TWO. Claim priority: interaction act >
    # required content > held object > pose > self_actions; over budget,
    # the low-priority claims are PRUNED FROM THE PLAN and the NL is
    # re-rendered from the pruned plan -- the extra LLM call is paid
    # only on violation.
    _HANDS_RE = re.compile(
        r"\bhands?\b|\barms?\b|\bholds?\b|holding|grip|strok\w+|"
        r"adjust\w+|clasp\w*|carr(?:y|ies|ying)|wield|fanning", re.I)
    _hand_pruned = False
    n_ps = len(plan.get("subjects") or [])
    inter_hand = [0] * max(n_ps, 1)
    for e in (plan.get("interactions") or []):
        if _HANDS_RE.search(str(e.get("act") or "")):
            for p in e.get("participants") or []:
                if isinstance(p, int) and 1 <= p <= n_ps:
                    inter_hand[p - 1] += 1
    req_hand = sum(1 for t9 in (opts.get("_required_content") or [])
                   if _HANDS_RE.search(t9) or t9 == "masturbation")
    for i, ps in enumerate(plan.get("subjects") or []):
        used = inter_hand[i] + (req_hand if i == 0 else 0)
        held9 = str(ps.get("held_object") or "").strip().lower()
        held_h = 0 if held9 in ("", "null", "none") else 1
        pose_h = 1 if _HANDS_RE.search(str(ps.get("pose") or "")) else 0
        selfs = ps.get("self_actions") if \
            isinstance(ps.get("self_actions"), list) else []
        self_h = [a for a in selfs if _HANDS_RE.search(str(a))]
        while used + held_h + pose_h + len(self_h) > 2 and self_h:
            selfs.remove(self_h.pop())
            _hand_pruned = True
        if used + held_h + pose_h + len(self_h) > 2 and pose_h:
            ps.pop("pose", None)
            pose_h = 0
            _hand_pruned = True
        if used + held_h + pose_h > 2 and held_h:
            ps.pop("held_object", None)
            _hand_pruned = True
    # THE MODEL SOMETIMES JUST OMITS `nl`. Measured at 33% of LLM
    # generations (4 of 12) -- the plan comes back complete and valid in
    # every other field, with no paragraph at all, and the prompt shipped
    # with an empty prose half. Nothing flagged it, because the re-render
    # below only ran when the ENGINE had changed the plan; a paragraph
    # that was never written in the first place set neither flag.
    # A SENTENCE SAID TWICE IS SAID ONCE (2026-09-06: the 4B repeated its
    # closing sentences verbatim)
    plan["nl"] = _dedupe_sentences(plan.get("nl"))
    _nl_missing = (not str(plan.get("nl") or "").strip()
                   or _is_refusal(plan.get("nl")))
    if _nl_missing:
        plan["nl"] = ""            # a refusal is not prose; treat as absent
    # THE PROSE MAY NOT CONTRADICT THE PLAN (2026-09-15): a paragraph that
    # undresses a dressed subject or names a style the plan does not carry
    # is dirty and is rendered again from the plan
    _nl_contra = None if (fast or _nl_missing) else _prose_contradicts(plan.get("nl"), plan)
    # ... NOR TELL IT AS SOMETHING THAT HAPPENED (the author, 2026-10-03: "llm
    # writes nl in past tense sometimes"): a picture is a moment that is. A
    # paragraph in the past tense is rendered again, and the template stands in
    # if the model insists.
    if not _nl_contra and not fast and not _nl_missing and prose_in_past_tense(plan.get("nl")):
        _nl_contra = "the paragraph is in the past tense"
    if _nl_contra:
        _nl_dirty = True
        opts.setdefault("_prose_contradictions", []).append(_nl_contra)
    if fast:
        # the plan changed under the fast path too (hand budget, pruning),
        # so the prose is simply rebuilt from the corrected plan -- the
        # template reads the plan, so it cannot drift from it.
        if _hand_pruned or _nl_dirty or _nl_missing:
            plan["nl"] = _fastplan.render_nl(plan, mode)
    elif _hand_pruned or _nl_dirty or _nl_missing:
        _NL_RERENDER = (
            "You are an EXPERT PROMPT WRITER for image-generation "
            "models. Write ONLY the natural-language paragraph for the "
            "FINAL scene plan given as JSON: one small story of the "
            "image, 3-6 flowing sentences, in the PRESENT TENSE only "
            "('sits', 'is holding', 'wears' -- never 'sat', 'was "
            "holding', 'wore'). Every subject is named in "
            "full (noun or alias) in every sentence that mentions them "
            "-- a pronoun may appear only in a sentence that already "
            "names its subject. NEVER invent proper names. Do NOT add "
            "any hand or arm action that is not in the plan -- each "
            "subject has exactly two hands and the plan already spends "
            "them. Render every act in the plan at its full "
            "explicitness -- never soften or euphemize. Output plain "
            "text only, no JSON, no commentary.")
        try:
            nl2 = chat(_NL_RERENDER, stage="prose", user=json.dumps(
                {k: plan.get(k) for k in
                 ("count_sentence", "subjects", "interactions", "action",
                  "setting", "lighting", "effects", "palette", "style",
                  "mood") if plan.get(k)},
                ensure_ascii=False), temp=0.7, max_tokens=400).strip()
            if nl2 and "{" not in nl2[:2] and not _is_refusal(nl2):
                plan["nl"] = _dedupe_sentences(nl2)
        except Exception:
            pass    # the original NL stands; the tag line is verified
        # the model insists on a fact the plan does not carry: the
        # template's paragraph, plainer and true to the plan
        _nl_contra2 = _prose_contradicts(plan.get("nl"), plan) or (
            "the paragraph is in the past tense" if prose_in_past_tense(plan.get("nl")) else None)
        if _nl_contra2:
            opts.setdefault("_prose_contradictions", []).append(_nl_contra2 + " (template)")
            plan["nl"] = _fastplan.render_nl(plan, mode)
        # A PROMPT IS NEVER SHIPPED WITH HALF OF IT MISSING. If the model
        # gave no paragraph and the re-render also came back empty, the
        # mechanical template writes one from the same plan. It is plainer
        # prose than the model's, and it is not nothing.
        if not str(plan.get("nl") or "").strip():
            plan["nl"] = _fastplan.render_nl(plan, mode)

    # A TYPED BACKGROUND IS THE PLACE (the author's live test, 2026-09-19: 'simple
    # background, white background' typed, and the model wrote a mecha hangar --
    # "simple colored backgrounds ARE a location (and it was typed)"): the model's
    # own setting and scene details leave the plan, the tag line carries no second place
    _loc9 = str((opts.get("_location") or (None, None, None))[1] or "")
    if _loc9.endswith("background"):
        if plan.get("setting") and not str(plan.get("setting")).lower().endswith("background"):
            plan["setting"] = ""
        if plan.get("scene_details"):
            plan["scene_details"] = []
    concepts, action_concepts, group_map, origins, anchors, single = \
        plan_concepts(plan, time_now=(opts.get("_lighting") or {}).get("time"))
    # TYPED LEFTOVERS (2026-09-11): the words of the user's own text that
    # no tag uses, in their runs ('moss-covered dais', 'vellichor'), join
    # the concepts as unmapped: a phrase on the line, an entry in the
    # ledger for review -- on both engines, never dropped silently
    _LEFTOVER_CONCEPTS.clear()
    try:
        for _lo in reversed([x for x in (opts.get("_leftovers") or typed_leftovers(base, user_tags))
                             if x not in (opts.get("_place_phrases") or [])]):
            if _lo not in origins:
                concepts.insert(0, _lo)          # the user's words come first
                origins[_lo] = ("scene", 0)
                _LEFTOVER_CONCEPTS.add(_lo)
    except Exception:
        pass
    # a persona's name words are cast facts, not retrieval material
    name_block = frozenset(
        w for s2 in subjects
        for w in re.findall(r"[a-z]{4,}",
                            str(s2.get("persona") or "").lower()))
    # comic-adjacent context legalizes emote/symbol vocabulary ('!' over
    # a surprised head belongs in a chibi scene, not a photograph)
    _ctx_names = " ".join(str(x) for x in (
        (opts.get("_style") or (None,))[0], genre_name,
        (opts.get("_medium") or (None, None))[1],
        (plan.get("style") or {}).get("name"), base)).lower()
    comic_ok = bool(re.search(
        r"chibi|comic|cartoon|toon|manga|meme|kawaii|oekaki|rubber hose|"
        r"superflat|4koma|comedy|comedic|comical|parody|emoji|sticker",
        _ctx_names))
    # A GENERIC PLACE IS A PHRASE, NEVER MAPPED (2026-09-15: the ruled
    # generic 'home' went through the mapper and came out as the meme tag
    # 'take it home'): the rulings' generic places and the user's own
    # place phrases stay in the user's words on the line
    _gen_ph = {str(x).lower() for x in ((opts.get("_phrases_typed") or []) + (opts.get("_place_phrases") or []))}
    _held_ph = [c for c in concepts if str(c).lower() in _gen_ph]
    tags, unmapped, stripped, mapping = bridge2([c for c in concepts if str(c).lower() not in _gen_ph],
                                                action_concepts,
                                                origins, anchors, single,
                                                name_block, comic_ok,
                                                use_llm=not fast)
    unmapped = list(unmapped) + [c for c in _held_ph if c not in unmapped]

    # SOLO-ARITY VERIFICATION (individual-action subsection rule 1): a tag
    # born from an individual pose/action/object concept must not need a
    # partner -- arity.json is the authority. 'hug' from a self_action drops.
    dropped_arity = []
    group_tags = {}
    for c, n_p in group_map.items():
        for t in mapping.get(c, []):
            group_tags[t] = max(group_tags.get(t, 0), n_p)
    solo_only = {t for c in action_concepts for t in mapping.get(c, [])
                 if t not in group_tags}
    kinds = [s.get("kind") for s in subjects]
    has_penis = any(k in ("male", "futanari") for k in kinds)
    has_female = any(k in ("female", "futanari") for k in kinds)
    for t in list(tags):
        if t in solo_only:
            need = sm.arity_of(t)
            if need and need >= 2:
                tags.remove(t)
                dropped_arity.append(t)
        elif t in group_tags:
            need = sm.arity_of(t)
            # a group-edge act must fit its OWN participant count, and its
            # anatomy must exist in the cast (a penis-act with no
            # penis-haver is the old 'together - handjob' bug reborn)
            if ((need and need > group_tags[t])
                    or (sm.MALE_ANATOMY.search(t) and not has_penis)
                    or (sm.FEMALE_ANATOMY.search(t) and not has_female)
                    or (sm.MALE_NAMED.search(t) and "male" not in kinds)
                    or (sm.FEMALE_NAMED.search(t) and not has_female)
                    or (sm.FUTA_NAMED.search(t) and "futanari" not in kinds)):
                tags.remove(t)
                dropped_arity.append(t)

    # persona names and their LOCKED canon go straight into the tag part --
    # they ARE danbooru tags, pre-verified by measurement; bridge 2 only
    # handles what the LLM invented. Canon still respects the level floors.
    canon_tags = []
    canon_by_subj = [[] for _ in subjects]
    # AGE tags come from the fixed whitelist, engine-mapped off the model's
    # age/alias fields -- never from fuzzy retrieval (the author's).
    for i, ps in enumerate(plan.get("subjects") or []):
        meta = subjects[i] if i < len(subjects) else None
        if meta and (ps.get("age") or ps.get("alias")):
            t = map_age(str(ps.get("age") or ""), str(ps.get("alias") or ""),
                        meta["kind"], typed=base)
            # the model's own STORY outranks its age field: an NL that
            # says 'young' while the age field said mature is the model
            # disagreeing with itself, and a maturity tag on a scene the
            # NL calls young is the wrong resolution (sweep: 'mature
            # female' on 'A young girl dances')
            if t and re.search(r"\byoung|\bteenag|\byouthful",
                               str(plan.get("nl") or ""), re.I):
                t = None
            if t:
                canon_tags.append(t)
                if i < len(canon_by_subj):
                    canon_by_subj[i].append(t)   # subject band, owned
    for s_i, s in enumerate(subjects):
        if s.get("persona"):
            canon_tags.append(s["persona"])
            canon_by_subj[s_i].append(s["persona"])
            if s.get("series"):
                canon_tags.append(s["series"])
                canon_by_subj[s_i].append(s["series"])
        # FUTANARI IS AN ENGINE-EMITTED IDENTITY TAG (the author's: it was
        # inconsistent -- present only when bridge2 happened to map it,
        # and mis-placed when it did). A futa subject ALWAYS carries
        # 'futanari' at nsfw+ (below that the futa demoted to female),
        # owned by its subject block so it sits right after the persona.
        if s.get("kind") == "futanari":
            canon_tags.append("futanari")
            canon_by_subj[s_i].append("futanari")
        # THE CANON GARMENTS YIELD TO THE UNDRESS (2026-09-16: melantha's
        # canon 'jacket' beside 'nude' in a bath): the canon list goes to
        # the line whole, past the plan's outfit, so a whole nudity state in
        # the subject's plan strips its body garments here as it does there
        _ps_o = ((plan.get("subjects") or [{}] * (s_i + 1))[s_i] or {}).get("outfit")             if s_i < len(plan.get("subjects") or []) else None
        _undressed = any(_is_whole_nudity(x) for x in (_ps_o or []) if x)
        # SHUT EYES TAKE THE CANON EYE STATES TOO (2026-09-16: 'blue eyes,
        # star in eye, closed eyes' on a sleeper): a colour or a pupil
        # nobody can see stays off the line
        _ps_ex = ((((plan.get("subjects") or [])[s_i] if s_i < len(plan.get("subjects") or []) else {}) or {}).get("body") or {}).get("expression") or []
        _shut = "closed eyes" in [str(x).lower() for x in _ps_ex]
        for t in s.get("locked_canon") or {}:
            if _shut and t != "closed eyes" and re.search(r"\b(?:eyes?|eyed|pupils)\b", str(t)):
                continue
            if sm.SPICE_ORDER[sm.safety_floor(t)] <= sm.SPICE_ORDER[level]:
                if _undressed and implies_clothing(t):
                    continue
                canon_tags.append(t)
                canon_by_subj[s_i].append(t)
    # TYPED TAGS ARE LAW (the project's first law, found broken in
    # current-scope testing): a typed REAL tag goes straight into the tag
    # part -- it must not depend on the LLM re-emitting it through the NL
    # ('summer festival' survived one run by luck and vanished the next).
    # First position = typed wins every earlier-wins verifier (clusters,
    # contradictions). Counts and level names ride their own channels.
    _v9 = _vocab()
    _cnt_re = re.compile(r"^\d*(?:girls?|boys?|futas?|others?|futanari)$|"
                         r"^(?:solo|multiple girls|multiple boys)$")
    typed_direct = [t for t in user_tags
                    if _v9.get(t, 0) >= 100 and not _cnt_re.match(t)
                    and t not in sm.LEVELS]
    # pre-injected spice is an engine fact: into the line ahead of the
    # mapped tags (it wins earlier-wins verifiers over LLM content)
    tags = list(dict.fromkeys(typed_direct + canon_tags
                              + pre_injected + tags))
    # THE USER'S OWN PHRASE SAYS IT ONCE (2026-09-21, his bug hunt: 'a man
    # walking down a rainy street' carried the mapped `street` AND the phrase
    # `rainy street'). A leftover phrase is kept whole, by the studio's own
    # rule; a tag that is one of its words only says it again.
    _ph_all = [str(x).lower() for x in ((opts.get("_leftovers") or [])
                                        + (opts.get("_place_phrases") or []))]
    if _ph_all:
        tags = [t for t in tags
                if not any(str(t).lower() in ph.split() and str(t).lower() != ph for ph in _ph_all)]

    # A PRE-INJECTED GARMENT NOBODY WOULD WEAR (the author's bug hunt,
    # 2026-09-21: an injected 'garter belt' over a 'string bikini' at the
    # beach). The level's spice is injected BEFORE anyone is dressed -- by
    # design, so the story can carry it -- so the check has to happen here,
    # where the outfits are settled: a worn word that the booru does not wear
    # beside ANY subject's clothes is not in this picture. Typed words stand.
    try:
        _typed_now = {str(x).lower() for x in (opts.get("_user_tags") or [])}
        _slots_now = (_clothes_table() or {}).get("slots") or {}
        _fits = []
        for _t9 in tags:
            if str(_t9).lower() in _typed_now or not (_is_garment(_t9) or implies_clothing(_t9)):
                _fits.append(_t9)
                continue
            _outs = [list(ps.get("outfit") or []) for ps in (plan.get("subjects") or [])]
            if _outs and not any(_pairs_with_worn(_t9, _o, _slots_now) for _o in _outs if _o):
                continue
            _fits.append(_t9)
        tags = _fits
    except Exception:
        pass

    if loc_name and loc_kind != "artificial":
        _typed_pc = {str(t).lower() for t in user_tags}
        tags = [t for t in tags if str(t).lower() not in _PLACE_CLASS_WORDS or str(t).lower() in _typed_pc]
    # engine-resolved location and genre ARE tags already
    if loc_name and loc_kind == "artificial":
        tags.append(loc_name)          # 'white background' is the tag
    elif loc_name:
        # ... unless the USER'S OWN PHRASE already says this place (2026-09-21:
        # 'a man walking down a rainy street' carried `street` AND the phrase
        # `rainy street`). His wording is the more specific of the two and it
        # is what he asked for; the bare word would only say it again.
        _ph_place = [str(x).lower() for x in ((opts.get("_leftovers") or [])
                                              + (opts.get("_place_phrases") or []))]
        if not any(says_the_place_again(_x, loc_name) and _x != str(loc_name).lower()
                   for _x in _ph_place):
            tags.append(loc_name)
        if loc_kind == "indoor":
            tags.append("indoors")
        elif loc_kind and loc_kind.startswith("outdoor"):
            tags.append("outdoors")
    genre_tags = []
    if genre_name:
        # a LOCATION-SHAPED genre tag ('school' is also the place tag)
        # must not sit next to a typed location and re-place the scene;
        # the genre still colours the cast through bridge 1
        if not (gen_mode == "rolled" and loc_mode == "typed" and
                _genre_pool().get(genre_name, {}).get("location_shaped")):
            # THE LEAF'S ANCHOR IS THE TAG. 'ancient egypt' emits
            # `ancient egyptian`; 'ancient japan' emits `miko` or `samurai`
            # (one, chosen here); a plain leaf emits its own name.
            _anc = _genre_anchors(genre_name) if genre_name in _genre_pool() else [genre_name]
            # A GENDERED ANCHOR FOLLOWS THE CAST: 'cavewoman' for a girl,
            # 'caveman' for a boy; with no gender word in the anchors
            # (miko / samurai) the pick is the dice's.
            _fem = cast.get("female", 0) + cast.get("futa", 0)
            _mal = cast.get("male", 0)
            _gendered = [t for t in _anc if any(w in t for w in ("woman", "girl", "man", "boy"))]
            _pref = [t for t in _gendered if (("woman" in t or "girl" in t) and _fem)
                     or (("man" in t and "woman" not in t or "boy" in t) and _mal)]
            if opts.get("_genre_anchor"):
                genre_tags = [opts["_genre_anchor"]]
            elif _pref:
                genre_tags = list(dict.fromkeys(_pref))   # a mixed cast gets both
            else:
                genre_tags = [rng.choice(_anc)]
            # tags that ride with every anchor ('prehistoric' itself)
            genre_tags += [t for t in (_genre_pool().get(genre_name, {}).get("always") or [])
                           if t not in genre_tags]
            # NOBODY IN FRAME, NO BEING ON THE LINE: 'a bowl of fruit on a
            # table' rolled the witch leaf and wrote `witch` beside
            # `no humans`. An anchor that names a person or a race is
            # dropped when the cast is empty; the leaf's other tags stay.
            if not _fem and not _mal and not cast.get("other"):
                genre_tags = [t for t in genre_tags
                              if sm.subject_kind(t) not in ("person", "race")
                              and not _is_humanlike(t)]
            # THE WORLD'S BEING IS NOT ANOTHER CAST MEMBER (the author's live tests,
            # 2026-09-19: 'slime boy' and 'dragon' beside a slime girl; a girl and
            # a boy in a park came out tengu, shinigami AND 'werewolf'): when the
            # subjects already carry races of their own, an anchor that names a
            # being is theirs only if it IS one of those races -- otherwise it is
            # a third body on the tag line that nobody asked for
            try:
                _sraces = {str((ps_ or {}).get("race") or "").lower()
                           for ps_ in (plan.get("subjects") or []) if isinstance(ps_, dict)} - {""}
            except Exception:
                _sraces = set()
            # ... AND WITH NO 'OTHER' IN THE CAST THERE IS NOBODY FOR IT TO BE (the
            # author, 2026-09-19: '1girl, hatsune miku' came out with 'zombie' -- "for
            # imagegen model it will turn hatsune miku in zombie, creatures should be
            # added ONLY when there is 1other in cast detected")
            if _sraces or not cast.get("other"):
                genre_tags = [t for t in genre_tags
                              if str(t).lower() in _sraces
                              or (sm.subject_kind(t) not in ("person", "race", "creature", "humanoid")
                                  and not _is_humanlike(t))]
            if genre_tags:
                opts["_genre_anchor"] = genre_tags[0]
            for t in genre_tags:
                if t not in tags:
                    tags.append(t)
    _ev = opts.get("_event") or (None, None, None)
    # AN OCCASION THE HOUR CONTRADICTS IS NOT THIS PICTURE'S (2026-09-21): a
    # moon-viewing festival and its full moon stood in a gym at noon. The hour
    # is already drawn; the same measured test guards the event, its props and
    # the scene's own sights.
    _hour = (opts.get("_lighting") or {}).get("time")
    if _ev[1] and not fits_hour(_ev[1], _hour):
        _ev = (None, None, {})
    if _ev[1]:
        if _ev[1] not in tags and _vocab().get(_ev[1], 0) >= 100:
            tags.append(_ev[1])
        _props = [p for p in (_ev[2].get("props") or []) if _vocab().get(p, 0) >= 100
                  and p not in tags and fits_hour(p, _hour)]
        try:
            _r = rng
        except NameError:
            _r = random.Random(17)
        for p in _props[:(1 if _r.random() < 0.6 else 2)]:
            try:
                if sm.above_ceiling(p, level):
                    continue
            except Exception:
                pass
            tags.append(p)
    # MEDIUM tag + the traditional-media rider (danbooru convention:
    # watercolor posts carry 'traditional media' alongside)
    med_tag = (opts.get("_medium") or ("none", None))[1]
    if med_tag:
        tags.append(med_tag)
        mp9 = _medium_pool()
        if med_tag in (mp9.get("rider") or {}).get(
                "implies_traditional", []):
            tags.append("traditional media")
    # scene-type tags: 'no humans' is the count statement of a
    # subjectless scene; the rest are scene facts
    for t9 in scene_tags:
        tags.append(t9)
    # PAIRING + SEX POSITION (engine facts, the author's studio round 7):
    # the participant kinds x a sexual act decide the pairing tag; the
    # position comes from the plan's closed menu, anatomy-gated.
    has_penis9 = any(s2.get("kind") in ("futanari", "male")
                     for s2 in subjects)

    def _sexual9(tag_list):
        return any(sm.SPICE_ORDER[sm.safety_floor(t)] >=
                   sm.SPICE_ORDER["nsfw"] and (sm.arity_of(t) or 0) >= 2
                   for t in tag_list)
    pair_tags = []
    for e9 in (plan.get("interactions") or []):
        acts9 = mapping.get(str(e9.get("act") or "").strip(), [])
        parts9 = [p for p in (e9.get("participants") or [])
                  if isinstance(p, int) and 1 <= p <= len(subjects)]
        edge_sexual = _sexual9(acts9)
        if len(parts9) >= 2 and edge_sexual:
            t9p = _PAIRING.get(frozenset(
                subjects[p - 1].get("kind") for p in parts9))
            if t9p:
                pair_tags.append(t9p)
        sp9 = str(e9.get("sex_position") or "").strip().lower()
        # the model's own position counts only when the engine fixed none:
        # both on the line ('missionary' rolled, 'doggystyle' planned) was
        # the contradiction the author's saw
        if sp9 in sex_position_names() and not opts.get("_sex_position") and \
                (has_penis9 or sp9 not in _REQUIRES_PENIS) and \
                (sp9 in spice_positions(cast) or sp9 in _SEX_POSITIONS):
            pair_tags.append(sp9)
    # the engine-rolled position is guaranteed into the line
    if opts.get("_sex_position"):
        pair_tags.append(opts["_sex_position"])
    # A MALE-RECEIVER ACT NEEDS A MALE OR FUTA RECEIVER. 'prostate
    # milking' was mapped onto an elf milf being fucked; the direction is
    # in the user's sentence, and prostate is not hers.
    _dir9 = _typed_direction(base, subjects)
    _m9 = re.search(r"subject (\d+) \(the [^)]*\) does it to subject (\d+)", _dir9 or "")
    if _m9:
        _rk = (subjects[int(_m9.group(2)) - 1].get("kind")
               if int(_m9.group(2)) <= len(subjects) else None)
        if _rk == "female":
            _drop9 = [t for t in tags if t in _MALE_RECEIVER_ACTS]
            if _drop9:
                tags = [t for t in tags if t not in _MALE_RECEIVER_ACTS]
                pair_tags = [t for t in pair_tags if t not in _MALE_RECEIVER_ACTS]
    # a sexual scene without a sexual EDGE act still has a pairing. The
    # STRONGEST authority is the user's own syntax: 'tifa lockhart
    # fucks aerith gainsborough' names the pair around the verb (the
    # first-edge guess had paired Tifa with the WATCHER). Then the
    # first real edge; then a two-subject cast.
    if not any(t in _PAIRING.values() for t in pair_tags) and \
            _sexual9(tags):
        kinds9f = None
        low9p = (base or "").lower()
        vm9 = re.search(r"\b(?:fuck\w*|rail\w*|breed\w*|pound\w*|"
                        r"penetrat\w*|scissor\w*|having sex|sex with)\b",
                        low9p)
        if vm9:
            bef9 = aft9 = None
            for i9, s2 in enumerate(subjects):
                p9 = str(s2.get("persona") or "").split(" (")[0]
                if not p9:
                    continue
                pos9 = low9p.find(p9)
                if pos9 < 0:
                    continue
                if pos9 < vm9.start() and (bef9 is None or
                                           pos9 > bef9[0]):
                    bef9 = (pos9, i9)
                elif pos9 > vm9.start() and (aft9 is None or
                                             pos9 < aft9[0]):
                    aft9 = (pos9, i9)
            if bef9 and aft9:
                kinds9f = frozenset((subjects[bef9[1]].get("kind"),
                                     subjects[aft9[1]].get("kind")))
        if kinds9f is None:
            for e9 in (plan.get("interactions") or []):
                parts9 = [p for p in (e9.get("participants") or [])
                          if isinstance(p, int) and
                          1 <= p <= len(subjects)]
                if len(parts9) >= 2:
                    kinds9f = frozenset(subjects[p - 1].get("kind")
                                        for p in parts9)
                    break
        if kinds9f is None and len(subjects) == 2:
            kinds9f = frozenset(s2.get("kind") for s2 in subjects)
        # THE VIEWER IS A PARTNER HERE TOO. With one subject and a POV
        # act there is no second participant to pair with, so this rule
        # emitted nothing and the parser's phrase maps were the only
        # source of `hetero` -- a second owner of the same fact, and a
        # wrong one for "two girls having anal sex". The viewer's gender
        # is `female pov` when written, male otherwise (the measured
        # default the partner rule already assumes).
        if kinds9f is None and len(subjects) == 1 and                 any(t in ("pov", "male pov", "female pov", "futanari pov")
                    for t in tags):
            # `futanari pov` (1,422 posts) is the camera holder being a
            # futa; `female pov` a woman; bare `pov` the measured default,
            # a man. (`male pov` is not a booru tag, kept for typed input.)
            _vg = ("futanari" if "futanari pov" in tags else
                   "female" if "female pov" in tags else "male")
            kinds9f = frozenset((subjects[0].get("kind"), _vg))
        t9p = _PAIRING.get(kinds9f) if kinds9f else None
        if t9p:
            pair_tags.append(t9p)
    for t9 in pair_tags:
        if t9 not in tags:
            tags.append(t9)
    for f in (opts.get("_focus") or []):
        if sm.SPICE_ORDER[sm.safety_floor(f)] <= sm.SPICE_ORDER[level]:
            tags.append(f)
    cam2 = opts.get("_camera") or (None, None, None)
    if cam2[0]:
        tags.append(cam2[0])
    if cam2[1]:
        tags.append(cam2[1])
    if cam2[0] and opts.get("_camera_extra"):
        tags.append(opts["_camera_extra"])
    st_name2 = (opts.get("_style") or (None,))[0]
    if st_name2 and st_name2 in _vocab() and _vocab()[st_name2] >= 100:
        tags.append(st_name2)
        # THE PALETTE HAS ONE OWNER (the author's 2026-09-13: greyscale on .31
        # of the lines, monochrome .25, against .04 and .05 of 1girl
        # posts): this block also appended the style's first two palette
        # tags, on every line, beside the plan's own palette. The plan
        # owns it -- the fast path draws each palette tag at P(tag |
        # style), the writer model chooses its own with the measured
        # palette as a bias -- and this block adds the style name only.
    # WINDOW MEDIATOR (the author's): sky and rain CAN be seen from indoors --
    # through a window. An indoor scene with sky/weather tags gains 'window'
    # instead of losing the sky.
    lt2 = opts.get("_lighting") or {}
    if lt2.get("time") and lt2["time"] not in ("day",):
        tags.append({"noon": "day", "evening": "evening"}.get(
            lt2["time"], lt2["time"]))
    if lt2.get("weather"):
        wmap = {"rainy": "rain", "raining": "rain", "snowy": "snow",
                "snowing": "snow", "foggy": "fog", "misty": "mist",
                "stormy": "storm", "windy": "wind", "sunny": "sunny",
                "clear sky": "clear sky"}
        tags.append(wmap.get(lt2["weather"], lt2["weather"]))
    for _lg in (lt2.get("lighting") or []):
        if _lg not in tags and fits_hour(_lg, lt2.get("time")):
            tags.append(_lg)
    _SKYISH = ("sky", "blue sky", "cloud", "clouds", "rain", "snow", "sunset",
               "sunrise", "starry sky", "night sky", "moon", "sun",
               "sunlight", "moonlight", "night", "overcast", "fog", "storm")
    # 'building' is ambiguous -- you can be ON it (a rooftop sunset needs
    # no window); the mediator applies to true interiors only
    if loc_kind == "indoor" and             any(t in _SKYISH for t in tags) and "window" not in tags:
        tags.append("window")
    tags = list(dict.fromkeys(tags))

    # BACKGROUND ENTITY (the author's): crowd-family tags instead of counts;
    # a solo main on a crowded street is 1girl + solo focus + crowd
    if tier_meta.get("background"):
        tags.append("crowd")
        counted = sum(cast.get(k, 0)
                      for k in ("female", "male", "futa", "other"))
        if counted == 1:
            tags.append("solo focus")

    # verifier: one_of clusters hold whatever bridge 2 did -- two breast
    # sizes arrived in one prompt through the 0-2-picks rule, and the
    # measured contradiction table happens not to carry that pair.
    # PER SUBJECT (current-scope testing, the author's): the cluster is a fact
    # about ONE body -- two girls legitimately have two eye colours, and
    # the global version silently ate the second girl's.
    _CLUSTERS = ONE_OF_FAMILIES
    # which subject owns each mapped tag, from the origin tracking + canon
    tag_subj = {}
    for c9, tl9 in mapping.items():
        o9 = origins.get(str(c9).lower())
        if o9 and o9[0] == "subject":
            for t9 in tl9:
                tag_subj.setdefault(t9, o9[1])
    for si9, ct9 in enumerate(canon_by_subj):
        for t9 in ct9:
            tag_subj.setdefault(t9, si9)
    # A TYPED SUBJECT TAG BELONGS TO ITS SUBJECT (2026-09-16: 'milf' typed
    # in a tag list sat in the scene band, after the subjects): only the
    # one-of families were owned here, and a typed tag never goes through
    # the mapper that owns the rest. The planner's own owner rule --
    # _typed_subject_tags, which reads the text's subject spans -- answers
    # for every typed tag, so the bands agree with the plan.
    try:
        for _si9 in range(len(subjects)):
            for _t9 in _typed_subject_tags(base, user_tags, _si9, len(subjects),
                                           kinds=[s9.get("kind") for s9 in subjects]):
                if _t9 in tags:
                    tag_subj.setdefault(_t9, _si9)
    except Exception:
        pass
    # TYPED BODY SIZES OVERRIDE CANON (the author's typed 'small breasts' for
    # Tifa; canon 'large breasts' must yield): a typed cluster tag binds
    # to the nearest PRECEDING persona in the user's text, claims that
    # subject's cluster slot, and -- sitting first in the line -- wins
    # the per-subject dedup over the canon entry.
    low0 = (base or "").lower()
    for t9 in typed_direct:
        if not any(t9 in fam for fam in _CLUSTERS):
            continue
        tp9 = low0.find(t9)
        own9, best9 = None, -1
        if tp9 < 0:
            # 'blonde milf' yields the tag `blonde hair`, which is nowhere
            # in the text to anchor on -- it still describes the one being
            # the user wrote, whoever the dice cast for her
            if sum(c for _n, c in _subject_spans(base)) == 1:
                tag_subj[t9] = 0
            continue
        for i9, s9 in enumerate(subjects):
            p9 = str(s9.get("persona") or "").split(" (")[0]
            pp9 = low0.rfind(p9, 0, tp9) if p9 else -1
            if p9 and pp9 >= 0 and pp9 > best9:
                best9, own9 = pp9, i9
        if own9 is None and len(subjects) == 1:
            own9 = 0
        # GENERATED personas are not in the text to anchor on; when the
        # user described exactly one being ("blonde milf ..."), a typed
        # trait is that being's, whoever the dice cast for it
        if own9 is None and sum(c for _n, c in _subject_spans(base)) == 1:
            own9 = 0
        if own9 is not None:
            tag_subj[t9] = own9
    # VIEWPOINTS COMPOSE (the author's): from above + from behind is a legitimate
    # angled view between the axes -- only OPPOSITE-axis pairs are illegal.
    _OPPOSITES = ({"from above", "from below"},
                  {"from behind", "from front"},
                  {"from behind", "straight-on"},
                  {"high angle", "low angle"})
    for pair in _OPPOSITES:
        if pair <= set(tags):
            later = max(pair, key=tags.index)
            tags.remove(later)

    for fam in _CLUSTERS:
        # heterochromia legitimately carries two eye colours; a marked
        # multicolour hairstyle legitimately carries two hair colours
        if "blue eyes" in fam and any("heterochromia" in t for t in tags):
            continue
        if "blonde hair" in fam and any(
                m in tags for m in ("multicolored hair", "two-tone hair",
                                    "streaked hair", "gradient hair")):
            continue
        # ONE VALUE PER BODY -- and an unattributed value belongs to SOME
        # body. This used to give every unattributed tag one shared slot,
        # so 'a woman with short hair and a man with long hair' (no
        # persona to bind either hair to) kept one length and dropped the
        # other, on a typed pair, in a cast of two. Unattributed values
        # now get as many slots as there are heads not already claimed
        # by an attributed one: a family holds at most one value per
        # body, which is what the rule always meant.
        _heads9 = sum(cast.get(k, 0) for k in ("female", "male", "futa",
                                              "other")
                      if isinstance(cast.get(k, 0), int)) or 1
        seen_fam = set()
        free9 = 0
        pruned = []
        for t in tags:
            if t in fam:
                owner = tag_subj.get(t)
                if owner is None:
                    if len(seen_fam) + free9 >= _heads9:
                        continue
                    free9 += 1
                else:
                    if owner in seen_fam:
                        continue
                    seen_fam.add(owner)
            pruned.append(t)
        tags = pruned

    # THE VOLUNTEER CHANNEL (round 11): the model writes content outside
    # the offered slots, and pruning the plan does not stop the mapped
    # tags. Two scrubs, both ownership-based:
    # (a) a BYSTANDER's genital tags go -- the brief suppression stops
    #     the offered path, this stops the volunteered one;
    # (b) a persona's SIZE-cluster tags must be canon or typed -- 'flat
    #     chest' volunteered onto Aerith is neither.
    # A TAG BOUND TO A BODY MUST FIT THAT BODY. `male chest` reached a
    # blonde milf through bridge 2, and the gates above only see group
    # acts and injections. Two axes per subject: what the body IS (a tag
    # named `male ...` needs a male subject; `female ...` a female or
    # futa) and what it HAS (penis-anatomy needs male or futa; breast-
    # anatomy needs female or futa). Unbound tags are judged against the
    # whole cast, so nothing a real body could carry is lost.
    _stage("verifying the tags")
    _kinds_all = [s2.get("kind") for s2 in subjects]
    for t in list(tags):
        if t in typed_direct:
            continue                      # typed is law
        j9 = tag_subj.get(t)
        ks = ([subjects[j9].get("kind")] if j9 is not None and
              j9 < len(subjects) else _kinds_all)
        if not ks:
            continue
        is_m = any(k == "male" for k in ks)
        has_p = any(k in ("male", "futanari") for k in ks)
        is_f = any(k in ("female", "futanari") for k in ks)
        is_fu = any(k == "futanari" for k in ks) or "futanari pov" in tags
        # THE MEASURED LEAN, at the gate as well as at the draw: bridge 2
        # mapped the plan's 'twin braids' to `braid` (worn by women on 96%
        # of its posts) on one of two men. Same table, same thresholds as
        # the fast planner's veto; unmeasured stays neutral.
        _fs = _fastplan._tag_female_share(t, level)
        _lean_bad = (_fs is not None and ((_fs >= 0.95 and not is_f)
                                          or (_fs <= 0.05 and not is_m)))
        if (_lean_bad or (sm.FUTA_NAMED.search(t) and not is_fu)
                or (sm.MALE_NAMED.search(t) and not is_m)
                or (sm.MALE_ANATOMY.search(t) and not has_p)
                or (sm.FEMALE_NAMED.search(t) and not is_f)
                or (sm.FEMALE_ANATOMY.search(t) and not is_f)):
            tags.remove(t)
            stripped.append((t, "does not fit the body it is bound to"))
    # THE ACT RULES THE POSES (the author's 2026-09-06, measured): a proposed
    # pose, pair act or gaze whose share beside the scene's act falls
    # under .6 of its share at the level is not in this picture -- 'head
    # between breasts', 'head back', 'looking down' beside a pov fellatio;
    # a second PAIR posture beside the act needs to be measured with it
    _act_v = opts.get("_act")
    if _act_v:
        _cam_v = (opts.get("_camera") or (None, None, None))[1] or ""
        _req_v = set(opts.get("_required_content") or [])
        # the typed act's own pose (its stance, its measured family pick)
        # is the act's fact, measured where it was drawn (2026-09-11)
        _derived_v = {str(x) for x in (opts.get("_act_derived") or ())}
        for t in list(tags):
            if t in typed_direct or t == _act_v or t in _req_v or t in _derived_v:
                continue
            fl = _gloss_flags(t)
            _pair_t = (sm.arity_of(t, banks) or 1) > 1
            _gaze = bool(fl & {"expression"}) and _GAZE_TAG_RE.search(t)
            if not (fl & {"pose", "act"} or _pair_t or _gaze):
                continue
            lf = act_lift(_act_v, t, level, view=_cam_v, live=True)
            if (lf is not None and lf < 0.6) or (_pair_t and (lf is None or lf < 0.6)):
                tags.remove(t)
                stripped.append((t, "does not fit the act (measured)"))
    # THE USER'S EXCLUSIONS GATE THE LINE (typed words still win)
    _xterms_v = opts.get("_excluded_terms") or exclusion_terms(opts)
    if _xterms_v:
        for t in list(tags):
            if t in typed_direct:
                continue
            if _hits_exclusion(t, _xterms_v):
                tags.remove(t)
                stripped.append((t, "excluded by the user"))
    _GENITAL_RE = re.compile(
        r"\bpenis\b|\bpussy\b|\btesticles\b|pubic hair|\berection\b")
    _SIZE_FAMS = [f for f in _CLUSTERS
                  if "small breasts" in f or "small penis" in f]
    edges9 = bool(plan.get("interactions"))
    for t in list(tags):
        j9 = tag_subj.get(t)
        if j9 is None or j9 >= len(subjects):
            continue
        if edges9 and busy and j9 not in busy and \
                sm.SPICE_ORDER[level] >= sm.SPICE_ORDER["nsfw"] and \
                _GENITAL_RE.search(t):
            tags.remove(t)
            stripped.append((t, "bystander"))
            continue
        if subjects[j9].get("persona") and \
                any(t in f for f in _SIZE_FAMS) and \
                t not in canon_by_subj[j9] and \
                not (t in typed_direct and tag_subj.get(t) == j9):
            tags.remove(t)
            stripped.append((t, "neither canon nor typed for persona"))

    # SEX POSITION COMPATIBILITY (the author's: 'all fours + girl on top is
    # implausible', but doggystyle + sex from behind + all fours COMBINE
    # -- they are the same rear position from different aspects). Prune
    # only ACROSS incompatible families; keep any combination WITHIN one.
    # The first-present family wins (engine-rolled/typed position leads).
    _POS_FAM = {}
    for fam, members in (
        ("rear", ("doggystyle", "all fours", "sex from behind",
                  "prone bone", "bent over", "top-down bottom-up")),
        ("top", ("girl on top", "cowgirl position",
                 "reverse cowgirl position", "upright straddle",
                 "straddling")),
        ("front", ("missionary", "mating press", "full nelson")),
        ("side", ("spooning",)),
        ("stand", ("standing sex",))):
        for m in members:
            _POS_FAM[m] = fam
    present = [t for t in tags if t in _POS_FAM]
    if present:
        keep_fam = _POS_FAM[present[0]]
        for t in present:
            if _POS_FAM[t] != keep_fam:
                tags.remove(t)
                stripped.append((t, "sex position clashes with %s family"
                                 % keep_fam))

    # AN ACT CONSTRAINS THE BODY (the author's: "standing and doing blowjob to
    # someone is hard"). The general rule is that acts and postures are
    # not independent -- fellatio wants kneeling or sitting, missionary
    # wants lying -- and a line carrying both leaves the model to pick
    # one and quietly drop the other.
    #
    # Which pairs clash is MEASURED, not listed: act_posture.json holds,
    # per act, how often each pose co-occurs with it against that pose's
    # average across all acts. A pose far below its own average is one
    # the act excludes. An act with no row is unmeasured and therefore
    # neutral, the same rule the affinity tables follow.
    #
    # A TYPED POSE IS NEVER PRUNED. Under the standing precedence rule
    # what the user wrote is the request; if they ask for a standing
    # fellatio they get one.
    # THE RULED WORDS ARE RULED ON EVERY PATH (2026-09-16). ruled_out() was
    # consulted by each emitter that draws, and the language model is not
    # one of them: it writes the plan's palette itself, and 'spot color'
    # reached the line that way. What the request typed stays -- the words
    # are typed-only, not banned.
    _ruled_v = ruled_out()
    _derived_ok = {str(x) for x in ((opts or {}).get("_act_derived") or ())}
    for t in list(tags):
        if t in _ruled_v and t not in typed_direct and t not in _derived_ok:
            tags.remove(t)
            stripped.append((t, "typed only"))

    # A PICTURE WITHOUT COLOUR HAS NO COLOURS TO DESCRIBE (the author,
    # 2026-09-16). With a colourless palette typed, a rolled colour
    # statement goes where the booru measures the pair far under chance:
    # 'greyscale' beside 'blonde hair' 213 posts of ~37,000 expected, beside
    # 'blue eyes' 'monochrome' 4,045 of ~44,000 -- while 'dark skin' stays
    # (6,857, near chance: shading still shows a skin tone). Typed colours
    # are the request's.
    _mono_typed = [t for t in _COLOURLESS if t in tags and t in typed_direct]
    if _mono_typed:
        for t in list(tags):
            if t in typed_direct or t in _COLOURLESS:
                continue
            _w9 = t.split()
            if not _w9 or _w9[0] not in _CANON_COLOURS:
                continue
            _lf = pair_lift(_mono_typed[0], t)
            if _lf is not None and _lf < 0.5:
                tags.remove(t)
                stripped.append((t, "%s leaves no colour to see (%.0f%% of chance on the booru)"
                                 % (_mono_typed[0], _lf * 100)))

    # WHOSE BODY A TAG IS ON (2026-09-16: two girls picnicking, one
    # 'standing', one 'sitting' -- both sat in the act band with no owner,
    # defaulted to the first girl, and the stance rule struck one). The
    # subject owner first; then the plan's own act owner (a pose or
    # self-action carries its subject); a tag with neither is unowned in a
    # scene of several, and the per-body rules leave it alone.
    _act_owner9 = {}
    for _c9, _tl9 in mapping.items():
        _o9 = origins.get(str(_c9).lower())
        if _o9 and _o9[0] == "act":
            for _t9 in _tl9:
                _act_owner9.setdefault(_t9, _o9[1])

    def _owner9(t):
        if t in tag_subj:
            return tag_subj[t]
        if t in _act_owner9:
            return _act_owner9[t]
        return 0 if len(subjects) <= 1 else None

    # A SUBJECT HAS TWO HANDS (the author, 2026-09-16: 'holding, heart
    # hands, hand on own thigh' on one woman). What occupies a hand is the
    # booru's own vocabulary, not a list: the tag groups 'hands' and
    # 'gestures', the arm and hand sections of 'posture', and the holding
    # verbs. Two rules, per subject, in claim order (typed, the typed act's
    # own, the character's canon, then the line):
    #   * a claim the booru measures against a kept one goes -- 'heart
    #     hands' beside 'hand on own thigh' is 5% of chance, beside
    #     'holding' 15%, while 'v' beside 'hand on own hip' is 188%;
    #   * past two claims, the rest go: every claim spends at least one
    #     hand. The holding verbs are one claim ('holding, holding gun' is
    #     one grip), so the count can only err towards keeping.
    _canon_all = {x for _cs in canon_by_subj for x in _cs}
    _hand_by_subj = {}
    for t in tags:
        if _hand_claim(t) and _owner9(t) is not None:
            _hand_by_subj.setdefault(_owner9(t), []).append(t)
    for _sj, _claims in _hand_by_subj.items():
        _rank = lambda t: (0 if t in typed_direct else 1 if t in _derived_ok
                           else 2 if t in _canon_all else 3)
        _kept, _keys = [], []
        for t in sorted(_claims, key=lambda t: (_rank(t), tags.index(t))):
            _key = "holding" if _is_holding(t) else t
            _typed9 = t in typed_direct or t in _derived_ok
            _clash = None
            for k in _kept:
                if _is_holding(t) and _is_holding(k):
                    continue                       # the same grip
                _lf = pair_lift(k, t)
                if _lf is not None and _lf < 0.5:
                    _clash = (k, _lf)
                    break
            if not _typed9 and _clash:
                tags.remove(t)
                stripped.append((t, "%s occupies the hands (%.0f%% of chance on the booru)"
                                 % (_clash[0], _clash[1] * 100)))
                continue
            if not _typed9 and _key not in _keys and len(_keys) >= 2:
                tags.remove(t)
                stripped.append((t, "both hands are taken (%s)" % ", ".join(_keys)))
                continue
            _kept.append(t)
            if _key not in _keys:
                _keys.append(_key)

    # ONE BODY, ONE STANCE (2026-09-16: 'lying, on side, the pose' and
    # 'on back ... the pose' on one woman). 'the pose' is a real booru pose
    # -- flat on the stomach, feet in the air -- and the booru says it
    # implies 'on stomach'. Each pose tag stands for the base stances it
    # implies (tree_vocab's bases: standing, sitting, lying, on back, on
    # side, on stomach, kneeling ...); two tags on one subject clash when a
    # stance of one and a stance of the other are different, neither
    # implies the other, and the booru shows them together under half of
    # chance IN SOLO POSTS -- one body; across 1girl posts comics and
    # multiple views put 'on stomach' beside 'on back' at 153% of chance,
    # in solo posts at 26%. Claim order as for the hands:
    # typed, the typed act's own, canon, then the line.
    _bases9 = set((_json_table(_TREE_VOCAB, "tree_vocab.json") or {}).get("bases") or [])
    if _bases9 and _implications():
        _st_by_subj = {}
        for t in tags:
            fl9 = _gloss_flags(t) or set()
            if not (fl9 & {"pose"} or t in _bases9):
                continue
            _stances9 = ({t} | set(implied(t))) & _bases9
            if _stances9 and _owner9(t) is not None:
                _st_by_subj.setdefault(_owner9(t), []).append((t, _stances9))
        _canon_all9 = {x for _cs in canon_by_subj for x in _cs}
        for _sj, _items in _st_by_subj.items():
            _rank9 = lambda it: (0 if it[0] in typed_direct else 1 if it[0] in _derived_ok
                                 else 2 if it[0] in _canon_all9 else 3, tags.index(it[0]))
            _kept9 = []
            for t, sts in sorted(_items, key=_rank9):
                _clash9 = None
                for k, ksts in _kept9:
                    for a in sts:
                        for b in ksts:
                            if a == b or a in implied(b) or b in implied(a):
                                continue
                            lf = pair_lift(a, b, "solo")
                            if lf is not None and lf < 0.5:
                                _clash9 = (k, a, b, lf)
                                break
                        if _clash9:
                            break
                    if _clash9:
                        break
                if _clash9 and t not in typed_direct and t not in _derived_ok:
                    tags.remove(t)
                    stripped.append((t, "%s says %s, this says %s (%.0f%% of chance on the booru)"
                                     % (_clash9[0], _clash9[2], _clash9[1], _clash9[3] * 100)))
                    continue
                _kept9.append((t, sts))

    # ONE SUBJECT IS DRESSED OR UNDRESSED, NEVER BOTH (the author,
    # 2026-09-16): a whole nudity state and a garment on the same subject
    # contradict each other. What the request wrote wins; with neither
    # typed, the nudity wins and the clothes go (two subjects may of
    # course differ -- this reads each subject's own tags).
    _nude_by_subj = {}
    for t in tags:
        if _is_whole_nudity(t):
            _nude_by_subj.setdefault(tag_subj.get(t, 0), []).append(t)
    for _sj, _nudes in _nude_by_subj.items():
        _clothes = [t for t in tags if implies_clothing(t) and tag_subj.get(t, 0) == _sj]
        if not _clothes:
            continue
        _typed_clothes = [t for t in _clothes if t in typed_direct]
        _typed_nude = [t for t in _nudes if t in typed_direct]
        if _typed_clothes and _typed_nude:
            # the request wrote both: its call for what it WROTE -- the
            # garments the engine rolled on top still go (2026-09-16:
            # 'completely nude, white thighhighs' typed kept a rolled
            # 'skirt' and 'see-through clothes' beside them)
            for t in [c for c in _clothes if c not in _typed_clothes]:
                tags.remove(t)
                stripped.append((t, "%s leaves no clothes on this subject" % _typed_nude[0]))
            _clothes = [c for c in _clothes if c in tags]
            # and where the engine already formed the booru's compound for
            # a garment ('nude, naked shirt'), the bare garment is redundant
            for t in _clothes:
                if ("naked " + t.split()[-1]) in tags:
                    tags.remove(t)
                    stripped.append((t, "said by naked %s" % t.split()[-1]))
            continue
        if _typed_clothes:
            for t in _nudes:
                tags.remove(t)
                stripped.append((t, "the request dressed this subject (%s)" % _typed_clothes[0]))
        else:
            for t in _clothes:
                tags.remove(t)
                stripped.append((t, "%s leaves no clothes on this subject" % _nudes[0]))

    # ONE NUDITY STATE PER BODY (2026-09-17: 'bottomless' beside 'nude' on
    # one woman). The nudity slot holds whole states (nude, bottomless,
    # naked shirt) and partial ones (no bra, breasts out, undressing); beside
    # a kept one, another the booru shows with it under half of chance IN
    # SOLO POSTS goes -- nude with bottomless 28%, with no panties 25%, with
    # no bra 20%, completely nude with undressing 38% -- while what a nude
    # body shows stays (nipples 999%, pussy 1,141%). Claim order: typed, the
    # typed act's own, canon, then the line.
    _nude_items = {}
    for t in tags:
        if (_is_nudity_tag(t) or _is_whole_nudity(t)) and _owner9(t) is not None:
            _nude_items.setdefault(_owner9(t), []).append(t)
    for _sj, _items in _nude_items.items():
        if len(_items) < 2:
            continue
        _rank_n = lambda t: (0 if t in typed_direct else 1 if t in _derived_ok
                             else 2 if t in _canon_all else 3, tags.index(t))
        _kept_n = []
        for t in sorted(_items, key=_rank_n):
            _clash_n = None
            for k in _kept_n:
                if t in implied(k) or k in implied(t):
                    continue
                _lf = pair_lift(k, t, "solo")
                if _lf is not None and _lf < 0.5:
                    _clash_n = (k, _lf)
                    break
            if _clash_n and t not in typed_direct and t not in _derived_ok:
                tags.remove(t)
                stripped.append((t, "%s is this body's state (%.0f%% of chance in solo posts)"
                                 % (_clash_n[0], _clash_n[1] * 100)))
                continue
            _kept_n.append(t)

    # ONE PAIR OF LEGS, ONE LEGWEAR ANSWER (2026-09-17: 'pants, socks,
    # thighhighs' on one woman). Garments of one body that sit on the legs
    # and feet -- the clothes table's bottom, legwear and feet slots, by the
    # tag or its head word -- are measured against each other in solo posts;
    # beside a kept one, a garment the booru shows with it under half of
    # chance goes (the roll's own gate measures only what it rolled, in the
    # 1girl universe, and skipped legwear against legwear). Two colours of
    # one garment are not a clash. Claim order: typed, the typed act's own,
    # canon, then the line; a typed garment always stays.
    # BARE LEGS ARE AN ANSWER TOO (2026-09-17: 'barefoot, pantyhose,
    # sandals, bare legs' on one subject from the LLM plan). The booru's own
    # tag groups name the states of the legs and feet -- nudity/legs
    # (barefoot, bare legs, no pants) and feet/style (barefoot, no shoes) --
    # and they join the same measure: barefoot with pantyhose is 6% of
    # chance, bare legs with thighhighs 2%, while barefoot with sandals
    # (78%) and bare legs with shorts (218%) stay.
    _cl_slots9 = ((_clothes_table() or {}).get("slots") or {})
    _tree9 = _tree_tags()

    def _leg_head(t):
        for _sl in ("bottom", "legwear", "feet"):
            _items = _cl_slots9.get(_sl) or {}
            if t in _items:
                return t
            if t.split()[-1] in _items:
                return t.split()[-1]
        _g9 = (_tree9.get(t) or {}).get("groups") or {}
        if _g9.get("nudity") == "legs" or _g9.get("feet") == "style":
            return t
        return None
    _leg_items = {}
    for t in tags:
        if _leg_head(t) and _owner9(t) is not None:
            _leg_items.setdefault(_owner9(t), []).append(t)
    for _sj, _items in _leg_items.items():
        if len(_items) < 2:
            continue
        _rank_l = lambda t: (0 if t in typed_direct else 1 if t in _derived_ok
                             else 2 if t in _canon_all else 3, tags.index(t))
        _kept_l = []
        for t in sorted(_items, key=_rank_l):
            _clash_l = None
            for k in _kept_l:
                if _leg_head(k) == _leg_head(t) or t in implied(k) or k in implied(t):
                    continue
                _lf = pair_lift(_leg_head(k), _leg_head(t), "solo")
                if _lf is not None and _lf < 0.5:
                    _clash_l = (k, _lf)
                    break
            if _clash_l and t not in typed_direct and t not in _derived_ok:
                tags.remove(t)
                stripped.append((t, "these legs already have %s (%.0f%% of chance in solo posts)"
                                 % (_clash_l[0], _clash_l[1] * 100)))
                continue
            _kept_l.append(t)

    # SHUT EYES ARE THE ONLY EYE STATE (the author, 2026-09-16: 'closed
    # eyes, slit pupils' on one subject): a colour, a pupil, a gaze or a
    # second eye state cannot be seen on shut eyes. The rule runs per
    # subject and the request wins, as everywhere.
    _shut_by_subj = {}
    for t in tags:
        if t in _SHUT_EYES:
            _shut_by_subj.setdefault(tag_subj.get(t, 0), []).append(t)
    for _sj, _shuts in _shut_by_subj.items():
        _states = [t for t in tags if t not in _SHUT_EYES
                   and tag_subj.get(t, 0) == _sj and is_eye_state(t)]
        if not _states:
            continue
        _typed_states = [t for t in _states if t in typed_direct]
        _typed_shut = [t for t in _shuts if t in typed_direct]
        if _typed_states and _typed_shut:
            continue                       # the request wrote both: its call
        if _typed_states:
            for t in _shuts:
                tags.remove(t)
                stripped.append((t, "the request wants this subject's eyes seen (%s)" % _typed_states[0]))
        else:
            for t in _states:
                tags.remove(t)
                stripped.append((t, "closed eyes leave no eye state to see"))

    # THE MEDIUM AND ITS DECORATION MUST BE MEASURED TOGETHER (the author,
    # 2026-09-16: '3d, spot color' -- 63,216 posts for the one, 29,907 for
    # the other, 14 for the pair, about a fourteenth of chance). The medium
    # is a resolved axis of the picture; a palette or technique word beside
    # it is decoration, so the decoration yields when the booru does not
    # show the two together. Typed words are law, as everywhere.
    _med9 = (opts.get("_medium") or (None, None))[1] if opts else None
    if _med9 and _med9 in tags:
        for t in list(tags):
            if t == _med9 or t in typed_direct:
                continue
            _fl9 = _gloss_flags(t) or set()
            if not (_fl9 & {"style", "medium"}) and t not in (plan.get("palette") or []):
                continue                       # only the decoration is gated
            _lf9 = pair_lift(_med9, t)
            if _lf9 is not None and _lf9 < 0.5:
                tags.remove(t)
                stripped.append((t, "%s does not go with it (%.0f%% of chance on the booru)"
                                 % (_med9, _lf9 * 100)))

    _ap = _act_posture()
    if _ap:
        _acts_here = [t for t in tags if t in _ap]
        if _acts_here:
            # THE TYPED ACT'S OWN POSE OUTRANKS A ROLLED EXCLUDER
            # (2026-09-11): 'smile', rolled, suppressed 'stretching', the
            # stance the typed 'yoga' pins -- the line lost the pose the
            # prose still told. A tag the typed act implies (its stance,
            # its family pose, its object) is the typed act's; a rolled
            # row that excludes it yields itself.
            _derived = {str(x) for x in ((opts or {}).get("_act_derived") or ())}
            for _a9 in list(_acts_here):
                if _a9 in typed_direct or _a9 in _derived:
                    continue
                _hit = [t for t in (_ap[_a9].get("suppressed") or {}) if t in _derived and t in tags]
                if _hit:
                    tags.remove(_a9)
                    _acts_here.remove(_a9)
                    stripped.append((_a9, "excludes the typed act's %s" % _hit[0]))
            for t in list(tags):
                if t in typed_direct or t in _derived:
                    continue
                # AN ACT EXCLUDES POSTURES, NOT LOOKS (the author, 2026-09-16:
                # "why does ponytail veto holding bottle?" -- the table's
                # posture universe took hair styles and expressions along, so
                # 'holding bottle' struck 'ponytail' at 14% of its usual rate
                # and 'closed eyes' at 9%). Only a pose, a position or an act
                # can be excluded here; hair, eyes, faces and garments are
                # not postures and keep their own owners.
                _tf = _gloss_flags(t) or set()
                if not (_tf & {"pose", "act", "position"}):
                    continue
                for _a9 in _acts_here:
                    if t == _a9:
                        continue
                    _lift = (_ap[_a9].get("suppressed") or {}).get(t)
                    if _lift is not None:
                        tags.remove(t)
                        stripped.append(
                            (t, "%s excludes it (%.0f%% of its usual rate)"
                             % (_a9, _lift * 100)))
                        break

    # BALD MEANS NO HAIR, and that is one rule rather than a list of
    # pairs. the author's saw `bald female`, `bob cut` and "very long hair" in
    # one line -- three mutually exclusive states of the same head.
    #
    # The measured route cannot settle this one: `bald female` has 177
    # posts, so `bald female` + `bob cut` expects 1.5 co-occurrences and
    # observing 0 is not evidence (build_contradictions rightly refuses
    # anything under 5 expected). What IS certain is semantic -- a bald
    # head has no hair to cut, colour or tie -- so the whole hair
    # vocabulary is excluded at once. `_hair_vocab` already defines it,
    # which keeps this right for hair tags nobody has added yet.
    #
    # Typed wins, as always: ask for a bald woman and the rolled
    # hairstyle goes; ask for a bob and a rolled `bald` goes. When the
    # engine rolled both, `bald` is the one to drop -- it is far the
    # rarer roll and the hairstyle carries more of the picture.
    _BALD = [t for t in tags if t in ("bald", "bald female", "bald girl")]
    if _BALD:
        _hv = _hair_vocab()
        # hair ACCESSORIES too: 'blue hairband' rolled onto a typed bald
        # girl (harness 38). Body hair is not head hair and stays.
        _hair_else = [t for t in tags if t not in _BALD
                      and (t in _hv or (re.search(r"(?<![a-z])hair", t)
                           and not re.search(r"(pubic|armpit|chest|body|facial|"
                                             r"leg|arm|ass|anal|excessive) hair", t)))]
        if _hair_else:
            _bald_typed = any(t in typed_direct for t in _BALD)
            _hair_typed = any(t in typed_direct for t in _hair_else)
            if _bald_typed and not _hair_typed:
                _drop, _why = _hair_else, "bald head has no hair"
            else:
                _drop, _why = _BALD, "hair is present"
            for t in _drop:
                if t in tags:
                    tags.remove(t)
                    stripped.append((t, _why))

    # a monochrome scene has no coloured palette (sweep: monochrome +
    # greyscale + 'warm colors' in one line); earlier wins, and 'spot
    # color' is exempt -- it IS monochrome plus one colour
    _MONO = [t for t in ("monochrome", "greyscale") if t in tags]
    # every coloured palette word, not only 'X colors' (2026-09-13:
    # 'greyscale' beside 'colorful' three times in a hundred lines)
    _COLS = [t for t in tags if t.endswith(" colors") or t.endswith(" theme")
             or t in ("colorful", "multicolored", "rainbow", "pastel", "neon", "vivid colors")]
    if _MONO and _COLS:
        if tags.index(_MONO[0]) < tags.index(_COLS[0]):
            drop9, why9 = _COLS, "monochrome scene"
        else:
            drop9, why9 = _MONO, "coloured palette present"
        for t in drop9:
            tags.remove(t)
            stripped.append((t, why9))
    # AT MOST TWO palette tags (the author's: 'warm/pastel/muted colors
    # almost in every generation' -- three generic palettes in one line
    # is padding, and often contradiction); the first two win, which
    # favours the style's own measured palette
    _pal9 = [t for t in tags
             if t.endswith(" colors") or t.endswith(" theme")
             or t in ("colorful", "limited palette", "spot color",
                      "high contrast")]
    for t in _pal9[2:]:
        tags.remove(t)
        stripped.append((t, "palette cap (2)"))

    # verifier: no measured-contradiction pairs survive (the author's rule 15 --
    # 'no contradictory descriptions'); the earlier tag wins, the later drops
    # THIS VERIFIER WAS DEAD. The bank loads the table's lists as SETS,
    # and the guard below accepted only list/dict -- so every lookup
    # resolved to [] and no measured pair was ever enforced. Found during
    # the contradiction audit: 'bob cut, very long hair' typed together
    # both survived, and the two-subject A/B that "proved" the hair pairs
    # harmless proved nothing, because nothing was being applied.
    #
    # AND ONCE ALIVE IT ATE THE SECOND SUBJECT. The table is measured
    # conditioned on `solo` -- one body -- but was applied to the whole
    # line, so 'tifa lockhart with short hair and aerith gainsborough
    # with very long hair' kept one length and threw the other away:
    # exactly the cross-subject error _CLUSTERS was made per-subject to
    # prevent, and on TYPED tags, which the first law says are never
    # lost. Two rules, both already stated elsewhere in this function,
    # now applied here too:
    #   * PER SUBJECT: a pair contradicts only when both tags belong to
    #     the same subject, or when at least one is unbound (a scene
    #     fact like 'outdoors' against 'bed' is still one image).
    #   * TYPED WINS: when exactly one of the pair is typed, the other
    #     one drops whatever the order; two typed, or two rolled -- the
    #     earlier one wins, as before.
    contra = banks.get("_contradictions") or {}
    _COLL = (list, dict, set, frozenset, tuple)
    _typed_set = set(typed_direct)
    _n_heads = sum(cast.get(k, 0) for k in ("female", "male", "futa", "other")
                   if isinstance(cast.get(k, 0), int))

    def _clash(x, y):
        ox = contra.get(x)
        oy = contra.get(y)
        if not ((ox and y in (ox if isinstance(ox, _COLL) else ()))
                or (oy and x in (oy if isinstance(oy, _COLL) else ()))):
            return False
        sx, sy = tag_subj.get(x), tag_subj.get(y)
        if sx is not None and sy is not None:
            return sx == sy
        # UNBOUND PERSON-DESCRIPTORS IN A CAST OF TWO OR MORE cannot be
        # proven to be the same body: 'a woman with short hair and a man
        # with long hair' binds neither hair to anyone (no persona to
        # anchor on), and calling them a clash ate the man's. Unmeasured
        # is neutral here as everywhere else. Scene facts ('bed' against
        # 'outdoors') are not person-descriptors and still clash.
        if _n_heads >= 2 and _describes_a_person(x)                 and _describes_a_person(y):
            return False
        return True

    kept, dropped_contra = [], []
    for t in tags:
        loser = None
        for prev in kept:
            if not _clash(prev, t):
                continue
            if t in _typed_set and prev not in _typed_set:
                loser = prev            # the typed newcomer displaces it
            else:
                loser = t
            break
        if loser is None:
            kept.append(t)
        elif loser is t:
            dropped_contra.append(t)
        else:
            kept.remove(loser)
            dropped_contra.append(loser)
            kept.append(t)
    tags = kept

    # HAND OCCUPANCY (the author's, current-scope testing): 'holding hands'
    # claims one hand of EVERY participant; in a cast of <=2 nobody has
    # both own hands free, so both-own-hands tags are anatomy errors
    # ('own hands together' next to 'holding hands' says the girls cannot
    # actually hold hands with each other). One-hand tags (hand on own
    # hip) stay legal -- the other hand does the holding.
    _BOTH_OWN_HANDS = {
        "own hands together", "own hands clasped", "hands in pockets",
        "hands on own hips", "hands on own face", "hands on own cheeks",
        "crossed arms", "arms behind back", "arms behind head",
        "hugging own legs", "hands in hair", "praying"}
    n_cast = sum(cast.get(k, 0) for k in ("female", "male", "futa", "other"))
    if "holding hands" in tags and n_cast <= 2:
        for t in [x for x in tags if x in _BOTH_OWN_HANDS]:
            tags.remove(t)
            stripped.append((t, "hands occupied by 'holding hands'"))

    # A COVER STATE STANDS FOR ITS PART (the author, 2026-09-19: "'covered
    # navel' or 'covering navel' should not add pure 'navel' tag to the prompt
    # even if it was rolled prior ... with the pure 'navel' tag image gen model
    # wants to draw it in its primal state -- showing"). The bare part word
    # yields to the state that covers it; a typed part stays (typed is law).
    _typed_low = {str(t).lower() for t in (user_tags or [])}
    for _cv in [t for t in tags if re.match(r"^(covered|covering)\s", t)]:
        _part = re.sub(r"^(covered|covering)\s+(own\s+|another's\s+|erect\s+)?", "", _cv).strip()
        _forms = {_part, _part + "s", _part[:-1] if _part.endswith("s") else _part}
        for t in [x for x in tags if x in _forms and x != _cv and x not in _typed_low]:
            tags.remove(t)
            stripped.append((t, "covered: '%s' stands for it" % _cv))

    # verifier: the tag part must measure at the target level
    line = list(dict.fromkeys(conditioning(mode, level, _cond_rng) + count_tags(cast, mode) + tags))
    measured = sm.implied_spice(tags)
    dropped_for_level = []
    if sm.SPICE_ORDER[measured] > sm.SPICE_ORDER[level]:
        keep = []
        for t in tags:
            if sm.SPICE_ORDER[sm.safety_floor(t)] > sm.SPICE_ORDER[level]:
                dropped_for_level.append(t)
            else:
                keep.append(t)
        tags = keep
        line = list(dict.fromkeys(conditioning(mode, level, _cond_rng) + count_tags(cast, mode) + tags))
        measured = sm.implied_spice(tags)

    # UP-INJECTION (defect #2 closed): a target ABOVE the measured level
    # gets content injected from the floor table itself, cast-checked and
    # arity-checked, so injection can neither invent nor overshoot nor
    # need a missing partner.
    # with pre-injection carrying the level, this net should rarely fire
    injected = []
    if sm.SPICE_ORDER[measured] < sm.SPICE_ORDER[level]:
        # the late net knows what the picture already shows: an act that needs a
        # partner's part bare ('penis awe') is offered only when it is (2026-09-20)
        for t_inj in draw_injectables(level, cast, mode, banks, rng, 2, ctx={
                "place": (opts.get("_location") or (None, None, None))[1],
                "genre": (opts.get("_genre") or (None, None))[1],
                "bare": list(tags), "nude": any(_is_whole_nudity(t) for t in tags),
                # what the picture already wears, so the net can ask the booru
                # whether its own pick belongs beside it (2026-09-21)
                "worn": [t for t in tags if _is_garment(t) or implies_clothing(t)]}):
            if t_inj in tags:
                continue
            # ... NOR A SECOND STANCE (the author's live test, 2026-09-27: 'on side'
            # injected first, then 'top-down bottom-up' by this net): a body holds
            # one posture, and the one already on the line stands
            if typed_stance([t_inj]) and typed_stance(tags):
                continue
            if implies_clothing(t_inj) and any(_is_whole_nudity(t) for t in tags):
                continue                       # no clothes on an undressed subject (2026-09-16)
            # ...AND NO UNDRESSED SUBJECT IN CLOTHES (2026-09-18): this net
            # fires AFTER the verifier's nudity-versus-clothes rule, so a
            # whole state injected here never met it -- 'nude' landed beside
            # 'collared shirt, open shirt, bloomers'. The outfit is settled
            # by now; the state that would contradict it is not drawn.
            if _is_whole_nudity(t_inj) and any(implies_clothing(t) for t in tags):
                continue
            # ... NOR A SECOND OUTFIT ON A DRESSED ONE (2026-09-19: 'bodysuit,
            # camisole' injected beside a typed sweater and pants): a garment is
            # the clothes roll's to give, never the level net's
            _cs9 = (_clothes_table() or {}).get("slots") or {}
            if any(t_inj in (_cs9.get(sl9) or {}) for sl9 in ("top", "bottom", "swim", "uniform", "traditional", "sexual"))                     and any(implies_clothing(t) for t in tags):
                continue
            tags.append(t_inj)
            injected.append(t_inj)
            measured = sm.implied_spice(tags)
            if sm.SPICE_ORDER[measured] >= sm.SPICE_ORDER[level]:
                break

    # (after the late net above: it is the net that adds 'completely nude' and
    # 'undressing' to a line already settled)
    # THE WHOLE UNDRESS SAYS IT ALL (the author, 2026-10-03, contradictory tags on
    # one body: 'topless female ... completely nude'). A body that is nude is not
    # also topless or bottomless: on the subject that carries a whole undress her
    # partial bare states go, unless the user typed them.
    try:
        _typed_wn = {str(x).lower() for x in (user_tags or [])}
        _PARTIAL_BARE = ("topless female", "topless male", "topless", "bottomless", "no bra", "no panties")
        # (only a COMPLETE undress subsumes: _is_whole_nudity also counts 'topless
        # female' and 'bottomless', which replace half the outfit and may stand
        # together)
        _nud_wn = ((_spice_table() or {}).get("slots") or {}).get("nudity") or {}
        _complete_wn = [t for t in tags if t in ("nude", "completely nude", "naked")
                        or (_nud_wn.get(t) or {}).get("section") == "complete"]
        for _wn in _complete_wn:
            _s_wn = tag_subj.get(_wn, 0)
            for t in list(tags):
                if t in _PARTIAL_BARE and t not in _typed_wn and tag_subj.get(t, 0) == _s_wn:
                    tags.remove(t)
                    stripped.append((t, "the whole undress says it"))
        # ... AND NOTHING IS TAKEN OFF A BODY WITH NOTHING ON ('bottomless, topless
        # female, undressing'; 'completely nude, undressing'): the act needs a
        # garment over the chest or the hips on its subject
        if "undressing" in tags and "undressing" not in _typed_wn:
            _s_un = tag_subj.get("undressing", 0)
            _own_un = [t for t in tags if tag_subj.get(t, 0) == _s_un]
            _cs_un = (_clothes_table() or {}).get("slots") or {}
            if not any((_cl_slots_of(_cs_un, t) - {"legwear"}) or t in (_cs_un.get("sexual") or {})
                       for t in _own_un):
                tags.remove("undressing")
                stripped.append(("undressing", "nothing on to take off"))
    except Exception:
        pass
    # WHAT BELONGS TO ANOTHER GENRE LEAVES THE LINE, by whatever door it came -- a
    # roll or the mapper's pick. What the user's own words say stays (typed is
    # law: 'a detective examining a corpse' has its corpse in any genre); a
    # comparison -- 'like a dead body' -- is not read as a tag in the first place.
    try:
        _segs_go = {str(x).lower() for x in (user_tags or [])}
        for t in list(tags):
            if str(t).lower() in _segs_go:
                continue
            if not genre_only_ok(t, genre_name):
                tags.remove(t)
                stripped.append((t, "only in its own genre (%s is not it)" % (genre_name or "no genre")))
    except Exception:
        pass
    # ... AND NO PART OR STATE OF A GARMENT WITHOUT THE GARMENT (lone_garment_states)
    try:
        _keep_lg = {str(x).lower() for x in (user_tags or [])} | {str(x).lower() for _cs in canon_by_subj for x in _cs}
        # each subject's tags: the ones filed under her, and what her own plan
        # fields map to -- a garment two subjects wear is filed under one of them
        # (the boy's `open fly` was judged without the `pants` both wore)
        _grp_lg = {}
        for t9 in tags:
            _grp_lg.setdefault(tag_subj.get(t9, 0), []).append(t9)
        for _si_lg, _ps_lg in enumerate(plan.get("subjects") or []):
            if not isinstance(_ps_lg, dict):
                continue
            _vals_lg = list(_ps_lg.get("outfit") or [])
            if _si_lg < len(subjects):
                _vals_lg += [str(x9) for x9 in (subjects[_si_lg].get("locked_canon") or {})]
            for _c_lg in _vals_lg:
                for t9 in (mapping.get(str(_c_lg)) or mapping.get(str(_c_lg).lower()) or [str(_c_lg).lower()]):
                    if t9 in tags and t9 not in _grp_lg.setdefault(_si_lg, []):
                        _grp_lg[_si_lg].append(t9)
        _lack_lg = []
        for _t_lg, _why_lg in lone_garment_states(_grp_lg, keep=_keep_lg, lacking=_lack_lg):
            if _t_lg in tags:
                tags.remove(_t_lg)
                stripped.append((_t_lg, _why_lg))
        # THE USER'S DECORATION BRINGS ITS GARMENT (a typed 'hat ribbon' with no hat
        # typed or rolled: the ribbon is law, and it is tied to a hat)
        for _sj_lg, _d_lg, _g_lg in _lack_lg:
            if _g_lg not in tags:
                tags.insert(tags.index(_d_lg) if _d_lg in tags else len(tags), _g_lg)
                tag_subj[_g_lg] = _sj_lg
                opts.setdefault("_brought_garments", {})[_g_lg] = _sj_lg
    except Exception:
        pass

    # PRESENTATION SORT (Scheme 1): rebuild the final tag order from the
    # origins -- contiguous per-subject blocks, then actions, interactions,
    # and the general bands. Engine-added tags classify by context; anything
    # unknown lands in the scene band rather than getting lost.
    _BAND = {"subject": 2, "act": 3, "inter": 4, "medium": 4.6,
             "style": 5, "genre": 6,
             "scene": 7, "light": 8, "camera": 9, "fx": 10}
    tag_meta = {}
    for c, tl in mapping.items():
        o = origins.get(str(c).lower())
        if o:
            for t2 in tl:
                tag_meta.setdefault(t2, o)
    for si2, ct in enumerate(canon_by_subj):
        for t2 in ct:
            tag_meta.setdefault(t2, ("subject", si2))
    # the garment a typed decoration brought is its subject's
    for t2, si2 in (opts.get("_brought_garments") or {}).items():
        tag_meta[t2] = ("subject", si2)
    # engine-resolved facts outrank concept-origin stamps for their own
    # tags ('science fiction' had inherited a subject block from a
    # subject-origin concept that happened to map it)
    # A GENRE WORD THAT NAMES A PERSON IS A SUBJECT'S (the author, 2026-09-27:
    # 'cavewoman', rolled with the prehistoric genre, sat among the scene tags):
    # the gloss or the lexicon says it is a person, and it is not the genre's
    # own name. It joins the first subject of its sex ('cavewoman' a woman,
    # 'caveman' a man), else the first subject, and ranks as an occupation.
    _genre_people = {}
    try:
        from promptstudio.engine import lexicon as _lx9
        # the genres' own names only: an alias may be a person ('cavewoman' is
        # one of prehistoric's aliases)
        _gnames9 = {g.split(" (")[0].lower() for g in _genre_pool()}
    except Exception:
        _lx9, _gnames9 = None, set()
    for _gt in (genre_tags or ([genre_name] if genre_name else [])):
        # a word typed for a subject stays the subject's even when it names
        # the genre too ('pirate captain', 'cyborg girl'; 2026-09-14)
        if (tag_meta.get(_gt) or ("",))[0] == "subject":
            continue
        _gl9 = str(_gt).lower()
        _person9 = bool(subjects) and _gl9 not in _gnames9 and (
            _describes_a_person(_gl9)
            or (_lx9 is not None and "person" in (_lx9.main_classes(_gl9) or ())))
        if _person9:
            # 'woman' is tried first: it ends in 'man' too
            _want9 = "female" if re.search(r"(woman|girl|female|lady|wife|queen|priestess)$", _gl9) else None
            if _want9 is None and re.search(r"(man|boy|male|king|priest)$", _gl9):
                _want9 = "male"
            _si9 = next((i9 for i9, s9 in enumerate(subjects) if _want9 and s9.get("kind") == _want9), 0)
            tag_meta[_gt] = ("subject", _si9)
            _genre_people[_gt] = _si9
        else:
            tag_meta[_gt] = ("genre", 0)
    if loc_name:
        tag_meta[loc_name] = ("scene", 0)
    _ctx = {(opts.get("_genre_anchor") or genre_name): ("genre", 0), loc_name: ("scene", 0),
            "indoors": ("scene", 0), "outdoors": ("scene", 0),
            "window": ("scene", 0), "crowd": ("scene", 0),
            (med_tag or ""): ("medium", 0),
            "traditional media": ("medium", 0),
            "no humans": ("subject", 0), "scenery": ("scene", 0),
            "landscape": ("scene", 0), "cityscape": ("scene", 0),
            "still life": ("scene", 0), "close-up": ("camera", 0),
            "solo focus": ("camera", 0),
            (framing or ""): ("camera", 0), (viewpoint or ""): ("camera", 0),
            (opts.get("_camera_extra") or ""): ("camera", 0)}
    for f2 in (opts.get("_focus") or []):
        _ctx[f2] = ("camera", 0)
    lt3 = opts.get("_lighting") or {}
    for w2 in [lt3.get("time"), lt3.get("weather")] + list(lt3.get("lighting") or []):
        if w2:
            _ctx[w2] = ("light", 0)
    st3 = (opts.get("_style") or (None, {}))
    if st3[0]:
        _ctx[st3[0]] = ("style", 0)
        for p3 in list((st3[1] or {}).get("palette") or {}):
            _ctx[p3] = ("style", 0)
    # ARTIST SUBSECTION (v1 rules carried whole): typed artists ALWAYS
    # honoured, the checkbox governs generation only; derivation follows
    # the resolved style (its measured artists, share-weighted), else a
    # uniform draw over the >=100-post pool; 1-3 at 50/30/20; per-mode
    # formatting ('@name' anima, descending '(name:w)' illustrious).
    art_in = list(user_tags)
    if st3[0]:
        art_in.append(st3[0])
    _ext_used = []
    # A WORD THE REQUEST CLAIMED AS A PLACE IS NOT AN ARTIST (2026-09-13:
    # 'in a cluttered cottage' drew the artist @cottage, 126 posts): the
    # artist scan reads the text without its place phrases, as it already
    # reads it without its character names
    _art_text = _sans_characters(base) or ""
    for _ph in (opts.get("_place_phrases") or []):
        _art_text = re.sub(r"(?<![a-z0-9])" + re.escape(str(_ph)) + r"(?![a-z0-9])", " ", _art_text, flags=re.I)
    # THE ARTIST FIELD'S ENTRIES ARE ARTISTS, WHOLE (2026-09-19): one entry per
    # comma, the '@' optional, a name the pool does not know kept as written,
    # parentheses and all; '(name:0.8)' keeps its weight. They never touch the
    # brief, so no scanner can read a name as a concept.
    _field_art = []
    for _e in re.split(r",(?![^()]*\))", str(opts.get("artist_hint") or "")):
        _e = _e.strip()
        _mw = re.match(r"^\((.+):\s*([0-9.]+)\)$", _e)
        _nm = (_mw.group(1) if _mw else _e).strip().lstrip("@").replace("_", " ").strip()
        if not _nm or not re.search(r"[A-Za-z0-9]", _nm):
            continue
        if _nm.lower() in pe.artist_denylist():
            continue
        _fa = pe._artist_format(_nm.lower(), mode)
        if _mw:
            _fa = "(%s:%s)" % (_fa, _mw.group(2))
        if _fa not in _field_art:
            _field_art.append(_fa)
    art_in = [t for t in art_in if str(t) not in _field_art]
    typed_art, gen_art, _art_note = pe.pick_artists(
        art_in, banks, mode, rng, bool(opts.get("gen_artists")) and not _field_art,
        base_text=_art_text, spice=level, fast=bool(opts.get("fast")),
        scene_tags=list(tags), look_text=", ".join(opts.get("_look_phrases") or []))
    typed_art = list(dict.fromkeys(_field_art + list(typed_art)))
    for a3 in typed_art + gen_art:
        if a3 not in tags:
            tags.append(a3)
        _ctx[a3] = ("style", 0)

    # NON-BOORU CONCEPTS THE CHECKPOINT KNOWS (the author's). These are written
    # by their own convention, not booru's: an artist here is '<name>
    # style', because '@name' asks for a danbooru artist tag that does not
    # exist. A style/genre/medium is written literally -- "new locations
    # though look the same as tags". They land in the band their KIND
    # belongs to, so the assembler orders them like any other concept.
    _ext_band = {"artist": "style", "style": "style", "genre": "genre",
                 "medium": "medium", "lighting": "light"}
    try:
        from promptstudio.library import external as _ext
        # a phrase booru already covers resolves to ITS tag, so the
        # measured co-occurrence behind that tag does the work
        _seen_low = {str(t).lower() for t in tags}
        for _p, _t in _ext.find_equivalents(base or ""):
            if _t.lower() not in _seen_low:
                tags.append(_t)
                _seen_low.add(_t.lower())
                _ctx[_t] = ("style", 0)
                _ext_used.append((_p, "booru-equivalent", _t))
        # TYPED IS HONOURED EVEN WHILE PENDING. 'pending' means the concept's
        # dependency fields are not filled yet, which is a reason the DICE
        # may not roll it -- not a reason to drop what the user wrote:
        # "in the style of van gogh" emitted nothing in 8 of 8 runs while
        # the UI promises typed concepts are always honoured. All 4,315
        # external artists are pending today; their render forms exist.
        _pool_art = pe._artist_pool()[0]
        for _n, _rec in _ext.find_typed(_sans_characters(base) or "", include_pending=True):
            # ONE ARTIST, ONE TAG (2026-09-16: 'drawn by wlop' wrote '@wlop,
            # WLOP style'). An artist the booru tags is the booru's: the
            # '@name' tag already carries it, measured; the registry's
            # '<name> style' form is for artists the booru never tagged.
            if (_rec or {}).get("kind") == "artist" and str(_n).lower().strip() in _pool_art:
                continue
            if (_rec or {}).get("prose_only"):
                _ext_used.append((_n, _rec.get("kind"), "(prose only)"))
                continue                          # said in the prose, never a tag (lighting rulings)
            _written = _ext.render(_n, "tag")
            # CASE-INSENSITIVE. The registry keeps display casing ('Anime
            # style') while our own prompt-craft vocabulary carries the
            # lowercase form, so an exact check emitted both: 'anime style,
            # Anime style'. Same concept, one tag.
            if _written and _written.lower() not in _seen_low:
                tags.append(_written)
                _seen_low.add(_written.lower())
                _ctx[_written] = (_ext_band.get(_rec.get("kind"), "style"), 0)
                _ext_used.append((_n, _rec.get("kind"), _written))
    except Exception:
        pass
    # PHRASES ON THE LINE (2026-09-11): an
    # unmapped concept that is VISUAL stays on the line as a phrase in its
    # own band, within a budget by detail -- the prose kept it, the tag
    # line lost it. Fillers, review-settled prose-only concepts and
    # phrases the tags already say stay off.
    _ph_budget = {"minimal": 0, "standard": 2, "detailed": 4}.get(
        str(opts.get("detail") or "standard"), 2)
    _phr = []
    if _ph_budget and unmapped:
        try:
            from promptstudio.library import concepts as _cl
            _settled = _cl.nl_only()
        except Exception:
            _settled = set()
        _seen_ph = {str(t).lower() for t in tags}
        _tagwords = {w for t in tags for w in str(t).lower().split()}
        # THE USER'S CANDIDATES FIRST, MAPPED OR NOT (2026-09-16). A candidate
        # formed whole ('sun-warmed wooden dock') maps to its head's tag
        # ('dock') and so left the unmapped list with its modifiers lost;
        # the check below still refuses one whose words the tags already
        # say. The user's own order decides the budget.
        _cands_ph = [c for c in (opts.get("_leftovers") or []) if c in _LEFTOVER_CONCEPTS]
        for _c in _cands_ph + [c for c in unmapped if c not in _cands_ph]:
            if len(_phr) >= _ph_budget:
                break
            _cl0 = str(_c).lower().strip(" ,.;")
            if not _cl0 or _cl0 in _seen_ph or _cl0 in _settled or len(_cl0) > 60:
                continue
            _ws = [w for w in re.findall(r"[a-z0-9-]+", _cl0) if w not in _PHRASE_STOP]
            if not _ws or len(_ws) > 4 or all(w in _PHRASE_FILLER for w in _ws):
                continue                    # a phrase is short; a sentence stays prose
            if set(_ws) <= _tagwords:
                continue
            if set(_ws) & {"viewer", "camera", "you"}:
                continue                    # a relation to the viewer is the gaze's and the pov's ('invites viewer' is no scene)
            _o = origins.get(_cl0)
            if not _o:
                continue
            # A PHRASE ON THE LINE IS THE USER'S OWN WORDS (2026-09-16: the
            # model's descriptor 'rotten' -- not a tag, not typed -- reached
            # the line as a phrase): a leftover run of the request, a ruled
            # generic place, a subjective phrase; never the model's invention,
            # which the prose keeps
            if _cl0 not in _LEFTOVER_CONCEPTS and _cl0 not in _gen_ph:
                continue
            try:
                from promptstudio.library import external as _extp
                if str(_cl0).lower() in {str(n).lower() for n in _extp.all_typed_names()}:
                    continue                # the external registry's (a prose-only light): said, never a phrase
            except Exception:
                pass
            tags.append(_cl0)
            _seen_ph.add(_cl0)
            _phr.append(_cl0)
            _ctx[_cl0] = _o
    # THE STYLE BOX'S PHRASES ARE THE USER'S LOOK, ALL OF THEM, WHOLE
    # (2026-09-17): no budget -- the box holds nothing but looks
    _seen_lk = {str(t).lower() for t in tags}
    for _lk in (opts.get("_look_phrases") or []):
        if _lk.lower() not in _seen_lk:
            tags.append(_lk)
            _seen_lk.add(_lk.lower())
            _phr.append(_lk)
            _ctx[_lk] = ("style", 0)
    opts["_phrases"] = list(_phr)
    for t_inj3 in injected + pre_injected:
        # A STATE THAT NAMES A SEX IS THAT SEX'S (2026-09-28: 'topless male'
        # injected for the level landed in the woman's part of a man-and-woman
        # scene): it goes to the first subject of the sex it names
        _kind9 = "male" if sm.MALE_NAMED.search(str(t_inj3)) else \
            ("female" if re.search(r"\bfemale\b", str(t_inj3)) else None)
        _si9 = next((i9 for i9, s9 in enumerate(subjects) if _kind9 and s9.get("kind") == _kind9), 0)
        _ctx.setdefault(t_inj3, ("act", _si9))
    # pairing/position tags are interaction facts; the penis family
    # belongs in its owner's subject block, not adrift in the scene band
    for t9 in pair_tags:
        _ctx.setdefault(t9, ("inter", 0))
    ph9 = next((i for i, s2 in enumerate(subjects)
                if s2.get("kind") in ("futanari", "male")), None)
    if ph9 is not None:
        for t9 in _GENITAL_FAMILY:
            _ctx.setdefault(t9, ("subject", ph9))
    for k3, v3 in _ctx.items():
        if k3:
            tag_meta.setdefault(k3, v3)
    # the futanari rider belongs beside its subject -- stamped BEFORE
    # the typed classifier can shove it into the scene band
    fu4 = next((i for i, s3 in enumerate(subjects)
                if s3.get("kind") == "futanari"), None)
    if fu4 is not None:
        tag_meta.setdefault("futanari", ("subject", fu4))
    # typed tags no concept stamped get a best-effort band so the sort
    # can place them; a mapped duplicate keeps its own stamp (setdefault)
    for t9 in typed_direct:
        if t9 in tag_meta:
            continue
        if t9 in tag_subj:               # typed body tag bound to its
            tag_meta[t9] = ("subject", tag_subj[t9])   # subject's block
            continue
        ar9 = sm.arity_of(t9)
        if ar9 and ar9 >= 2:
            tag_meta[t9] = ("inter", 0)
        elif t9 in _light_vocab():
            tag_meta[t9] = ("light", 0)
        elif t9 in _style_vocab():
            tag_meta[t9] = ("style", 0)
        elif t9 in _GENITAL_FAMILY:
            # THESE ALREADY HAVE AN OWNERSHIP RULE, just above: they go
            # in the male or futa subject's block. Reaching here means
            # there is no such subject -- a POV act, where the penis
            # belongs to the person holding the camera -- and stamping
            # it onto subject 0 put `penis` directly after `1girl`,
            # which reads as the girl having one. No owner, no block.
            tag_meta[t9] = ("inter", 0)
        elif set(_gloss_flags(t9) or ()) & {"pose", "act"}:
            # A TYPED POSE OR ACT IS THE SUBJECT'S (the author's live test,
            # 2026-09-28: 'spread legs' and 'provocative pose' typed came out in
            # the scene part, after the place): it stands in its subject's part
            # with her other acts
            tag_meta[t9] = ("act", tag_subj.get(t9, 0))
        elif _describes_a_person(t9):
            # A DESCRIPTION OF A PERSON BELONGS IN A PERSON'S BLOCK.
            # the author's: "subject names are not at the start of their
            # subsections but just at the start of the prompt" -- the
            # names were fine; what scattered was everything meant to sit
            # under them. 'a blonde elf ranger and a red-haired dwarf'
            # put `elf`, `dwarf`, `pointy ears` and `blonde hair` in the
            # SCENE band, so they sorted past the act, the style and the
            # location and washed up at the very end of the line, leaving
            # the persona leading a block with nothing in it.
            #
            # The gloss library already knows which tags describe a
            # person (body / person / creature); hair is the one family
            # it files as `object`, and _hair_vocab covers that. No new
            # list, and every future tag with those flags lands right.
            tag_meta[t9] = ("subject", tag_subj.get(t9, 0))
        else:
            tag_meta[t9] = ("scene", 0)

    # WITHIN a subject block the order is ROLE, not insertion: the
    # persona LEADS (a stray 'penis' had sorted ahead of 'tifa
    # lockhart'), the series follows, everything else after
    _role_rank = {}
    for si3, s3 in enumerate(subjects):
        if s3.get("persona"):
            _role_rank[s3["persona"]] = 0
            if s3.get("series"):
                _role_rank.setdefault(s3["series"], 1)
        # THE RACE IS WHO THE SUBJECT IS (2026-09-19: 'tengu' and 'shinigami'
        # washed up after the place, at the end of the line): it leads its
        # subject's block, right after the name
        _ps3 = (plan.get("subjects") or [])[si3] if si3 < len(plan.get("subjects") or []) else {}
        # WHO BEFORE WHAT (the author, 2026-09-19: "tags like 'milf', 'futanari',
        # race / occupation / profession predate the general description tags like
        # eye / hair colors" -- a persona's canon hair and eyes had pushed them
        # away): name, series, futanari, race, maturity, occupation, THEN the look
        _o3 = str((_ps3 or {}).get("occupation") or s3.get("occupation") or "").lower()
        if _o3 and _o3 in tags:
            tag_meta[_o3] = ("subject", si3)
            _role_rank[_o3] = 1.7
        _r3 = str(s3.get("race") or (_ps3 or {}).get("race") or "").lower()
        if _r3 and _r3 in tags:
            tag_meta[_r3] = ("subject", si3)
            _role_rank[_r3] = 1.2
    for _gp9 in _genre_people:
        _role_rank.setdefault(_gp9, 1.7)
    # the futanari rider is a subject IDENTITY tag (like the persona and
    # the count): it belongs right after the persona in its block, not at
    # the tail of the appearance tags (the author's: 'milf places correctly'
    # via canon; futanari sorted late because bridge2 appended it mid-
    # list). Stamp the band AND give it an early role rank.
    fu3 = next((i for i, s3 in enumerate(subjects)
                if s3.get("kind") == "futanari"), None)
    if fu3 is not None:
        tag_meta["futanari"] = ("subject", fu3)
        _role_rank["futanari"] = 0.3   # right after the persona name

    def _layer_rank(t2):
        """the subject band in layers (the author's 2026-09-16: "actions go
        earlier than body descriptions, clothes out of order"): name,
        series, hair, eyes, body, clothes and the undress, then the face
        (expression, gaze) and the states -- canon bundles arrive
        alphabetical, the model's plan in its own order"""
        if t2 in _role_rank:
            return _role_rank[t2]
        if t2 in _AGE_TAG_SET:
            return 1.5                 # maturity is identity: right after the name
        try:
            sl = sm.slot_of(t2, banks)
        except Exception:
            sl = None
        if sl == "hair":
            return 2
        if str(t2).endswith(" eyes") or str(t2).endswith(" pupils"):
            return 3
        if sl == "body":
            return 4
        if sl == "clothing" or _is_nudity_tag(t2) or _is_garment(t2):
            return 5
        if sl in ("expression", "gaze"):
            return 6
        return 7

    def _order_key(pair):
        idx, t2 = pair
        band, subj = tag_meta.get(t2, ("scene", 0))
        # WHAT A SUBJECT DOES SITS IN THAT SUBJECT'S BLOCK (the author's live test,
        # 2026-09-19: "individual actions / poses are not placed within their
        # subjects subsection but at the end of the all subjects section (making
        # it impossible to distinguish what subject does what)"): an act or a
        # pose closes its own subject's block, after the face and the states;
        # what two subjects do together (the 'inter' band) still follows them all
        if band == "act":
            return (_BAND["subject"], subj, 8, idx)
        return (_BAND.get(band, 7), subj, _layer_rank(t2) if band == "subject" else _role_rank.get(t2, 2), idx)

    # EACH FACT ONCE (2026-09-16, danbooru's implications). 'holding gun'
    # already says 'holding', 'holding weapon', 'gun' and 'weapon'; 'completely
    # nude' says 'nude'; 'white thighhighs' says 'thighhighs'. A general tag
    # another tag of the same subject implies leaves the line -- the booru
    # adds it to every such post by itself. A typed parent goes too: the
    # brief's 'holding a gun' parses to 'holding' and 'gun' beside 'holding
    # gun', and the child still says what was typed. What stays: a
    # character or series (the persona block names both on purpose), an
    # artist, a meta tag.
    if _implications():
        _subj_of = {t2: tag_meta.get(t2, ("scene", 0))[1] for t2 in tags}
        _said = {}
        # ... EXCEPT THE GARMENT UNDER ITS DECORATION (the author, 2026-10-03: "hat
        # feather and hat ribbon cannot exist without a hat"). 'skirt lift' is the
        # skirt, lifted, and says it; a 'hat ribbon' is a ribbon ON a hat and does
        # not wear one. The hat it implies stays on the line beside it -- as it
        # always did beside 'hat feather', for which the booru files no implication.
        _decor_ef = garment_decorations()
        for t2 in tags:
            if t2 in _decor_ef:
                continue
            for c2 in implied(t2):
                _said.setdefault((c2, _subj_of.get(t2, 0)), t2)
        _typed_set = set(typed_direct)
        _kept_line = []
        for t2 in tags:
            by = _said.get((t2, _subj_of.get(t2, 0)))
            if (by and t2 not in _role_rank
                    and not str(t2).startswith("@")
                    and _general_tag(t2)):
                continue
            _kept_line.append(t2)
        tags = _kept_line
    # A FRAME WITHOUT THE FACE DOES NOT DESCRIBE IT (the author, 2026-09-19: "a
    # lower body frame is not respected even if typed (because describing eyes /
    # hair / face makes imagegen confused even if a tag like ass focus +
    # close up are typed)" -- a persona's canon hair and eyes above all). When
    # the frame is below the head, or the head is out of it, the hair, the eyes,
    # the expression and the gaze leave every subject's block; typed words stay.
    _FRAME_NO_FACE = {"lower body", "head out of frame", "feet only", "legs only", "faceless"}
    _LOW_FOCUS = {"ass focus", "crotch focus", "foot focus", "feet focus", "thigh focus", "pussy focus",
                  "navel focus", "hip focus", "leg focus", "penis focus", "back focus"}
    _tl9 = {str(t2).lower() for t2 in tags}
    if (_tl9 & _FRAME_NO_FACE) or ((_tl9 & _LOW_FOCUS) and "close-up" in _tl9):
        _typed9 = {str(t2).lower() for t2 in typed_direct}
        _nf = []
        for t2 in tags:
            band9 = tag_meta.get(t2, ("scene", 0))[0]
            if band9 == "subject" and t2 not in _typed9 and t2 not in _role_rank \
                    and _layer_rank(t2) in (2, 3, 6):
                stripped.append((t2, "the frame does not show the face"))
                continue
            _nf.append(t2)
        tags = _nf
        # ... and the PARAGRAPH says the same (it still described the hair and the
        # eyes of a picture that shows neither): the plan's face fields are
        # emptied, typed ones kept, and the prose is rendered again from it
        try:
            for ps9 in plan.get("subjects") or []:
                if not isinstance(ps9, dict):
                    continue
                for k9 in ("hair", "eyes"):
                    _keep9 = [x.strip() for x in str(ps9.get(k9) or "").split(",") if x.strip().lower() in _typed9]
                    ps9[k9] = ", ".join(_keep9)
                if isinstance(ps9.get("body"), dict):
                    ps9["body"]["expression"] = [x for x in (ps9["body"].get("expression") or []) if str(x).lower() in _typed9]
            if opts.get("fast"):
                plan["nl"] = _fastplan.render_nl(plan, mode)
        except Exception:
            pass
    tags = [t2 for _, t2 in
            sorted(enumerate(tags), key=_order_key)]
    # THE LINE IN PARTS (the author, 2026-09-19: "1 - before subjects part, 2 -
    # each individual subject's part, 3 - after subjects part", each subject's
    # part led by its identity: the persona, the race, else its own count tag;
    # a separator the image models read as a break, and that shows the two of
    # us how the line was composed). opts['block_sep'] turns it on.
    _sep = str(opts.get("block_sep") or "")
    _n_subj9 = sum(int(cast.get(k9) or 0) for k9 in ("female", "male", "futa", "other"))
    _parts, _cur_key = [], None
    for t2 in tags:
        band9, subj9 = tag_meta.get(t2, ("scene", 0))
        key9 = ("s", subj9) if band9 in ("subject", "act") else ("after", 0)
        if key9 != _cur_key:
            _parts.append([key9, []])
            _cur_key = key9
        _parts[-1][1].append(t2)
    # WHAT TWO SUBJECTS BOTH DO IS SAID FOR EACH (the author, 2026-09-19: "individual
    # acts and poses may be duplicated for each subject"). A tag stood on the line
    # once, so when both girls sat -- or one sat by the dice and the other by the
    # brief -- 'sitting' showed in one part only and the other subject had no pose.
    # Each subject's own pose and acts (the plan's) are written in ITS part, again
    # if another part already carries them.
    _dup_ok = set()
    if _n_subj9 > 1:
        _blk_of = {k9[1]: blk9 for k9, blk9 in _parts if k9[0] == "s"}
        for si9, ps9 in enumerate(plan.get("subjects") or []):
            if not isinstance(ps9, dict):
                continue
            # ... and what she IS and WEARS as well (the author, same day: "both
            # subjects cant have red eyes in tags? - if so this also needs fixing"):
            # the subject's own hair, eyes, body, clothes and held thing
            _own9 = [ps9.get("pose"), ps9.get("held_object")] + list(ps9.get("self_actions") or [])                 + [x.strip() for x in str(ps9.get("hair") or "").split(",")]                 + [x.strip() for x in str(ps9.get("eyes") or "").split(",")]                 + list(ps9.get("outfit") or []) + list(ps9.get("descriptors") or [])
            # her typed and canon tags ride as the subject's locked canon, not in the plan
            if si9 < len(subjects):
                _own9 += [str(x) for x in (subjects[si9].get("locked_canon") or {})]
            for _bv9 in (ps9.get("body") or {}).values() if isinstance(ps9.get("body"), dict) else []:
                _own9 += list(_bv9) if isinstance(_bv9, (list, tuple)) else [_bv9]
            for c9 in _own9:
                if not c9:
                    continue
                for t9 in (mapping.get(str(c9 or "")) or mapping.get(str(c9 or "").lower())
                           or ([str(c9).lower()] if str(c9).lower() in tags else [])):
                    if t9 not in tags or tag_meta.get(t9, ("", 0))[0] not in ("act", "subject"):
                        continue
                    if si9 in _blk_of and t9 not in _blk_of[si9]:
                        _blk_of[si9].append(t9)
                        _dup_ok.add(t9)
                    elif si9 not in _blk_of:
                        _parts.insert(max(1, sum(1 for k9, _b9 in _parts if k9[0] == "s")), [("s", si9), [t9]])
                        _blk_of[si9] = _parts[[k9 for k9, _b9 in _parts].index(("s", si9))][1]
                        _dup_ok.add(t9)
    # a repeated tag takes its layer's place in the part, not the tail
    if _dup_ok:
        for k9, blk9 in _parts:
            if k9[0] == "s":
                blk9.sort(key=lambda t9: 8 if tag_meta.get(t9, ("", 0))[0] == "act" else _layer_rank(t9))
    tags = []
    for key9, blk in _parts:
        if _sep:
            tags.append(_sep)
        if _sep and key9[0] == "s" and _n_subj9 > 1:
            _s9 = subjects[key9[1]] if key9[1] < len(subjects) else {}
            _lead = blk[0] if blk else ""
            if not (_lead in _role_rank and _role_rank[_lead] <= 1.2):
                _own = {"female": "1girl", "male": "1boy", "futanari": "1futa"}.get(_s9.get("kind"), "1other")
                blk = [_own] + [b for b in blk if b != _own]
        tags += blk
    opts["_dup_ok"] = _dup_ok
    # EACH TAB IN ITS OWN BOORU'S SPELLING (the author's). The scene is the
    # same; 'bound leg' is simply how danbooru writes what gelbooru calls
    # 'bound legs', and the model that was trained on the other form does
    # not recognise it.
    line = (conditioning(mode, level, _cond_rng,
                         opts.get("quality", "standard"),
                         opts.get("period") if mode == "anima" else None)
            + count_tags(cast, mode)
            + [spell_for(t, mode, banks) for t in tags])
    # the count channel and the scene-type channel can both say 'no
    # humans'; the line says everything once
    _seen9, _l9 = set(), []
    for t2 in line:
        if (_sep and (t2 == _sep or re.match(r"^1(girl|boy|futa|other)$", str(t2)))) \
                or t2 not in _seen9 or t2 in _dup_ok:
            _l9.append(t2)
        _seen9.add(t2)
    line = _l9

    # ...and what the contradiction prune dropped from the line leaves the
    # prose too (2026-09-15: 'very long hair' stayed in the sentence beside
    # 'short hair with long locks' after the line lost it)
    # ... AND SO DOES EVERY GARMENT THE LINE LOST, WHATEVER CHECK DROPPED IT (the
    # author's live test, 2026-09-28: the prose dressed the woman in 'gloves, a hair
    # ribbon, and a pink ribbon' the line never carried; measured on the fast path,
    # 35 garments in 32 prompts). The prose is written from the plan before the
    # line's checks run, and only the contradiction prune told it what went: each
    # subject's plan outfit is held to the finished line here.
    _final9 = {str(t2).lower() for t2 in line}
    _gone9 = []
    for ps9 in plan.get("subjects") or []:
        if not isinstance(ps9, dict) or not isinstance(ps9.get("outfit"), list):
            continue
        _keep9 = []
        for g9 in ps9["outfit"]:
            _m9 = mapping.get(str(g9)) or mapping.get(str(g9).lower()) or [str(g9).lower()]
            if any(str(x9).lower() in _final9 for x9 in _m9):
                _keep9.append(g9)
            else:
                _gone9.append(str(g9))
        ps9["outfit"] = _keep9
    # ... AND EVERY OTHER FACT OF HERS THE LINE LOST (2026-10-03: the prose said
    # 'symbol in eye' and 'looking at the viewer', 'dirty' beside 'dirty face'; the
    # line had dropped each -- a gate, an implication -- and told nobody). The fast
    # plan's values are tags: one that is a tag and is not on the finished line
    # leaves the plan.
    if opts.get("fast"):
        try:
            _on9 = _final9 | {str(t2).lower() for t2 in tags}
            _voc9 = _vocab()

            def _stays9(v9):
                v9 = str(v9 or "").strip()
                if not v9:
                    return False
                _m9 = mapping.get(v9) or mapping.get(v9.lower())
                if not _m9 and not _voc9.get(v9.lower()):
                    return True                 # a phrase, not a tag: the prose's own
                return any(str(x9).lower() in _on9 for x9 in (_m9 or [v9.lower()]))
            for ps9 in plan.get("subjects") or []:
                if not isinstance(ps9, dict):
                    continue
                for k9 in ("hair", "eyes"):
                    _p9 = [x9.strip() for x9 in str(ps9.get(k9) or "").split(",") if x9.strip()]
                    _k9 = [x9 for x9 in _p9 if _stays9(x9)]
                    if len(_k9) != len(_p9):
                        ps9[k9] = ", ".join(_k9)
                        _gone9 += [x9 for x9 in _p9 if x9 not in _k9]
                if isinstance(ps9.get("body"), dict):
                    for _bk9, _bv9 in list(ps9["body"].items()):
                        if isinstance(_bv9, list):
                            _k9 = [x9 for x9 in _bv9 if _stays9(x9)]
                            if len(_k9) != len(_bv9):
                                _gone9 += [str(x9) for x9 in _bv9 if x9 not in _k9]
                                ps9["body"][_bk9] = _k9
                if ps9.get("pose") and not _stays9(ps9["pose"]):
                    _gone9.append(str(ps9["pose"]))
                    ps9["pose"] = ""
                if isinstance(ps9.get("self_actions"), list):
                    _k9 = [x9 for x9 in ps9["self_actions"] if _stays9(x9)]
                    if len(_k9) != len(ps9["self_actions"]):
                        _gone9 += [str(x9) for x9 in ps9["self_actions"] if x9 not in _k9]
                        ps9["self_actions"] = _k9
        except Exception:
            pass
    # ... AND THE PROSE GAINS WHAT THE LINE SAYS OF A SUBJECT BY THE USER'S WORD OR
    # THE LEVEL'S (the author, 2026-10-03: "fix other typed facts" -- 'a woman with
    # dark skin and large breasts' carried `dark skin, large breasts` on the line,
    # the level added `completely nude` and `grabbing own breast`, and the paragraph
    # told none of them). Typed tags ride outside the plan and the level's tags join
    # after it is written; the template only knows the plan. Each such tag of a
    # subject's own part is entered where the template reads it -- hair, eyes,
    # clothes and states of undress, what she does, else the body -- unless a plan
    # field already says it. A tag that names who she is (the persona, the race,
    # the job, 'mature female') is the sentence's noun, not a descriptor.
    _added9 = False
    if opts.get("fast"):
        try:
            _src9 = {str(x).lower() for x in list(typed_direct or []) + list(pre_injected or []) + list(injected or [])}
            _worn9 = {t9 for _it9 in ((_clothes_table() or {}).get("slots") or {}).values() for t9 in _it9}
            _und9 = set(((_spice_table() or {}).get("slots") or {}).get("nudity") or {})
            _stv9 = stance_vocab()
            _who9 = {m9 for _ms9 in _fastplan._ADULT_MOD.values() for m9 in _ms9}
            for k9, blk9 in _parts:
                if k9[0] != "s" or k9[1] >= len(plan.get("subjects") or []):
                    continue
                ps9 = (plan.get("subjects") or [])[k9[1]]
                if not isinstance(ps9, dict):
                    continue
                _said9 = set()
                _vals9 = [ps9.get("pose"), ps9.get("held_object"), ps9.get("occupation"), ps9.get("race")] \
                    + list(ps9.get("self_actions") or []) + list(ps9.get("outfit") or []) \
                    + list(ps9.get("descriptors") or []) \
                    + [x9.strip() for x9 in str(ps9.get("hair") or "").split(",")] \
                    + [x9.strip() for x9 in str(ps9.get("eyes") or "").split(",")]
                for _bv9 in (ps9.get("body") or {}).values() if isinstance(ps9.get("body"), dict) else []:
                    _vals9 += list(_bv9) if isinstance(_bv9, (list, tuple)) else [_bv9]
                for c9 in _vals9:
                    if c9:
                        _said9.add(str(c9).lower())
                        _said9.update(str(x9).lower() for x9 in (mapping.get(str(c9)) or mapping.get(str(c9).lower()) or []))
                _acted9 = str(plan.get("action") or "").lower().split(" to ")[0]
                for t9 in blk9:
                    tl9 = str(t9).lower()
                    # (a persona's canon stays the line's: the name says it in the prose)
                    _mine9 = tl9 in _src9 or not (subjects[k9[1]].get("persona") if k9[1] < len(subjects) else None)
                    if not _mine9 or tl9 in _said9 or t9 in _role_rank or tl9 == _acted9:
                        continue
                    _fl9 = _gloss_flags(tl9) or set()
                    if tl9 in _who9 or ("person" in _fl9 and not _fastplan._GENDER_SUFFIX.search(tl9)) or re.match(r"^(\d\+?(girls?|boys?|others?|futas?)|solo)$", tl9):
                        continue
                    _said9.add(tl9)
                    _added9 = True
                    if "hair" in tl9.split() or tl9 in _hair_vocab():
                        ps9["hair"] = ", ".join(x9 for x9 in (str(ps9.get("hair") or "").strip(), tl9) if x9)
                    elif re.search(r"\b(eyes?|pupils)\b", tl9):
                        ps9["eyes"] = ", ".join(x9 for x9 in (str(ps9.get("eyes") or "").strip(), tl9) if x9)
                    elif tl9.startswith("holding ") and ps9.get("held_object"):
                        continue                    # the held thing is told already
                    elif (_fastplan._GENDER_SUFFIX.sub("", tl9) in _fastplan._NUDE_STATES
                          or (not tl9.split()[0].endswith("ing")
                              and (names_garment(tl9) or (tl9 in _und9 and is_clothing_state(tl9))))):
                        # worn, a state of undress ('is completely nude'), or a garment's
                        # own state ('shirt lift', 'no bra'): the clothes' sentence
                        ps9["outfit"] = list(ps9.get("outfit") or []) + [tl9]
                    elif tl9 in _stv9 and not ps9.get("pose"):
                        ps9["pose"] = tl9
                    elif (_fastplan._is_feeling_verb(tl9) or tl9.split()[0].endswith("ing") or tag_meta.get(t9, ("", 0))[0] == "act"
                          or _fl9 & {"pose", "act"}):
                        ps9["self_actions"] = list(ps9.get("self_actions") or []) + [tl9]
                    else:
                        if not isinstance(ps9.get("body"), dict):
                            ps9["body"] = {}
                        ps9["body"].setdefault("desc", []).append(tl9)
        except Exception:
            _added9 = False
    if _gone9 or _added9:
        if opts.get("fast"):
            plan["nl"] = _fastplan.render_nl(plan, mode)
        stripped = list(stripped) + [(g9, "not on the line") for g9 in _gone9]
    nl = _scrub_stripped(str(plan.get("nl") or "").strip(),
                         list(stripped) + [(t, "contradiction") for t in (dropped_contra or [])])
    # the mood sentence says the chain's mood whatever the model wrote there
    # (2026-09-28: told 'innocent', a model may still write 'joyful')
    if nl and opts.get("_mood"):
        nl = re.sub(r"(?i)\b(the mood is )[^.]+\.", lambda m9: m9.group(1) + str(opts["_mood"]) + ".", nl)
    # NAME SCRUBBER: the model invents names for unnamed subjects despite
    # instruction ("Lila with lavender hair and Mira with...") -- naming
    # is the checkbox's job. Mechanical, like every guardrail that
    # matters, and MULTI-SUBJECT since the two-girl christening: a
    # capitalized mid-sentence token that is not in the user's text, a
    # persona or the style, seen twice, is an invented name; invented
    # names map to unnamed subjects in order of first appearance and are
    # replaced by a deterministic alias ('the first girl').
    if nl and subjects and not all(s2.get("persona") for s2 in subjects):
        prot = {w.lower() for w in re.findall(r"[A-Za-z]+", base or "")}
        for s2 in subjects:
            prot |= {w.lower() for w in
                     re.findall(r"[A-Za-z]+", str(s2.get("persona") or ""))}
        prot |= {w.lower() for w in re.findall(
            r"[A-Za-z]+", str((plan.get("style") or {}).get("name") or ""))}
        # EVERY WORD OF EVERY EMITTED TAG IS PROTECTED. This strips names
        # the model invents for a subject ("Elara") by spotting capitalised
        # words the prompt never contained -- but plenty of ordinary tags
        # are written with a capital in prose. `dutch angle` appears as
        # "Dutch angle", so "from a low Dutch angle" was read as an
        # invented name and rewritten to "from a low the girl angle".
        # With one subject a SINGLE occurrence is enough to trigger it, so
        # there was nothing to make it think twice.
        #
        # The tag line is exactly the list of things the prose is meant to
        # be describing, so no word in it can be an invented name. That
        # covers Dutch, French, Victorian, Gothic and every proper noun a
        # future tag brings with it, without a list to maintain.
        prot |= {w.lower() for w in re.findall(
            r"[A-Za-z]+", " ".join(str(t) for t in tags))}
        nl = re.sub(r"\s*,?\s*\b(?:named|called)\s+[A-Z][a-z]+", "", nl)
        nl = re.sub(r"^([A-Z][a-z]+), (?=an? |the )", "", nl)
        nl = nl[:1].upper() + nl[1:] if nl else nl
        starts = {m.start(1) for m in
                  re.finditer(r"(?:^|[.!?]\s+)([A-Z][a-z]+)", nl)}
        seen9, order9 = {}, []
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", nl):
            w = m.group(1)
            if w.lower() in prot:
                continue
            e = seen9.setdefault(w, [0, False])
            e[0] += 1
            e[1] = e[1] or (m.start(1) not in starts)
            if w not in order9:
                order9.append(w)
        names9 = [w for w in order9 if seen9[w][1]
                  and (seen9[w][0] >= 2 or len(subjects) == 1)]
        unnamed9 = [i for i, s2 in enumerate(subjects)
                    if not s2.get("persona")]
        for w, si in zip(names9, unnamed9):
            alias9 = _subject_alias(subjects, si, plan)
            nl = re.sub(r"\b" + re.escape(w) + r"\b", alias9, nl)
        if names9:   # a replacement at a sentence start needs its capital
            nl = re.sub(r"(^|[.!?]\s+)([a-z])",
                        lambda m: m.group(1) + m.group(2).upper(), nl)
    # A NUMERAL NEVER PRECEDES A DEFINITE ARTICLE. The model likes to
    # open with the cast count ("1 girl stands..."), and the brief now
    # tells it to call an unnamed subject "the girl" -- so it wrote
    # "1 the girl stands on a green soccer field". Both halves are
    # doing what they were told; the join is what is wrong, and
    # "<number> the <noun>" is never grammatical English whatever
    # produced it. The count is the redundant half -- the tag line
    # already carries `1girl` -- so it is the half that goes.
    if nl:
        nl = re.sub(r"\b\d+\s+(the\s)", r"\1", nl)
        nl = re.sub(r"(^|[.!?]\s+)([a-z])",
                    lambda m: m.group(1) + m.group(2).upper(), nl)

    # PRONOUN ENFORCER, SENTENCE-SCOPED (the author's refined rule: 'the girl
    # places her hand on the table' is FINE -- a pronoun is bad only in a
    # sentence that never defines its subject). Per sentence: named
    # subject present -> untouched; no naming -> the FIRST referential
    # pronoun becomes the name, which defines the subject for the rest.
    # Reflexives (herself/himself) are always fine (in-clause referent).
    _NOUNS_RE = re.compile(
        r"\b(girls?|boys?|wom[ae]n|m[ae]n|lad(?:y|ies)|futanaris?|"
        r"figures?|group|pair)\b", re.I)

    def _sentences(text):
        return re.findall(r"[^.!?]*[.!?]?\s*", text)

    def _fix_unnamed(text, named_re, sub1):
        out = []
        for s9 in _sentences(text):
            if not s9 or named_re.search(s9):
                out.append(s9)
                continue
            state = {"done": False}
            out.append(re.sub(
                r"\b(?:[Ss]he|[Hh]e|[Hh]er|[Hh]is|him|[Tt]hey|"
                r"[Tt]heir|them)\b",
                lambda m: sub1(m, s9, state), s9))
        return "".join(out)

    if nl and len(subjects) == 1:
        s0 = subjects[0]
        # DISPLAY NAME, NOT TAG: substituting the persona tag for a pronoun
        # wrote "ereshkigal (fate)'s long blonde hair" into the paragraph.
        # The qualifier is booru bookkeeping; the tag line keeps it.
        who = _display_name(s0.get("persona")) if s0.get("persona") else None
        if not who:
            # THE THIRD COPY OF THE ALIAS RULE, and the buggy one:
            # `alias.lstrip("the ")` strips CHARACTERS in {t,h,e, }
            # from the left, not the prefix "the ". An alias of
            # "a young woman" sailed through untouched and became
            # "the a young woman" three times in one paragraph; an
            # alias of "teacher" would have become "acher". One
            # definition, three users, as it should have been.
            who = _subject_alias(subjects, 0, plan)
        whoc = who[0].upper() + who[1:]
        named_re = _NOUNS_RE if not s0.get("persona") else re.compile(
            re.escape(s0["persona"]) + "|"
            + re.escape(_display_name(s0["persona"])) + "|"
            + _NOUNS_RE.pattern, re.I)

        def sub_solo(m, sent, state):
            if state["done"]:
                return m.group(0)
            state["done"] = True
            w = m.group(0)
            low, cap = w.lower(), w[0].isupper()
            rep = whoc if cap else who
            if low in ("her", "his") and \
                    re.match(r"\s+\w", sent[m.end():]):
                return rep + "'s"        # possessive defines the subject
            return rep
        nl = _fix_unnamed(nl, named_re, sub_solo)
    elif nl and len(subjects) >= 2:
        # plural pronouns cover the WHOLE cast, so attribution is safe;
        # singular her/his in a multi cast is only repaired when the
        # sentence names nobody (rare; the alias rule covers the rest)
        kinds0 = {s2.get("kind") for s2 in subjects}
        # TWO PEOPLE ARE A PAIR. "the group's faces" for a boy and a girl
        # was this line's doing, not the model's.
        group = ("the girls" if kinds0 == {"female"} else
                 "the boys" if kinds0 == {"male"} else
                 "the pair" if len(subjects) == 2 else "the group")
        groupc = group[0].upper() + group[1:]
        pers = [str(s2.get("persona") or "") for s2 in subjects
                if s2.get("persona")]
        pers += [_display_name(p) for p in pers]
        named_re = _NOUNS_RE if not pers else re.compile(
            "|".join(re.escape(p) for p in pers) + "|" + _NOUNS_RE.pattern,
            re.I)

        def sub_multi(m, sent, state):
            if state["done"]:
                return m.group(0)
            w = m.group(0)
            low, cap = w.lower(), w[0].isupper()
            if low not in ("they", "their", "them"):
                return w                 # singular w/o naming: leave to NL
            state["done"] = True
            rep = groupc if cap else group
            if low == "their":
                # "the girls'" but "the group's" -- the possessive
                # suffix depends on the noun's ending
                return rep + ("'" if rep.endswith("s") else "'s")
            return rep
        nl = _fix_unnamed(nl, named_re, sub_multi)
    # PRONOUN GENDER AGREES WITH THE SENTENCE'S SUBJECT. "Camille ... holding
    # a cowgirl position with HIS legs" on a woman: when a sentence names
    # exactly one gender of subject, a singular pronoun of the other gender
    # is the model's slip, and it is corrected. Two genders in one
    # sentence -> left alone (the reference could be either).
    if nl and subjects:
        nl = _agree_pronouns(nl, subjects, plan)
        # A TYPED BODY SIZE IS LAW IN THE PROSE TOO: Tifa typed with small
        # breasts was written "with large breasts" while the tag line kept
        # `small breasts`. A sentence that names ONE subject and a size
        # from that subject's typed one-of family gets the typed value.
        try:
            nl = _agree_typed_sizes(nl, subjects, plan, tag_subj, typed_direct)
        except NameError:
            pass
    # A named style that bridge 2 could not map is the flagship NL-only
    # concept -- it must survive even when the model's own render drops it.
    st = (plan.get("style") or {}).get("name")
    # the prose never carries a booru qualifier: 'western comics (style)'
    # is written 'western comics' (same rule as fastplan._style_display)
    if st:
        for _q in (" (style)", " (medium)", " (theme)"):
            if str(st).lower().endswith(_q):
                st = str(st)[:-len(_q)]
    if st and st.lower() not in nl.lower() and st.lower() not in             {t.lower() for t in tags} and             not any(t.lower().startswith(st.lower() + " (") for t in tags):
        # adjective style names need the noun ('Rendered in cinematic.'
        # is not a sentence; 'Rendered in cinematic style.' is)
        st_phrase = st if st.lower().endswith(
            ("style", "art", "painting", "render", "aesthetic")) \
            else st + " style"
        nl = (nl + (", " if mode != "anima" and nl else " ") +
              ("" if mode == "anima" else "") + st) if mode != "anima"              else (nl + " Rendered in " + st_phrase + ".")
    # THE FLOOR HOLDS AT THE END TOO. The plan-stage floor guarantees a
    # paragraph exists, but the scrubbers and enforcers above can still
    # leave it blank (one run shipped an empty prose half after a
    # non-deterministic model output). A prompt never leaves here with
    # half of it missing: the template writes one from the plan.
    if not (nl or "").strip():
        nl = _fastplan.render_nl(plan, mode)
    if not (nl or "").strip() and subjects:
        # THE PLAN CAME BACK WITHOUT SUBJECTS (the model dropped the list),
        # so the template had nobody to write about. The engine's own
        # subject list always exists: give the template those, named by
        # display name or kind, and it writes the count, setting and mood.
        _p2 = dict(plan)
        _p2["subjects"] = [{"who": (_display_name(s2.get("persona"))
                                    if s2.get("persona") else
                                    {"female": "girl", "male": "boy",
                                     "futanari": "futanari"}.get(
                                         s2.get("kind"), "figure")),
                            "hair": "", "eyes": "", "outfit": [], "body": {},
                            "pose": "", "self_actions": [], "held_object": None}
                           for s2 in subjects]
        nl = _fastplan.render_nl(_p2, mode)

    # AN EXTERNAL CONCEPT THE TAG LINE CARRIES REACHES THE PROSE TOO, in
    # its own prose form ("in the style of Vincent van Gogh"): the
    # template only knows the resolved booru style, so a typed external
    # artist was in the tags and nowhere in the paragraph.
    try:
        from promptstudio.library import external as _extn
        _adds = []
        for _n, _kind, _written in _ext_used:
            if _kind == "booru-equivalent":
                continue
            _nlf = str(_extn.render(_n, "nl") or "").strip()
            if not _nlf or _nlf.lower() in (nl or "").lower() or str(_n).lower() in (nl or "").lower():
                continue
            _adds.append(_nlf[:1].upper() + _nlf[1:] if _kind != "artist"
                         else "Rendered " + _nlf)
        if _adds and mode == "anima":
            nl = ((nl or "").rstrip() + " " + ". ".join(_adds) + ".").strip()
        elif _adds:
            nl = ((nl or "").rstrip(", ") + ", " + ", ".join(_adds)).strip(", ")
    except Exception:
        pass

    # ONE PROMPT, RENDERED FOR BOTH MODELS (the author's dual-tab design).
    # The content above is resolved ONCE; only the packaging differs --
    # quality/meta/year tags, the count-tag vocabulary, the '@' on artist
    # names, and whether the NL channel is prose (anima) or comma phrases
    # (illustrious). Running the whole pipeline twice cannot work: the
    # cast distributions are measured per booru, so the two modes disagree
    # about who is even in the picture -- 'a knight in a castle' came out
    # 6 girls for anima and 1 girl + 2 boys for illustrious on one seed.
    def _assemble(m, line_m, nl_m):
        # AN EMOTICON TAG KEEPS ITS UNDERSCORE (2026-09-15: a canon '|_|'
        # reached the line as '| |'): the library writes every tag with
        # spaces, which is the booru's word form; a tag with no letter or
        # digit is a symbol the booru writes as one token
        line_m = [(str(t).replace(" ", "_") if not re.search(r"[a-z0-9]", str(t), re.I) else t) for t in line_m]
        _sp = str(opts.get("block_sep") or "")

        def _join(parts_):
            out_ = ", ".join(str(x) for x in parts_)
            if _sp:
                out_ = out_.replace(", %s, " % _sp, "%s " % _sp).replace(", %s" % _sp, _sp)
            return out_
        if m == "anima":
            return _join(line_m) + ".\n" + nl_m
        # UNMAPPED CONCEPTS ARE ILLUSTRIOUS PHRASE MATERIAL (the author's,
        # current-scope testing): 'wholesome', 'elegant', 'soft golden
        # light from the left' carry meaning the tag part lost -- exactly
        # the NL channel's job in this mode. HEADLESS phrases are
        # excluded: 'warm and diffused' names no referent, so a phrase
        # must be a single word or contain a word our vocabulary knows
        # as a head noun ('light', 'pinks'->pink).
        heads = {t.split()[-1] for t in _vocab()}
        # colour words are referents too ('pastel pinks' names a palette)
        # but never end a booru tag, so the head set misses them
        heads |= {"pink", "blue", "red", "green", "purple", "lavender",
                  "gold", "silver", "orange", "yellow", "teal", "white",
                  "black", "brown", "grey", "gray", "aqua", "crimson"}
        _stop9 = {"and", "or", "the", "a", "an", "of", "with", "in",
                  "from", "to", "on", "at", "very"}

        def _phrase_ok(c9):
            ws = [w for w in re.split(r"[^a-z-]+", str(c9).lower())
                  if w and w not in _stop9]
            if not ws:
                return False
            if len(ws) == 1:
                return True
            return any(w in heads or w.rstrip("s") in heads for w in ws)

        seen9 = {t.lower() for t in line_m}
        extra = []
        for c9 in unmapped:
            c9 = str(c9).strip()
            if c9 and c9.lower() not in seen9 and _phrase_ok(c9):
                extra.append(c9)
                seen9.add(c9.lower())
        extra = extra[:8]
        # SHORT PHRASES ONLY (the mode's NL contract): the model writes
        # story prose anyway, so the engine cuts it -- split on sentence
        # and clause boundaries, keep 2-6 word fragments, drop the rest
        phrases = []
        for p in re.split(r"[,.;\n]+", nl_m):
            p = p.strip()
            # a CUT fragment loses the sentence that named its subject,
            # so the phrase-level pronoun rule applies (the author's: '...or
            # phrase or part of a complex sentence'): a fragment carrying
            # a referential pronoun without its own naming is dropped
            if re.search(r"\b(she|he|her|his|him|they|their|them)\b",
                         p, re.I) and not _NOUNS_RE.search(p):
                continue
            if (p and p.lower() not in seen9
                    and 2 <= len(p.split()) <= 6):
                phrases.append(p)
                seen9.add(p.lower())
        return _join(line_m + extra + phrases[:12 - len(extra)])

    prompt = _assemble(mode, line, nl)

    # THE OTHER MODEL'S TAB: same tags, same cast, same plan, repackaged.
    _other = "illustrious" if mode == "anima" else "anima"
    try:
        _bare = []
        _wt = re.compile(r"^\((.+?):(\d*\.?\d+)\)$")
        for _t in tags:
            _b = str(_t)
            _m = _wt.match(_b)
            _inner = _m.group(1) if _m else _b
            if _inner.startswith("@") or (chr(92) + "(") in _inner \
                    or (_m and pe._is_artist_name(_inner)):
                # an artist name is a name, not a tag: it is re-formatted
                # (the '@' is anima's, per its README) but never respelled.
                # A WEIGHTED artist '(@name:0.8)' starts with '(' and slipped
                # past the '@' test, so the other tab kept the '@'.
                _nm = pe._artist_format(
                    _inner.lstrip("@").replace(chr(92), ""), _other)
                _bare.append("(%s:%s)" % (_nm, _m.group(2)) if _m else _nm)
            elif str(_t) in (opts.get("_phrases") or ()):
                _bare.append(str(_t))        # a phrase is not a dead token
            elif str(_t) in _role_rank and _role_rank[str(_t)] <= 1:
                # A CHARACTER AND ITS SERIES ARE NAMES, NOT GENERAL TAGS (2026-09-19:
                # the second tab dropped 'hatsune miku, vocaloid' as "unknown to
                # this booru" -- the known-tag test reads the general-tag counts,
                # where no character lives)
                _bare.append(str(_t))
            elif str(_t) == str(opts.get("block_sep") or "<no separator>") or re.match(r"^1(girl|boy|futa|other)$", str(_t)):
                _bare.append(str(_t))        # the part separator and a block's own count tag
            else:
                _sp9 = spell_for(_t, _other, banks)
                # a tag this booru has no posts for is a dead token there,
                # typed or not (the role 'pirate' on Illustrious: danbooru
                # has no such tag; the prose keeps the concept and the
                # card says what was dropped)
                if not known_to(_other, _sp9):
                    opts.setdefault("_dropped_unknown", {}).setdefault(_other, []).append(str(_t))
                    continue
                _bare.append(_sp9)
        _line_o = (conditioning(
            _other, level,
            random.Random((int(seed) if seed is not None else 0) * 7919 + 29),
            opts.get("quality", "standard"),
            opts.get("period") if _other == "anima" else None)
            + count_tags(cast, _other) + _bare)
        _so, _lo = set(), []
        for _x in _line_o:
            if (opts.get("block_sep") and (_x == opts["block_sep"] or re.match(r"^1(girl|boy|futa|other)$", str(_x)))) \
                    or _x not in _so or _x in (opts.get("_dup_ok") or ()):
                _lo.append(_x)
            _so.add(_x)
        _line_o = _lo
        # the structured plan renders in either shape, so the second tab
        # never costs a second model call
        _nl_o = _fastplan.render_nl(plan, _other)
        _prompts = {mode: prompt, _other: _assemble(_other, _line_o, _nl_o)}
    except Exception as _e9:
        _prompts = {mode: prompt, "_error": "%s: %r" % (type(_e9).__name__, _e9)}

    # CONCEPT LIBRARY capture (best-effort, never blocks a prompt): log
    # the concepts bridge2 could not map, with their origin band + mode,
    # so the nightly classifier can route recurring ones into the pools.
    _ledger_new = []
    try:
        from promptstudio.library import concepts as concept_library
        _ledger_new = concept_library.log_unmapped(
            unmapped, origins, mode, base,
            style_names=[(plan.get("style") or {}).get("name")])
    except Exception:
        pass

    # THE WHY LINE: what the engine resolved, for the card under each
    # variant (genre and its bucket, place, occupations, event, season,
    # weather, races) -- every one of these is a decision the dice made
    # before any tag was picked, and the user should see it
    _why = {}
    try:
        _g = opts.get("_genre") or (None, None)
        _why["genre"] = _g[1]
        _why["genre_src"] = "selected" if opts.get("_genre_selected") else _g[0]
        _why["bucket"] = ((_genre_pool().get(_g[1] or "") or {}).get("bucket"))
        _why["anchor"] = opts.get("_genre_anchor")
        _l = opts.get("_location") or (None, None, None)
        _why["place"] = _l[1]
        _why["place_src"] = _l[0]
        _e = opts.get("_event") or (None, None, None)
        _why["event"] = _e[1] if len(_e) > 1 else None
        _lg = opts.get("_lighting") or {}
        _why["season"] = _lg.get("season")
        _why["weather"] = _lg.get("weather")
        _why["time"] = _lg.get("time")
        _why["act"] = opts.get("_act")
        _why["style"] = (opts.get("_style") or (None, {}))[0]
        _why["style_src"] = opts.get("_style_mode")
        _subs = (plan.get("subjects") or []) if isinstance(plan, dict) else []
        _why["occupations"] = [s9.get("occupation") for s9 in _subs if isinstance(s9, dict) and s9.get("occupation")]
        if not _why["occupations"]:
            try:                                  # the typed job (the planner's own detector)
                _tocc = _fastplan._occupation_from_prompt(_fastplan._seeds(base, cast, opts, banks))
                if _tocc:
                    _why["occupations"] = [_tocc + " [typed]"]
            except Exception:
                pass
        _why["races"] = [s9.get("race") for s9 in _subs if isinstance(s9, dict) and s9.get("race")]
        _why["engine"] = "fast" if opts.get("fast") else ("llm" if not opts.get("_plan_fallback") else "llm (mechanical fallback)")
    except Exception:
        pass
    _stage("done")
    tags = list(dict.fromkeys(tags))      # one tag once, whatever band stamped it twice
    return {"prompt": prompt, "negative": negative_line(mode, opts),
            "coverage": _coverage(user_tags, tags, opts, unmapped),
            "why": _why,
            "llm_calls": [dict(c) for c in _CALLS],
            "ledger_new": list(_ledger_new or []),
            "phrases": list(opts.get("_phrases") or []),
            "looks": {"tags": list(opts.get("_look_tags") or []), "phrases": list(opts.get("_look_phrases") or []),
                      "yielded": list(opts.get("_look_yielded") or [])},
            "refused": list(opts.get("_refused_youth") or []),
            "dropped_unknown": opts.get("_dropped_unknown") or {},
            # BOTH TABS. One resolution, two packagings.
            "prompts": _prompts,
            # which non-booru concepts were recognised, and how written
            "external": _ext_used,
            "negatives": {m9: negative_line(m9, opts)
                          for m9 in ("anima", "illustrious")},
            "artists": typed_art + gen_art,
            "cast": cast, "level": level,
            "subjects": [{k: v for k, v in s.items() if k != "locked_canon"}
                         for s in subjects],
            "switched": switched, "measured": measured, "plan": plan,
            "tags": tags, "unmapped": unmapped,
            "stripped_by_verifier": stripped,
            "dropped_for_level": dropped_for_level,
            "injected_for_level": pre_injected + injected,
            "dropped_solo_arity": dropped_arity,
            # debug internals for current-scope testing: which concept
            # produced which tag, and which (band, subject) each concept
            # was born in -- attribution failures are invisible without
            # them and this phase is ABOUT attribution failures
            "debug_mapping": {str(k): v for k, v in mapping.items()},
            "debug_origins": {str(k): v for k, v in origins.items()},
            "debug_tag_meta": {t9: tag_meta.get(t9, ("scene", 0))
                               for t9 in tags}}


_generate_drawn = generate


def generate(base, mode="anima", spice="auto", opts=None, seed=None):
    """generate() with the booru shut out: every count read while a prompt is
    drawn is the cache's (see _count_cached); what was wanted is noted for the
    harvest tool when the prompt is done."""
    _IN_DRAW["depth"] += 1
    try:
        return _generate_drawn(base, mode, spice, opts, seed)
    finally:
        _IN_DRAW["depth"] -= 1
        if _IN_DRAW["depth"] <= 0:
            _IN_DRAW["depth"] = 0
            _flush_wanted()


# -------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--mode", choices=["anima", "illustrious"], default="anima")
    ap.add_argument("--spice", default="auto")
    ap.add_argument("--no-clothes", action="store_true")
    ap.add_argument("--gen-characters", action="store_true")
    ap.add_argument("--detail", choices=["minimal", "standard", "detailed"],
                    default="standard")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    r = generate(args.base, args.mode, args.spice,
                 {"gen_clothes": not args.no_clothes,
                  "gen_characters": args.gen_characters,
                  "detail": args.detail}, args.seed)
    print("=== PROMPT ===")
    print(r["prompt"])
    print()
    print("cast=%s  level=%s measured=%s switched=%s" %
          (r["cast"], r["level"], r["measured"], r["switched"]))
    for i, sub in enumerate(r.get("subjects") or [], 1):
        print("subject %d: %s persona=%s slots=%s" %
              (i, sub["kind"], sub.get("persona"), sub.get("slots")))
    print("tags: %s" % ", ".join(r["tags"]))
    print("unmapped (NL-only): %s" % ", ".join(r["unmapped"]))
    if r["stripped_by_verifier"]:
        print("VERIFIER stripped: %s" % r["stripped_by_verifier"])
    if r["dropped_for_level"]:
        print("dropped for level: %s" % r["dropped_for_level"])


if __name__ == "__main__":
    main()
