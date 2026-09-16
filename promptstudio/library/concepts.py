#!/usr/bin/env python
"""
concept_library.py -- the CONCEPT LIBRARY (the author's integrate-and-expand
design, 2026-08-30). Complement to the gloss/flag library: that one
describes EXISTING booru tags; this one remembers the concepts the
generator INVENTS that have no clean tag (the `unmapped` list) and routes
each into machinery the generator ALREADY reads -- never a bolt-on layer.

LIFECYCLE (the algorithm):
  encounter -> LOG to concept_ledger.json (freq + context)
            -> nightly CLASSIFY (local LLM + measurement)
            -> human+Claude REVIEW
            -> INTEGRATE into the real file (alias_map / decompositions /
               a pool / the canonical-NL set)
            -> the generator REUSES it, never re-deriving.

FIVE INTEGRATION TYPES (each writes into existing machinery):
  alias   -> alias_map.json          (retrieval finds it next time)
  decomp  -> concept_decomp.json     (bridge2 expands it to real tags)
  pool    -> style/location/genre/medium pool (measured, REVIEW-gated)
  nl      -> concept_nl_only.json    (legitimately NL-only; stop flagging)
  reject  -> the flag layer          (never a new blocklist)

STYLE-HANDLING IS FIRST-CLASS (the author's: design it in now, styles feature
next): a style-band concept is classified against the style sphere --
alias to an existing style_pool entry, decompose into descriptors +
measured artists/palette, or promote to a new MEASURED style_pool entry.

PHASE A here = CAPTURE. classify/integrate follow.
"""

import datetime
import json
import os
import re
from promptstudio import paths as _paths

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = _paths.data("concept_ledger.json")

# concepts too generic/structural to be library-worthy even when unmapped
_SKIP = re.compile(
    r"^(the |a |an )?(girl|boy|woman|man|figure|subject|scene|image|"
    r"background|foreground|light|lighting|color|colour)s?$", re.I)


_LOCK = LEDGER + ".lock"


class _Lock:
    """a lock file around read-modify-write of the ledger (two processes
    wrote it at once tonight and one clobbered the other)"""
    def __enter__(self):
        import time
        for _ in range(200):
            try:
                self.fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(_LOCK) > 60:
                        os.remove(_LOCK)          # a stale lock from a dead process
                        continue
                except Exception:
                    pass
                time.sleep(0.05)
        self.fd = None
        return self

    def __exit__(self, *a):
        try:
            if self.fd is not None:
                os.close(self.fd)
            os.remove(_LOCK)
        except Exception:
            pass


def _load(strict=False):
    """the ledger; None (strict) or an EMPTY default when the file cannot
    be read -- a caller that will WRITE must use strict=True and skip its
    write on None, never replace an existing ledger it could not read"""
    if not os.path.exists(LEDGER):
        return {"_note": "concept library ledger: unmapped concepts the "
                "generator invented, with frequency + context, awaiting "
                "classification (see concept_library.py classify).",
                "concepts": {}}
    for _ in range(3):
        try:
            with open(LEDGER, encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            import time
            time.sleep(0.1)
    return None if strict else {"_note": "unreadable ledger", "concepts": {}}


def _save(blob):
    """atomic: the previous ledger becomes .bak, the new one lands whole"""
    tmp = LEDGER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False)
    try:
        if os.path.exists(LEDGER):
            import shutil
            shutil.copyfile(LEDGER, LEDGER + ".bak")
    except Exception:
        pass
    os.replace(tmp, LEDGER)


NL_ONLY = _paths.data("concept_nl_only.json")
DECOMP = _paths.data("concept_decomp.json")
ALIAS_LOCAL = _paths.data("alias_map_local.json")


def _json(path, default):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def nl_only():
    """the concepts reviewed as legitimately natural-language (or junk):
    the engine keeps them in the prose and never logs them again"""
    return set((_json(NL_ONLY, {}).get("concepts") or {}).keys())


def decompositions():
    """concept -> [booru tags] reviewed as its visible parts; bridge 2
    maps the concept to them directly"""
    return {k: v.get("tags") or [] for k, v in (_json(DECOMP, {}).get("concepts") or {}).items()}


def log_unmapped(unmapped, origins=None, mode="anima", base="",
                 style_names=None):
    """append this generation's unmapped concepts to the ledger. Cheap,
    best-effort; the caller wraps it so a failure never blocks a prompt.
    -> the names seen for the FIRST time (the card reports them as
    stored for review)"""
    if not unmapped:
        return []
    try:
        from promptstudio.llm import config as _cfg
        if not (_cfg.load() or {}).get("ledger", True):
            return []                      # a release gathers nothing
    except Exception:
        pass
    origins = origins or {}
    style_names = set(s.lower() for s in (style_names or []))
    with _Lock():
        return _log_unmapped_locked(unmapped, origins, mode, base, style_names)


def _log_unmapped_locked(unmapped, origins, mode, base, style_names):
    blob = _load(strict=True)
    if blob is None:
        return []                       # unreadable: never overwrite it
    con = blob.setdefault("concepts", {})
    now = datetime.datetime.now().isoformat(timespec="minutes")
    snippet = (base or "")[:90]
    settled = nl_only()
    try:
        from promptstudio.engine import vocab as _vc
        _known_tag = _vc._vocab()
    except Exception:
        _known_tag = {}
    try:
        from promptstudio.engine import bridge as _lb
        _style_names = {str(n).lower() for n in _lb._style_pool2() if not str(n).startswith("_")}
    except Exception:
        _style_names = set()
    new_names = []
    for c in unmapped:
        c = str(c).strip()
        cl = c.lower()
        if not c or _SKIP.match(cl) or len(cl) > 90 or cl in settled:
            continue
        if (_known_tag.get(cl) or 0) >= 100:
            continue                    # a real tag is not an unknown concept
        if cl in _style_names:
            continue                    # a style the pool renders is not unknown either
        if cl not in con:
            new_names.append(cl)
        band = (origins.get(cl) or ("?", 0))[0]
        # a style-band concept, or one the plan named as a style, is
        # flagged so the classifier routes it through the style sphere
        is_style = band == "style" or cl in style_names
        e = con.get(cl)
        if e is None:
            e = con[cl] = {"n": 0, "bands": {}, "modes": {},
                           "examples": [], "first_seen": now,
                           "is_style": is_style, "status": "new"}
        e["n"] += 1
        e["bands"][band] = e["bands"].get(band, 0) + 1
        e["modes"][mode] = e["modes"].get(mode, 0) + 1
        e["is_style"] = e.get("is_style") or is_style
        e["last_seen"] = now
        if snippet and snippet not in e["examples"] and len(e["examples"]) < 3:
            e["examples"].append(snippet)
    _save(blob)
    return new_names


_COLOR_WORDS = {"pink", "blue", "red", "green", "purple", "lavender",
                "gold", "golden", "silver", "orange", "yellow", "teal",
                "white", "black", "brown", "grey", "gray", "aqua", "cyan",
                "crimson", "navy", "beige", "cream", "pastel", "neon",
                "muted", "warm", "cool", "monochrome", "sepia", "vibrant",
                "saturated", "earth", "earthy", "dark", "bright"}
_PALETTE_RE = re.compile(r"colou?rs?|palette|hues?|tones?|shades?", re.I)
_THEME_WORDS = {"cozy", "cosy", "holiday", "festive", "seasonal",
                "occasion", "themed", "celebration", "christmas",
                "halloween", "easter", "valentine"}


def _classify_style(concept, lb):
    from promptstudio.engine import vocab as vc
    """route a style-band concept through the STYLE sphere -- but the
    style BAND also carries PALETTES and THEMES (the author's review caught
    'pastel pinks'/'warm tones'/'cozy holiday' wrongly proposed as new
    styles). Disambiguate FIRST: palette -> colour tags; theme -> reject
    (it is the bogus-style artifact, already guarded at generation);
    only a genuine style NAME -> alias/pool."""
    cl = concept.lower()
    words = set(re.findall(r"[a-z]+", cl))
    # PALETTE: colour words / palette nouns -> not a style. Map to a
    # measured colour tag if the single family is clear, else NL.
    if _PALETTE_RE.search(cl) or (words & _COLOR_WORDS):
        # A COLOUR TONE IS PROSE (the author's 2026-09-15: "soft, desaturated,
        # muted, deep, cool are subjective fluff for colours; we won't
        # classify every colour tone"). A phrase that IS a measured colour
        # tag ('warm colors', 'blue theme') was caught above as a real tag;
        # everything else stays in the prose, never a tag.
        return {"type": "nl", "auto": True, "why": "a colour tone: prose only (2026-09-15)"}
    # THEME/OCCASION masquerading as a style -> reject (holidays are
    # handled in the genre channel; 'cozy holiday' is not a style)
    if words & _THEME_WORDS:
        return {"type": "reject", "auto": True,
                "why": "theme/occasion, not a style"}
    sp = lb._style_pool2()
    for name in sp:
        if name == cl or (len(name) >= 5 and name in cl):
            return {"type": "alias", "to": name, "why": "known style",
                    "auto": True}
    if cl in vc._style_vocab():
        return {"type": "alias", "to": cl, "why": "style-vocab tag",
                "auto": True}
    # a genuine NEW style: pool promotion is REVIEW-gated, measured later
    return {"type": "pool", "pool": "style", "measured": None,
            "why": "candidate new style -- needs measurement + review",
            "auto": False}


_CLASSIFY_SYS = (
    "You route an image-prompt concept that has NO booru tag. Answer ONE "
    "JSON object: {\"type\": \"decomp\"|\"nl\"|\"reject\", \"tags\": "
    "[booru tags]}. 'decomp' = the concept is a BUNDLE of concrete "
    "visible things, give 2-5 REAL booru tags that together convey it "
    "(only for concepts that decompose into drawable parts). 'nl' = a "
    "mood/atmosphere/abstract quality with no tag representation (keep "
    "it in the prose). 'reject' = meaningless or junk. Prefer 'nl' over "
    "forcing a bad decomposition.")


_SUBJECTIVE = {"data": None}


def subjective_words():
    """danbooru's tag group:subjective -- 'common subjective tags' (beautiful,
    cute, cool, attractive...) plus the fluff the ledger keeps collecting
    (gentle, subtle, slight, faint, delicate): modifiers that ride on a
    description, never a tag of their own (the author's 2026-09-15). The group's
    'exceptions' (colorful, fluffy, manly...) ARE tags and stay."""
    if _SUBJECTIVE["data"] is None:
        words = {"gentle", "subtle", "slight", "faint", "delicate", "lovely", "nice", "gorgeous", "stunning",
                 "amazing", "epic", "elegant", "graceful", "dreamy", "serene", "cozy", "moody",
                 "stately", "tough", "healthy", "robust"}       # the author's 2026-09-15
        try:
            w = _json(_paths.data("danbooru_wiki.json"), {}).get("groups") or {}
            for sec, tags in (w.get("subjective") or {}).items():
                if not str(sec).startswith("exception"):
                    words |= {str(t).lower() for t in tags if " " not in str(t)}
        except Exception:
            pass
        words -= {"color", "colour"}          # a palette word, not a judgement
        _SUBJECTIVE["data"] = words
    return _SUBJECTIVE["data"]


def strip_subjective(concept):
    """-> (core phrase, [modifiers]): the concept without its subjective
    words ('gentle rim light' -> 'rim light', ['gentle'])"""
    words = str(concept or "").lower().split()
    subj = subjective_words()
    mods = [w for w in words if w in subj]
    core = " ".join(w for w in words if w not in subj).strip()
    return core, mods


def _classify_one(concept, entry, lb):
    from promptstudio.engine import vocab as vc
    cl = concept.lower()
    if entry.get("is_style"):
        return _classify_style(concept, lb)
    # MEASUREMENT FIRST (no LLM): is it a real general tag we simply
    # failed to retrieve? -> ALIAS to itself (retrieval fix).
    v = vc._vocab()
    if v.get(cl, 0) >= 200 and vc._tag_cat(cl) == 0:
        return {"type": "alias", "to": cl, "why": "real tag, retrieval "
                "missed it", "auto": True}
    # A COLOUR TONE IS PROSE whatever band it came from (the author's 2026-09-15):
    # 'muted blue', 'desaturated greens', 'cool daylight tones' are said,
    # never tagged, never classified one by one
    _cw = set(re.findall(r"[a-z]+", cl))
    _TONE = {"pastel", "pastels", "muted", "desaturated", "saturated", "deep", "cool", "warm", "tone",
             "tones", "hue", "hues", "palette", "shade", "shades", "tint", "tints", "grading", "vivid",
             "vibrant", "dull", "faded", "washed", "rich", "bold", "neon", "earthy", "monochromatic"}
    if _PALETTE_RE.search(cl) or (_cw & _COLOR_WORDS) or (_cw & _TONE):
        return {"type": "nl", "auto": True, "why": "a colour tone: prose only (2026-09-15)"}
    # A SUBJECTIVE WORD RIDES ON A DESCRIPTION, IT IS NO TAG (the author's
    # 2026-09-15, tag group:subjective): 'gentle rim light' is the light
    # with a modifier; the core decides -- a real tag, decomposed to it
    # with the modifier kept for the prose; nothing real, junk
    core, mods = strip_subjective(cl)
    if mods and core != cl:
        if core and v.get(core, 0) >= 100 and vc._tag_cat(core) == 0:
            return {"type": "decomp", "tags": [core], "modifier": mods, "auto": True,
                    "why": "a subjective modifier on the tag '%s'" % core}
        try:
            cand = vc.confident_pick(core) if core else None
        except Exception:
            cand = None
        if cand and vc._tag_cat(cand) == 0:
            return {"type": "decomp", "tags": [cand], "modifier": mods, "auto": True,
                    "why": "a subjective modifier on '%s' (retrieves cleanly)" % cand}
        if not core:
            return {"type": "reject", "auto": True, "why": "subjective words only"}
    # does a confident retrieval now exist for it? -> ALIAS to that tag.
    try:
        cand = vc.confident_pick(concept)
    except Exception:
        cand = None
    if cand and vc._tag_cat(cand) == 0:
        return {"type": "alias", "to": cand, "why": "retrieves cleanly",
                "auto": True}
    # ambiguous -> ask the local model (decomp / nl / reject)
    try:
        cands = [t for _, t in vc.candidates_for(concept)[:6]]
        out = lb.chat(_CLASSIFY_SYS,
                      "%s   (candidate tags: %s)" % (concept,
                                                     ", ".join(cands)),
                      temp=0.1, max_tokens=120)
        pick = lb.parse_json_block(out)
        typ = pick.get("type")
        if typ == "decomp":
            tags = [t for t in (pick.get("tags") or [])
                    if isinstance(t, str) and v.get(t.lower(), 0) >= 100
                    and vc._tag_cat(t.lower()) == 0]
            if tags:
                return {"type": "decomp", "tags": tags[:5], "auto": True,
                        "why": "decomposes to real tags"}
        if typ == "reject":
            return {"type": "reject", "auto": True, "why": "junk"}
    except Exception:
        pass
    return {"type": "nl", "auto": True, "why": "legitimately NL-only"}


def classify(min_n=2, limit=0):
    """propose an integration type for every un-classified recurring
    concept. Writes proposals into the ledger and a review file."""
    import sys
    sys.path.insert(0, HERE)
    from promptstudio.engine import bridge as lb
    from promptstudio.engine import vocab as vc
    blob = _load(strict=True)
    if blob is None:
        print('ledger unreadable; nothing written')
        return
    con = blob.get("concepts", {})
    todo = [(k, v) for k, v in con.items()
            if v.get("status") == "new" and v.get("n", 0) >= min_n]
    todo.sort(key=lambda kv: -kv[1]["n"])
    if limit:
        todo = todo[:limit]
    print("classifying %d concepts (n>=%d)" % (len(todo), min_n))
    for i, (k, v) in enumerate(todo, 1):
        v["class"] = _classify_one(k, v, lb)
        v["status"] = "classified"
        if i % 10 == 0 or i == len(todo):
            with _Lock():
                _save(blob)
            print("  [%d/%d]" % (i, len(todo)))
    _write_review(blob)


_UA = {"User-Agent": "PromptStudio/1.0 (concept review)"}


def _get(url):
    import urllib.request
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=30) as r:
                return json.load(r)
        except Exception:
            import time
            time.sleep(1.5 * (attempt + 1))
    return None


def evidence(concept):
    """LIVE BOORU EVIDENCE for one concept (the author's: always check the
    boorus): danbooru's tag row (count, category), an active alias to a
    canonical tag, a wiki page by title, and gelbooru's count with the
    user's env key. Errors are reported as None, never as absence."""
    import urllib.parse
    q = concept.strip().lower().replace(" ", "_")
    out = {"danbooru": None, "alias_to": None, "wiki": None, "gelbooru": None}
    rows = _get("https://danbooru.donmai.us/tags.json?" + urllib.parse.urlencode({"search[name]": q}))
    if isinstance(rows, list):
        out["danbooru"] = ({"count": int(rows[0].get("post_count") or 0), "category": rows[0].get("category")}
                           if rows else {"count": 0})
    al = _get("https://danbooru.donmai.us/tag_aliases.json?" + urllib.parse.urlencode(
        {"search[antecedent_name]": q, "search[status]": "active"}))
    if isinstance(al, list) and al:
        out["alias_to"] = str(al[0].get("consequent_name") or "").replace("_", " ") or None
    wk = _get("https://danbooru.donmai.us/wiki_pages.json?" + urllib.parse.urlencode({"search[title]": q}))
    if isinstance(wk, list):
        out["wiki"] = (str(wk[0].get("body") or "")[:160] if wk else "")
    key, uid = os.environ.get("GELBOORU_API_KEY"), os.environ.get("GELBOORU_USER_ID")
    if key and uid:
        d = _get("https://gelbooru.com/index.php?page=dapi&s=tag&q=index&json=1&limit=1&names=%s&api_key=%s&user_id=%s"
                 % (urllib.parse.quote(q), key, uid))
        tg = ((d or {}).get("tag") or [])
        out["gelbooru"] = {"count": int(tg[0].get("count") or 0)} if tg else {"count": 0}
    return out


def gather_evidence(limit=0, pause=0.8):
    """attach live evidence to every classified entry without it; a real
    tag found live corrects the class to an alias (retrieval fix)"""
    import time
    blob = _load(strict=True)
    if blob is None:
        print('ledger unreadable; nothing written')
        return
    con = blob.get("concepts", {})
    todo = [(k, v) for k, v in con.items() if v.get("status") == "classified" and "evidence" not in v]
    todo.sort(key=lambda kv: -kv[1]["n"])
    if limit:
        todo = todo[:limit]
    print("evidence for %d concepts" % len(todo))
    for i, (k, v) in enumerate(todo, 1):
        ev = evidence(k)
        v["evidence"] = ev
        dan = (ev.get("danbooru") or {}).get("count") or 0
        if ev.get("alias_to"):
            # only an alias onto a GENERAL tag is a spelling; 'archer' ->
            # 'archer (fate)' is a character and would rename every archer
            import urllib.parse
            rows = _get("https://danbooru.donmai.us/tags.json?" + urllib.parse.urlencode(
                {"search[name]": ev["alias_to"].replace(" ", "_")}))
            cat = rows[0].get("category") if isinstance(rows, list) and rows else None
            ev["alias_category"] = cat
            if cat == 0:
                v["class"] = {"type": "alias", "to": ev["alias_to"], "auto": True, "why": "danbooru alias (live)"}
            else:
                v["class"] = {"type": "nl", "auto": True,
                              "why": "danbooru aliases it onto a non-general tag (%s); kept as prose" % ev["alias_to"]}
        elif dan >= 200 and (ev.get("danbooru") or {}).get("category") == 0 and v["class"].get("type") in ("nl", "reject", "pool"):
            v["class"] = {"type": "alias", "to": k, "auto": True, "why": "real danbooru tag (live), %d posts" % dan}
        if i % 10 == 0 or i == len(todo):
            with _Lock():
                _save(blob)
            print("  [%d/%d] %s" % (i, len(todo), k))
        time.sleep(pause)
    _write_review(blob)


def integrate(apply_pool=False):
    """INTEGRATE the reviewed ledger into the machinery the engine reads:
    alias -> alias_map_local.json, decomp -> concept_decomp.json, nl and
    reject -> concept_nl_only.json. Automatic classes apply as they are;
    a pool promotion applies only when its entry carries approved:1 (or
    apply_pool). Every integrated entry is marked with its date."""
    blob = _load(strict=True)
    if blob is None:
        print('ledger unreadable; nothing written')
        return
    con = blob.get("concepts", {})
    aliases = _json(ALIAS_LOCAL, {"_note": "aliases integrated from the concept ledger (review_ledger.py); "
                                          "merged into retrieval beside alias_map.json", "aliases": {}})
    decomp = _json(DECOMP, {"_note": "concept -> visible parts, integrated from the concept ledger; bridge 2 "
                                     "maps the concept to these tags directly", "concepts": {}})
    nlo = _json(NL_ONLY, {"_note": "concepts reviewed as natural-language only (or junk): kept in the prose, "
                                   "never logged again", "concepts": {}})
    now = datetime.datetime.now().isoformat(timespec="minutes")
    done = {"alias": 0, "decomp": 0, "nl": 0, "reject": 0, "pool": 0, "known": 0, "held": 0}
    try:
        from promptstudio.engine import vocab as _vc
        _vocab, _cat = _vc._vocab(), _vc._tag_cat
    except Exception:
        _vocab, _cat = {}, (lambda t: 0)
    for k, v in con.items():
        if v.get("status") != "classified":
            continue
        c = v.get("class") or {}
        typ = c.get("type")
        if typ == "alias" and c.get("to"):
            to = str(c["to"]).lower().strip()
            if to == k and (_vocab.get(k) or 0) >= 100:
                # a real tag retrieval missed: nothing to write, the miss
                # is the retrieval's (recorded, not integrated)
                v["status"], v["integrated"], v["note"] = "known", now, "real tag; retrieval missed it"
                done["known"] += 1
                continue
            if _cat(to) not in (0, None) or to not in _vocab:
                v["status"] = "held"           # not a general tag: never an alias
                done["held"] += 1
                continue
            aliases["aliases"][k] = to
        elif typ == "decomp" and c.get("tags"):
            # a decomposition is REVIEW-GATED like a pool promotion (the
            # model proposed 'cool tones -> cool colors, explosion background')
            if not (apply_pool or v.get("approved")):
                done["held"] += 1
                continue
            decomp["concepts"][k] = {"tags": list(c["tags"]), "why": c.get("why"), "when": now}
        elif typ in ("nl", "reject"):
            nlo["concepts"][k] = {"type": typ, "why": c.get("why"), "when": now}
        elif typ == "pool":
            if not (apply_pool or v.get("approved")):
                done["held"] += 1
                continue
            # a pool promotion is a writer's job (measured, the pool's own
            # rules); the ledger only records the approval for the writer
            v["pool_approved"] = now
        else:
            continue
        v["status"] = "integrated"
        v["integrated"] = now
        done[typ] += 1
    for path, tbl in ((ALIAS_LOCAL, aliases), (DECOMP, decomp), (NL_ONLY, nlo)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tbl, f, ensure_ascii=False, indent=1)
    with _Lock():
        _save(blob)
    print("integrated:", done)
    return done


def _write_review(blob):
    con = blob.get("concepts", {})
    rows = [(k, v) for k, v in con.items()
            if v.get("status") == "classified"]
    path = _paths.data("concept_review.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("CONCEPT LIBRARY REVIEW -- proposed integrations.\n")
        f.write("AUTO (alias to a general tag / nl / reject) apply on `integrate`; POOL and DECOMP "
                "entries wait for your approval (approved:1 on the ledger entry).\n")
        f.write("=" * 92 + "\n")
        for typ in ("pool", "decomp", "alias", "nl", "reject"):
            grp = [(k, v) for k, v in rows if v["class"]["type"] == typ]
            if not grp:
                continue
            f.write("\n--- %s (%d)%s\n" % (
                typ.upper(), len(grp),
                "  [REVIEW-GATED: set approved:1 on the ledger entry]" if typ in ("pool", "decomp") else ""))
            for k, v in sorted(grp, key=lambda kv: -kv[1]["n"]):
                c = v["class"]
                detail = (c.get("to") or ", ".join(c.get("tags") or [])
                          or c.get("why") or "")
                ev = v.get("evidence") or {}
                dan = ev.get("danbooru") or {}
                gel = ev.get("gelbooru") or {}
                evs = []
                if dan:
                    evs.append("dan %d" % (dan.get("count") or 0))
                if gel:
                    evs.append("gel %d" % (gel.get("count") or 0))
                if ev.get("alias_to"):
                    evs.append("alias->%s" % ev["alias_to"])
                if ev.get("wiki"):
                    evs.append("wiki: %s" % ev["wiki"][:70].replace("\n", " "))
                f.write("  n=%-3d %-32s -> %s%s\n" % (v["n"], k, detail,
                                                       ("   [" + "; ".join(evs) + "]") if evs else ""))
    print("review written: %s" % path)


def stats():
    blob = _load()
    con = blob.get("concepts", {})
    from collections import Counter
    by_status = Counter(v.get("status") for v in con.values())
    recurring = sum(1 for v in con.values() if v.get("n", 0) >= 3)
    styles = sum(1 for v in con.values() if v.get("is_style"))
    return {"total": len(con), "recurring(>=3)": recurring,
            "style-flagged": styles, "by_status": dict(by_status)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print(json.dumps(stats(), indent=1, ensure_ascii=False))
    else:
        print("concept_library: %s" % json.dumps(stats(), ensure_ascii=False))
