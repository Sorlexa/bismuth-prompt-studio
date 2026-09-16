#!/usr/bin/env python
"""
scene.py -- the cast of a prompt, decided once.

STAGE 1: THIS MODULE OBSERVES AND CHANGES NOTHING. It is built from the parser
and compared against the existing behaviour; no generation path imports it yet.
That is deliberate -- the point of stage 1 is to find out where the structured
reading disagrees with the current one before anything depends on it.

The problem it exists to solve: every rule in the generator changes meaning with
the number of people in frame, and today that number is re-derived at each step
from whatever tags happen to be present. Deciding it once and holding it fixed
is what stops per-subject rules from multiplying with the subject count.

Four subject kinds, not six. An animal that acts like a character is an `other`
(danbooru's `1other`, 127,133 posts); an animal that is merely depicted is not a
subject at all and the scene carries `no humans` instead. There is no `1animal`
or `1object` tag on either booru, so those cannot be kinds -- they are only ever
a flavour of emptiness, deciding which focus tag the scene wants.

    from promptstudio.engine.scene import Scene
    s = Scene.build("futanari Tifa Lockhart fucks Aerith", banks)
    s.summary()   # 'read as: 2 subjects - 1 futa (tifa lockhart) - 1 girl ...'
"""

import re
from dataclasses import dataclass, field

from promptstudio.engine import enhancer as pe
from promptstudio.engine import slots as sm

FEMALE, MALE, FUTA, OTHER = "female", "male", "futa", "other"

KIND_NOUN = {FEMALE: "girl", MALE: "boy", FUTA: "futa", OTHER: "other"}

# A typed POV compound names the VIEWER'S TYPE, and that viewer is a subject of
# its own. Measured over all 1,382 `futanari pov` posts: 67% also carry `1girl`
# and only 3% carry `solo`, so the tag overwhelmingly means "a futa is looking at
# somebody else". Bare `pov` says nothing about who is holding the camera and is
# a property of the scene instead.
TYPED_POV = {
    "futanari pov": FUTA,
    "male pov": MALE,
    "female pov": FEMALE,
}

# Slots whose tags belong to the picture rather than to any one person.
CAMERA_SLOTS = ("viewpoint", "framing", "focus")
SCENE_SLOTS = ("scene", "lighting")
# Slots whose tags describe a person.
SUBJECT_SLOTS = ("hair", "body", "clothing", "expression", "gaze", "pose")


@dataclass
class Subject:
    kind: str
    name: str = None
    descriptors: list = field(default_factory=list)
    is_viewer: bool = False

    def label(self):
        return self.name or KIND_NOUN.get(self.kind, self.kind)

    def describe(self):
        # always state the kind, even for a named character. Whether the model
        # thinks Tifa is a girl or a futa is exactly the thing that needs to be
        # visible, and a bare name hides it.
        bits = self.label()
        if self.name:
            bits += f" [{KIND_NOUN.get(self.kind, self.kind)}]"
        if self.is_viewer:
            bits += " (viewer)"
        if self.descriptors:
            bits += ": " + ", ".join(self.descriptors[:5])
        return bits


@dataclass
class Scene:
    subjects: list = field(default_factory=list)
    empty_kind: str = None        # animal | object | scenery, when nobody is here
    interactions: list = field(default_factory=list)
    camera: list = field(default_factory=list)
    place: list = field(default_factory=list)
    loose: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    forced: dict = None

    # -- counts ------------------------------------------------------------
    def count(self, kind):
        return sum(1 for s in self.subjects if s.kind == kind)

    @property
    def total(self):
        return len(self.subjects)

    def census(self):
        return {"females": self.count(FEMALE), "males": self.count(MALE),
                "futa": self.count(FUTA), "others": self.count(OTHER),
                "total": self.total}

    def summary(self):
        if not self.subjects:
            what = self.empty_kind or "scenery"
            return f"read as: no subjects ({what})"
        parts = []
        for kind in (FEMALE, FUTA, MALE, OTHER):
            n = self.count(kind)
            if n:
                noun = KIND_NOUN[kind] + ("s" if n > 1 else "")
                parts.append(f"{n} {noun}")
        head = f"read as: {self.total} subject{'s' if self.total > 1 else ''}"
        return head + " - " + " - ".join(parts)

    def reading(self, emitted_tags=None):
        """The interpretation, in a form the studio and the API can show.

        This is the whole point of stage 2. Committing to a cast makes a wrong
        reading produce a COHERENTLY wrong prompt, which is harder to notice
        than scattered tags -- so the reading has to be visible before anything
        depends on it. `ambiguous` marks the prompts where a different reading
        was genuinely defensible.
        """
        out = {
            "summary": self.summary(),
            "cast": [{"kind": s.kind,
                      "name": s.name,
                      "label": s.label(),
                      "viewer": s.is_viewer,
                      "descriptors": list(s.descriptors)} for s in self.subjects],
            "shared": list(self.interactions),
            "camera": list(self.camera),
            "empty_kind": self.empty_kind,
            "notes": list(self.notes),
            "ambiguous": bool(self.notes),
            "mismatch": None,
        }
        if emitted_tags is not None:
            # Normalise whatever the caller hands over. A raw comma split leaves
            # ' (1girl:1.15)' with a leading space and emphasis syntax attached,
            # and the census then matches nothing and reports every prompt as
            # subject-less -- which is how this check first flagged 89 of 100.
            clean = []
            for t in emitted_tags:
                t = str(t).strip().strip("()").split(":")[0].strip().lower()
                t = t.replace(chr(92), "")
                if t:
                    clean.append(t)
            got = sm.subject_census(clean)
            if got["total"] != self.total:
                out["mismatch"] = (f"the prompt names {got['total']} subject(s) "
                                   f"but the reading found {self.total}")
                out["ambiguous"] = True
        return out

    def detail(self):
        lines = [self.summary()]
        for s in self.subjects:
            lines.append("   " + s.describe())
        if self.interactions:
            lines.append("   shared: " + ", ".join(self.interactions))
        if self.camera:
            lines.append("   camera: " + ", ".join(self.camera))
        for n in self.notes:
            lines.append("   ? " + n)
        return "\n".join(lines)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, text, banks):
        tags, _ = pe.parse_input(text, banks)
        return cls.from_tags(text, tags, banks)

    @classmethod
    def from_tags(cls, text, tags, banks, cast=None):
        low = [str(t).lower() for t in tags]
        sc = cls(tags=list(low))
        sc.forced = cast or None
        # A HAND-CORRECTED CAST IS NOT A HINT. When one is given the scene is
        # built to match it exactly, or the serialiser would re-infer the cast
        # from the tags at the end and quietly overrule the correction.
        cen = (dict(sm.subject_census(low),
                    females=int(cast.get("female", 0) or 0),
                    males=int(cast.get("male", 0) or 0),
                    futa=int(cast.get("futa", 0) or 0),
                    others=int(cast.get("futa", 0) or 0)
                           + int(cast.get("other", 0) or 0))
               if cast else sm.subject_census(low))

        # --- who is here ---------------------------------------------------
        names = _named_characters(text, low, banks)
        said = pe.character_attributes(text or "", list(names), banks)

        futa_names = {n for n in names
                      if any(a in ("futanari", "1futa") for a in said.get(n, []))}

        # A named character takes its type FROM THE PROMPT, never from its
        # profile: 'futanari Tifa' is a futa however Tifa is usually tagged.
        for n in names:
            kind = FUTA if n in futa_names else _kind_of_name(n, banks)
            sc.subjects.append(Subject(kind=kind, name=n,
                                       descriptors=list(said.get(n, []))))

        # unnamed subjects, from the count tags
        want = {FEMALE: cen["females"], MALE: cen["males"],
                FUTA: cen.get("futa", 0), OTHER: cen["others"] - cen.get("futa", 0)}
        for kind, n in want.items():
            have = sc.count(kind)
            for _ in range(max(0, n - have)):
                sc.subjects.append(Subject(kind=kind))

        # --- the viewer ----------------------------------------------------
        viewer_kind = next((k for t, k in TYPED_POV.items() if t in low), None)
        if viewer_kind:
            claimed = next((s for s in sc.subjects
                            if s.kind == viewer_kind and not s.name
                            and not s.is_viewer), None)
            if claimed is not None:
                claimed.is_viewer = True
            elif sc.forced:
                pass          # the cast was set by hand; do not add to it
            else:
                sc.subjects.append(Subject(kind=viewer_kind, is_viewer=True))
                sc.notes.append(
                    f"a {KIND_NOUN[viewer_kind]} viewer was added by the typed "
                    f"pov tag; the alternative reading merges it with a subject "
                    f"already in frame")

        # --- nobody at all -------------------------------------------------
        # ...unless the tags plainly describe a body. A comma-separated list
        # like 'colored skin, red skin, black horns, flying whale' carries no
        # count tag at all, and reading it as an empty scene because it mentions
        # a whale ignores the three tags describing somebody's skin and horns.
        # Personal descriptors are themselves evidence that a person is present.
        if not sc.subjects:
            personal = [t for t in low
                        if sm.slot_of(t, banks) in ("hair", "body", "clothing")]
            if personal and "no humans" not in low and not sc.forced:
                sc.subjects.append(Subject(kind=FEMALE))
                sc.notes.append(
                    "no count tag, but personal descriptors are present "
                    f"({', '.join(personal[:3])}) -- assumed one subject")
            else:
                sc.empty_kind = _empty_kind(text, low)

        # --- what the tags are about ---------------------------------------
        owned = {d.lower() for s in sc.subjects for d in s.descriptors}
        for t in low:
            if t in owned or _is_count_tag(t):
                continue
            slot = sm.slot_of(t, banks)
            if t in TYPED_POV or slot in CAMERA_SLOTS:
                sc.camera.append(t)
            elif (slot in SCENE_SLOTS or t in pe.INDOOR_PLACES
                  or t in pe.OUTDOOR_PLACES):
                sc.place.append(t)
            elif _is_shared_act(t, banks):
                sc.interactions.append(t)
            elif slot in SUBJECT_SLOTS:
                sc.loose.append(t)
            else:
                sc.loose.append(t)

        # --- give every personal tag an owner ------------------------------
        # A trait with nobody to belong to is how 'purple eyes' ended up reading
        # as a property of the alley. Assign to the first subject it does not
        # contradict; conflicts use the ordinary one-of groups.
        still_loose = []
        for t in sc.loose:
            # A tag DERIVED from a stated attribute belongs to whoever owns that
            # attribute: 'penis' comes from 'futanari', so it follows the futa
            # rather than waiting to be assigned on its own merits.
            owner = next((s for s in sc.subjects
                          if any(sm.implied_by(d, t) for d in s.descriptors)),
                         None)
            if owner is not None:
                owner.descriptors.append(t)
                continue
            if sm.slot_of(t, banks) not in SUBJECT_SLOTS:
                still_loose.append(t)
                continue
            keys = set(pe.Conflicts()._keys(t))
            placed = False
            for subj in sc.subjects:
                if any(keys & set(pe.Conflicts()._keys(d))
                       for d in subj.descriptors):
                    continue
                subj.descriptors.append(t)
                placed = True
                break
            if not placed:
                still_loose.append(t)
        sc.loose = still_loose

        # --- flag the reading that could honestly have gone the other way ---
        # A typed pov beside a SOLO action is the genuinely undecidable case:
        # 'elf princess masturbates, futanari pov' is either one futa seeing
        # herself or a futa watching a girl. The measured default is two, since
        # only 3% of futanari pov posts are solo -- but 5% are masturbation, so
        # this particular shape is exactly where the default is weakest.
        if viewer_kind and not sc.interactions:
            sc.notes.append(
                f"typed pov with no shared action: read as a separate "
                f"{KIND_NOUN[viewer_kind]} viewer, but the subject in frame "
                f"could be the {KIND_NOUN[viewer_kind]} seeing themselves")

        # --- does the cast support what is happening? ----------------------
        heads = sm.heads_available(sm.subject_census(low))
        for act in sc.interactions:
            need = sm.arity_of(act, banks)
            if need > max(heads, 1):
                sc.notes.append(
                    f"{act!r} needs {need} people but the cast has {heads}")
                break
        return sc


def _is_count_tag(t):
    return bool(re.match(r"^(\d+\+?(girls?|boys?|others?|futas?)|solo|"
                         r"multiple (girls|boys|others|futa)|no humans|1futa)$", t))


def _is_shared_act(tag, banks):
    """an action that takes more than one person belongs to the scene, not to
    any single subject -- which is the split stage 4 serialises on"""
    return sm.arity_of(tag, banks) >= 2


def _named_characters(text, tags, banks):
    """character names the user actually used, in the order they were written"""
    profiles = banks.get("_profiles") or {}
    if not profiles:
        return []
    found = []
    for t in tags:
        if t in profiles and t not in found:
            found.append(t)
    words = re.findall(r"[a-z0-9'\-]+", (text or "").lower())
    for i in range(len(words)):
        for size in (4, 3, 2, 1):
            cand = " ".join(words[i:i + size])
            if cand in profiles and cand not in found:
                found.append(cand)
                break
    return found


def _kind_of_name(name, banks):
    """a profile says whether a character is usually drawn as a girl or a boy"""
    rel = dict((t, f) for t, f in (banks.get("_profiles") or {}).get(name, [])[:12])
    if rel.get("1boy", 0) > rel.get("1girl", 0):
        return MALE
    return FEMALE


def _empty_kind(text, tags):
    """with nobody in frame, what is the picture of?"""
    joined = " ".join(tags)
    if sm.NONHUMAN_SUBJECT.search(text or "") or sm.NONHUMAN_SUBJECT.search(joined):
        return "animal"
    if "scenery" in tags or any(sm.slot_of(t) == "scene" for t in tags):
        return "scenery"
    return "object"


if __name__ == "__main__":
    import sys
    banks = pe.load_all_banks()
    for line in (sys.argv[1:] or [
            "pov, futanari Tifa Lockhart fucks Aerith Gainsborough",
            "elf princess with huge tits masturbates sitting on the throne, futanari pov",
            "two girls sitting on a bench, one smiling and one annoyed",
            "the owl is soft and fluffy",
            "a girl in a red dress doing a handjob to the viewer"]):
        print(f"\n{line}")
        print(Scene.build(line, banks).detail())


# Count vocabulary differs by booru, so it differs by MODE. Gelbooru pairs
# `futanari` with `1futa` on 81% of its 99,280 posts; danbooru has no `1futa`
# tag at all (0 posts), so there a futa is counted into the girl total and
# `futanari` rides along as a descriptor. Emitting `2futas` -- which exists
# nowhere -- would just spend conditioning on a token no model has seen.
COUNT_MODIFIERS = ("multiple girls", "multiple boys", "multiple others",
                   "multiple futa", "futa with female", "futa with male",
                   "solo focus", "solo")


def _plural(n, noun):
    if n == 1:
        return "1" + noun
    return (f"{n}{noun}s" if n < 6 else f"6+{noun}s")


class _Serializer:
    """Turns a Scene back into an ordered tag list.

    The order IS the structure: count, then one contiguous run per subject, then
    what they do together, then where the camera is, then the world. Reading a
    prompt should tell you who owns what without having to guess.
    """

    def __init__(self, scene, mode, banks):
        self.sc, self.mode, self.banks = scene, mode, banks

    def counts(self):
        sc = self.sc
        if getattr(sc, "forced", None):
            return count_tags_for(sc.forced, self.mode)
        out = []
        futa = sc.count(FUTA)
        girls = sc.count(FEMALE)
        if self.mode == "anima":
            if futa:
                out.append("1futa" if futa == 1 else "multiple futa")
        else:
            girls += futa            # danbooru counts a futa among the girls
        if girls:
            out.append(_plural(girls, "girl"))
        if sc.count(MALE):
            out.append(_plural(sc.count(MALE), "boy"))
        if sc.count(OTHER):
            out.append(_plural(sc.count(OTHER), "other"))
        if not out:
            out.append("no humans")
        return out

    def run(self, all_tags, quality, style):
        sc = self.sc
        seen = set()
        out = []

        def take(seq):
            for t in seq:
                tl = str(t).lower()
                if tl and tl not in seen:
                    seen.add(tl)
                    out.append(t)

        low = [str(t) for t in all_tags]
        lowset = {t.lower() for t in low}
        qs = {str(q).lower() for q in (quality or ())}
        st = {str(x).lower() for x in (style or ())}

        take([t for t in low if t.lower() in qs])
        take([t for t in low if t.lower() in st])
        take(self.counts())
        take([t for t in COUNT_MODIFIERS if t in lowset])

        for subj in sc.subjects:
            if subj.name:
                take([subj.name])
            take(subj.descriptors)

        take(sc.interactions)
        take(sc.camera)
        take(sc.place)
        # anything left keeps its original order rather than being dropped --
        # except the count tags we already re-emitted in the right vocabulary,
        # which would otherwise come back and contradict them ('2girls' followed
        # by the '1girl, 1futa' the parser had produced).
        take([t for t in low if not _is_count_tag(str(t).lower())])
        return out


def count_tags_for(cast, mode):
    """{'female':1,'futa':1,...} -> the count tags this mode's booru would use.

    Same vocabulary rule as the serialiser: gelbooru (anima) has `1futa`,
    danbooru (illustrious) has none and counts a futa among the girls.
    """
    girls = int(cast.get("female", 0) or 0)
    futa = int(cast.get("futa", 0) or 0)
    males = int(cast.get("male", 0) or 0)
    others = int(cast.get("other", 0) or 0)
    out = []
    if mode == "anima":
        if futa:
            out.append("1futa" if futa == 1 else "multiple futa")
    else:
        girls += futa
    if girls:
        out.append(_plural(girls, "girl"))
    if males:
        out.append(_plural(males, "boy"))
    if others:
        out.append(_plural(others, "other"))
    if not out:
        out.append("no humans")
    return out


COUNT_RE = _COUNT_PATTERN = None
