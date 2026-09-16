#!/usr/bin/env python
"""
tag_net.py — the tag RELATIONSHIP NETWORK, as a network.

tag_graph.json holds, for ~30k danbooru tags, the tags that actually co-occur with
them and how often. Reading one row out of that file is a lookup, not a network:
it copies one tag's neighbours and stops. This module treats the same data as the
graph it is.

Four things make it behave like a network instead of a lookup table:

1. HUBNESS CORRECTION. Danbooru's `frequency` is P(other | tag), so `1girl`, `solo`
   and `long hair` are "strongly related" to virtually everything. Raw frequency
   therefore ranks the most generic tag first every single time. Each tag gets a
   specificity weight from how promiscuous it is across the whole graph, and edges
   are scaled by it — the association between `throne` and `crown` survives, the
   association between `throne` and `1girl` does not.

2. MUTUAL EDGES. An edge that exists in both directions is far better evidence than
   a one-way one. a->b and b->a combine as a geometric mean; a one-way edge is
   discounted.

3. SPREADING ACTIVATION over several hops. One hop from `throne` gives `crown`.
   Two hops reaches `stone wall`, `red carpet`, `tapestry` — tags no single row
   contains but which genuinely belong to the scene. Activation decays per hop.

4. MUTUAL COHERENCE, not chain-following. A candidate is scored by how much of the
   set already chosen supports it, so a tag tied hard to one seed but alien to the
   rest loses to a tag tied moderately to many. After each acceptance the set is
   re-scored, so the prompt converges on something that hangs together instead of
   drifting tag by tag away from what was asked for.

The network never outranks the user: callers pass seeds (the user's own tags and
any explicitly selected category) and a veto, and the net only ever proposes tags
for slots the user left open.
"""

import json
import math
import os
import re
from collections import defaultdict
from promptstudio import paths as _paths

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = _paths.data("tag_graph.json")

# What role a tag plays. Ordered — first match wins.
ROLE_PATTERNS = [
    # Unambiguous lighting/atmosphere phrases are claimed FIRST. Otherwise the
    # scene pattern wins on a shared word - "window light" was filed as a place
    # because it contains "window" - and weather, which the user wants treated
    # as atmosphere, was filed as scenery.
    ("lighting", r"window light|golden hour|blue hour|afternoon sun|morning sun|"
                 r"candlelight|firelight|moonlight|sunlight|backlight|rim light|"
                 r"god ?rays?|dappled|volumetric|cinematic light|dramatic light|"
                 r"soft light|hard light|studio light|neon light|"
                 r"^(rain|snow|snowing|fog|mist|haze|overcast|storm|cloudy sky|"
                 r"blizzard|drizzle|sunshower)$"),
    ("scene", r"outdoor|indoor|beach|ocean|\bsky$|cloud|forest|\bcity\b|street|"
              r"\broom$|bedroom|bathroom|\bpool\b|onsen|\bbath\b|office|classroom|"
              r"school|window|\btree\b|\bgrass\b|mountain|castle|dungeon|\bbar$|"
              r"kitchen|festival|\bwater$|underwater|scenery|architecture|building|"
              r"\bthrone\b|\bcarpet\b|\bwall$|\bfloor$|\bcurtain|\bfurniture|shelf|"
              # boundaries matter: unanchored 'rain' matches 'film grain', which
              # is a render effect, not a place
              r"\bdesk$|\bbed$|\bcouch$|\bstairs$|\bpillar|ruins|\bsnow|\brain\b"),
    # Expressions are tested BEFORE lighting: 'light' as a substring otherwise
    # claims 'light smile' and files a facial expression as a lighting tag.
    ("expression", r"smile|blush|expression|grin|\beyes$|smirk|frown|open mouth|"
                   r"tongue|wink|crying|tears|ahegao|embarrass|\bshy\b|seductive|"
                   r"naughty|looking|\bgaze\b|pout"),
    # Objects with 'light' in the name are not lighting: 'bow of light' is a
    # weapon, not an illumination style, and it was being emitted as one. This
    # role is claimed first so those tags never reach the lighting pattern.
    ("object", r"\b(bow|sword|blade|staff|wand|orb|arrow|spear|gun|rifle|ring|"
               r"crystal|shield|armou?r|amulet|gem|lightsaber|lantern|torch)\b"),
    ("lighting", r"light|\blit\b|glow|shadow|backlit|contrast|sunbeam|lens flare|"
                 r"bokeh|sunlight|moonlight|\blamp\b|neon|chiaroscuro|god ?rays?|"
                 r"\bdusk$|\bdawn$|sunset|sunrise|\bnight$|\bday$|twilight|"
                 r"golden hour|blue hour|afternoon sun|morning sun|window light|"
                 r"candlelight|firelight|dappled|god ?ray|rim ?light|spotlight|"
                 r"ambient|luminous|radiance|silhouette|overcast|gloom"),
    ("camera", r"^from |\bfocus$|close-?up|depth of field|blurr|^pov$|\bangle\b|"
               r"\bshot\b|\bportrait$|foreshortening|out of frame|straight-on|"
               r"\bprofile$|wide angle|fisheye|upper body|full body|dutch angle"),
    ("hair", r"\bhair\b|\bhair$|\bbangs$|ahoge|twintail|ponytail|braid|\bbun$|sidelocks"),
    ("pose", r"sitting|standing|lying|kneeling|squat|\bpose$|posing|leaning|\barms |"
             r"\blegs |hand on|hands on|crossed|spread|arched|bent over|all fours|"
             r"straddl|stretch|walking|running|\bjump|reclin|sprawl"),
    ("clothing", r"\bdress$|\bshirt$|skirt|bikini|panties|\bbra$|lingerie|uniform|"
                 r"thighhigh|stocking|pantyhose|garter|swimsuit|leotard|jacket|"
                 r"\bcoat$|sweater|hoodie|apron|corset|kimono|yukata|\bmaid\b|"
                 r"topless|bottomless|\bnude\b|underwear|clothes|clothing|"
                 r"see-through|fishnet|choker|\bgloves$|\bboots$|\bheels$|\bsocks$|"
                 r"\bcrown$|\bjewelry$|earrings|necklace|\btiara$|\bcape$|\bveil$|\brobe$"),
    ("body", r"\bbreasts$|\bskin$|\bthighs$|\bhips$|\bwaist\b|midriff|\bnavel\b|"
             r"collarbone|freckle|\bmole\b|\bsweat\b|\bwet\b|\babs$|muscul|curvy|"
             r"\bslim\b|petite|\btall\b|\btan\b|tanline|cleavage|sideboob|underboob|"
             r"skindentation|armpit|barefoot|\bfeet$|\btoes$|\bass\b|\bthick (thighs|body|hips)\b|"
             r"\bhorns$|\btail$|animal ears|pointy ears|wings"),
]
_ROLES = [(r, re.compile(p, re.I)) for r, p in ROLE_PATTERNS]

# Tags that are structural rather than descriptive — the net must never propose
# these, they are decided by the caller (subject counts, ratings, meta).
NEVER_PROPOSE = re.compile(
    # YOUTH-CODED VOCABULARY IS NEVER GENERATED (the author's standing rule:
    # only typed input may carry it; no rolled pool, network proposal or
    # persona may). The act pool held 'onee-shota' and 'kodomo doushi' with
    # nothing refusing them. One list, matched anywhere in the tag; this
    # regex is the gate every rolled pool passes (fastplan._draw,
    # propose(), the enhancer's network route).
    r".*\b(?:loli|lolis|lolicon|shota|shotas|shotacon|kodomo|toddlercon|"
    r"age regression|aged down|underage|onii-shota|onee-shota|oppai loli)\b|"
    # a file property, never a picture (the author's: "useless")
    r"^transparent background$|"
    # horror rolls (2026-09-04), but never these (the author's: "no body horror or guro as rolls")
    r"^(guro|body horror|gore)$|"
    # 'alternate costume/hairstyle/hair length (shorter)...' are a
    # CHARACTER'S variations on canon -- meaningless unprompted
    r"^alternate\b|"
    r"^\d\+?(girls?|boys?)$|^(solo|multiple (girls|boys)|no humans|solo focus|"
    # ensemble bookkeeping: a character's measured canon carried these
    r"everyone|multiple others)$|"
    # subject-structure tags: the caller decides who is in frame, not the network.
    # 'male focus' leaking in turned every female-subject prompt into a male one.
    r"^(male focus|female focus|1other|others|couple|group)$|"
    # implied by any specific value, so emitting them adds nothing and invites
    # 'breasts' next to 'large breasts'
    r"^(breasts|hair|eyes|clothes|clothing|body|skin|legs|arms|hands|feet)$|"
    r"^(highres|absurdres|lowres|commentary|translated|artist request|"
    r"bad id|bad link|md5 mismatch|revision|third-party edit|check \w+|"
    r"tagme|character request|source request|duplicate|resolution)|"
    # tags that DESCRIBE A DRAWING ERROR. Asking a model for these asks for the
    # error: 'twisted breasts' was appearing in prompt after prompt.
    # Narrowly: only tags that name a MISTAKE. 'floating hair' and 'impossible
    # geometry' are deliberate effects, not errors, so no broad 'floating \\w+' or
    # 'impossible \\w+' patterns here.
    r"^(twisted (breasts|torso|neck)|bad (anatomy|hands|proportions|perspective)|"
    r"anatomically incorrect|extra (arms|legs|digits|fingers|breasts|heads)|"
    r"missing (limb|arm|leg|finger|hand)s?|disembodied (hand|limb|head|penis))$|"
    # booru event/meme tags: 'parsee day', 'nue day' are fan-celebration dates and
    # were being read as lighting because they end in 'day'
    r"^(?!(?:sunny|rainy|cloudy|clear|bright|overcast|summer|winter|spring|"
    r"autumn|hot|cold|snowy|windy|stormy|mid)\b)\w+ day$|"
    r"\b(day|week|month|anniversary)\b.*\b\d{4}\b|'s pose$|\bmeme\b",
    re.I)

# A tag with brackets cannot go into a prompt: '(' and ')' are emphasis syntax, so
# 'fuuka school uniform (hara hara!!)' is read as markup, not as a costume.
BRACKETED = re.compile(r"[()\[\]{}<>|]")


# Species/kind traits change WHAT the subject is, not how it looks. The network
# may add description; it may not turn a woman into a kemonomimi because 'animal
# ears' happens to co-occur with everything. These are allowed through only when
# a seed already establishes that kind — ask for an elf and you get pointy ears.
SPECIES_RE = re.compile(
    r"\b(animal (?:hands|feet|legs|nose|head)|"
    r"animal ears|cat ears|fox ears|dog ears|rabbit ears|mouse ears|"
    r"wolf ears|horse ears|bear ears|cow ears|animal ear\w*|kemonomimi|"
    r"pointy ears|elf|elve|horns?|antlers|halo|wings?|\btail\b|tails|"
    r"monster girl|demon|succubus|angel|mermaid|harpy|slime|centaur|"
    r"orc|goblin|dragon|robot|android|cyborg|ghost|vampire|zombie|"
    r"furry|anthro|scales|fangs)\b", re.I)


def role_of(tag):
    for role, pat in _ROLES:
        if pat.search(tag):
            return role
    return None


class TagNet:
    """Spreading-activation view over tag_graph.json."""

    # Danbooru's co-occurrence data is the only edge source. A second layer
    # measured from harvested civitai prompts used to sit on top of it; it was
    # retired once the banks had absorbed everything it taught us.

    def __init__(self, graph, vocab=None, counts=None, min_posts=1500):
        self.fwd = {}
        self.rev = defaultdict(dict)
        for tag, rels in graph.items():
            t = tag.lower()
            row = {}
            for r in rels:
                if not r:
                    continue
                rel, freq = r[0].lower(), float(r[1])
                row[rel] = freq
                self.rev[rel][t] = freq
            if row:
                self.fwd[t] = row

        # 1. HUBNESS. A tag listed as "related" by half the vocabulary tells us
        #    nothing about any particular tag. Mass, not count: a tag that shows up
        #    everywhere AND with high frequency is the most useless of all.
        self.spec = {}
        for tag, incoming in self.rev.items():
            mass = sum(incoming.values())
            self.spec[tag] = 1.0 / (1.0 + math.log1p(mass))
        self._default_spec = 1.0

        # Everything danbooru has an opinion about. If a tag is in here and
        # danbooru still did not rank it beside another tag, that silence is a
        # judgement over millions of images and nothing gets to overrule it —
        # 'cow print bikini' has 14,712 danbooru posts and a 0.0000 edge to
        # 'bikini', and that zero is the correct answer.
        self.danbooru_vocab = set(self.fwd) | set(self.rev)

        # A POPULARITY FLOOR. Danbooru has a long tail of hyper-specific tags —
        # 'nine ball maid uniform', "dio brando's pose" — that are real tags but
        # that no checkpoint has meaningfully learned and that no one wants in a
        # generated prompt. Reverse traversal reaches them easily, so proposals
        # are floored by how much the tag is actually used. The floor applies
        # to every candidate; nothing is exempt from it any more.
        self.counts = {k.replace('_', ' ').lower(): v
                       for k, v in (counts or {}).items()}
        self.min_posts = min_posts

        # tags we are allowed to emit at all (danbooru + bank vocabulary)
        self.vocab = {v.lower() for v in vocab} if vocab else None
        self._act_cache = {}
        self._rev_cache = {}
        self._row_cache = {}

    def __len__(self):
        return len(self.fwd)

    def edge(self, a, b):
        """2. MUTUAL EDGES — both directions is much stronger evidence than one."""
        ab = self.fwd.get(a, {}).get(b, 0.0)
        ba = self.fwd.get(b, {}).get(a, 0.0)
        return math.sqrt(ab * ba) if (ab and ba) else 0.6 * (ab or ba)

    REVERSE_DISCOUNT = 0.5   # b->a is real evidence that a and b belong together,
                             # but it is P(a|b), not P(b|a), so it is worth less
    REVERSE_WIDTH = 50       # a hub's reverse index is the whole vocabulary

    def _reverse_row(self, node):
        """Edges pointing AT this node.

        Danbooru returns only ~30 neighbours per tag, and for a common tag those
        slots are entirely consumed by hubs: 'bikini' lists breasts, 1girl, solo,
        long hair... and never reaches 'beach'. But 'beach' DOES list bikini at
        0.64. The relationship is in the graph, just stored one way round, so
        traversing only forward edges throws half the graph away — that is why
        seeding 'bikini' could not find a beach.
        """
        hit = self._rev_cache.get(node)
        if hit is not None:
            return hit
        inc = self.rev.get(node)
        if not inc:
            self._rev_cache[node] = {}
            return {}
        # A hub's reverse index holds most of the vocabulary, and re-sorting it on
        # every traversal step was the single biggest cost in generating a prompt.
        # It never changes, so it is computed once per tag.
        ranked = sorted(inc.items(),
                        key=lambda kv: -kv[1] * self.spec.get(kv[0], 1.0))
        out = {src: freq * self.REVERSE_DISCOUNT
               for src, freq in ranked[:self.REVERSE_WIDTH]}
        self._rev_cache[node] = out
        return out

    def _row(self, node):
        """Every edge touching this node, forward and reverse, merged.
        Static per tag, so it is built once and reused."""
        hit = self._row_cache.get(node)
        if hit is not None:
            return hit
        row = dict(self.fwd.get(node) or {})
        for src, w in self._reverse_row(node).items():
            if w > row.get(src, 0.0):
                row[src] = w
        out = {k: w for k, w in row.items() if w > 0}
        self._row_cache[node] = out
        return out

    def activate(self, seeds, hops=2, decay=0.45, width=40):
        """3. SPREADING ACTIVATION. seeds -> {tag: activation}, hubness-corrected.

        Cached: one enhance() asks for the same seed set once per slot and once per
        category, so without this the identical spread is recomputed a dozen times
        and a single prompt costs seconds."""
        ck = (tuple(sorted((k.lower(), round(v, 3)) for k, v in seeds.items())),
              hops, decay, width)
        hit = self._act_cache.get(ck)
        if hit is not None:
            return hit
        act = defaultdict(float)
        frontier = {s.lower(): w for s, w in seeds.items()}
        seen = set(frontier)
        for _ in range(max(1, hops)):
            nxt = defaultdict(float)
            for node, energy in frontier.items():
                row = self._row(node)
                if not row:
                    continue
                # only the strongest edges propagate, or everything reaches everything
                top = sorted(row.items(), key=lambda kv: -kv[1] * self.spec.get(kv[0], 1.0))[:width]
                for rel, freq in top:
                    gain = energy * freq * self.spec.get(rel, self._default_spec) * decay
                    if gain > 1e-4:
                        act[rel] += gain
                        nxt[rel] += gain
            frontier = {k: v for k, v in nxt.items() if k not in seen}
            seen.update(frontier)
        for s in seeds:
            act.pop(s.lower(), None)
        if len(self._act_cache) > 256:      # per-process, bounded
            self._act_cache.clear()
        self._act_cache[ck] = act
        return act

    def support(self, cand, chosen):
        """4. MUTUAL COHERENCE — how much of the current set actually backs this tag.
        A tag welded to one seed and alien to the rest is what causes drift."""
        if not chosen:
            return 1.0
        hits = 0.0
        for c in chosen:
            e = self.edge(cand, c.lower())
            if e:
                hits += min(1.0, e * 4.0)
        return hits / len(chosen)

    def propose(self, seeds, n, role=None, veto=None, exclude=(), rng=None,
                hops=2, temperature=0.6, pool=60):
        """Top candidates for a slot, sampled (so variants differ) rather than argmax.

        seeds   {tag: weight} — the user's tags and any selected category.
        role    restrict to one slot ('hair', 'clothing', 'lighting', ...).
        veto    callable(tag) -> True to reject (the caller's conflict engine).
        """
        if not self.fwd:
            return []
        act = self.activate(seeds, hops=hops)
        if not act:
            return []
        low_ex = {e.lower() for e in exclude} | {s.lower() for s in seeds}
        # only a prompt that already establishes a non-human kind may acquire more
        # of one; otherwise species traits are off the table entirely
        seeded_species = any(SPECIES_RE.search(s) for s in seeds)
        cands = []
        for tag, a in act.items():
            if tag in low_ex or NEVER_PROPOSE.match(tag) or BRACKETED.search(tag):
                continue
            if self.counts and self.counts.get(tag, 0) < self.min_posts:
                continue
            if not seeded_species and SPECIES_RE.search(tag):
                continue
            if self.vocab is not None and tag not in self.vocab:
                continue
            if role and role_of(tag) != role:
                continue
            if veto and veto(tag):
                continue
            cands.append((tag, a))
        if not cands:
            return []
        cands.sort(key=lambda kv: -kv[1])
        cands = cands[:pool]

        picks, chosen = [], [s for s in seeds]
        for _ in range(n):
            scored = []
            for tag, a in cands:
                if tag in picks:
                    continue
                # coherence re-scored against the growing set after every
                # acceptance
                s = a * (0.35 + 0.65 * self.support(tag, chosen))
                if s > 0:
                    scored.append((tag, s))
            if not scored:
                break
            scored.sort(key=lambda kv: -kv[1])
            head = scored[:12]
            if rng is None or temperature <= 0:
                pick = head[0][0]
            else:
                top = head[0][1] or 1.0
                weights = [(t, (s / top) ** (1.0 / temperature)) for t, s in head]
                total = sum(w for _, w in weights)
                r, acc = rng.uniform(0, total), 0.0
                pick = weights[-1][0]
                for t, w in weights:
                    acc += w
                    if r <= acc:
                        pick = t
                        break
            picks.append(pick)
            chosen.append(pick)
            if veto and veto(pick):       # caller state may have moved
                picks.pop()
        return picks


def load(vocab=None, path=GRAPH_PATH):
    """the danbooru co-occurrence graph"""
    graph = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            graph = json.load(f)
    return TagNet(graph, vocab)


if __name__ == "__main__":
    import sys
    net = load()
    print(f"network: {len(net.fwd)} danbooru rows, "
          f"{len(net.rev)} tags reachable")
    seeds = {t.strip(): 1.0 for t in (sys.argv[1] if len(sys.argv) > 1
                                      else "throne, elf, 1girl").split(",")}
    print("seeds:", ", ".join(seeds))
    for role in (None, "scene", "clothing", "lighting", "expression"):
        got = net.propose(seeds, 6, role=role)
        print(f"  {str(role or 'any'):11s} -> {', '.join(got) if got else '(nothing yet)'}")
