# Bismuth Prompt Studio: the manual

How to write a brief that gives you what you meant, and what every control does. Installing, updating and the language model are in README.md.

---

## Writing a good brief

The studio reads whatever you type: booru tags, plain sentences, or a mix. A
few habits make the result much more predictable.

**Typed is law.** Anything you type is kept and placed; the dice only fill
what you left open. If you want something, say it. If you do not want
something, put it in **exclude**.

**One subject needs no ceremony.** `1girl, nurse, hospital` or `a nurse
walking down a hospital corridor at night` both work.

**Several subjects: write one sentence per subject.** This is the single most
useful habit. Separate the sentences with full stops:

1. **First sentence: the scene, and how many are in it** (as tags or in
   words).
2. **Then one sentence for each subject**, opening on that subject (`First
   girl ...`, `Second girl ...`, `The boy ...`, or simply `1girl, ...`).
3. **Then anything else**: style, light, weather, mood.

```
Two girls playing water volleyball in the ocean. First girl has big breasts
and is dressed in a red bikini. Second girl has blue eyes. Sunny weather.
```

or with tags

```
2girls, water, ocean, playing volleyball. 1girl, large breasts, red bikini.
1girl, blue eyes, black hair. sunny
```

or mixed. Written this way, each subject's sentence owns its words: the red
bikini lands on the first girl and the blue eyes on the second, what a
sentence says a subject *does* (sitting, reading, drinking) is bound to that
subject, and the cast is counted from the first sentence only. Without it the
studio still reads the brief, but has to guess which description belongs to
whom.

**Say the count once.** `2girls` or `two girls` in the scene sentence is
enough; repeating it elsewhere invites a miscount.

**Races, jobs and characters.** A typed race (`elf`, `cat girl`, `lion girl`)
brings its own ears, tail or horns. A typed job brings its outfit. A typed
character brings its canon look. The **fantasy races** and **professions**
boxes only decide whether the dice may add one you did not type.

**Framing decides what is described.** If you type a frame that leaves the
face out (`lower body`, `head out of frame`, or a low focus such as `ass
focus` together with `close-up`), the studio stops describing hair, eyes and
expression on the tag line, because those words pull the image model back to
the face.

**Avoid ambiguity.** The studio does not guess what an unclear word means.
`a girl and her friend` names one subject and a vague companion: say `two
girls` or `a girl and a boy`. `cowgirl` can be a job or a position: say
`cowgirl (western)` or `cowgirl position`. A word that names both a character
and an ordinary thing (`a female doctor`) may be read as the character. When
a result surprises you, the brief usually left something open; say it
plainly and it will be kept.

**What the style hint understands** is listed in full in
[STYLE_TERMS.md](STYLE_TERMS.md): styles, mediums, lighting, techniques,
palettes, and the looks read off the artists' own work.

**Artists go in the artist field**, one per comma, spelled as on the booru.
The `@` is optional; parentheses and weights like `(wlop:0.8)` are kept.

**Reading the tag line.** The line is written in parts, separated by `;`:
everything before the subjects (quality, level, count), then one part per
subject, then everything after them (style, artists, place, light, camera).
Inside a subject's part the order is *who* before *what*: name and series,
race, maturity, profession; then hair, eyes, body, clothes and undress, face;
then what that subject is doing.

---

### Name the gender when you can

A role on its own -- "a student", "a knight", "a barista" -- names somebody but
not which somebody, so the studio picks the gender from what the booru measures
for that word. That is a good guess, not your intent. Writing "a female
student", "1girl, knight" or "a male barista" removes the guess entirely, and
the same goes for every other thing you actually care about: the studio never
overrides a word you typed, and anything you leave out it has to decide for
you.

## A first walk-through

1. Type `a girl reading in a library` into the big box.
2. Leave every control at its default and press **Generate**.
3. A card appears with two tabs, **anima** and **illustrious**. Each shows the
   prompt for that model family: the tag line, and for Anima a paragraph
   under it. Both are editable in place.
4. Press **copy prompt** and paste it into your image tool.

Try the same brief with **spice** set to `nsfw`, or with **generate random
characters** ticked, and compare.

---

## Every control, explained

### The brief (the big text box)

This is where you say what you want. Any of these works:

- A sentence: `a nurse and a doctor in a hospital at night`
- Booru tags: `1girl, solo, long hair, blue eyes, school uniform, classroom`
- A character by name, spelled as on danbooru: `hatsune miku singing on stage`
- An artist: `drawn by wlop`, or `@incase, 1girl, tavern`
- Nothing. An empty brief is "surprise me": the studio invents the whole
  scene within the controls you set.

Words that match a measured tag become tags. Words that do not are kept in
the paragraph, or placed on the tag line as a phrase in your own words. For
example `cluttered desk` gives the tag `desk` and keeps "cluttered" as a phrase.

Type two letters and a suggestion list opens, with the number of pictures on
danbooru or gelbooru that carry each tag. Up and down choose, Tab or Enter
inserts, Esc closes the list.

### let the LLM invent it

The button above the brief asks the local language model for a scene idea,
about thirty-five words of plain English, and puts it in the brief for you to
edit or generate. Use it when you want a starting point rather than a finished
prompt.

It never reads what is already in the brief, so pressing it again cannot hand
you back the idea it just wrote. Instead it takes the spice level, the genre
and the period from the controls, rolls a place (and sometimes a job or an
action) from the measured tables, and tells the model which ideas and which
words it has already used. What comes back is your text, not the studio's: it
is not verified, mapped to tags or corrected in any way. Pressing **Generate**
afterwards runs the normal pipeline on it, exactly as if you had typed it.

Two limits worth knowing. The button is greyed out without a language model,
because there is nothing to invent with. And a small model has favourite
images: pinned to one genre and one spice level for many presses in a row, it
will circle back to similar scenes. Leaving the genre on `random` gives the
widest spread.

If the brief already had text in it, a link appears next to the button to put
your own words back.

### exclude

Things that must not appear. Whatever you write here goes to the negative
line of both output formats, is removed from the plan before the paragraph is
written, and is kept out of the dice. A word you typed in the brief still wins
over this box.

### style hint

A look for the picture, kept separate from the scene: the name of a style the
studio knows (`art nouveau`, `western comics`, `1990s`), a medium
(`watercolor`, `pixel art`, `oil painting`), or a palette or technique. Styles
that are not booru tags but that image models understand (`ukiyo-e`, `in the
style of van gogh`) are written in the form the models expect.

Separate entries with commas; each entry is read whole. An entry the studio
knows (a style name, a medium, a booru tag) is used as such. An entry that a
booru tag means as a whole becomes that tag (`glossy skin` gives `shiny
skin`). Anything else is kept as your own phrase: `loose linework` goes on the
tag line and into the paragraph exactly as written ("Drawn with loose
linework"), because image models read such phrases well. Every entry also
steers which artists are drawn. A translation never raises the spice level
you chose; above it, the entry stays your phrase.

Your look also shapes the style the dice would otherwise pick. A rolled style
or medium that the boorus (or the artists' descriptions) rarely show with one
of your entries is dropped: `chiaroscuro` removes a rolled `retrofuturism`.
Pairs that were never measured are left alone.

This box has its own suggestion list, so you do not have to guess what the
studio knows. Type two letters and it offers two kinds of entry:

- **Styles**, with a post count: the style names the studio has measured, the
  mediums, the studio and cultural looks, and the palette and technique tags
  those styles are made of. A style showing no number is a look the image
  models know that danbooru never tagged; it still works.
- **Looks**, marked *look* with a number of artists: the terms the artists'
  cleaned style descriptions use ("smooth linework", "muted palette",
  "dramatic lighting"), with how many artists are described that way. A look matches
  those artists exactly. Looks that belong to explicit work are offered only
  at the `nsfw` and `explicit` spice levels.

### artist hint

Artist names, comma separated, spelled as on danbooru (spaces or underscores
both work). Put `@` in front of a name when it is also an ordinary phrase, so
the studio knows you mean the artist: `@naked_cat`. For a name that is not an
ordinary phrase the `@` is optional.

Type two letters and the box suggests artists with their post counts. The list
holds every artist the studio can actually steer -- those with at least 100
pictures on danbooru, plus the artists the image models know from elsewhere.
A rarer name can still be typed by hand with `@` in front; there is simply no
measured evidence behind it.

### remember inputs

Keeps everything you typed and selected in this browser and restores it the
next time you open the page. Stored only in the browser.

### engine

- **fast**: no language model. One to two seconds. The paragraph is assembled
  from a template, so it is plain but always consistent with the tags.
- **llm**: the local language model writes the plan and the paragraph. Richer
  prose, 15 to 40 seconds, needs the setup above. If the model fails on a
  prompt, that prompt falls back to the fast engine.

### genre

The world the picture belongs to: fantasy, dark fantasy, cyberpunk,
historical periods, everyday life and so on. `random` lets the dice pick. A
chosen genre shapes everything downstream: the places, the jobs people have,
the races, the poses, the clothes. If your brief names a genre in words,
that wins over the dropdown.

### spice

How explicit the picture may be. The four levels follow the booru rating
system:

| level | roughly |
|---|---|
| **safe** | nothing sexual; everyday clothing |
| **sensitive** | swimwear, lingerie, mild suggestion, no nudity |
| **nsfw** | nudity and sexual framing, no explicit acts |
| **explicit** | visible genitals and sex acts |

`auto` reads the level from your words: "a nude woman" is at least nsfw,
"having sex" is explicit, a plain sentence stays safe. Every tag has a
measured floor (the mildest rating it actually appears under), and the level
gates them: a tag whose floor is above the chosen level is not rolled. The
level also decides how likely an undress is (measured by level, place, genre
and occupation: an onsen undresses, a hospital does not) and which artists
may be drawn (see *generate random artists*).

### quality

The quality tags at the head of the line (`masterpiece, best quality, …`).
`standard` is the usual set, `maximum` the long one, `off` none.

### period

The era band the Anima models understand (`newest`, a year, a decade). Only
the Anima format uses it.

### variants

How many prompts to make from this one brief, one card each (1 to 10). Each
variant rolls its own dice.

### seed

A number that makes the dice repeatable. Leave it blank for a fresh roll
every time. With the fast engine the same seed and inputs give the same
prompt; with the llm engine the structure repeats but the model's wording may
differ.

### generate random characters

When ticked, subjects you did not name become existing characters, matched to
the gender your brief implies and, if you named a series, taken from that
series. A character brings its canon look: hair, eyes, outfit. Unticked, the
subjects stay anonymous ("the girl", "the knight").

At sensitive and above only characters nothing measured shows as young are
drawn: not those the booru's wiki calls minors or high-school students, nor
those whose pictures the booru often tags `loli` / `shota`. A character like
that typed by name at those levels is dropped, and the subject stays unnamed.

### generate random artists

When ticked, the studio adds one to three artist references it thinks fit.
"Fit" is measured: an artist whose own posts carry the same kinds of tags as
this picture (the same acts, garments, world) is drawn more often, and an
artist is only drawn from the pool of the spice level. Safe and sensitive
prompts draw artists whose measured work is safe or sensitive; nsfw and
explicit prompts draw artists whose work is nsfw or explicit. Artists you
typed yourself are always kept, whatever the box says.

### fantasy races

Ticked (the default), the dice may give an unnamed subject a race of the
picture's genre: an elf, an oni, an android. Unticked, every rolled subject is
human. A race you type is always honoured.

### professions

Ticked (the default), the dice may give a subject a profession with its
outfit: a nurse, a knight, an idol. Unticked, no profession is rolled. One you
type is always honoured.

### appearance

Where a subject's look comes from when you did not spell it out.

- **automatic**: your brief first; then the character's canon if it is a
  known character; then the ordinary dependencies (a nurse wears a nurse's
  outfit, a beach means swimwear, the level and the action shape the rest).
- **random**: your brief first, then the dice for everything else.

### character detail

How much description each subject gets: `minimal`, `standard` or `detailed`.
`minimal` keeps who the subject is and what they wear (hair, eyes, the
outfit); `standard` adds the extras the booru usually shows with them --
legwear, shoes, headwear, neckwear, accessories -- at their measured rates;
`detailed` draws those extras nearly twice as often, adds body marks and more
of the parts the frame is about, and keeps more of your unmapped phrases on
the tag line.

### Generate / cancel

**cancel** stops after the current model call; variants already finished stay.

### release LLM

Unloads the language model and frees its video memory, for when you want to
generate images now. The next Generate loads it again (about 20 seconds).

---

## The result card

Each variant is one card with two tabs:

- **anima**: a tag line with `@artist` references, then a paragraph.
- **illustrious**: the same scene as a tag line with weighted tags and no
  paragraph.

Both are editable right on the card; edits survive switching tabs.

- **copy prompt** copies what you see: tag line and paragraph.
- **copy story** copies the paragraph alone.
- **use story as start** puts the paragraph (edited or not) back into the
  brief, for a second pass that starts from it.
- **reuse inputs** restores every control that made this card, seed included.
- **download bundle** saves every result of this run as one JSON file, with
  the tags, the plan and the reasons behind the choices.

Under the prompt, a note line says what happened to your words: which typed
tags were kept as is, which were mapped to a booru tag, what was dropped
because of the level, which character or artist was chosen.

### How a tag line is ordered

See "Reading the tag line" at the top of this manual. The paragraph under the
tag line (Anima) opens with the place, describes each subject by name or
noun, and ends with the style, the lighting and the mood.

---

