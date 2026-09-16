# Bismuth Prompt Studio (beta)

Bismuth Prompt Studio writes prompts for anime-style image models. You describe a
picture in a few words, or paste some tags, or type nothing at all, and it
gives you back a complete prompt: a line of booru tags plus a short paragraph
describing the scene. Two output formats are produced from every request,
one for the **Anima** model family and one for the **Illustrious** family.

Everything runs on your own computer. There is no account, no cloud service
and nothing is uploaded. The only network use is optional and small: when the
studio meets a pair of tags it has never measured before, it asks danbooru
for a post count once and remembers the answer.

---

## What "measured" means

Bismuth Prompt Studio does not guess what goes with what. Every choice it makes is
drawn from tables built from booru statistics: which tags exist, how often
each one appears, how often two tags appear together, what tends to be worn
in a given place, what an occupation implies, how explicit a tag is, which
artists draw what. When something has not been measured, the studio does not
roll it; it only appears when you type it yourself.

| what | count |
|---|---|
| booru tags known (danbooru / gelbooru) | 30,625 / 33,302 |
| tags with a measured explicitness floor | 6,666 |
| artists in the pool / drawable (100+ posts) | 576,754 / 24,928 |
| artists with a measured subject fingerprint (their top tags) | 24,928 |
| artists with a written style description | 25,050 |
| artists with a measured explicitness level (safe / sensitive / nsfw / explicit) | 25,054 (11,788 / 8,074 / 1,741 / 3,451) |
| characters with a canon appearance | 21,996 |
| non-booru concepts the checkpoints know (artists, series, lighting, fashion) | 4,473 (4,315 artists) |
| styles with measured artists | 56 |
| genres / locations / activities | 23 / 264 / 221 |
| places with a measured tag profile | 282 |
| occupations with measured places and undress states | 254 |
| garments in the clothes table | 1,104 |
| spice-table states (undress, acts, positions, toys, cum, attire) | 349 |

The two tag counts are the vocabularies read from each booru (a tag appears in both when both sites use it). Every roll the studio makes draws from these tables; what is not in them is not rolled.

Data snapshot: 2026-09-18.

---

## What the studio does, in short

- **Reads what you type.** Plain English ("a knight resting by a campfire at
  night"), booru tags ("1girl, sitting, cafe"), or a mix. Anything you type is
  kept: a tag you write is never dropped, a character you name is used, a
  pose you name is the pose.
- **Fills in the rest from measured tables.** Who is in the picture and how
  many, the genre, the place, the time of day and the weather, the lighting,
  the camera, what the subject is doing, the pose, the outfit, the body and
  the face, the artists. Each of these follows the others: an onsen makes a
  towel likely and a hospital does not, a librarian is in a library, a
  fisherman is by the water.
- **Writes the result two ways.** As a tag line, ordered the way image models
  read best, and as a paragraph. Every tag on the line is a real booru tag
  that was checked against the vocabulary.
- **Two engines.** The *fast* engine needs no language model and answers in
  one or two seconds; its paragraph is assembled from a template. The *llm*
  engine uses a local language model (through llama.cpp) to write the plan
  and the paragraph in richer prose; it needs a GPU and a model file, and
  takes 15 to 40 seconds.

## What it does not do

- It does not generate images. It writes prompts for image models.
- It does not invent tags. A word it cannot match to a measured tag stays in
  the paragraph, or goes on the tag line as a plain phrase in your own words.
- It does not learn from you. Nothing you type is stored, apart from the
  browser's "remember inputs" box if you tick it, and the tag-pair counts it
  looks up.
- It does not know your image model's quirks. What a checkpoint does with a
  tag is the checkpoint's business.
- It is not a content filter. It refuses sexualised-minor content at every
  level and never rolls a fixed list of extreme content (see *Limitations*).
  Beyond that, what you generate is your responsibility.

---

## Installing

**You need**

- Windows 10 or 11. (The code is plain Python; on Linux or macOS use
  `run_studio.sh` instead of the `.cmd` file.)
- Python 3.10 or newer, installed so that typing `python` in a terminal works.
  No extra packages are needed.
- For the *llm* engine only: an NVIDIA graphics card. 8 GB of video memory is
  comfortable for an 8-billion-parameter model; smaller models need less.

**Steps**

1. Unpack (or clone) this folder anywhere you like. Keep its contents as they are.
2. Double-click `run_studio.cmd`. A browser tab opens at http://localhost:7801.
   Without a language model the studio runs on the fast engine, and that is
   a complete, working setup.
3. If you want the *llm* engine, run `setup.cmd` and pick what to install.
   It downloads llama.cpp and a language model for you (see *Setting up the
   language model* if you would rather do it by hand).
4. To check an installation, run `python tools/selfcheck.py` from this
   folder. It lists any missing data file, generates one prompt with the fast
   engine, and tells you whether a model and the llama.cpp server were found.

## Updating

Close the studio, then double-click `update.cmd`. It shows what changed since
your build and asks before touching anything.

It downloads only the files that differ. Every build carries a list of its
files with a fingerprint for each (`release_manifest.json`), and the updater
compares that list from GitHub with the files on your disk. A typical update
is a few files, not the whole studio. Files the new build no longer has are
removed.

What it never touches:

- Your language model, llama.cpp and anything else in `LLM/`, the logs, and
  the optional embeddings pack.
- Files you edited. An edited file is kept as it is, and the new version is
  saved beside it with `.new` added to the name, so you can compare the two.
- Your language-model settings in `data/library/llm_config.json`.

The two caches the studio fills while it runs are combined with the new
build's instead of replaced, so nothing it has already looked up is lost.

The update is all or nothing. Every file is downloaded to a temporary folder
and checked first, and only then moved into place. If the connection drops,
the studio stays exactly as it was; run the updater again.

Useful options, from a terminal in this folder:

```
python tools/update.py --check            show what would change, change nothing
python tools/update.py --yes              update without asking
python tools/update.py --replace-edited   take the new version of edited files too
```

A copy downloaded before the updater existed has no file list, so the updater
cannot tell your edits from old files. Run it once with `--replace-edited`.

If you cloned the repository with git, the updater works the same way. Note
that `git pull` does not: each release replaces the published history.

---

## Setting up the language model

The studio runs its own copy of `llama-server` (the server program from the
llama.cpp project) and talks to it locally. Nothing else is supported: there
is no place to enter an API key and no online service is used.

### The easy way: setup.cmd

Double-click `setup.cmd` (or run `python tools/setup_models.py`). It shows
what is installed, asks what you want, and fetches it:

- **llama.cpp** for your machine -- it looks for an NVIDIA card and takes the
  CUDA build, otherwise a Vulkan or CPU build -- from llama.cpp's own GitHub
  releases.
- **A language model**, from its Hugging Face repository. The default is the
  uncensored Qwen3.5 4B (2.7 GB) the studio was tuned against; the official
  Qwen3 4B and 8B are offered as censored alternatives.
- **The embedding model** for semantic style matching (84 MB).
- **The artist embeddings pack** (179 MB, 424 MB unpacked).

It then writes the file names into `data/library/llm_config.json`, so the
studio is ready the next time you start it. Downloads resume: if one is
interrupted, run it again. Nothing is uploaded and no account is needed.

Useful flags:

```
python tools/setup_models.py --list                  what is offered, what is here
python tools/setup_models.py --all                   llama.cpp + the first model + embeddings
python tools/setup_models.py --model qwen3-8b        one model by name
python tools/setup_models.py --llama-cpp cuda        force a build (cuda / vulkan / cpu)
python tools/setup_models.py --pack                  the artist embeddings pack
python tools/setup_models.py --latest                newest llama.cpp instead of the tested build
python tools/setup_models.py --manifest <url>        a different list of downloads
```

The list itself is `data/library/setup_manifest.json`. Add your own model to
it with a direct download link (a Hugging Face `resolve` URL works well), or
set `_manifest_url` to a copy you host, and the installer will read that
instead.

### By hand

The `LLM` folder already has the structure the studio expects. You only add
files; there is a `README.txt` in each folder saying what goes there.

```
LLM/
  your-model.gguf              <- 1. your language model
  llama.cpp/
    backend/                   <- 2. llama-server.exe and the .dll files that came with it
    vendor/                    <- 3. only if your llama.cpp build ships a separate CUDA runtime folder
```

1. **The model.** Download a chat model in GGUF format and put the file in
   `LLM/`. The studio was developed and tested with Qwen3-family models of 4
   to 8 billion parameters at Q4 to Q5 quantisation. Any instruction-tuned
   model can be tried; the studio asks it for strictly formatted answers, and
   the smaller the model, the more often it fails to produce one (the studio
   then answers that prompt with the fast engine instead).
2. **llama.cpp.** Download a llama.cpp release build matching your graphics
   card (for NVIDIA cards, the CUDA build) from the llama.cpp releases page.
   Unpack it and copy `llama-server.exe` together with all its `.dll` files
   into `LLM/llama.cpp/backend/`. If the download also contains a separate
   CUDA runtime folder (named like `cudart-…`), put that whole folder into
   `LLM/llama.cpp/vendor/`.
3. **Tell the studio the model's name.** Open `data/library/llm_config.json`
   in a text editor. It already exists and looks like this:

   ```jsonc
   {
     "main":     "your-model.Q4_K_M.gguf",   // the file name of your model inside LLM/
     "ctx":      8192,                       // how much text the model can hold at once; 8192 is right for the studio
     "model_id": "your-model",               // a label for the server; any short name
     "thinking": false,                      // true only for Qwen3 "thinking" models (adds /no_think, strips the think block)
     "embed":    null                        // optional: an embedding model file for semantic style matching (next section)
   }
   ```

   Change `main` to your file name and save. That is the whole configuration.
   (If you prefer environment variables: `PROMPTSTUDIO_LLM_MAIN`,
   `PROMPTSTUDIO_LLM_CTX` and `PROMPTSTUDIO_LLM_MODEL_ID` override the file.)
4. Start the studio again. The line under the title tells you what it found:
   *built-in engine, auto-starts on first generate* means the model and the
   server are in place. The first generation then takes about 20 seconds
   longer while the model loads. The model unloads itself after ten minutes
   without use, and the **release LLM** button unloads it at once when you
   want the video memory back for image generation.

**Changing the model later** is the same edit of `main`. The chat template
comes from the model file itself, so nothing else needs to change.

### Semantic style matching (optional)

The studio picks artists two ways, and which way depends on what you have
installed.

**Always, with nothing extra:** when you type a style name the studio knows
(`art nouveau`, `western comics`, `1990s`), it draws that style's measured
artists. When you describe a look in your own words ("painterly, muted
colours, melancholic"), it compares your words with the written style
descriptions of the artists and prefers the artists whose descriptions
answer them.

Those descriptions were written by a vision model that looked at each
artist's own pictures, for 25,000 artists, and then cleaned into one
terminology: about 8 terms per artist, 180 looks shared by ten artists
or more, each written one way ("cinematic
mood", never also "cinematic atmosphere"), a booru tag where one means the
whole phrase ("backlit lighting" is `backlighting`), and nothing that does
not describe how a picture is drawn (clothing, eyes and hair, bare body
parts, genres, ages, the model's own ratings). Terms that mean the same look
were merged by hand, so an artist described as "curvaceous" and one described
as "curvy" answer the same word. The vocabulary is small and plain --
"soft shading", "pastel palette", "melancholic mood", "sensual mood",
"exaggerated anatomy", "painterly". So a word of yours counts when a
description uses it or one of its dictionary synonyms (from WordNet):
"gloomy" is written nowhere, and reaches the 1,279 artists described as
"dark"; "sexy"
reaches "seductive", "sensual" and "erotic"; "sad" reaches "melancholic". A
colour word stands only for a colour, because the descriptions use colours
for palettes ("blue" is no synonym of "gloomy" here). A word the dictionary
does not know ("hypersexualised") and no description uses changes nothing.

**With the embeddings pack installed:** the same comparison is done by
meaning instead of by exact words. Your description and every artist's
description are turned into vectors by a small embedding model, and the
closest artists win: "hypersexualised" reaches artists described as
"erotic", "sensual" or "exaggerated" (all of the closest twenty), and
"gloomy" mostly those described as "melancholic" or "dark" (thirteen of
twenty; an artist's name counts too, so one called "gloomyowl" ranks high).
A described look is matched to the style pool the same way, less reliably:
a word or two rarely lands on the right one of 78 styles. Artists are then
written with a strength that follows how well they match.

To turn it on you need all three of:

1. llama.cpp installed as described above (the embedding model is served by
   the same `llama-server`).
2. The embedding model **nomic-embed-text-v1.5** in GGUF form (`setup.cmd`
   fetches this for you): the file
   `nomic-embed-text-v1.5.Q4_K_M.gguf` (84 MB) from the
   `nomic-ai/nomic-embed-text-v1.5-GGUF` repository on Hugging Face. Put it
   in `LLM/` and write its name into `data/library/llm_config.json`:
   `"embed": "nomic-embed-text-v1.5.Q4_K_M.gguf"`. The studio itself never
   downloads model files; only `setup.cmd` does, when you ask it to.
3. The embeddings pack (`PromptStudio-embeddings.zip`, the precomputed
   vectors of 24,927 artists, 179 MB zipped, 424 MB unpacked), from the
   studio's release page, unzipped into the studio folder so that
   `data/embeddings/artist_embeddings.json` exists. It ships separately
   because the unpacked file is larger than a git repository allows;
   `setup.cmd` installs it for you (`python tools/setup_models.py --pack`).
   If you installed an older pack, install it again: the updater never
   touches the pack.

When all three are present the studio uses them automatically; the status
of the embedding server is logged in `logs/llm_server.log`. When any is
missing, the exact-word matching above is used and nothing else changes.

---

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

### generate random artists

When ticked, the studio adds one to three artist references it thinks fit.
"Fit" is measured: an artist whose own posts carry the same kinds of tags as
this picture (the same acts, garments, world) is drawn more often, and an
artist is only drawn from the pool of the spice level. Safe and sensitive
prompts draw artists whose measured work is safe or sensitive; nsfw and
explicit prompts draw artists whose work is nsfw or explicit. Artists you
typed yourself are always kept, whatever the box says.

### appearance

Where a subject's look comes from when you did not spell it out.

- **automatic**: your brief first; then the character's canon if it is a
  known character; then the ordinary dependencies (a nurse wears a nurse's
  outfit, a beach means swimwear, the level and the action shape the rest).
- **random**: your brief first, then the dice for everything else.

### character detail

How much description each subject gets: `minimal`, `standard` or `detailed`.

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

Quality and level first, then the count (`1girl`, `2girls`), then for each
subject: name and series, hair, eyes, body, clothes and undress, expression.
After the subjects: poses and activities, what happens between subjects,
medium, style and artists, genre, the place and its details, light and
weather, camera. The paragraph opens with the place, describes each subject
by name or noun, and ends with the style, the lighting and the mood.

---

## Limitations

- **Measured, not imaginative.** The studio draws what the booru shows. Rare
  combinations are rare in its output, and a concept the booru does not tag
  cannot become a tag. Some creative requests come back plainer than you
  wrote them; the paragraph keeps your words, the tag line keeps only what
  is measured.
- **It does not guess intent.** "salon" is a hair salon, "sitting" is the
  pose, an artist name made of ordinary words needs the `@`. Ambiguous words
  are taken at their booru meaning.
- **Some content is never rolled, some only when typed.** Refused even when
  typed: sexualised minors, at every level; youth-coded words ("child",
  "young girl", and the dictionary words that share their meaning, such as
  "lass" or "sonny") at sensitive and above. Never
  rolled by the dice: mutilation, torture, scat, guro, bestiality and the
  like. Rolled only when you type them: bdsm, rape, incest, pet play, public
  use, extreme body types, and a list of niche tags. Typed content within the
  chosen level is honoured as written.
- **The language model is a model.** It can be slow, it can fail on a hard
  brief (the fast engine then answers), and it can contradict the plan; the
  studio re-renders the paragraph when it catches a contradiction and uses
  the template if the model insists. Wording changes from run to run even
  with a seed.
- **The vocabulary is a snapshot.** Tags added to the boorus after the data
  was built are unknown until the data is rebuilt, which this release cannot
  do.
- **Artists are booru artists.** `@artist` references are danbooru artist
  tags. Whether your image model knows an artist is up to the model.
- **Describing a look in free words matches by words and their dictionary
  synonyms unless the embeddings pack is installed** (see *Semantic style
  matching*): "gloomy" finds "dark", but a synonym can carry
  another of the word's meanings ("low" is one of "gloomy"'s), and a word the
  dictionary does not know finds only itself. Typing a style name works fully
  either way.
- **Scenes with several subjects** take longer with the llm engine (around
  two minutes) and are the least stable part of the generator.
- **Network.** One post-count lookup per never-seen tag pair, cached. Offline,
  those pairs count as unmeasured. Nothing else leaves the machine.
- **Speed.** The first generation of a session loads the tables (a few
  seconds). A brief whose tag pairs were never counted is slower the first
  time; the same brief is fast afterwards.
- **No image preview, no command line, no batch API.** The web page is the
  tool; the JSON bundle is the export.

---

## Folder layout

```
run_studio.cmd / run_studio.sh   start the web UI (http://localhost:7801)
setup.cmd                        install llama.cpp, a language model, the embedding model
update.cmd                       update to the latest build (downloads only what changed)
release_manifest.json            this build's file list and fingerprints, for the updater
promptstudio/                    the program
  engine/     scene resolution, the two engines, the verifiers
  library/    artist and concept helpers
  llm/        the llama-server driver and its config loader
  ui/         the web page (one file, standard-library HTTP server)
data/                            what the engine reads (never edited by the studio,
                                 except two caches: tag-pair counts and slot hints)
  tags/       booru vocabularies, definitions, explicitness floors, aliases
  pools/      characters, artists, locations, genres, styles, media, quality
  library/    the measured tables and llm_config.json
  embeddings/ artist fingerprints (the top tags of each artist's posts), the
              style vectors, and the artist vectors once the pack is installed
LLM/                             your model and llama.cpp (folders ready, files not shipped)
logs/                            the language-model server log
tools/setup_models.py            the installer behind setup.cmd
tools/update.py                  the updater behind update.cmd
tools/selfcheck.py               install check
```

## Settings you may edit

- `data/library/llm_config.json` — the language model (above).
- `data/library/artist_denylist.json` — artists never drawn by the dice.
  Add names to taste; typing a listed artist still works.
- `data/library/location_rulings.json` — the place rulings the engine honours
  (aliases, generic words, genre binding). Keep the shapes if you edit it.
- `data/library/alias_map_local.json` — extra spellings the parser maps to
  booru tags.
- The UI port is 7801. To use another: `python -c "from promptstudio.ui import
  studio as s; s.PORT = 7899; s.main()"`.

## Credits

The measured data derives from public booru tag statistics (danbooru,
gelbooru) and from the definitions on their wikis. llama.cpp, the language
model and any image checkpoint you use are separate projects under their own
licenses and are not part of Prompt Studio.

`data/library/english_lexicon.json` is derived from
[Open English WordNet](https://github.com/globalwordnet/english-wordnet)
(2025 edition), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Kept from it: each
word's parts of speech, the categories of its noun senses, its listed
inflected forms, the synonyms and broader terms that are booru words or
words of the artists' style descriptions, and
lists of the words that name a person (with the gender and age their senses
and definitions state). The studio uses it to read a brief as a sentence:
which words are verbs, what a conjugated verb comes from, whether a noun
names someone or something, what a word is called on the booru ('crimson' is
red), and who a word like 'seamstress' or 'mariner' is.
