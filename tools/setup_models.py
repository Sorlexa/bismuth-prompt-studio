#!/usr/bin/env python
"""
setup_models.py -- fetch what the language-model engine needs.

Run it from the studio folder:

    python tools/setup_models.py                 # ask, then install
    python tools/setup_models.py --list          # what is offered, what is here
    python tools/setup_models.py --all           # llama.cpp + a model + the embed model
    python tools/setup_models.py --model qwen3-4b --llama-cpp auto --embed
    python tools/setup_models.py --manifest https://example/manifest.json

What it installs, and from where (nothing is mirrored -- every file comes
from the project that publishes it):

    LLM/llama.cpp/backend/      llama-server.exe and its libraries, from
    LLM/llama.cpp/vendor/       llama.cpp's own GitHub releases (MIT)
    LLM/<model>.gguf            a language model, from its Hugging Face repo
    LLM/nomic-embed-...gguf     the embedding model for style matching

Then it writes the file names into data/library/llm_config.json, so the
studio is ready on the next start.

Downloads resume: interrupt it and run it again. Nothing is uploaded and
no account is needed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

# the studio folder: this file ships as <studio>/tools/setup_models.py and
# lives in the workshop as <studio>/tools/release/setup_models.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(ROOT) == "tools":
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
UA = {"User-Agent": "bismuth-prompt-studio-setup/1.0"}


def _paths_data(name):
    try:
        from promptstudio import paths
        return paths.data(name, expect=False)
    except Exception:
        return os.path.join(ROOT, "data", "library", name)


def manifest(src=""):
    if src.startswith("http"):
        with urllib.request.urlopen(urllib.request.Request(src, headers=UA), timeout=30) as f:
            return json.loads(f.read().decode("utf-8"))
    p = src or _paths_data("setup_manifest.json")
    with open(p, encoding="utf-8-sig") as f:
        m = json.load(f)
    url = (m.get("_manifest_url") or "").strip()
    if url.startswith("http") and not src:
        try:                                   # a newer list, if one is published
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15) as f:
                return json.loads(f.read().decode("utf-8"))
        except Exception:
            pass
    return m


def llm_dir():
    d = os.path.join(ROOT, "LLM")
    os.makedirs(d, exist_ok=True)
    return d


def have_gpu():
    """-> 'cuda', 'vulkan' or 'cpu', by what the machine answers"""
    try:
        r = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                           shell=True, capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return "cuda"
    except Exception:
        pass
    try:
        r = subprocess.run("wmic path win32_VideoController get name", shell=True,
                           capture_output=True, text=True, timeout=20)
        blob = (r.stdout or "").lower()
        if "radeon" in blob or "intel" in blob or "arc" in blob:
            return "vulkan"
    except Exception:
        pass
    return "cpu"


def human(n):
    return "%.0f MB" % (n / 1e6) if n < 1e9 else "%.1f GB" % (n / 1e9)


def download(url, dest, expect_mb=0):
    """resumable GET with a progress line; returns the path"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers=dict(UA))
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        if have:                                # the server refused the resume: start over
            os.remove(part)
            return download(url, dest, expect_mb)
        raise e
    total = int(r.headers.get("Content-Length") or 0) + (have if r.status == 206 else 0)
    if r.status != 206:
        have = 0
    mode = "ab" if have else "wb"
    done = have
    last = -1
    with open(part, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = int(done * 100 / total)
                if pct != last:
                    last = pct
                    sys.stdout.write("\r    %s  %3d%%  (%s)" % (os.path.basename(dest), pct, human(done)))
                    sys.stdout.flush()
    sys.stdout.write("\r    %s  done  (%s)%s\n" % (os.path.basename(dest), human(done), " " * 12))
    if expect_mb and abs(done / 1e6 - expect_mb) > max(8, expect_mb * 0.05):
        print("    warning: expected about %d MB, got %s -- keeping it anyway" % (expect_mb, human(done)))
    os.replace(part, dest)
    return dest


def unzip_flat(zip_path, dest_dir):
    """extract a llama.cpp release zip: its files sit at the archive root"""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            target = os.path.join(dest_dir, os.path.basename(n))
            with z.open(n) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return dest_dir


def github_assets(repo, build=""):
    """-> (tag, {asset name: url}) for a build tag, or the newest release"""
    api = "https://api.github.com/repos/%s/releases" % repo
    api += ("/tags/" + build) if build else "?per_page=5"
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=30) as f:
        d = json.loads(f.read().decode("utf-8"))
    if isinstance(d, list):                     # newest release that actually ships binaries
        d = next((r for r in d if len(r.get("assets") or []) > 3), d[0])
    return d.get("tag_name"), {a["name"]: a["browser_download_url"] for a in (d.get("assets") or [])}


def install_llama_cpp(m, which="auto", latest=False):
    spec = m.get("llama_cpp") or {}
    builds = spec.get("builds") or {}
    if which in ("auto", "", None):
        which = have_gpu()
        print("  detected: %s" % (builds.get(which, {}).get("label") or which))
    if which not in builds:
        print("  no such build: %s (have: %s)" % (which, ", ".join(builds)))
        return False
    tag, assets = github_assets(spec.get("github_repo") or "ggml-org/llama.cpp",
                                "" if latest else (spec.get("pinned") or ""))
    b = builds[which]
    names = [b["asset"].replace("{build}", tag or "")]
    if b.get("runtime"):
        names.append(b["runtime"])
    tmp = tempfile.mkdtemp(prefix="ps-llama-")
    try:
        for i, name in enumerate(names):
            url = assets.get(name)
            if not url:                          # the tag names its own files
                cand = [n for n in assets if name.split("-bin-")[-1] in n and n.startswith(name.split("-")[0])]
                url = assets.get(cand[0]) if cand else None
                name = cand[0] if cand else name
            if not url:
                print("  not in release %s: %s" % (tag, name))
                return False
            z = download(url, os.path.join(tmp, name))
            dest = os.path.join(llm_dir(), "llama.cpp", "backend" if i == 0 else
                                os.path.join("vendor", "cudart"))
            unzip_flat(z, dest)
            print("    -> %s" % os.path.relpath(dest, ROOT))
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_config(**kw):
    p = _paths_data("llm_config.json")
    try:
        cfg = json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        cfg = {}
    cfg.update({k: v for k, v in kw.items() if v is not None})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("  config: " + ", ".join("%s = %s" % (k, v) for k, v in kw.items() if v is not None))


def install_model(m, mid):
    ent = next((x for x in (m.get("models") or []) if x.get("id") == mid), None)
    if not ent:
        print("  no such model: %s" % mid)
        return False
    dest = os.path.join(llm_dir(), ent["file"])
    if os.path.exists(dest):
        print("  already here: %s" % ent["file"])
    else:
        download(ent["url"], dest, ent.get("size_mb", 0))
    write_config(main=ent["file"], ctx=ent.get("ctx", 8192),
                 model_id=ent.get("model_id") or mid, thinking=bool(ent.get("thinking")))
    return True


def install_embed(m):
    ent = m.get("embed") or {}
    if not ent.get("url"):
        return False
    dest = os.path.join(llm_dir(), ent["file"])
    if os.path.exists(dest):
        print("  already here: %s" % ent["file"])
    else:
        download(ent["url"], dest, ent.get("size_mb", 0))
    write_config(embed=ent["file"])
    print("  note: semantic style matching also needs the embeddings pack "
          "(README, 'Semantic style matching')")
    return True


def install_pack(m):
    """the artist embeddings pack: a zip that unpacks into data/embeddings/"""
    ent = m.get("embeddings_pack") or {}
    target = os.path.join(ROOT, *(ent.get("installs") or "data/embeddings/artist_embeddings.json").split("/"))
    if os.path.exists(target):
        print("  already here: %s" % os.path.basename(target))
        return True
    if not (ent.get("url") or "").startswith("http"):
        print("  not published yet: download the embeddings pack from the studio's release page "
              "and unzip it here so %s exists" % os.path.relpath(target, ROOT))
        return False
    tmp = tempfile.mkdtemp(prefix="ps-pack-")
    try:
        z = download(ent["url"], os.path.join(tmp, ent.get("file") or "pack.zip"), ent.get("size_mb", 0))
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith("/") or "artist_embeddings" not in n:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(n) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        print("    -> %s" % os.path.relpath(target, ROOT))
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def status(m):
    L = llm_dir()
    exe = os.path.join(L, "llama.cpp", "backend", "llama-server.exe")
    print("llama.cpp:      %s" % ("installed" if os.path.exists(exe) else "missing"))
    emb = (m.get("embed") or {}).get("file") or ""
    print("embed model:    %s" % ("installed" if emb and os.path.exists(os.path.join(L, emb)) else "missing"))
    print("language models:")
    for e in (m.get("models") or []):
        here = os.path.exists(os.path.join(L, e["file"]))
        print("  %-10s %-64s %s" % (e["id"], e["label"], "installed" if here else human(e.get("size_mb", 0) * 1e6)))
    vec = os.path.join(ROOT, "data", "embeddings", "artist_embeddings.json")
    print("embeddings pack: %s" % ("installed" if os.path.exists(vec) else "not installed (optional)"))


def ask(m):
    status(m)
    print("\nWhat should I install? (nothing is uploaded; downloads resume if interrupted)")
    opts = [("llama.cpp for this machine", "llama")] + \
           [(e["label"], "model:" + e["id"]) for e in (m.get("models") or [])] + \
           [((m.get("embed") or {}).get("label") or "embedding model", "embed")]
    if (m.get("embeddings_pack") or {}).get("url"):
        opts.append((m["embeddings_pack"].get("label") or "artist embeddings pack", "pack"))
    for i, (label, _k) in enumerate(opts, 1):
        print("  %d) %s" % (i, label))
    print("  a) all of the above      q) quit")
    pick = input("choice: ").strip().lower()
    if pick in ("q", ""):
        return []
    if pick == "a":
        return [k for _l, k in opts]
    out = []
    for part in pick.replace(",", " ").split():
        if part.isdigit() and 1 <= int(part) <= len(opts):
            out.append(opts[int(part) - 1][1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--pack", action="store_true", help="the artist embeddings pack")
    ap.add_argument("--llama-cpp", dest="llama", default="", choices=["", "auto", "cuda", "vulkan", "cpu"])
    ap.add_argument("--latest", action="store_true", help="newest llama.cpp build instead of the tested one")
    args = ap.parse_args()
    m = manifest(args.manifest)
    if args.list:
        status(m)
        return 0
    jobs = []
    if args.all:
        jobs = ["llama", "model:" + ((m.get("models") or [{}])[0].get("id") or ""), "embed"]
        if (m.get("embeddings_pack") or {}).get("url"):
            jobs.append("pack")
    else:
        if args.llama:
            jobs.append("llama")
        if args.model:
            jobs.append("model:" + args.model)
        if args.embed:
            jobs.append("embed")
        if args.pack:
            jobs.append("pack")
    if not jobs:
        jobs = ask(m)
    for j in jobs:
        if j == "llama":
            print("llama.cpp:")
            install_llama_cpp(m, args.llama or "auto", args.latest)
        elif j.startswith("model:"):
            print("language model:")
            install_model(m, j.split(":", 1)[1])
        elif j == "embed":
            print("embedding model:")
            install_embed(m)
        elif j == "pack":
            print("artist embeddings pack:")
            install_pack(m)
    print("\nDone. Start the studio (run_studio.cmd) -- the engine line under the "
          "title should say the built-in engine is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
