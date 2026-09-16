#!/usr/bin/env python
"""
update.py -- bring this copy of Bismuth Prompt Studio up to the latest build.

Run it from the studio folder (or double-click update.cmd):

    python tools/update.py            # show what changed, ask, then update
    python tools/update.py --check    # only show what would change
    python tools/update.py --yes      # update without asking

HOW IT DECIDES WHAT TO DOWNLOAD. Every build ships release_manifest.json: a
list of the files the build owns, each with a fingerprint (sha256). The
updater reads the newest manifest from GitHub, fingerprints the files on
this disk, and downloads only the files whose fingerprints differ. Files a
new build no longer has are removed. Nothing else is touched: the language
models and llama.cpp in LLM/, the logs, the embeddings pack and anything you
added yourself are not in the manifest.

YOUR CHANGES ARE KEPT. A file you edited (its fingerprint matches neither
the build you have nor the new one) is never overwritten: the new version is
saved beside it as <name>.new, and the updater tells you which. Two files
are handled by name, as the manifest's "policies" say:

    keep    data/library/llm_config.json -- your language-model settings.
            Replaced only if you never changed it.
    merge   the booru count caches the studio fills while it runs. Your
            cached counts and the new build's are combined.

NOTHING IS HALF-DONE. Every file is downloaded into .update_tmp/ and its
fingerprint checked first; only when all of them arrived intact are they
moved into place, and the manifest is written last. A dropped connection
leaves the studio exactly as it was -- run the updater again.

Pure standard library. No account, no token, nothing uploaded.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import urllib.request

# the studio folder: this file ships as <studio>/tools/update.py and lives in
# the workshop as <studio>/tools/release/update.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(ROOT) == "tools":
    ROOT = os.path.dirname(ROOT)
MANIFEST = "release_manifest.json"
STAGE = ".update_tmp"
UA = {"User-Agent": "bismuth-prompt-studio-update/1.0"}


# ---------------------------------------------------------------- fingerprints
def fingerprint_bytes(data):
    """sha256 of the content with Windows line endings read as Unix ones: a
    git clone on Windows may have turned every LF into CRLF, and that is not
    a change anyone made"""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def fingerprint(path):
    try:
        with open(path, "rb") as f:
            return fingerprint_bytes(f.read())
    except OSError:
        return None


def build_manifest(root, meta):
    """-> the manifest for the files under `root` (used by the release
    builder; the updater itself only reads manifests)"""
    skip_dirs = {".git", "__pycache__", STAGE, "logs"}
    files = {}
    for dp, dns, fs in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in skip_dirs)
        for fn in sorted(fs):
            rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
            if rel == MANIFEST or fn.endswith((".pyc", ".new", ".part")):
                continue
            if rel.startswith("LLM/") and not fn.endswith(".txt"):
                continue                      # models and binaries are not the build's
            if rel == "data/embeddings/artist_embeddings.json":
                continue                      # the optional pack, installed separately
            p = os.path.join(dp, fn)
            files[rel] = {"sha256": fingerprint(p), "size": os.path.getsize(p)}
    out = dict(meta)
    out["files"] = files
    return out


# ---------------------------------------------------------------- the remote
def _get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


class Remote:
    """the newest build: a GitHub repository (pinned to one commit, so every
    file comes from the same build even while the CDN catches up), or a
    local folder holding a build (--source, for testing)"""

    def __init__(self, local_manifest, source=""):
        self.folder = source if source and os.path.isdir(source) else ""
        if self.folder:
            self.label = self.folder
            return
        repo = (source or local_manifest.get("repo") or "").strip("/")
        branch = local_manifest.get("branch") or "main"
        if not repo:
            raise SystemExit("this copy has no repository to update from "
                             "(release_manifest.json has no 'repo')")
        self.ref = branch
        try:
            info = json.loads(_get("https://api.github.com/repos/%s/commits/%s" % (repo, branch), 30))
            self.ref = info.get("sha") or branch
        except Exception:
            pass                               # rate-limited or offline API: the branch name will do
        self.base = "https://raw.githubusercontent.com/%s/%s/" % (repo, self.ref)
        self.label = "github.com/%s (%s)" % (repo, self.ref[:10])

    def read(self, rel):
        if self.folder:
            with open(os.path.join(self.folder, *rel.split("/")), "rb") as f:
                return f.read()
        return _get(self.base + urllib.request.quote(rel))

    def manifest(self):
        return json.loads(self.read(MANIFEST).decode("utf-8-sig"))


# ---------------------------------------------------------------- the plan
def plan_update(local, remote):
    """-> {"download": [rel], "add": [rel], "remove": [rel], "kept": [rel],
    "merge": [rel], "same": n} comparing the disk against both manifests"""
    lf = local.get("files") or {}
    rf = remote.get("files") or {}
    pol = dict(local.get("policies") or {})
    pol.update(remote.get("policies") or {})
    out = {"download": [], "add": [], "remove": [], "kept": [], "merge": [], "same": 0}
    for rel, rec in sorted(rf.items()):
        disk = fingerprint(os.path.join(ROOT, *rel.split("/")))
        want = rec.get("sha256")
        if disk == want:
            out["same"] += 1
            continue
        if disk is None:
            out["add"].append(rel)
            continue
        shipped = (lf.get(rel) or {}).get("sha256")
        if shipped == want:
            # THE NEW BUILD DID NOT CHANGE IT; this copy did (the language-
            # model settings setup wrote, a cache that grew). Nothing to do.
            out["same"] += 1
            continue
        if pol.get(rel) == "merge":
            out["merge"].append(rel)
            continue
        if disk == shipped:
            out["download"].append(rel)       # untouched here, changed upstream
        else:
            out["kept"].append(rel)           # edited here: never overwritten
    for rel, rec in sorted(lf.items()):
        if rel in rf:
            continue
        p = os.path.join(ROOT, *rel.split("/"))
        if os.path.exists(p) and fingerprint(p) == rec.get("sha256"):
            out["remove"].append(rel)
    return out


def _merge_json(mine, theirs):
    """a cache of counts: every key either side knows, the larger count where
    both do (a booru count only grows)"""
    try:
        a = json.loads(mine.decode("utf-8-sig"))
        b = json.loads(theirs.decode("utf-8-sig"))
    except Exception:
        return None
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    out = dict(b)
    for k, v in a.items():
        if k not in out:
            out[k] = v
        elif isinstance(v, (int, float)) and isinstance(out[k], (int, float)):
            out[k] = max(v, out[k])
        elif isinstance(v, dict) and isinstance(out[k], dict):
            merged = dict(out[k])
            merged.update({kk: vv for kk, vv in v.items() if kk not in merged})
            out[k] = merged
    return (json.dumps(out, ensure_ascii=False, indent=0) + "\n").encode("utf-8")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0


def studio_running(port=7801):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------- apply
def apply(plan, remote, remote_manifest):
    rf = remote_manifest.get("files") or {}
    stage = os.path.join(ROOT, STAGE)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    staged = []                                   # (rel, staged path, final path)
    todo = plan["download"] + plan["add"] + plan["kept"] + plan["merge"]
    for i, rel in enumerate(todo, 1):
        sys.stdout.write("\r  downloading %d/%d  %s%s" % (i, len(todo), rel[-60:], " " * 10))
        sys.stdout.flush()
        data = remote.read(rel)
        if fingerprint_bytes(data) != (rf.get(rel) or {}).get("sha256"):
            shutil.rmtree(stage, ignore_errors=True)
            raise SystemExit("\n  %s arrived damaged (fingerprint mismatch). Nothing was changed; "
                             "run the updater again." % rel)
        final = os.path.join(ROOT, *rel.split("/"))
        if rel in plan["kept"]:
            final += ".new"                       # beside your edited file
        elif rel in plan["merge"]:
            try:
                with open(final, "rb") as f:
                    merged = _merge_json(f.read(), data)
            except OSError:
                merged = None
            data = merged if merged is not None else data
        sp = os.path.join(stage, "%05d" % i)
        with open(sp, "wb") as f:
            f.write(data)
        staged.append((rel, sp, final))
    sys.stdout.write("\r  downloaded %d file(s)%s\n" % (len(todo), " " * 60))
    # everything is here and intact: now, and only now, the studio changes
    for rel, sp, final in staged:
        os.makedirs(os.path.dirname(final), exist_ok=True)
        os.replace(sp, final)
    for rel in plan["remove"]:
        try:
            os.remove(os.path.join(ROOT, *rel.split("/")))
        except OSError:
            pass
    with open(os.path.join(ROOT, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        json.dump(remote_manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    shutil.rmtree(stage, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="update Bismuth Prompt Studio to the latest build")
    ap.add_argument("--check", action="store_true", help="only show what would change")
    ap.add_argument("--yes", action="store_true", help="update without asking")
    ap.add_argument("--replace-edited", action="store_true",
                    help="replace edited files too (your language-model settings are still kept)")
    ap.add_argument("--source", default="",
                    help="update from this folder holding a build, or from OWNER/REPO on GitHub "
                         "(default: the repository this copy came from)")
    args = ap.parse_args()

    print("Bismuth Prompt Studio updater")
    try:
        with open(os.path.join(ROOT, MANIFEST), encoding="utf-8-sig") as f:
            local = json.load(f)
    except Exception:
        local = {"files": {}}
        print("  this copy has no %s (it predates the updater), so the updater cannot tell "
              "your edits from older files. Run it with --replace-edited to take the latest "
              "build's version of every file that differs (your language-model settings are "
              "kept either way)." % MANIFEST)
    if not args.source and not local.get("repo"):
        local["repo"] = "Sorlexa/bismuth-prompt-studio"

    print("  this copy : build %s" % (local.get("build") or "unknown"))
    try:
        remote = Remote(local, args.source)
        latest = remote.manifest()
    except SystemExit:
        raise
    except Exception as e:
        print("  could not reach the latest build: %s" % e)
        return 2
    print("  latest    : build %s  (%s)" % (latest.get("build") or "unknown", remote.label))

    plan = plan_update(local, latest)
    if args.replace_edited:
        _pol = dict(local.get("policies") or {})
        _pol.update(latest.get("policies") or {})
        keep = [r for r in plan["kept"] if _pol.get(r) == "keep"]
        plan["download"] += [r for r in plan["kept"] if r not in keep]
        plan["kept"] = keep
    rf = latest.get("files") or {}
    size = sum(int((rf.get(r) or {}).get("size") or 0)
               for r in plan["download"] + plan["add"] + plan["kept"] + plan["merge"])
    changes = sum(len(plan[k]) for k in ("download", "add", "remove", "kept", "merge"))
    if not changes:
        print("\n  Up to date: nothing in the latest build is newer than this copy "
              "(%d files checked)." % plan["same"])
        return 0
    print("\n  %d file(s) already match." % plan["same"])
    for key, title in (("download", "changed, will be replaced"), ("add", "new in this build"),
                       ("merge", "caches, will be combined with yours"),
                       ("remove", "no longer in the build, will be removed"),
                       ("kept", "EDITED HERE -- kept; the new version is saved beside it as .new")):
        if plan[key]:
            print("  %s (%d):" % (title, len(plan[key])))
            for rel in plan[key][:40]:
                print("    %s" % rel)
            if len(plan[key]) > 40:
                print("    ... and %d more" % (len(plan[key]) - 40))
    print("  download size: %s" % human(size))
    if args.check:
        return 0
    if studio_running():
        print("\n  The studio is running (port 7801). Close it first, then run the updater again.")
        return 3
    if not args.yes:
        try:
            if input("\n  Update now? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  Nothing was changed.")
                return 0
        except EOFError:
            print("  Nothing was changed (no answer; use --yes to update without asking).")
            return 0
    apply(plan, remote, latest)
    print("  Updated to build %s." % (latest.get("build") or "unknown"))
    if plan["kept"]:
        print("  Your edited files were kept. Compare each with its .new copy and keep what you want.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
