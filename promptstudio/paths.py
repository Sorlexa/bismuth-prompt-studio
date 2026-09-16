"""Central path resolution for PromptStudio.

Data lives under data/<group>/, but code should never care WHICH group --
call data("style_pool.json"). The index is built once by scanning data/
recursively, so files can be regrouped without touching a single import.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
_INDEX = None


def _index():
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for dirpath, _dirs, files in os.walk(DATA_DIR):
            for f in files:
                _INDEX.setdefault(f, os.path.join(dirpath, f))
    return _INDEX


# Names a caller asked for that do not exist under data/. A miss is almost
# always a bug -- every loader in this codebase guards its read with
# os.path.exists() and silently falls back to an empty default, so a wrong
# path is indistinguishable from an empty file. That cost us a tag graph
# once (0 rows instead of 29,599, no error anywhere). Misses are recorded
# here and reported by the studio at startup; tools/dev/audit_paths.py
# fails on them.
MISSES = {}


def data(name, expect=True):
    """-> absolute path of a data file, wherever it sits under data/.

    Unknown names resolve into data/library/ so new files land somewhere
    sensible when written. Pass expect=False for a file that is legitimately
    optional or about to be created; anything else is recorded in MISSES."""
    hit = _index().get(name)
    if hit:
        return hit
    if expect:
        MISSES[name] = MISSES.get(name, 0) + 1
    return os.path.join(DATA_DIR, "library", name)


def refresh():
    """re-scan data/ (after a build tool writes a new file)"""
    global _INDEX
    _INDEX = None
    MISSES.clear()


# --- non-data locations -------------------------------------------------
# The GGUF models are large (~11 GB) and live outside the repo. Look in the
# project first, then fall back to the legacy folder, so they can be moved
# later without touching code.
_MODEL_CANDIDATES = [
    os.path.join(ROOT, "LLM"),
]


def models_dir():
    for c in _MODEL_CANDIDATES:
        if os.path.isdir(c):
            return c
    return _MODEL_CANDIDATES[0]


def model(*parts):
    """path to a model file inside the models dir"""
    return os.path.join(models_dir(), *parts)


def logs(name):
    d = os.path.join(ROOT, "logs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def tool(group, name):
    """path to a tool script, e.g. tool('harvest', 'gallery_collector.py')"""
    return os.path.join(ROOT, "tools", group, name)
