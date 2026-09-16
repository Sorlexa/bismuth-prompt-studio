#!/usr/bin/env python
"""
llm_server.py -- the generator's OWN LLM engine, no third-party app.

the author's: "Can we use llm without the LM Studio?" -- yes: LM Studio is a
GUI around llama-server.exe, and that binary already lives on this disk
(its backend extension). This module launches it directly as a child
process pointed at the same GGUF.

Contract for llm_bridge.chat():
    ensure_up() -> the chat-completions URL, starting the engine if
    needed (~20s cold). Resolution order:
      1. our own llama-server on PORT (reuse if already up)
      2. spawn it (backend exe + main GGUF found)
      (there is no third option: the engine is ours)

IDLE AUTO-UNLOAD: a watchdog kills the child after IDLE_TTL seconds
without a generate, freeing the whole card for image generation -- the
manual 'lms unload' dance this session kept doing, automated. The next
generate pays the ~20s reload.
"""

import atexit
import glob
import json
import os
import subprocess
import threading
import time
import urllib.request
from promptstudio import paths as _paths
from promptstudio.llm import config as _config

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8090
BASE = "http://127.0.0.1:%d" % PORT
# LM STUDIO IS GONE (the author's 2026-09-02: "remove the lm fallback
# dependency entirely. llama.cpp files/dependencies should also be part
# of the project folder structure"). The engine is our own llama-server
# child on PORT, launched from binaries that live inside the project.
# With no second endpoint to fall back to, a failed start is reported as
# a failure instead of silently handing the work to someone else's server.
# WHICH MODEL IS A SETTING, NOT A CONSTANT (see promptstudio/llm/config.py):
# data/llm_config.json, or PROMPTSTUDIO_LLM_* in the environment. With
# neither present these resolve to exactly the previous hard-coded paths.
_CFG = _config.load()
MAIN_GGUF = _CFG["main_path"]
# the embedding model powers the style/artist matcher (the author's describe-
# the-style feature + the artist-fit fix). Tiny (Q4, ~80MB); served on
# its own port, spun up only for the offline fingerprint build and the
# occasional freeform 'describe the style' query -- the artist-random
# fix itself uses PRECOMPUTED vectors and needs no runtime embedding.
EMBED_PORT = 8091
EMBED_BASE = "http://127.0.0.1:%d" % EMBED_PORT
# copied into the project with the rest of the engine -- it was the last
# file still being read out of an LM Studio install
EMBED_GGUF = _CFG.get("embed_path") or ""   # llm_config.json 'embed'; none = no matcher
LOG = _paths.logs("llm_server.log")
IDLE_TTL = 600          # seconds without a generate before auto-unload
CREATE_NO_WINDOW = 0x08000000

_proc = None
_last_use = 0.0
_lock = threading.RLock()
_watchdog_on = False


def _backend():
    """the project's own llama-server + the vendor DLL dirs it loads from.

    The binaries live in LLM/llama.cpp inside the project: backend/ holds
    llama-server.exe with its ggml and llama DLLs, vendor/ the CUDA
    runtime it links against. Copied in rather than borrowed from an LM
    Studio install, so the generator owns its whole engine.
    """
    root = os.path.join(_paths.models_dir(), "llama.cpp")
    exe = os.path.join(root, "backend", "llama-server.exe")
    if os.path.exists(exe):
        vendors = sorted(glob.glob(os.path.join(root, "vendor", "*")))
        return exe, ([os.path.join(root, "backend")]
                     + [v for v in vendors if os.path.isdir(v)])
    return None, []


def _alive(url, timeout=2):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _spawn():
    """launch llama-server; full config first, minimal on a fast death"""
    global _proc
    exe, dll_dirs = _backend()
    if not exe or not os.path.exists(MAIN_GGUF):
        return False
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + env.get("PATH", "")
    # ctx 8192: a 3-subject bridge-1 call can brush past 4096
    common = [exe, "-m", MAIN_GGUF, "--host", "127.0.0.1",
              "--port", str(PORT), "-ngl", "99",
              "-c", str(_CFG.get("ctx") or 8192), "--no-ui"]
    for cmd in (common,):
        try:
            log = open(LOG, "ab")
            _proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env,
                                     creationflags=CREATE_NO_WINDOW,
                                     cwd=os.path.dirname(exe))
        except Exception:
            _proc = None
            continue
        t0 = time.time()
        while time.time() - t0 < 180:
            if _alive(BASE + "/health"):
                return True
            if _proc.poll() is not None:
                break               # died -- retry with the minimal cmd
            time.sleep(1.0)
        stop()
    return False


_embed_proc = None


def ensure_embed_up():
    """-> the embeddings URL; spawns the nomic embed server if needed."""
    global _embed_proc
    with _lock:
        if _alive(EMBED_BASE + "/health"):
            return EMBED_BASE + "/v1/embeddings"
        exe, dll_dirs = _backend()
        if not exe or not os.path.exists(EMBED_GGUF):
            raise RuntimeError("embed model not found: %s" % EMBED_GGUF)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + \
            env.get("PATH", "")
        cmd = [exe, "-m", EMBED_GGUF, "--host", "127.0.0.1",
               "--port", str(EMBED_PORT), "--embeddings", "-ngl", "99",
               "-c", "2048", "--no-ui"]
        log = open(LOG, "ab")
        _embed_proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                                       env=env,
                                       creationflags=CREATE_NO_WINDOW,
                                       cwd=os.path.dirname(exe))
        t0 = time.time()
        while time.time() - t0 < 120:
            if _alive(EMBED_BASE + "/health"):
                return EMBED_BASE + "/v1/embeddings"
            if _embed_proc.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError("embed server failed to start (see llm_server.log)")


def embed(texts, batch=64):
    """embed a list of strings -> list of vectors (lists of float)."""
    url = ensure_embed_up()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        body = json.dumps({"model": "nomic", "input": chunk}).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        out.extend(row["embedding"] for row in d["data"])
    return out


def stop_embed():
    global _embed_proc
    with _lock:
        if _embed_proc and _embed_proc.poll() is None:
            try:
                _embed_proc.terminate()
                _embed_proc.wait(timeout=8)
            except Exception:
                try:
                    _embed_proc.kill()
                except Exception:
                    pass
        _embed_proc = None


def _watchdog():
    while True:
        time.sleep(30)
        with _lock:
            if _proc and _proc.poll() is None and \
                    time.time() - _last_use > IDLE_TTL:
                stop()              # free the card for image generation


def ensure_up():
    """-> chat-completions URL; starts the engine when needed"""
    global _last_use, _watchdog_on
    with _lock:
        _last_use = time.time()
        if _alive(BASE + "/health"):
            return BASE + "/v1/chat/completions"
        if _spawn():
            if not _watchdog_on:
                threading.Thread(target=_watchdog, daemon=True).start()
                _watchdog_on = True
            return BASE + "/v1/chat/completions"
        raise RuntimeError(
            "no LLM engine: llama-server could not start. Check "
            "logs/llm_server.log, that LLM/llama.cpp/backend/"
            "llama-server.exe exists, and that the model named in "
            "llm_config.json is present under LLM/.")


def status():
    """'own' | 'ready-to-start' | ''"""
    if _alive(BASE + "/health"):
        return "own"
    exe, _ = _backend()
    if exe and os.path.exists(MAIN_GGUF):
        return "ready-to-start"
    return ""


def stop():
    global _proc
    with _lock:
        if _proc and _proc.poll() is None:
            try:
                _proc.terminate()
                _proc.wait(timeout=10)
            except Exception:
                try:
                    _proc.kill()
                except Exception:
                    pass
        _proc = None
        # an engine started by ANOTHER process (a CLI test run) is still
        # ours -- nothing else runs a standalone llama-server here; the
        # release button must free the card no matter who spawned it
        if _alive(BASE + "/health"):
            try:
                subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                               capture_output=True, timeout=15,
                               creationflags=CREATE_NO_WINDOW)
            except Exception:
                pass


atexit.register(stop)
atexit.register(stop_embed)


if __name__ == "__main__":
    print("backend:", _backend()[0])
    print("main gguf:", os.path.exists(MAIN_GGUF), MAIN_GGUF)
    print("status:", status())
    url = ensure_up()
    print("engine up at", url)
    body = json.dumps({"model": "any", "messages": [
        {"role": "system", "content": "You reply with one word. /no_think"},
        {"role": "user", "content": "Say READY"}],
        "max_tokens": 200}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    print("reply (%.1fs):" % (time.time() - t0),
          d["choices"][0]["message"]["content"].split("</think>")[-1].strip())
