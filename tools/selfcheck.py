#!/usr/bin/env python
"""selfcheck.py -- does this install work? Run from the project folder:

    python tools/selfcheck.py

Checks that every data file the engine asks for is present, generates one
prompt with the no-LLM engine, and reports the language-model setup."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from promptstudio import paths                                 # noqa: E402
from promptstudio.engine import bridge, enhancer as pe        # noqa: E402
from promptstudio.llm import config, server                   # noqa: E402

pe.load_all_banks()
misses = list(paths.MISSES) if hasattr(paths, "MISSES") else []
print("data files missing:", len(misses), misses[:10] if misses else "")
t0 = time.time()
r = bridge.generate("a girl reading in a library", "anima", "safe", {"fast": True}, seed=1)
print("fast engine: ok (%.1fs)" % (time.time() - t0))
print("  ", r["prompt"].split("\n")[0][:160])
cfg = config.load()
main = config._resolve(cfg.get("main"))
print("language model:", cfg.get("main"), "->", "found" if main and os.path.exists(main) else "NOT FOUND (fast engine only)")
print("llama-server:", "found" if (server._backend() or (None,))[0] else "NOT FOUND under LLM/llama.cpp (fast engine only)")
print("status:", server.status() or "no engine possible")
