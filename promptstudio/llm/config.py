"""
config.py -- which local model the generator runs on, as a setting.

Everything here used to be module constants in server.py and bridge.py, so
changing quant or model meant editing Python. It is now:

    data/llm_config.json          (checked in to data/, edit freely)
    PROMPTSTUDIO_LLM_MAIN=...     (environment wins over the file)

Nothing is required. With no config file and no environment, the defaults
are exactly the previous constants, so an untouched install behaves
identically.

    {
      "main":     "heretic/qwen3-8b-heretic/qwen3-8b-heretic-Q5_K_M.gguf",
      "ctx":      8192,
      "model_id": "qwen3-8b-heretic",
      "thinking": true
    }

Paths are relative to the models directory (paths.models_dir()), or
absolute.

WHY `thinking` EXISTS. Qwen3 needs ` /no_think` appended to the system
prompt to skip its reasoning block, and the reply has to be split on
`</think>` defensively. Both are meaningless for other families -- harmless,
but they put stray tokens in every system prompt. Set false for a
non-thinking model.
"""

import json
import os

from promptstudio import paths as _paths

DEFAULTS = {
    "main": "heretic/qwen3-8b-heretic/qwen3-8b-heretic-Q5_K_M.gguf",
    "ctx": 8192,
    "model_id": "qwen3-8b-heretic",
    "thinking": True,
    # the defaults are a release's: no embed model, no gathering, no
    # embedding matcher (the workshop's llm_config.json turns them on)
    "embed": None,
    # the concept ledger: log the concepts no tag mapped, for review (a
    # development aid; nothing is gathered when false)
    "ledger": False,
}

_ENV = {
    "main": "PROMPTSTUDIO_LLM_MAIN",
    "ctx": "PROMPTSTUDIO_LLM_CTX",
    "model_id": "PROMPTSTUDIO_LLM_MODEL_ID",
}

_CFG = None


def _resolve(rel):
    """a model path: absolute as given, otherwise under the models dir"""
    if not rel:
        return None
    if os.path.isabs(rel):
        return rel
    return os.path.join(_paths.models_dir(), *rel.replace("\\", "/").split("/"))


def load(reload=False):
    """-> the effective config, environment over file over defaults"""
    global _CFG
    if _CFG is not None and not reload:
        return _CFG
    cfg = dict(DEFAULTS)
    try:
        with open(_paths.data("llm_config.json"), encoding="utf-8") as f:
            for k, v in (json.load(f) or {}).items():
                if not k.startswith("_"):
                    cfg[k] = v
    except Exception:
        pass                       # no file is the normal case
    for k, env in _ENV.items():
        v = os.environ.get(env)
        if v:
            cfg[k] = int(v) if k == "ctx" else v
    cfg["main_path"] = _resolve(cfg.get("main"))
    cfg["embed_path"] = _resolve(cfg.get("embed"))
    _CFG = cfg
    return cfg


# ---------------------------------------------------------------- GGUF


def describe():
    """one line for the startup banner"""
    c = load()
    return "model %s | ctx %s" % (os.path.basename(c["main_path"] or "?"), c["ctx"])
