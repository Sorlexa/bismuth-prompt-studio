"""
aliases.py -- danbooru's alias table, one owner (2026-09-17).

The table lived twice: data/tags/danbooru_aliases.json (a cache tag_bridge.py
downloaded on 25 August, read by the engine, the scanner and the studio) and
data/tags/alias_map.json (tools/build/build_alias_map.py, 28 August, read by
retrieval). Two copies of one fact drift -- they already differed by 125
rows. alias_map.json is the owner now; every reader comes through here, and
the refresh tool rebuilds it with the other booru tables.

    booru_aliases()  -> {antecedent: tag it resolves to}, spaces not
                        underscores, lower case
"""

import json
import os

from promptstudio import paths as _paths

_CACHE = {"mtime": None, "data": {}}


def booru_aliases():
    p = _paths.data("alias_map.json")
    try:
        mt = os.path.getmtime(p)
    except OSError:
        return {}
    if _CACHE["mtime"] != mt:
        try:
            with open(p, encoding="utf-8-sig") as f:
                raw = json.load(f)
            raw = raw.get("aliases") if isinstance(raw, dict) and "aliases" in raw else raw
            _CACHE["data"] = {str(k).replace("_", " ").lower().strip():
                              str(v).replace("_", " ").lower().strip()
                              for k, v in (raw or {}).items()}
        except Exception:
            _CACHE["data"] = {}
        _CACHE["mtime"] = mt
    return _CACHE["data"]
