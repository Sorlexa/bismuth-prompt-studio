#!/bin/sh
# Bismuth Prompt Studio -- local web UI on http://localhost:7801
cd "$(dirname "$0")"
exec python3 -m promptstudio.ui.studio "$@"
