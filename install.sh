#!/bin/bash
# =========================================================================
# Executable Entrypoint Wrapper for src/bin/install.sh
# =========================================================================
exec "$(dirname "$0")/src/bin/install.sh" "$@"
