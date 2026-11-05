#!/bin/sh
set -e
DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
exec python3 "$DIR/../scenarios/base/inference.py" "$@"
