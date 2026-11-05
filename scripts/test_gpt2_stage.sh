#!/bin/sh
# Downloads GPT-2's real weights once, then tests the "Run GPT-2" stage (zr7):
#
#   1. The Go oracle must produce GPT-2's six ids for "The capital of France is".
#   2. The tester must pass the reference solution on stage zr7.
#
# Usage: scripts/test_gpt2_stage.sh [repository dir]
#
# The repository dir defaults to the reference solution in
# internal/test_helpers/pass_all. Pass your own solution's directory to test it
# instead. Set GPT2_WEIGHTS_DIR to keep the 548 MB file somewhere else.
set -eu

cd "$(dirname "$0")/.."

WEIGHTS_URL="https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
WEIGHTS_SHA256="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
WEIGHTS_DIR="${GPT2_WEIGHTS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/codecrafters/gpt2}"
WEIGHTS="$WEIGHTS_DIR/model.safetensors"
REPOSITORY_DIR="$(cd "${1:-internal/test_helpers/pass_all}" && pwd)"

sha256() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | cut -d' ' -f1
	else
		shasum -a 256 "$1" | cut -d' ' -f1
	fi
}

if [ ! -f "$WEIGHTS" ] || [ "$(sha256 "$WEIGHTS")" != "$WEIGHTS_SHA256" ]; then
	echo "Downloading GPT-2's weights (548 MB) to $WEIGHTS"
	mkdir -p "$WEIGHTS_DIR"
	curl -fL --progress-bar -o "$WEIGHTS.partial" "$WEIGHTS_URL"
	if [ "$(sha256 "$WEIGHTS.partial")" != "$WEIGHTS_SHA256" ]; then
		rm -f "$WEIGHTS.partial"
		echo "The downloaded file's SHA-256 doesn't match. Try again." >&2
		exit 1
	fi
	mv "$WEIGHTS.partial" "$WEIGHTS"
fi
echo "Using GPT-2's weights at $WEIGHTS"

echo
echo "== Checking the oracle on GPT-2's real weights"
GPT2_WEIGHTS="$WEIGHTS" go test ./internal/gpt2/ -run TestRealGPT2Greedy -count=1 -v

echo
echo "== Running stage zr7 against $REPOSITORY_DIR"
go build -o dist/main.out ./cmd/tester
CODECRAFTERS_GPT2_WEIGHTS_PATH="$WEIGHTS" \
	CODECRAFTERS_REPOSITORY_DIR="$REPOSITORY_DIR" \
	CODECRAFTERS_TEST_CASES_JSON='[{"slug":"zr7","tester_log_prefix":"stage-9","title":"Run GPT-2"}]' \
	dist/main.out
