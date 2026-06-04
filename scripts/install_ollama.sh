#!/usr/bin/env bash
set -uo pipefail

MIN_VERSION="0.20.7"

INSTALLED="$(ollama --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)"
if [ -n "$INSTALLED" ] && [ "$(printf '%s\n' "$MIN_VERSION" "$INSTALLED" | sort -V | head -1)" = "$MIN_VERSION" ]; then
    echo "Ollama v${INSTALLED} satisfies minimum v${MIN_VERSION}. Nothing to do."
    exit 0
fi

echo "Attempting system-wide install via official script..."
if curl -fsSL https://ollama.com/install.sh | sh; then
    echo "Ollama installed successfully ..."
    exit 0
fi

echo "Official installer failed (likely missing permissions). Falling back to manual install in $HOME/ollama..."

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$TMPDIR"

wget -q --show-progress https://ollama.com/download/ollama-linux-amd64.tar.zst
mkdir -p "$HOME/ollama"
tar --use-compress-program=unzstd -xf ollama-linux-amd64.tar.zst -C "$HOME/ollama"

BASHRC="$HOME/.bashrc"
LINE_PATH='export PATH="$HOME/ollama/bin:$PATH"'
LINE_ENV='. "$HOME/.local/bin/env"'

touch "$BASHRC"
grep -qxF "$LINE_PATH" "$BASHRC" || echo "$LINE_PATH" >> "$BASHRC"
grep -qxF "$LINE_ENV"  "$BASHRC" || echo "$LINE_ENV"  >> "$BASHRC"

echo ""
echo "Ollama installed to \$HOME/ollama/bin"
echo "Run 'source ~/.bashrc' or open a new shell to pick up PATH changes."
