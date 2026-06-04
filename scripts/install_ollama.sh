#!/usr/bin/env bash
set -uo pipefail

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
