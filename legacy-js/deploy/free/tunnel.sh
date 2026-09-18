#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-3005}"
echo "🌐 [Cloudflare Tunnel] Launching instant public URL for localhost:${PORT}..."
if ! command -v cloudflared &> /dev/null; then
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    ARCH="$(uname -m)"
    if [ "$ARCH" = "x86_64" ]; then ARCH="amd64"; fi
    if [ "$ARCH" = "aarch64" ]; then ARCH="arm64"; fi
    TMP_BIN="/tmp/cloudflared"
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-${OS}-${ARCH}" -o "$TMP_BIN"
    chmod +x "$TMP_BIN"
    CLOUDFLARED_BIN="$TMP_BIN"
else
    CLOUDFLARED_BIN="cloudflared"
fi
"$CLOUDFLARED_BIN" tunnel --url "http://localhost:${PORT}"
