#!/bin/bash
# Start the web receiver: waterfall, active channels, live audio and transcripts in the browser.
# It uses the RSPdx, so stop the other programs that use it first.
#
# GPU transcription (optional): run tools/whisper_service.py on a computer with an NVIDIA GPU,
# open a shared SSH connection to it, and set GPU_SSH (and GPU_SSH_SOCKET if not the default):
#   ssh -M -S ~/.ssh/cm-gpu -o ControlPersist=12h user@gpu-host
#   GPU_SSH=user@gpu-host ./rspdx-websdr.sh
# Then the transcription goes to the GPU through an SSH port forward; if not, it runs on this CPU.
#
# Usage: ./rspdx-websdr.sh [--center 14.225e6] [--rate 384000] [--antenna "Antenna C"] [--lna 5]
#        [--model small] [--lang en] [--max-channels 8] [--no-transcribe] [--no-save-audio]
#        [--no-prompt] [--remote http://127.0.0.1:8491] [--port 8490] [--bind 127.0.0.1]
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/local.env" ] && . "$DIR/local.env"   # your own settings, for example GPU_SSH; not in git
CM=${GPU_SSH_SOCKET:-~/.ssh/cm-gpu}
REMOTE=()
if [ -n "$GPU_SSH" ] && [ -S "$CM" ] && ssh -S "$CM" -O check "$GPU_SSH" 2>/dev/null; then
    ssh -S "$CM" -O forward -L 8491:127.0.0.1:8491 "$GPU_SSH" 2>/dev/null
    curl -s -m 3 http://127.0.0.1:8491/health >/dev/null && REMOTE=(--remote http://127.0.0.1:8491)
fi
exec python3 -u "$DIR/websdr/server.py" "${REMOTE[@]}" "$@"
