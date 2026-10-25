#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Stop the web receiver and wait until its port is free.
PIDFILE=$DIR/logs/websdr.pid
[ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null
until ! ss -ltn | grep -q ":8490 "; do sleep 0.5; done
rm -f "$PIDFILE"
