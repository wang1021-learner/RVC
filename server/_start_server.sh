#!/bin/bash
# 启动 RVC 推理服务器（可反复执行，自动重启）
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
pkill -f "rvc_server.py" 2>/dev/null
sleep 1

is_py311() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null
}

try_py() {
  if [ -x "$1" ] && is_py311 "$1"; then
    PY="$1"
    return 0
  fi
  return 1
}

PY=""
try_py "$ROOT/.venv/bin/python" || try_py "$ROOT/venv/bin/python" || true
if [ -z "$PY" ] && command -v python3.11 >/dev/null 2>&1; then
  if is_py311 python3.11; then PY=python3.11; fi
fi
if [ -z "$PY" ] && command -v python3 >/dev/null 2>&1; then
  if is_py311 python3; then PY=python3; fi
fi
if [ -z "$PY" ]; then
  echo "[ERROR] 需要 Python 3.11。请: python3.11 -m venv .venv"
  exit 1
fi
echo "using: $PY"

mkdir -p "$ROOT/logs"
setsid nohup "$PY" -u rvc_server.py --host 0.0.0.0 --port 8765 >> "$ROOT/logs/rvc_server.log" 2>&1 < /dev/null &
echo "started pid: $!"
for i in $(seq 1 45); do
  if command -v ss >/dev/null 2>&1 && ss -tln | grep -q 8765; then
    echo "LISTENING after ${i}x2s"
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT - 未监听"
tail -5 "$ROOT/logs/rvc_server.log"
