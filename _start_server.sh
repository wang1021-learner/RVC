#!/bin/bash
# 启动 RVC 推理服务器（可反复执行，自动重启）
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
pkill -f "rvc_server.py" 2>/dev/null
sleep 1

PY="$ROOT/venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$ROOT/.venv/bin/python"
fi
if [ ! -x "$PY" ]; then
  PY=python3
fi

mkdir -p "$ROOT/logs"
setsid nohup "$PY" -u server/rvc_server.py --host 0.0.0.0 --port 8765 >> "$ROOT/logs/rvc_server.log" 2>&1 < /dev/null &
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
