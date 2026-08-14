#!/bin/bash
# 启动 RVC 推理服务器（可反复执行，自动重启）
pkill -f "rvc_server.py" 2>/dev/null
sleep 1
cd /home/songwang/Retrieval-based-Voice-Conversion-WebUI
setsid nohup venv/bin/python -u server/rvc_server.py --port 8765 >> ~/rvc_server.log 2>&1 < /dev/null &
sleep 1
echo "started pid: $!"
# 等待监听（最多 90 秒，HuBERT 加载需要时间）
for i in $(seq 1 45); do
  if ss -tln | grep -q 8765; then
    echo "LISTENING after ${i}x2s"
    exit 0
  fi
  sleep 2
done
echo "TIMEOUT - 未监听"
tail -5 ~/rvc_server.log
