#!/bin/bash
cd /root/songwang/rvc-infer || exit 1
export RVC_ONNX=0
export RVC_INCREMENTAL_HUBERT=0
while true; do
  echo "start rvc_server" >> rvc_server.log
  .venv/bin/python -u server/rvc_server.py --host 0.0.0.0 --port 8765 >> rvc_server.log 2>&1
  echo "rvc_server exit, restart in 2s" >> rvc_server.log
  sleep 2
done
