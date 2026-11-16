#!/bin/bash
# 사용법: src/render.sh <이름> <가로>   →  ../<이름>.png (2배 해상도, 오른쪽·아래 여백 정리)
# 예: src/render.sh 1_basic 1240   (macOS, Google Chrome 필요)
set -e
cd "$(dirname "$0")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=2 --window-size="$2,${3:-1400}" --screenshot="../$1.png" "file://$PWD/$1.html" 2>/dev/null
"${PYTHON:-python}" trim.py "../$1.png" 56
