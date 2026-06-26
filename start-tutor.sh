#!/bin/bash
# English Tutor — 在 Sway + foot 中固定位置打开
# 用法: 直接运行，或绑定到 Sway 快捷键

TITLE="English Tutor"
WORKSPACE="1"          # 放在第一个工作区
WIDTH=900              # 像素宽
HEIGHT=600             # 像素高
POS_X=100              # 屏幕坐标 X
POS_Y=50               # 屏幕坐标 Y
APP_ID="english-tutor" # 用于 sway 规则匹配

cd /home/ubuntu/workspace/english-tutor || exit 1

# 切换到目标工作区并启动 foot
swaymsg workspace "$WORKSPACE"
foot --title="$TITLE" --app-id="$APP_ID" \
     --window-size-pixels="$WIDTH":"$HEIGHT" \
     python3 run.py &

# 等窗口出现后再定位
sleep 0.3
swaymsg "[app_id=\"$APP_ID\"]" move position "$POS_X" "$POS_Y"
