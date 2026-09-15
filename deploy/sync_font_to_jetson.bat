@echo off
chcp 65001 >nul
title 推送 Windows 中文字体到 Jetson

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_font_to_jetson.ps1" %*

echo.
echo 按任意键退出...
pause >nul
