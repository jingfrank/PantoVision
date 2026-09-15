@echo off
chcp 65001 >nul
title 增量同步至 Jetson 设备

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_to_jetson.ps1" %*

echo.
echo 按任意键退出...
pause >nul
