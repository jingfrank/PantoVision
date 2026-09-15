@echo off
chcp 65001 >nul
title 配置 Jetson SSH 免密登录

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_ssh_key.ps1" %*

echo.
echo 按任意键退出...
pause >nul
