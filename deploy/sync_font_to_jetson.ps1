<#
.SYNOPSIS
    一键将 Windows 自带中文字体 (黑体 SimHei) 推送至 Jetson 离线字体库
#>
param (
    [string]$JetsonIP = "172.10.1.5",
    [string]$JetsonUser = "hirain"
)

$fontPath = "C:\Windows\Fonts\simhei.ttf"

if (-not (Test-Path $fontPath)) {
    Write-Host "❌ 本地未找到 $fontPath" -ForegroundColor Red
    exit 1
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🔤 正在将 Windows 黑体 (simhei.ttf) 推送到 Jetson (${JetsonIP})..." -ForegroundColor Green
Write-Host "   (适用于 Jetson 无法访问外网安装 fonts-wqy-zenhei 的离线场景)" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan

$remoteCmd = "mkdir -p ~/.local/share/fonts && tar -xzf - -C ~/.local/share/fonts && fc-cache -f ~/.local/share/fonts"
$pipelineCmd = "tar.exe -czf - -C C:\Windows\Fonts simhei.ttf | ssh.exe -o StrictHostKeyChecking=accept-new ${JetsonUser}@${JetsonIP} `"${remoteCmd}`""

cmd.exe /c $pipelineCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "✅ 中文字体推送成功！Jetson 已支持完整的中文 OSD 水印与报警渲染。" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
} else {
    Write-Host "❌ 推送失败，请检查连接与密码。" -ForegroundColor Red
}
