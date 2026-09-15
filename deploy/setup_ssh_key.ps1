<#
.SYNOPSIS
    一键配置 Windows 到 Jetson 的 SSH 免密登录
#>
param (
    [string]$JetsonIP = "172.10.1.5",
    [string]$JetsonUser = "hirain"
)

$pubKeyPath = "$env:USERPROFILE\.ssh\id_ed25519.pub"

if (-not (Test-Path $pubKeyPath)) {
    Write-Host "🔑 正在生成本机 SSH 密钥对..." -ForegroundColor Cyan
    & ssh-keygen -t ed25519 -N '""' -f "$env:USERPROFILE\.ssh\id_ed25519"
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🚀 正在配置 SSH 免密登录至: ${JetsonUser}@${JetsonIP}" -ForegroundColor Green
Write-Host "👉 接下来请输入一次 Jetson 的登录密码 (输入时屏幕不会显示字符，输入完直接按回车):" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

$pubKey = (Get-Content $pubKeyPath -Raw).Trim()
$remoteCmd = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '${pubKey}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo 'SUCCESS_SSH_CONFIGURED'"

& ssh.exe -o StrictHostKeyChecking=accept-new "${JetsonUser}@${JetsonIP}" $remoteCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "🎉 免密配置成功！后续同步脚本将不再需要输入密码！" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
} else {
    Write-Host "❌ 配置失败，请核对密码是否正确。" -ForegroundColor Red
}
