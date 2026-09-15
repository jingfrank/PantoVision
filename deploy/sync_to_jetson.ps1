<#
.SYNOPSIS
    Windows 原生 Jetson 极速流式代码同步脚本
.DESCRIPTION
    1. 使用 Windows 自带 tar.exe 与 ssh.exe 管道流式传输，彻底绕过 Windows OpenSSH 9.5 的 scp SFTP 兼容性 bug。
    2. 全程仅需输入一次密码，无任何临时文件残留。
    3. 默认仅同步代码与配置 (~180 KB，瞬间完成)，自动跳过 models 目录。
    4. 若需同步模型权重，可追加 -IncludeModels 参数。
#>

param (
    [string]$JetsonIP = "172.10.1.5",
    [string]$JetsonUser = "hirain",
    [string]$TargetDir = "/home/hirain/Desktop/gongwang_clean",
    [switch]$IncludeModels
)

$ProjectDir = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🚀 开始同步工程至 Jetson 设备 (流式传输模式)" -ForegroundColor Green
Write-Host "   目标主机: ${JetsonUser}@${JetsonIP}:${TargetDir}" -ForegroundColor White
Write-Host "   本地目录: ${ProjectDir}" -ForegroundColor White
if ($IncludeModels) {
    Write-Host "   模型模式: [包含 models 权重 (~945 MB)] - 传输时间可能较长" -ForegroundColor Yellow
} else {
    Write-Host "   传输模式: [仅同步代码与配置 (~180 KB)] - 秒级完成 (已跳过 models)" -ForegroundColor Green
}
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. 连通性测试
Write-Host "📡 测试网络连通性 (${JetsonIP})..." -NoNewline
$pingOk = Test-Connection -ComputerName $JetsonIP -Count 1 -Quiet
if (-not $pingOk) {
    Write-Host " [失败]" -ForegroundColor Red
    Write-Host "❌ 无法 ping 通目标 IP: ${JetsonIP}" -ForegroundColor Red
    Write-Host "💡 提示: 请确认网线已插紧，本机已配置 172.10.1.x 同网段 IP。" -ForegroundColor Yellow
    exit 1
}
Write-Host " [在线]" -ForegroundColor Green

# 2. 准备排除项参数
$tarExcludes = "--exclude=.git --exclude=local_cache --exclude=events --exclude=__pycache__ --exclude=*.pyc --exclude=*.swp --exclude=*.tmp"
if (-not $IncludeModels) {
    $tarExcludes += " --exclude=models"
}

Write-Host "📦 正在流式打包并直接推送至 Jetson (只需输一次密码)..." -ForegroundColor Magenta

Push-Location $ProjectDir
try {
    $remoteCmd = "mkdir -p ${TargetDir} && tar -xzf - -C ${TargetDir}"
    $pipelineCmd = "tar.exe -czf - ${tarExcludes} . | ssh.exe -o StrictHostKeyChecking=accept-new ${JetsonUser}@${JetsonIP} `"${remoteCmd}`""
    
    cmd.exe /c $pipelineCmd
    $syncResult = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host "======================================================================" -ForegroundColor Cyan
if ($syncResult -eq 0) {
    Write-Host "✅ 同步成功！修改已在 Jetson 端实时生效。" -ForegroundColor Green
    if (-not $IncludeModels) {
        Write-Host "💡 提示: 若需同步 models 权重，请运行: .\deploy\sync_to_jetson.ps1 -IncludeModels" -ForegroundColor DarkGray
    }
} else {
    Write-Host "⚠️ 同步中断 (退出码: $syncResult)" -ForegroundColor Red
    Write-Host "💡 若提示 'No space left on device'，说明 Jetson 磁盘已满，可运行以下命令查看空间:" -ForegroundColor Yellow
    Write-Host "   ssh ${JetsonUser}@${JetsonIP} `"df -h`"" -ForegroundColor Yellow
}
Write-Host "======================================================================" -ForegroundColor Cyan
