param([int]$Port = 4317,[switch]$NoBuild,[switch]$NoUiBuild)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $NoBuild) {
    Push-Location $root
    try { & cargo build --release --features curriculum --bin axon --bin axon-observer --bin axon-uic-evolution-lab --bin axon-uic-reactive-lab --bin axon-uic-continuous-lab --bin axon-uic-seed-lab --bin axon-uic-controller-lab --bin axon-uic-genome-lab --bin axon-uic-language-lab --bin axon-uic-cumulative-lab --bin axon-uic-semantic-lab --bin axon-uic-nursery-lab --bin axon-uic-constructive-lab --bin axon-uic-curriculum-lab; if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' } }
    finally { Pop-Location }
}
$bun = (Get-Command bun -ErrorAction SilentlyContinue).Source
if (-not $bun) { $bun = Join-Path $env:USERPROFILE '.bun/bin/bun.exe' }
if (-not (Test-Path -LiteralPath $bun)) { throw 'Bun is required for the AXON workspace.' }
if (-not $NoUiBuild) {
    Push-Location (Join-Path $root 'ui')
    try {
        & $bun install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw 'UI dependency installation failed.' }
        & $bun run build
        if ($LASTEXITCODE -ne 0) { throw 'React workspace build failed.' }
    } finally { Pop-Location }
}
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Invalid port.' }
while ($Port -le 65535) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$Port)
    try { $probe.Start(); $probe.Stop(); break } catch { $probe.Stop(); $Port++ }
}
if ($Port -gt 65535) { throw 'No free port.' }
$logs = Join-Path $root '.axon-memory/observatory'
[System.IO.Directory]::CreateDirectory($logs) | Out-Null
$stamp = [guid]::NewGuid().ToString('N')
$stdout = Join-Path $logs "server-$Port-$stamp.stdout.log"
$stderr = Join-Path $logs "server-$Port-$stamp.stderr.log"
$server = Join-Path $root 'ui/server.mjs'
$process = Start-Process -FilePath $bun -ArgumentList @("`"$server`"",$Port) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$url = "http://127.0.0.1:$Port"
$ready = $false
for ($attempt=0; $attempt -lt 50; $attempt++) {
    if ($process.HasExited) { throw (Get-Content -LiteralPath $stderr -Raw) }
    try { $state = Invoke-RestMethod -Uri "$url/api/state" -TimeoutSec 2; $ready = $true; break } catch { Start-Sleep -Milliseconds 100 }
}
if (-not $ready) { Stop-Process -Id $process.Id; throw 'Observatory startup timed out.' }
[ordered]@{url=$url;pid=$process.Id;core_ready=$state.coreReady;stdout=$stdout;stderr=$stderr} | ConvertTo-Json
