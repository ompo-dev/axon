#Requires -Version 7.0
param(
    [string]$Binary = "target/release/axon.exe",
    [string]$Skill = ".axon-memory/text-learning-demo/contacts-v1.axtext",
    [ValidateRange(1, 100000)][int]$Rows = 20000,
    [string]$OutputRoot = "target/text-workflow-validation",
    [string]$Report = "target/text-workflow-validation.json"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
function Resolve-TaskPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $root $Value))
}
$exe = Resolve-TaskPath $Binary
$skillPath = Resolve-TaskPath $Skill
$output = Resolve-TaskPath $OutputRoot
$reportPath = Resolve-TaskPath $Report
foreach ($path in @($output, $reportPath)) { if (Test-Path -LiteralPath $path) { throw "Existing output is preserved: $path" } }
foreach ($path in @($exe, $skillPath)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" } }
[IO.Directory]::CreateDirectory($output) | Out-Null
[IO.Directory]::CreateDirectory((Split-Path -Parent $reportPath)) | Out-Null
$reserved = [IO.File]::Open($reportPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    $inputPath = Join-Path $output "input.csv"
    $resultPath = Join-Path $output "output.csv"
    0..($Rows - 1) | ForEach-Object {
        [PSCustomObject]@{ contact = "Name$_ Person|Family$_ Branch$($_ % 17)"; note = "record-$_" }
    } | Export-Csv -LiteralPath $inputPath -NoTypeInformation -Encoding utf8
    $inputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash
    $skillHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $skillPath).Hash
    function Invoke-Measured([string[]]$Arguments) {
        $start = [Diagnostics.ProcessStartInfo]::new($exe)
        $start.UseShellExecute = $false; $start.CreateNoWindow = $true
        $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
        $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
        foreach ($arg in $Arguments) { $start.ArgumentList.Add($arg) }
        $process = [Diagnostics.Process]::new(); $process.StartInfo = $start
        $timer = [Diagnostics.Stopwatch]::StartNew(); $peak = 0L; $started = $false
        try {
            $process.Start() | Out-Null; $started = $true
            $stdout = $process.StandardOutput.ReadToEndAsync(); $stderr = $process.StandardError.ReadToEndAsync()
            while (-not $process.WaitForExit(10)) {
                $process.Refresh()
                try { $peak = [Math]::Max($peak, $process.PeakWorkingSet64) } catch [InvalidOperationException] { }
            }
            $timer.Stop(); $text = $stdout.GetAwaiter().GetResult(); $errors = $stderr.GetAwaiter().GetResult()
            if ($process.ExitCode -ne 0) { throw "AXON failed ($($process.ExitCode)): $errors" }
            return [PSCustomObject]@{ elapsed_ms = $timer.Elapsed.TotalMilliseconds; sampled_peak_working_set_bytes = $peak; stdout = $text }
        } finally {
            if ($started -and -not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
            $process.Dispose()
        }
    }
    $common = @($skillPath, $inputPath, "--column", "contact", "--result-column", "normalized", "--context", "demo/contacts")
    $preview = Invoke-Measured (@("text-preview") + $common + @("--limit", "3"))
    $token = ($preview.stdout -split '\r?\n' | Where-Object { $_ -like "review_token=*" }) -replace '^review_token=', ''
    if (@($token).Count -ne 1 -or $preview.stdout -notmatch "(?m)^agreed=$Rows\r?$" ) { throw "Preview was not fully applicable" }
    # The bulk synthetic fixture has an independent, explicit expected transformation.
    $application = Invoke-Measured (@("text-apply") + $common + @("--output", $resultPath, "--confirm", $token))
    $verified = 0
    Import-Csv -LiteralPath $resultPath | ForEach-Object {
        $parts = $_.contact.Split('|')
        if ($_.normalized -cne ($parts[1] + ", " + $parts[0])) { throw "Incorrect output at row $verified" }
        if ($_.note -cne "record-$verified") { throw "Original field changed" }
        $verified++
    }
    if ($verified -ne $Rows) { throw "Incorrect row count" }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash -ne $inputHash -or (Get-FileHash -Algorithm SHA256 -LiteralPath $skillPath).Hash -ne $skillHash) { throw "Input or learned skill changed" }
    $json = [ordered]@{
        protocol = "axon-text-workflow-v1"
        recorded_utc = [DateTime]::UtcNow.ToString("o")
        synthetic_fixture = $true
        verified_rows = $verified
        input_bytes = (Get-Item -LiteralPath $inputPath).Length
        output_bytes = (Get-Item -LiteralPath $resultPath).Length
        input_sha256 = $inputHash
        output_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash
        skill_sha256 = $skillHash
        skill_bytes = (Get-Item -LiteralPath $skillPath).Length
        executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash
        executable_bytes = (Get-Item -LiteralPath $exe).Length
        preview = $preview
        application = $application
    } | ConvertTo-Json -Depth 5
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json)
    $reserved.Write($bytes, 0, $bytes.Length); $reserved.Flush($true)
    $json
} finally { $reserved.Dispose() }
