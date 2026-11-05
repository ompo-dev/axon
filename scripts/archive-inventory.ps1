[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$IndexDirectory,
    [Parameter(Mandatory = $true)][string]$DestinationPrefix,
    [Parameter(Mandatory = $true)][string]$Verification,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Get-Location).ProviderPath)
$rootPrefix = $root.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
function SafePath([string]$name) {
    $absolute = [IO.Path]::GetFullPath($name, $root)
    if (-not $absolute.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Inventory path outside workspace' }
    $part = $absolute
    while ($part.Length -gt $root.Length) {
        try { if ([IO.File]::GetAttributes($part) -band [IO.FileAttributes]::ReparsePoint) { throw 'Inventory path link' } }
        catch [IO.FileNotFoundException] {} catch [IO.DirectoryNotFoundException] {}
        $part = [IO.Path]::GetDirectoryName($part)
    }
    return $absolute
}
function Hash([string]$name) {
    $stream = [IO.File]::OpenRead((SafePath $name))
    try { return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant() } finally { $stream.Dispose() }
}
$verificationPath = SafePath $Verification
if ([IO.File]::Exists($verificationPath)) { throw 'Verification already exists' }
$indexPath = SafePath (Join-Path $IndexDirectory 'inventory.json')
$planPath = SafePath (Join-Path $IndexDirectory 'plan.json')
$index = Get-Content -Raw -LiteralPath $indexPath | ConvertFrom-Json
$plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
if ($index.schema -ne 'axon-byte-deduplicated-evidence/v1' -or $plan.schema -ne 'axon-evidence-shard-plan/v1' -or (Hash $indexPath) -ne $plan.index_sha256) { throw 'Inventory identity' }
if ($index.raw_files -gt 100000 -or $index.raw_bytes -gt 536870912 -or $index.files.Count -ne $index.raw_files) { throw 'Inventory limits/cardinality' }
$expected = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
$sourcePrefix = (SafePath $index.source).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
foreach ($record in $index.files) {
    foreach ($name in @($record.path, $record.content_path)) {
        if (-not (SafePath $name).StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Record outside declared source' }
    }
    if ($expected.ContainsKey($record.path)) { throw 'Duplicate logical path' }
    $expected[$record.path] = $record
}
$stored = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
$receipts = [Collections.Generic.List[object]]::new()
$metadata = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($shard in $plan.shards) {
    if ($shard.paths.Count -gt 4096) { throw 'Shard file quota' }
    $destination = "$DestinationPrefix-part-$($shard.index).zip"
    [void](SafePath $destination)
    foreach ($name in $shard.paths) {
        $absolute = SafePath $name
        if ($absolute -notin @($indexPath, $planPath) -and (-not $expected.ContainsKey($name) -or $expected[$name].content_path -ne $name)) { throw 'Undeclared shard input' }
    }
    if ($VerifyOnly) {
        $parsed = [pscustomobject]@{ archive = $destination; archive_bytes = [IO.FileInfo]::new((SafePath $destination)).Length; sha256 = Hash $destination; verified = $true }
    } else {
        $receipt = & (Join-Path $PSScriptRoot 'archive-evidence.ps1') -Destination $destination -Paths @($shard.paths)
        $parsed = $receipt | ConvertFrom-Json
    }
    if (-not $parsed.verified) { throw 'Unverified ZIP shard' }
    $receipts.Add($parsed)
    $zip = [IO.Compression.ZipFile]::OpenRead((SafePath $destination))
    try {
        $manifestEntry = $zip.GetEntry('manifest.json')
        if ($null -eq $manifestEntry -or $manifestEntry.Length -gt 67108864 -or $zip.Entries.Count -gt 4097) { throw 'ZIP manifest quota' }
        $stream = $manifestEntry.Open(); $reader = [IO.StreamReader]::new($stream)
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        if ($zip.Entries.Count -ne $manifest.files.Count + 1) { throw 'ZIP cardinality' }
        $sourceBytes = 0L
        foreach ($record in $manifest.files) {
            $sourceBytes += [long]$record.bytes
            if ($sourceBytes -gt 536870912) { throw 'ZIP source byte quota' }
            $entry = $zip.GetEntry($record.path)
            if ($null -eq $entry -or $entry.Length -ne $record.bytes -or $record.bytes -gt 67108864) { throw 'ZIP entry size' }
            $payload = $entry.Open()
            try { $digest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($payload)).ToLowerInvariant() } finally { $payload.Dispose() }
            if ($digest -ne $record.sha256) { throw 'ZIP entry bytes' }
            if ($expected.ContainsKey($record.path)) {
                $target = $expected[$record.path]
                if ($target.content_path -ne $record.path -or $target.sha256 -ne $record.sha256 -or $target.bytes -ne $record.bytes) { throw 'Representative ZIP identity' }
                $stored[$record.path] = $record
            } else {
                $absolute = SafePath $record.path
                if ($absolute -notin @($indexPath, $planPath) -or (Hash $absolute) -ne $record.sha256) { throw 'Unexpected ZIP content or metadata drift' }
                [void]$metadata.Add($absolute)
            }
        }
        if ($VerifyOnly) {
            $parsed | Add-Member -NotePropertyName files -NotePropertyValue $manifest.files.Count
            $parsed | Add-Member -NotePropertyName source_bytes -NotePropertyValue $sourceBytes
        }
    } finally { $zip.Dispose() }
}
if ($metadata.Count -ne 2) { throw 'Recoverability requires archived inventory and plan' }
$rawBytes = 0L
foreach ($record in $index.files) {
    if (-not $stored.ContainsKey($record.content_path)) { throw 'Missing representative; not recoverable' }
    $representative = $stored[$record.content_path]
    if ($representative.sha256 -ne $record.sha256 -or $representative.bytes -ne $record.bytes) { throw 'Alias content identity' }
    $file = [IO.FileInfo]::new((SafePath $record.path))
    if ($file.Length -ne $record.bytes -or (Hash $record.path) -ne $record.sha256) { throw 'Raw evidence changed after indexing' }
    $rawBytes += [long]$record.bytes
}
if ($rawBytes -ne $index.raw_bytes -or $stored.Count -ne $index.unique_files) { throw 'Byte/content cardinality mismatch' }
$result = [ordered]@{ schema = 'axon-deduplicated-evidence-verification/v1'; verified = $true; raw_files = $index.raw_files; raw_bytes = $rawBytes; unique_files = $stored.Count; index = $plan.index; index_sha256 = $plan.index_sha256; archives = @($receipts); scope = 'Every original path resolves to verified byte-identical ZIP content; raw aliases rehashed after capture. No source files deleted or modified. Snapshot byte recovery, not milestone approval or OS metadata backup.' }
$file = [IO.File]::Open($verificationPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $file.Write([Text.Encoding]::UTF8.GetBytes(($result | ConvertTo-Json -Depth 12))) } finally { $file.Dispose() }
$result | ConvertTo-Json -Depth 12
