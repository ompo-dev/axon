[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][string[]]$Paths
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Get-Location).ProviderPath)
$prefix = $root.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
function WorkspacePath([string]$value) {
    $absolute = [IO.Path]::GetFullPath($value, $root)
    if (-not $absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Evidence path must stay inside the workspace: $value"
    }
    $cursor = $absolute
    while ($cursor.Length -gt $root.Length) {
        if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -Force -LiteralPath $cursor).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Evidence cannot traverse a reparse point: $value"
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}
function Digest([byte[]]$bytes) {
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}
$destinationPath = WorkspacePath $Destination
if (Test-Path -LiteralPath $destinationPath) { throw 'Refusing to replace an existing evidence archive' }
$files = [Collections.Generic.SortedDictionary[string, object]]::new([StringComparer]::Ordinal)
foreach ($item in $Paths) {
    $resolved = WorkspacePath $item
    $entry = Get-Item -Force -LiteralPath $resolved
    $children = if ($entry.PSIsContainer) { Get-ChildItem -Force -Recurse -File -LiteralPath $resolved } else { @($entry) }
    foreach ($file in $children) {
        $full = WorkspacePath $file.FullName
        if ($full -eq $destinationPath) { throw 'Archive cannot include itself' }
        if ($file.Length -gt 67108864) { throw 'Single evidence-file quota' }
        $relative = [IO.Path]::GetRelativePath($root, $full).Replace('\', '/')
        $files[$relative] = [ordered]@{ path = $relative; bytes = $file.Length; sha256 = Digest ([IO.File]::ReadAllBytes($full)) }
        if ($files.Count -gt 4096) { throw 'Evidence file-count quota' }
    }
}
if ($files.Count -eq 0 -or $files.Count -gt 4096) { throw 'Evidence file-count quota' }
$total = 0L
foreach ($record in $files.Values) { $total += [long]$record.bytes }
if ($total -gt 536870912) { throw 'Evidence byte quota' }
$manifest = [ordered]@{ schema = 'axon-evidence-archive/v1'; files = @($files.Values); scope = 'Immutable file snapshot with byte/hash verification. Evidence retention is not milestone approval.' }
$stream = [IO.File]::Open($destinationPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
$zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($record in $files.Values) {
        $bytes = [IO.File]::ReadAllBytes((WorkspacePath $record.path))
        if ($bytes.Length -ne $record.bytes -or (Digest $bytes) -ne $record.sha256) { throw "Evidence changed during capture: $($record.path)" }
        $entryStream = $zip.CreateEntry($record.path, [IO.Compression.CompressionLevel]::Optimal).Open()
        try { $entryStream.Write($bytes) } finally { $entryStream.Dispose() }
    }
    $entryStream = $zip.CreateEntry('manifest.json').Open()
    try { $entryStream.Write([Text.Encoding]::UTF8.GetBytes(($manifest | ConvertTo-Json -Depth 8))) } finally { $entryStream.Dispose() }
} finally { $zip.Dispose(); $stream.Dispose() }
$zip = [IO.Compression.ZipFile]::OpenRead($destinationPath)
try {
    if ($zip.Entries.Count -ne $files.Count + 1) { throw 'Archive entry count mismatch' }
    $manifestStream = $zip.GetEntry('manifest.json').Open()
    $reader = [IO.StreamReader]::new($manifestStream)
    try { $stored = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    if ($stored.files.Count -ne $files.Count) { throw 'Archived manifest cardinality' }
    foreach ($record in $stored.files) {
        $expected = $files[$record.path]
        if ($null -eq $expected -or $expected.sha256 -ne $record.sha256 -or $expected.bytes -ne $record.bytes) { throw 'Archived manifest identity' }
        $entry = $zip.GetEntry($record.path)
        if ($null -eq $entry -or $entry.Length -ne $record.bytes) { throw "Archive missing or resized entry: $($record.path)" }
        $entryStream = $entry.Open(); $buffer = [IO.MemoryStream]::new()
        try {
            $entryStream.CopyTo($buffer)
            if ((Digest $buffer.ToArray()) -ne $record.sha256) { throw "Archive digest mismatch: $($record.path)" }
        } finally { $entryStream.Dispose(); $buffer.Dispose() }
    }
} finally { $zip.Dispose() }
[ordered]@{ archive = $Destination; files = $files.Count; source_bytes = $total; archive_bytes = (Get-Item -LiteralPath $destinationPath).Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationPath).Hash.ToLowerInvariant(); verified = $true } | ConvertTo-Json
