[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ExpectedDocumentSha256,
    [Parameter(Mandatory=$true)][string]$SnapshotReceipt,
    [Parameter(Mandatory=$true)][string]$MigrationReceipt
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Get-Location).ProviderPath)
$document=Join-Path $root 'docs/AXON.md'
function Digest([byte[]]$bytes) { [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant() }
$raw=[IO.File]::ReadAllBytes($document)
if ((Digest $raw) -ne $ExpectedDocumentSha256) { throw 'Canonical document changed before migration' }
$receipt=Get-Content -LiteralPath $SnapshotReceipt -Raw | ConvertFrom-Json
if ($receipt.verified -ne $true) { throw 'Unverified historical snapshot' }
$archive=[IO.Path]::GetFullPath($receipt.archive,$root)
if (-not $archive.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside workspace' }
if ((Digest ([IO.File]::ReadAllBytes($archive))) -ne $receipt.sha256) { throw 'Historical archive changed' }
$zip=[IO.Compression.ZipFile]::OpenRead($archive)
try {
    $entry=$zip.GetEntry('docs/AXON.md')
    if ($null -eq $entry) { throw 'Missing canonical snapshot' }
    $input=$entry.Open();$buffer=[IO.MemoryStream]::new()
    try { $input.CopyTo($buffer);$saved=$buffer.ToArray() } finally { $input.Dispose();$buffer.Dispose() }
    if ((Digest $saved) -ne (Digest $raw)) { throw 'Historical snapshot is not byte-identical' }
} finally { $zip.Dispose() }
$encoding=[Text.UTF8Encoding]::new($false,$true)
$text=$encoding.GetString($raw)
$markers=[regex]::Matches($text,'(?m)^<a id="historico-h[0-9]{3}"></a>\r?$')
if ($markers.Count -ne 50) { throw 'Unexpected historical inventory boundaries' }
$prefix=$text.Substring(0,$markers[0].Index)
$historical=$text.Substring($markers[0].Index)
$newline="`r`n"
$stub=$newline+'### 18.1 Evidencia Historica Arquivada'+$newline+$newline+
    'Os 50 blocos integrais e a revisao 26 completa estao em '+$receipt.archive+'.'+$newline+
    'O arquivo e evidencia imutavel, nao uma segunda fonte de instrucoes atuais.'+$newline+
    'SHA-256 do documento integral: '+$ExpectedDocumentSha256+'.'+$newline+
    'SHA-256 dos blocos historicos: '+(Digest ($encoding.GetBytes($historical)))+'.'+$newline+
    'Os identificadores abaixo preservam os links do inventario; leia o snapshot para o texto original.'+$newline+$newline
foreach ($marker in $markers) { $stub+=$marker.Value.TrimEnd()+$newline }
$updated=$encoding.GetBytes($prefix+$stub)
if ($updated.Length -ge $raw.Length) { throw 'History migration did not reduce hot document size' }
if (Test-Path -LiteralPath $MigrationReceipt) { throw 'Migration receipt already exists' }
# Mechanical byte move: the prefix is copied exactly, and the removed suffix is
# verified in the immutable ZIP before the only canonical source is replaced.
if ((Digest ([IO.File]::ReadAllBytes($document))) -ne $ExpectedDocumentSha256) { throw 'Canonical document changed during migration' }
[IO.File]::WriteAllBytes($document,$updated)
$result=[ordered]@{schema='axon-canonical-history-migration/v1';old_bytes=$raw.Length;new_bytes=$updated.Length;old_sha256=(Digest $raw);new_sha256=(Digest $updated);unchanged_prefix_sha256=(Digest ($encoding.GetBytes($prefix)));historical_suffix_sha256=(Digest ($encoding.GetBytes($historical)));historical_anchors=$markers.Count;archive=$receipt.archive;archive_sha256=$receipt.sha256;archive_verified=$true;scope='Only historical bodies moved to immutable evidence. Current architecture, milestone contracts, results and inventory table copied byte-for-byte.'}
$json=$result|ConvertTo-Json -Depth 6
[IO.File]::WriteAllText([IO.Path]::GetFullPath($MigrationReceipt,$root),$json,$encoding)
$json
