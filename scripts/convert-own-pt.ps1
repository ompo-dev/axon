param(
    [Parameter(Mandatory=$true)][string]$Xml,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $Xml).Path
$outputPath = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw 'Output already exists' }
if ((Get-Item -LiteralPath $source).Length -gt 67108864) { throw 'XML byte limit' }
$settings = [Xml.XmlReaderSettings]::new()
$settings.DtdProcessing = [Xml.DtdProcessing]::Ignore
$settings.XmlResolver = $null
$settings.MaxCharactersInDocument = 67108864
$reader = [Xml.XmlReader]::Create($source, $settings)
$document = [Xml.XmlDocument]::new()
$document.XmlResolver = $null
try { $document.Load($reader) } finally { $reader.Dispose() }
$lexicon = $document.LexicalResource.Lexicon
if ($lexicon.language -ne 'pt' -or $lexicon.id -ne 'own-pt' -or $lexicon.version -ne '1.0.0') { throw 'Expected OWN-PT 1.0.0 Portuguese lexicon' }
$rows = [Collections.Generic.List[object]]::new()
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$skipped = 0
$counts = @{ word = 0; definition = 0; 'is-a' = 0 }
function Add-Fact([string]$sense, [string]$relation, [string]$object) {
    if ([string]::IsNullOrWhiteSpace($object)) { return }
    $object = [string]::Join(' ', ($object -split '\s+' | Where-Object { $_ -ne '' }))
    $maximum = if ($relation -eq 'definition') { 4096 } else { 128 }
    if ([Text.Encoding]::UTF8.GetByteCount($object) -gt $maximum) { $script:skipped++; return }
    $identity = [string]::Join([char]0, @($sense, $relation, $object))
    if (!$seen.Add($identity)) { return }
    if ($rows.Count -ge 250000) { throw 'Dictionary fact limit' }
    $rows.Add([pscustomobject][ordered]@{ source = 'own-pt/1.0.0'; event = "fact-$($rows.Count)"; time = 1; sense = $sense; relation = $relation; object = $object })
    $counts[$relation]++
}
foreach ($entry in $lexicon.LexicalEntry) {
    foreach ($sense in $entry.Sense) { Add-Fact $sense.synset 'word' $entry.Lemma.writtenForm }
}
foreach ($synset in $lexicon.Synset) {
    $definitions = @($synset.SelectNodes('Definition') | ForEach-Object { $_.InnerText })
    if ($definitions.Count -gt 0) { Add-Fact $synset.id 'definition' ([string]::Join('; ', $definitions)) }
    foreach ($relation in $synset.SynsetRelation) {
        if ($relation.relType -eq 'hypernym' -or $relation.relType -eq 'instance_hypernym') {
            # Legacy 1.0.0 uses hypernymOf / hasInstance (parent -> child), not LMF target-parent semantics.
            $uri = $relation.GetAttribute('type', 'https://globalwordnet.github.io/schemas/dc/')
            $expected = if ($relation.relType -eq 'hypernym') { 'https://w3id.org/own/schema/hypernymOf' } else { 'https://w3id.org/own/schema/hasInstance' }
            if ($uri -ne $expected) { throw 'Unexpected legacy relation direction' }
            Add-Fact $relation.target 'is-a' $synset.id
        }
    }
}
$rows | Export-Csv -LiteralPath $outputPath -Delimiter "`t" -NoTypeInformation -Encoding utf8NoBOM -NoClobber
$manifest = [ordered]@{
    source = 'https://github.com/own-pt/openWordnet-PT/releases/tag/v1.0.0'
    download = 'https://github.com/own-pt/openWordnet-PT/releases/download/v1.0.0/own.tar.gz'
    authors = 'Alexandre Rademaker; Valeria de Paiva; Fredson Aguiar; OWN-PT contributors'
    license = 'CC-BY-4.0; inherited Princeton WordNet terms: see downloaded LICENSE'
    language = 'pt'; version = '1.0.0'; xmlSha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    factsSha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
    lexicalEntries = @($lexicon.LexicalEntry).Count; synsets = @($lexicon.Synset).Count
    exportedFacts = $rows.Count; relations = $counts; overlongSkipped = $skipped
    directionCorrection = 'Legacy 1.0.0 parent-to-child hypernymOf/hasInstance is inverted into child-is-a-parent; checked dc:type. See upstream fix_lmf.py.'
    note = 'Only declared senses, definitions and hypernym/instance_hypernym relations. No inferred relations from prose. Synonymy only through shared source synset. Other source relations remain in original XML.'
}
$json = $manifest | ConvertTo-Json -Depth 4
$manifestPath = "$outputPath.provenance.json"
$stream = [IO.File]::Open($manifestPath, [IO.FileMode]::CreateNew)
$writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
try { $writer.Write($json) } finally { $writer.Dispose() }
$json
