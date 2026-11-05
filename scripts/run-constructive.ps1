param(
    [ValidateRange(1024,65535)][int]$Port=4319,
    [ValidateRange(100,10000)][int]$Count=10000,
    [ValidateRange(1,20)][int]$MaxBatches=4,
    [ValidateRange(1,2147483647)][int]$Seed=75218341,
    [ValidateSet('elitist','diverse','reinforced')][string]$Strategy='diverse',
    [ValidateSet('legacy','arithmetic')][string]$Curriculum='arithmetic'
)
$ErrorActionPreference='Stop'
$runs=@()
$outcome='batch-quota'
for ($batch=0; $batch -lt $MaxBatches; $batch++) {
    $raw = & (Join-Path $PSScriptRoot 'run-observatory-workflow.ps1') -Port $Port -Mode constructive -Count $Count -Seed $Seed -Strategy $Strategy -Curriculum $Curriculum | Out-String
    $run = $raw | ConvertFrom-Json
    $runs += [ordered]@{id=$run.id;new_graphs=$run.result.new_candidates;admitted=$run.result.goal_admitted;goals=$run.result.goal_tasks;passed=$run.result.all_goals_passed;elapsed_ms=$run.elapsed_ms}
    Write-Host ("Batch {0}: {1} new graphs; {2}/{3} goals admitted; final passed={4}" -f ($batch+1),$run.result.new_candidates,$run.result.goal_admitted,$run.result.goal_tasks,$run.result.all_goals_passed)
    if ($run.result.all_goals_admitted -eq $true) {
        $outcome = if ($run.result.all_goals_passed -eq $true) { 'goals-passed' } else { 'final-refuted' }
        break
    }
    if ($run.result.new_candidates -eq 0) { $outcome='resource-limit-or-no-new-candidates'; break }
}
[ordered]@{outcome=$outcome;strategy=$Strategy;curriculum=$Curriculum;runs=$runs;all_curriculum_goals_passed=($outcome -eq 'goals-passed');negative_control_required=$false;note='Same durable scope; final failures never select replacements. No guarantee of general mathematical understanding.'} | ConvertTo-Json -Depth 6
