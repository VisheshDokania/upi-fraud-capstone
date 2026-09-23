[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('test', 'tabular', 'graph', 'fusion', 'nlp', 'api', 'dashboard')]
    [string]$Task
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found: $Python"
}

Push-Location $ProjectRoot
try {
    switch ($Task) {
        'test'      { & $Python -m pytest -q }
        'tabular'   { & $Python scripts/03b_tabular_timesplit.py }
        'graph'     { & $Python scripts/05b_graph_ablation_valsplit.py }
        'fusion'    { & $Python scripts/08_fusion_elliptic.py }
        'nlp'       { & $Python scripts/06b_export_nlp_models.py }
        'api'       { & $Python -m uvicorn app.api:app --reload --port 8000 }
        'dashboard' { & $Python -m streamlit run dashboard/streamlit_app.py }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
