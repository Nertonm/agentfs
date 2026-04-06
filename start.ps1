# start.ps1 - SmartContext Agent no Windows (PowerShell)
#
# Uso:
#   .\start.ps1 [workspace] [modelo.gguf]              -> chat interativo
#   .\start.ps1 [workspace] [modelo.gguf] "tarefa..."  -> execucao autonoma
#
# Exemplos:
#   .\start.ps1 . C:\models\mistral-7b-q4_k_m.gguf
#   .\start.ps1 .\projeto modelo.gguf "documentar as classes publicas"

param(
    [string]$Workspace = ".",
    [string]$Model     = "model.gguf",
    [string]$Task      = "",
    [int]   $Port      = 8080
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Config    = Join-Path $ScriptDir "config.toml"

function Log  { param($m) Write-Host "[start] $m" -ForegroundColor Green  }
function Warn { param($m) Write-Host "[start] $m" -ForegroundColor Yellow }
function Err  { param($m) Write-Host "[start] $m" -ForegroundColor Red    }

Log "SmartContext Agent"
Log "Workspace : $Workspace"
Log "Modelo    : $Model"
if ($Task) { Log "Tarefa    : $Task" }
Write-Host ""

# -- 1. Dependencias Python ---------------------------------------------------
Log "Verificando dependencias..."
python -c "import requests"              2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Warn "Instalando requests..."; pip install -q requests }
python -c "import numpy"                 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Warn "Instalando numpy...";    pip install -q numpy }
python -c "import sentence_transformers" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Warn "sentence-transformers ausente. pip install sentence-transformers" }
python -c "import watchdog"              2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Warn "watchdog ausente - usando polling. pip install watchdog" }

# -- 2. llama-server ----------------------------------------------------------
$health = try { Invoke-RestMethod "http://localhost:$Port/health" -EA Stop } catch { $null }
if ($health -and $health.status -eq "ok") {
    Log "llama-server ja rodando em :$Port OK"
} else {
    $llamaExe = Get-Command llama-server -EA SilentlyContinue
    if ($llamaExe) {
        Log "Iniciando llama-server ($Model)..."
        $llamaProc = Start-Process llama-server `
            -ArgumentList "-m `"$Model`" -ngl 99 -c 4096 --port $Port" `
            -PassThru -WindowStyle Minimized
        $llamaProc.Id | Out-File "$env:TEMP\smartctx_llama.pid"
        Log "Aguardando llama-server (PID $($llamaProc.Id))..."
        $ready = $false
        1..40 | ForEach-Object {
            Start-Sleep 1
            $h = try { Invoke-RestMethod "http://localhost:$Port/health" -EA Stop } catch { $null }
            if ($h -and $h.status -eq "ok") { $ready = $true }
        }
        if ($ready) { Log "llama-server pronto OK" }
        else { Err "Timeout - llama-server nao iniciou."; exit 1 }
    } else {
        Err "llama-server nao encontrado. Inicie manualmente antes de rodar este script."
        exit 1
    }
}

# -- 3. Vault Watcher (background) --------------------------------------------
Log "Iniciando vault watcher..."
$watcherProc = Start-Process python `
    -ArgumentList "`"$ScriptDir\vault_watcher.py`" `"$Workspace`" --config `"$Config`" --no-initial-scan" `
    -PassThru -WindowStyle Minimized -RedirectStandardOutput "$env:TEMP\smartctx_watcher.log"
$watcherProc.Id | Out-File "$env:TEMP\smartctx_watcher.pid"
Log "Watcher PID $($watcherProc.Id) OK"

# -- 4. Scan + embed inicial --------------------------------------------------
Log "Indexando workspace..."
python "$ScriptDir\vault_indexer.py" "$Workspace" --embed-all
if ($LASTEXITCODE -ne 0) {
    python "$ScriptDir\vault_indexer.py" "$Workspace"
}
Log "Indice pronto OK"
Write-Host ""

# -- 5. Agente ----------------------------------------------------------------
if ($Task) {
    Log "Modo autonomo: $Task"
    python "$ScriptDir\multi_agent.py" "$Task" `
        --workspace "$Workspace" `
        --url "http://localhost:$Port/v1"
} else {
    Log "Modo interativo (chat)"
    python "$ScriptDir\agent.py" `
        --workspace "$Workspace" `
        --url "http://localhost:$Port/v1" `
        --ctx 4096
}

# -- Cleanup ------------------------------------------------------------------
Log "Encerrando watcher..."
Stop-Process -Id $watcherProc.Id -EA SilentlyContinue
Remove-Item "$env:TEMP\smartctx_*.pid" -EA SilentlyContinue
Log "Encerrado!"
