$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "taales_automation.py"
$ConfigPath = Join-Path $ScriptDir "taales_automation_config.json"
$StopFile = Join-Path $ScriptDir "STOP_AUTOMATION"

if (-not (Test-Path $PythonScript)) {
    throw "Could not find: $PythonScript"
}

try {
    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
} catch {
    Write-Warning "Could not change sleep settings. Keep the PC awake manually."
}

$PythonCandidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
    (Get-Command python.exe -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
) | Where-Object { $_ -and (Test-Path $_) }

$Python = $PythonCandidates | Select-Object -First 1
if (-not $Python) {
    throw "No usable python.exe installation was found."
}

Write-Host "Using Python: $Python" -ForegroundColor Cyan

& $Python -c "import pyautogui, pygetwindow, pyperclip, psutil, pywinauto" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing GUI dependencies..." -ForegroundColor Cyan
    $PipProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "pip", "install",
            "--disable-pip-version-check",
            "--upgrade",
            "pyautogui", "pygetwindow", "pyperclip", "psutil", "pywinauto"
        ) `
        -WorkingDirectory $ScriptDir `
        -NoNewWindow `
        -Wait `
        -PassThru

    if ($PipProcess.ExitCode -ne 0) {
        throw "Dependency installation failed with exit code $($PipProcess.ExitCode)."
    }
}

function Invoke-TAALESPhase {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Mode,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $RestartCount = 0

    while ($true) {
        if (Test-Path $StopFile) {
            Write-Host "STOP_AUTOMATION exists. Supervisor is stopping." `
                -ForegroundColor Yellow
            exit 0
        }

        Write-Host ""
        Write-Host "============================================================" `
            -ForegroundColor Cyan
        Write-Host $Description -ForegroundColor Cyan
        Write-Host "============================================================" `
            -ForegroundColor Cyan

        $Process = Start-Process `
            -FilePath $Python `
            -ArgumentList @("`"$PythonScript`"", $Mode) `
            -WorkingDirectory $ScriptDir `
            -NoNewWindow `
            -Wait `
            -PassThru

        $ExitCode = [int]$Process.ExitCode

        if ($ExitCode -eq 0) {
            return
        }

        if ($ExitCode -eq 10) {
            Write-Warning "Another automation instance is using this work folder."
            exit 10
        }

        $RestartCount++
        Write-Warning "$Description exited unexpectedly with code $ExitCode."
        Write-Warning "Restarting this phase in 30 seconds. Restart number: $RestartCount"
        Start-Sleep -Seconds 30
    }
}

if (-not (Test-Path $ConfigPath)) {
    Invoke-TAALESPhase `
        -Mode "setup" `
        -Description "ONE-TIME TAALES SETUP"
    Write-Host ""
    Write-Host "Setup finished. Run this same launcher again to start processing." `
        -ForegroundColor Green
    exit 0
}

Invoke-TAALESPhase `
    -Mode "run" `
    -Description "PHASE 1/3: PROCESSING ALL ORDINARY REMAINING FILES"

Invoke-TAALESPhase `
    -Mode "retry-failed" `
    -Description "PHASE 2/3: RETRYING ALL FILES STILL MARKED FAILED"

Invoke-TAALESPhase `
    -Mode "cleanup" `
    -Description "PHASE 3/3: VERIFYING FINAL OUTPUT AND REMOVING TEMPORARY FILES"

Write-Host ""
Write-Host "TAALES ONE-CLICK WORKFLOW FINISHED." -ForegroundColor Green
