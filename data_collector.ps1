# Configuration
$ProjectDir = Join-Path $env:USERPROFILE 'bd-model-generations'
$StatusFile = Join-Path $ProjectDir 'status\data_collector.status'
$LogFile = Join-Path $ProjectDir 'logs\actions.log'

function Write-Log {
    param([string]$Message, [string]$Type = 'INFO')
    $timestamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
    Add-Content -Path $LogFile -Value "[$timestamp] $Type`: $Message"
}

# Ensure status directory exists
New-Item -ItemType Directory -Force -Path (Split-Path $StatusFile) | Out-Null

try {
    # Initialize status
    Set-Content -Path $StatusFile -Value 'Initializing data collection...'
    Write-Log 'Data collector started' 'INFO'
    
    # Simulated data collection progress (replace with actual logic)
    $progressSteps = @(
        @{ Status = 'Connecting to data sources...'; Duration = 2 },
        @{ Status = 'Fetching Bengali text corpus...'; Duration = 3 },
        @{ Status = 'Processing raw data...'; Duration = 2 },
        @{ Status = 'Cleaning and normalizing text...'; Duration = 2 },
        @{ Status = 'Preparing training dataset...'; Duration = 1 }
    )

    foreach ($step in $progressSteps) {
        Set-Content -Path $StatusFile -Value $step.Status
        Write-Log $step.Status 'INFO'
        Start-Sleep -Seconds $step.Duration
    }

    # Final status update
    Set-Content -Path $StatusFile -Value 'Data collection completed successfully'
    Write-Log 'Data collection completed' 'SUCCESS'
    Start-Sleep -Seconds 1

} catch {
    Write-Log "Error in data collection: $_" 'ERROR'
    Set-Content -Path $StatusFile -Value 'Error: Data collection failed'
    Start-Sleep -Seconds 1
} finally {
    # Cleanup status file
    if (Test-Path $StatusFile) {
        Remove-Item -Path $StatusFile
    }
}
