# PowerShell script to set up scheduled auto-commit task
$ErrorActionPreference = "Stop"

# Get the current directory
$scriptPath = $PWD.Path
$autoCommitPath = Join-Path $scriptPath "auto_commit.ps1"

# Create the scheduled task
$taskName = "GitAutoCommit"
$taskDescription = "Automatically commit and push Git changes"

# Create the task action
$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$autoCommitPath`""

# Create the task trigger (every hour)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 60)

# Register the task
Register-ScheduledTask `
    -TaskName $taskName `
    -Description $taskDescription `
    -Action $action `
    -Trigger $trigger `
    -RunLevel Highest `
    -Force

Write-Host "Scheduled task '$taskName' has been created successfully!"
Write-Host "The script will run every hour to check for and commit changes."
Write-Host "You can modify the schedule in Task Scheduler if needed." 