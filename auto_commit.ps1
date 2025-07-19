# PowerShell script for automated Git commits
# To set up automatic running:
# 1. Open Task Scheduler
# 2. Create Basic Task
# 3. Name: GitAutoCommit
# 4. Trigger: Daily, recur every 1 hour
# 5. Action: Start a program
# 6. Program: powershell.exe
# 7. Arguments: -ExecutionPolicy Bypass -File "FULL_PATH_TO_THIS_SCRIPT"

$ErrorActionPreference = "Stop"

# Change to the repository directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Function to get a meaningful commit message based on changes
function Get-CommitMessage {
    $changes = git diff --cached --name-status
    if (-not $changes) {
        $changes = git diff --name-status
    }
    
    $message = "Updates: "
    $fileTypes = @{}
    
    foreach ($change in $changes -split "`n") {
        if ($change -match "^([AMD])\s+(.+)$") {
            $type = $matches[1]
            $file = $matches[2]
            $ext = [System.IO.Path]::GetExtension($file)
            
            if (-not $fileTypes.ContainsKey($ext)) {
                $fileTypes[$ext] = @{
                    'added' = 0
                    'modified' = 0
                    'deleted' = 0
                }
            }
            
            switch ($type) {
                'A' { $fileTypes[$ext]['added']++ }
                'M' { $fileTypes[$ext]['modified']++ }
                'D' { $fileTypes[$ext]['deleted']++ }
            }
        }
    }
    
    foreach ($ext in $fileTypes.Keys) {
        $stats = $fileTypes[$ext]
        if ($stats['added'] -gt 0) {
            $message += "Added $($stats['added']) $ext file(s). "
        }
        if ($stats['modified'] -gt 0) {
            $message += "Modified $($stats['modified']) $ext file(s). "
        }
        if ($stats['deleted'] -gt 0) {
            $message += "Deleted $($stats['deleted']) $ext file(s). "
        }
    }
    
    return $message.Trim()
}

# Check if there are any changes
$hasChanges = $false
$status = git status --porcelain
if ($status) {
    $hasChanges = $true
}

if (-not $hasChanges) {
    Write-Host "No changes to commit."
    exit 0
}

# Stage all changes
git add .

# Generate commit message
$commitMessage = Get-CommitMessage

# Commit changes
Write-Host "Committing with message: $commitMessage"
git commit -m $commitMessage

# Push changes
Write-Host "Pushing changes to remote..."
git push

Write-Host "Auto-commit completed successfully!" 