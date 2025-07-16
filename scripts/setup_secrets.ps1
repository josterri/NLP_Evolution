# PowerShell script to help set up GitHub secrets
Write-Host "GitHub Secrets Setup Helper" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

# Function to get secret value
function Get-SecretValue {
    param (
        [string]$secretName,
        [string]$description
    )
    Write-Host "$description" -ForegroundColor Yellow
    Write-Host "Secret Name: $secretName" -ForegroundColor Cyan
    $secretValue = Read-Host "Enter value (input will be hidden)" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secretValue)
    return [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
}

# Get repository information
$repoOwner = Read-Host "Enter repository owner (username or organization)"
$repoName = Read-Host "Enter repository name"

# Get secrets
$secrets = @{
    "STREAMLIT_API_KEY" = "Your Streamlit API key for deployment"
    "SNYK_TOKEN" = "Your Snyk API token for security scanning"
}

Write-Host "`nPlease follow these steps to add the secrets:" -ForegroundColor Green
Write-Host "1. Go to: https://github.com/$repoOwner/$repoName/settings/secrets/actions" -ForegroundColor Yellow
Write-Host "2. Click 'New repository secret'" -ForegroundColor Yellow
Write-Host "3. Add each of the following secrets:`n" -ForegroundColor Yellow

foreach ($secret in $secrets.GetEnumerator()) {
    $value = Get-SecretValue -secretName $secret.Key -description $secret.Value
    Write-Host "`nFor $($secret.Key):" -ForegroundColor Cyan
    Write-Host "- Name: $($secret.Key)" -ForegroundColor White
    Write-Host "- Value: <hidden>" -ForegroundColor White
    Write-Host "------------------------" -ForegroundColor White
}

Write-Host "`nAdditional Setup Steps:" -ForegroundColor Green
Write-Host "1. Enable 'GitHub Actions' in repository settings" -ForegroundColor Yellow
Write-Host "2. Enable 'Dependency graph' in repository settings" -ForegroundColor Yellow
Write-Host "3. Enable 'Dependabot alerts' in repository settings" -ForegroundColor Yellow
Write-Host "4. Enable 'Code scanning' in repository settings" -ForegroundColor Yellow

Write-Host "`nSetup instructions have been provided. Would you like to open the repository settings page?" -ForegroundColor Green
$openBrowser = Read-Host "Enter 'y' to open browser"

if ($openBrowser -eq 'y') {
    Start-Process "https://github.com/$repoOwner/$repoName/settings"
} 