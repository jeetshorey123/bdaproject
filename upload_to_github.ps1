<#
PowerShell helper to prepare and push the project to GitHub.
USAGE (run from project root):
  1. Edit the variable $remoteUrl below to your repo (HTTPS or SSH).
  2. Open PowerShell (with Git installed), then run:
       .\upload_to_github.ps1

Notes:
- This script WILL NOT supply your GitHub credentials. You must authenticate (PAT or SSH key) when prompted.
- For very large CSVs, use Git LFS. This script can initialize git-lfs but you must install it first.
- If your CSVs exceed GitHub file size limits or exceed repo size quotas, consider uploading datasets to cloud storage and keeping only sample or metadata in repo.
#>

param(
    [string]$remoteUrl = "https://github.com/jeetshorey123/bdaproject.git",
    [switch]$useLFS
)

function Check-CommandExists {
    param([string]$cmd)
    $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

Write-Host "Preparing to upload repository to: $remoteUrl" -ForegroundColor Cyan

if (-not (Check-CommandExists git)) {
    Write-Error "git is not installed or not in PATH. Install Git and re-run the script."; exit 1
}

# Show detected large files (simple heuristic: files > 50MB)
Write-Host "Scanning for large files (>50 MB)..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -File | Where-Object { $_.Length -gt 50MB } | Select-Object FullName, @{Name='MB';Expression={ [math]::Round($_.Length/1MB,2) }} | Format-Table -AutoSize

if ($useLFS) {
    if (-not (Check-CommandExists git-lfs)) {
        Write-Host "git-lfs not detected. Attempt to initialize but you should install git-lfs first: https://git-lfs.github.com/" -ForegroundColor Yellow
    } else {
        Write-Host "Initializing Git LFS and tracking CSVs..." -ForegroundColor Green
        git lfs install
        git lfs track "*.csv"
        if (Test-Path ".gitattributes") { Write-Host ".gitattributes created/updated." -ForegroundColor Green }
    }
}

# Initialize repo if needed
if (-not (Test-Path ".git")) {
    Write-Host "Initializing new git repository..." -ForegroundColor Green
    git init
} else {
    Write-Host "Git repository detected." -ForegroundColor Green
}

# Add remote if not present or update it
$existing = git remote -v 2>$null
if (-not $existing) {
    Write-Host "Adding remote origin: $remoteUrl" -ForegroundColor Green
    git remote add origin $remoteUrl
} else {
    Write-Host "Remote(s) present:" -ForegroundColor Green
    Write-Host $existing
    Write-Host "If you want to change the remote, run: git remote set-url origin <url>" -ForegroundColor Yellow
}

# Add files (use careful add to avoid unintentionally large files)
Write-Host "Staging files... (this may take time)" -ForegroundColor Green
# By default we add everything. If you prefer to exclude raw CSVs, edit the gitignore before running.
git add -A

# Commit
$commitMsg = "Initial upload: full project and analysis data"
Write-Host "Committing changes: $commitMsg" -ForegroundColor Green
try {
    git commit -m "$commitMsg"
} catch {
    Write-Host "No changes to commit or commit failed." -ForegroundColor Yellow
}

# Push to remote
Write-Host "Pushing to remote origin (branch: main). You may be prompted to authenticate." -ForegroundColor Cyan
# Attempt to push to 'main' branch. If your repo uses 'master' or other default branch, change accordingly.
try {
    git push -u origin main
} catch {
    Write-Warning "Push failed. Common causes: authentication required, branch name mismatch, or remote not found. See GITHUB_UPLOAD_INSTRUCTIONS.md for manual steps and troubleshooting."
}

Write-Host "Done. Verify your repository on GitHub: $remoteUrl" -ForegroundColor Green
