# GitHub Repository Setup Script
# This script will help you push your code to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Repository Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Check current git status
Write-Host "Current git status:" -ForegroundColor Yellow
git status

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor White
Write-Host "   - Go to: https://github.com/new" -ForegroundColor Gray
Write-Host "   - Repository name: slp-text-summarizer" -ForegroundColor Gray
Write-Host "   - Choose Public or Private" -ForegroundColor Gray
Write-Host "   - DO NOT initialize with README, .gitignore, or license" -ForegroundColor Gray
Write-Host "   - Click 'Create repository'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. After creating the repository, run:" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/slp-text-summarizer.git" -ForegroundColor Green
Write-Host "   git branch -M main" -ForegroundColor Green
Write-Host "   git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "   (Replace YOUR_USERNAME with your actual GitHub username)" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to continue
$response = Read-Host "Have you created the GitHub repository? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    $username = Read-Host "Enter your GitHub username"
    $repoName = Read-Host "Enter repository name (default: slp-text-summarizer)" 
    if ([string]::IsNullOrWhiteSpace($repoName)) {
        $repoName = "slp-text-summarizer"
    }
    
    Write-Host ""
    Write-Host "Adding remote origin..." -ForegroundColor Yellow
    $remoteUrl = "https://github.com/$username/$repoName.git"
    git remote add origin $remoteUrl
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Remote added successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Renaming branch to main..." -ForegroundColor Yellow
        git branch -M main
        
        Write-Host ""
        Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
        Write-Host "You may be prompted for your GitHub credentials." -ForegroundColor Yellow
        git push -u origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "========================================" -ForegroundColor Green
            Write-Host "Success! Your code has been pushed to GitHub!" -ForegroundColor Green
            Write-Host "Repository URL: $remoteUrl" -ForegroundColor Green
            Write-Host "========================================" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "There was an error pushing to GitHub." -ForegroundColor Red
            Write-Host "Make sure you have the correct permissions and credentials." -ForegroundColor Red
        }
    } else {
        Write-Host ""
        Write-Host "Remote might already exist. Checking..." -ForegroundColor Yellow
        git remote -v
        Write-Host ""
        Write-Host "If the remote is correct, you can push directly with:" -ForegroundColor Yellow
        Write-Host "  git push -u origin main" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "Please create the repository first, then run this script again." -ForegroundColor Yellow
    Write-Host "Or follow the manual instructions in GITHUB_SETUP.md" -ForegroundColor Yellow
}
