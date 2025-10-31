Upload instructions for `jeetshorey123/bdaproject`

Overview
--------
This document explains how to upload your project (code + analysis outputs) to the GitHub repository `https://github.com/jeetshorey123/bdaproject` from your local machine.

Important: I cannot push to your GitHub repo from here because I don't have your credentials or SSH key. The helper script `upload_to_github.ps1` has been added to your project root; run it locally in PowerShell to perform the upload steps.

Pre-checks
----------
1. Ensure Git is installed and available in PATH.
   - Windows: https://git-scm.com/download/win

2. (Recommended) Install Git LFS for large CSVs:
   - https://git-lfs.github.com/
   - After installing, `git lfs install`.

3. Create a GitHub repo if not already created:
   - Visit https://github.com/new and create repository `bdaproject` under your account `jeetshorey123`.
   - Choose public/private as needed. Do not initialize with README if you plan to push an existing repo.

Large files & repository size
-----------------------------
- Your dataset contains very large CSVs (millions of rows). GitHub limits single files to 100 MB and has practical repo size limits.
- Recommended approaches:
  1. Use Git LFS to track large CSV files (`git lfs track "*.csv"`) and commit. LFS stores pointers in repo and moves large file storage to LFS (but LFS has storage/transfer quotas).
  2. Store large raw datasets outside the repo (Azure Blob, AWS S3, Google Cloud Storage, or a private file server) and include a small sample or the `powerbi_analysis_data.csv` summary in the repo.
  3. Use GitHub Releases to attach large assets (zipped datasets) outside the main commit history.

How to run the helper script (PowerShell)
-----------------------------------------
Open PowerShell and cd to the project root (where `upload_to_github.ps1` is located). Example:

```powershell
cd "C:\Users\91983\OneDrive\Desktop\bda"
# If your GitHub remote uses SSH, edit the script or pass remoteUrl as param
.\upload_to_github.ps1 -remoteUrl "https://github.com/jeetshorey123/bdaproject.git" -useLFS
```

Notes while running:
- The script will scan for large files (>50 MB) and list them.
- If you pass `-useLFS` the script will attempt to run `git lfs track "*.csv"` (git-lfs must be installed).
- You will be prompted to authenticate when pushing. Use one of the following:
  - HTTPS with a Personal Access Token (PAT): use PAT as password when prompted. Create PAT here: https://github.com/settings/tokens
  - SSH: ensure your SSH public key is added to your GitHub account and `remoteUrl` uses the SSH form: `git@github.com:jeetshorey123/bdaproject.git`.

Manual steps (if script fails)
----------------------------
1. Initialize git (if needed):
   ```powershell
   git init
   git remote add origin https://github.com/jeetshorey123/bdaproject.git
   git add -A
   git commit -m "Initial upload: code + analysis"
   git push -u origin main
   ```
2. If push fails due to branch name:
   ```powershell
   git branch -M main
   git push -u origin main
   ```
3. If files are too large, set up Git LFS and then re-add and commit large files:
   ```powershell
   # install Git LFS from https://git-lfs.github.com/ (one-time)
   git lfs install
   git lfs track "*.csv"
   git add .gitattributes
   git add <large-file.csv>
   git commit -m "Add large dataset with Git LFS"
   git push origin main
   ```

What I added to the repo for you
-------------------------------
- `upload_to_github.ps1` — PowerShell helper script to initialize repo, optionally configure Git LFS, add/commit, and push.
- `GITHUB_UPLOAD_INSTRUCTIONS.md` — this instructions file.
- `powerbi_analysis_data.csv` — structured summary CSV (already in project root).

Suggested repo layout before pushing (to reduce size):
- Keep lightweight code and documentation in the repo.
- Keep full raw CSVs out of the Git history or use LFS. Consider moving the raw data into a separate storage account or compressing and adding as a release asset.

Alternatives if you prefer I do the upload
-----------------------------------------
I cannot upload directly to GitHub for you because I don't have your credentials or explicit, secure authorization to act on your GitHub account. If you want me to produce a ZIP of everything (excluding very large raw CSVs) I can prepare it here and you can upload the ZIP to GitHub Releases or manually push it.

Checklist to mark as complete (once you run the script):
- [ ] Create GitHub repo `jeetshorey123/bdaproject` (if not exists)
- [ ] Install Git and (optional) Git LFS
- [ ] Run `upload_to_github.ps1` from project root
- [ ] Verify files visible on GitHub
- [ ] If large files were not uploaded, choose an alternative: LFS, cloud storage, or Releases

Troubleshooting
---------------
If you run into authentication errors, create a PAT and use HTTPS, or set up SSH keys. If you see errors about large files, follow the "Large files & repository size" section above.

If you want, tell me which approach you prefer for the raw dataset (LFS / cloud / release) and I will update the script or create additional helper scripts to automate that choice.
