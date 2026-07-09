Packaging and uploading model artifacts

1) Create the tarball locally (from repo root on Windows PowerShell):

```powershell
.
\scripts\package_models.ps1 -OutFile .\models.tar.gz
```

Or, from any Unix shell (Git Bash / WSL / macOS / Linux):

```bash
# run from repository root
tar -czf models.tar.gz -C project models
```

2) Upload options (choose one):

- GitHub Release (recommended for small private artifacts):
  - Create a new release in the GitHub UI and attach `models.tar.gz`.
  - Once uploaded, right-click the file link and copy the direct URL (it will be served with authentication; for public releases it is publicly accessible).

- GitHub Container Registry / GitHub Packages (not for arbitrary files):
  - Use a package registry if supported.

- S3 / GCS / Azure Blob (recommended for large or private artifacts):
  - Upload the tarball to a bucket and create a pre-signed URL (expiry as needed).

3) Set GitHub secret `MODEL_ARTIFACTS_URL` to the direct download URL (or signed URL) in your repository Settings → Secrets → Actions.

4) Push to `main` to trigger CI. The workflow will download the tarball and include `models/` in the built Docker image.
