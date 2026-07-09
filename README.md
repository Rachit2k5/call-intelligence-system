# CallIQ — Deployment & Local Run

This repository contains the FastAPI backend for the Intelligent Call Prioritization System.

Files added to help deployment:
- [Dockerfile](Dockerfile)
- [render.yaml](render.yaml)

Quick notes
- The app serves `project/appp.html` at `/` and exposes several API endpoints (health, score, predict, upload-score, etc.).
- If no trained models are present, the rule-based scorer is used — the app still runs without `models/best_model.pkl`.

Local (no Docker)
1. Create and activate a virtualenv (recommended).
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
3. Run the app:
```bash
uvicorn project.app:app --host 0.0.0.0 --port 8010
```
Open http://127.0.0.1:8010/ and check `GET /health`.

Local (Docker)
```bash
docker build -t calliq .
docker run -e PORT=8000 -p 8000:8000 calliq
```

Deploying to Render (recommended simple path)
1. Push this repository to GitHub.
2. Go to Render dashboard → New → Web Service.
3. Connect your GitHub repo and select the branch.
4. Render will detect `Dockerfile` and build the image. If you prefer the Python environment, set Environment to `Python` and use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn project.app:app --host 0.0.0.0 --port $PORT`
5. Deploy. After successful build, the service will be reachable at the Render domain.

Notes and troubleshooting
- Compiled packages such as `xgboost` and `scikit-learn` may take longer to build; using the `Dockerfile` (which installs `build-essential` and `libgomp1`) reduces platform build issues.
- If you want faster deploys, pre-build a Docker image and push it to a registry (Docker Hub / GitHub Container Registry), then configure Render to deploy from that image.
- To verify the service after deploy, call `GET /health` which reports model and processed-data availability.

Next steps I can help with
- Walk through creating the Render Web Service step-by-step.
- Create a GitHub Action to build and push the Docker image automatically.
- Add a small frontend or a one-click Render button.

CI / Automatic image build
- A GitHub Actions workflow is included at `.github/workflows/publish-and-deploy.yml`.
- On push to `main` the workflow builds the Docker image and publishes it to GitHub Container Registry as `ghcr.io/<owner>/calliq:latest` and a commit-SHA tag.

Secrets to set (optional)
- `GITHUB_TOKEN` — provided automatically by GitHub Actions; no action needed.
- `RENDER_API_KEY` — create a Render API key if you want the workflow to automatically trigger a Render deploy.
- `RENDER_SERVICE_ID` — the Render service id (e.g. `srv-xxxxx`) for your web service.

To add automatic Render deploys:
1. Create a Render API key from your Render dashboard (Account → API Keys).
2. In your GitHub repo, go to Settings → Secrets → Actions and add `RENDER_API_KEY` and `RENDER_SERVICE_ID`.
3. Push to `main`; the workflow will build and push the image and then call Render's deploy API if the secrets are present.

Baking model artifacts into the image
- If you want model artifacts baked into the Docker image automatically at CI build time, provide a tarball URL containing a `models/` directory with `best_model.pkl`, `scaler.pkl`, and any other artifacts.
- Set the repository secret `MODEL_ARTIFACTS_URL` to a direct download URL for a `.tar.gz` archive (public or signed). The workflow will download and extract it into the repo before building the image so the `models/` folder is included in the image.

Example tarball layout:
```
models/
   best_model.pkl
   scaler.pkl
   feature_columns.json
   metrics.json
```

If you prefer to commit the `models/` directory directly into the repo, the Docker image will include it automatically when the workflow builds.

