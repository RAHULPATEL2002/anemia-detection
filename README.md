# AnemiaVision AI


![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B3-red)
![Flask](https://img.shields.io/badge/Flask-Hospital%20Web%20App-black)
![SQLite](https://img.shields.io/badge/SQLite-Patient%20Records-0f766e)
![Docker](https://img.shields.io/badge/Docker-Production%20Ready-2496ED)

AnemiaVision AI is a hospital-grade, smartphone-first anemia screening platform built for conjunctiva and fingernail image analysis. It combines deep learning, Grad-CAM explainability, patient record management, PDF reporting, analytics, and a production-ready Flask deployment stack.

## Demo

`docs/demo.gif`

Demo GIF placeholder: add a short screen recording here showing upload, prediction, result review, and PDF export.

## Features

- 🩺 AI-assisted anemia screening from smartphone conjunctiva and fingernail images
- 📁 Smart dataset discovery that handles `anemic`, `Anemic`, `Anemia`, `non-anemic`, `Non-anemic`, and `Non_Anemia`
- 🧠 EfficientNet-B3 transfer learning with class balancing, mixed precision, warmup, cosine annealing, and early stopping
- 🔥 Grad-CAM explainability with side-by-side visual overlays
- 📄 Professional PDF report generation for hospitals and clinics
- 📊 Analytics dashboard with Chart.js visualizations
- 📱 Mobile-first capture workflow with rear-camera preference and upload fallback
- 🧾 SQLite-backed patient history, CSV export, and full-text patient-name search
- 🚦 Per-IP rate limiting for prediction endpoints
- 🐳 Docker and Gunicorn deployment support

## Project Overview

The system is designed for real-world screening assistance in hospitals, clinics, outreach camps, and home use. The workflow is:

1. Capture or upload a smartphone image of the eye conjunctiva or fingernail.
2. Validate image quality and reject unusable scans before inference.
3. Run the EfficientNet-B3 classifier and calibrated probability pipeline.
4. Generate a Grad-CAM focus image and patient-facing risk summary.
5. Save the scan record, export a PDF report, and view analytics over time.

## Tech Stack

- Backend: Flask, Flask-SQLAlchemy, Gunicorn
- Deep Learning: PyTorch, torchvision, EfficientNet-B3
- Imaging: Pillow, OpenCV
- Database: SQLite
- Frontend: Bootstrap 5, custom CSS, Chart.js, Lucide icons
- Reports: ReportLab
- Deployment: Docker, docker-compose

## Dataset Setup

Use the existing dataset structure exactly as below:

```text
dataset/
├── train/
│   ├── anemic/
│   └── non-anemic/
├── valid/
│   ├── Anemia/
│   └── Non_Anemia/
└── test/
    ├── Anemic/
    └── Non-anemic/
```

Important notes:

- The mixed capitalization is intentional and already handled in `dataset.py`.
- Eye conjunctiva and fingernail images are intentionally mixed in the same folders.
- The loader maps both modalities into the same anemia classification task:
  - class `0` = `non-anemic`
  - class `1` = `anemic`

To validate the dataset before training:

```bash
python utils/check_dataset.py
```

## Installation

### Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python utils/check_dataset.py
```

### Docker Setup

```bash
docker compose up --build
```

The production container serves the app on `http://127.0.0.1:5000`.

## Training

Start a fresh training run:

```bash
python train.py
```

CPU note:

- When running on CPU, training automatically uses a fast profile (EfficientNet-B0, 224px, fewer epochs) to avoid multi-hour runs.
- To force the full EfficientNet-B3 profile, set `ANEMIA_TRAIN_PROFILE=full` and `ANEMIA_MODEL_ARCH=efficientnet_b3` before running `train.py`.

Resume from the latest checkpoint:

```bash
python train.py --resume models/last_checkpoint.pth
```

Evaluate the trained model:

```bash
python evaluate.py --split test
```

Artifacts are saved to:

- `models/best_model.pth`
- `models/last_checkpoint.pth`
- `logs/training.log`
- `logs/evaluation/`

## Running the App

### Development

```bash
python app.py
```

### Mobile Use (Camera Capture)

- Mobile camera access via the **Camera Capture** tab requires HTTPS in most browsers.
- The upload button now uses the `capture="environment"` hint, which opens the camera on many phones even over HTTP.
- For full camera support, deploy behind HTTPS (for example via a reverse proxy with a TLS certificate).

### HTTPS Reverse Proxy (Caddy)

1. Point your domain DNS A record to the server IP.
2. Install Caddy and place your domain in `deploy/Caddyfile`.
3. Run:

```bash
caddy run --config deploy/Caddyfile
```

Your app will be available at `https://your-domain.example.com`, and mobile camera capture will work.

### Production

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
```

Health check:

```text
GET /health
```

### Render (persistent data + Grad-CAM)

Render web services use an ephemeral filesystem by default. Without a persistent disk, any new SQLite records, uploaded scan images, Grad-CAM outputs, and generated PDF reports can disappear after a restart or redeploy.

Recommended setup:

1. Use a Render plan that supports persistent disks.
2. Attach a disk with mount path `/var/data`.
3. Set `ANEMIA_RUNTIME_ROOT=/var/data/anemiavision`.
4. Set `ANEMIA_ENABLE_GRADCAM=true`.
5. Redeploy the service.

This repository also includes a sample [`render.yaml`](render.yaml) blueprint with the same settings.

## API Documentation

### POST `/api/predict`

Content type:

```text
application/json
```

Example request:

```json
{
  "image": "<base64-image-or-data-uri>",
  "patient_name": "Aarav Kumar",
  "age": 25,
  "gender": "M",
  "phone": "9999999999",
  "image_type": "Eye Conjunctiva",
  "notes": "Fatigue for 2 weeks"
}
```

Example success response:

```json
{
  "success": true,
  "scan_id": 42,
  "public_scan_id": "AV-000042",
  "prediction": "Non-Anemic",
  "confidence": 0.912,
  "risk_level": "Low Risk",
  "medical_advice": "Results look healthy at this screening level, with no strong visual signs of anemia detected.",
  "gradcam_url": "/media/gradcam/scan_042.jpg",
  "result_url": "/result/42",
  "pdf_url": "/export/pdf/42"
}
```

Example error response:

```json
{
  "success": false,
  "error": "Too many prediction requests from this device. Please wait about 18 seconds and try again."
}
```

Notes:

- Base64 payloads can be raw base64 strings or data URIs.
- Prediction routes are rate-limited to `10` requests per minute per IP.
- Uploads are resized to `300x300` before saving.

## Results

Target evaluation goals for the final trained checkpoint:

| Metric | Target |
|---|---:|
| Accuracy | > 90% |
| F1 Score | Strong clinical generalization target |
| AUC-ROC | > 0.93 |
| Sensitivity | > 88% |
| Specificity | > 88% |

When available, real metrics are loaded from the latest evaluation artifacts in `logs/evaluation/`.

## Deployment Notes

- `Dockerfile` uses `python:3.10-slim`
- Gunicorn runs with `2` workers on port `5000`
- Runtime data can be redirected with `ANEMIA_RUNTIME_ROOT`
- Uploaded scan images are served from `/media/uploads/<filename>`
- Grad-CAM images are served from `/media/gradcam/<filename>`
- `docker-compose.yml` mounts persistent volumes for:
  - `database/`
  - `models/`
  - `reports/`
  - `logs/`
  - `static/uploads/`
  - `static/gradcam/`
- `render.yaml` mounts a persistent disk at `/var/data` and stores database, reports, uploads, and Grad-CAM outputs under `/var/data/anemiavision`
- The `/health` endpoint is used by both Docker and application readiness checks

## Screenshots

Add project screenshots here:

- Dashboard screenshot placeholder
- Scan page screenshot placeholder
- Result page screenshot placeholder
- Analytics page screenshot placeholder

## Contributing

1. Fork the repository or create a feature branch.
2. Keep the mobile-first and hospital-grade design language intact.
3. Run dataset checks, compile checks, and route smoke tests before opening a PR.
4. Document any user-facing workflow changes in this README.

## License

Add your preferred project license here, for example `MIT` or `Apache-2.0`.

## Disclaimer

This tool is for screening assistance only and does not replace clinical diagnosis. Always consult a qualified medical professional.
