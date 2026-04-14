# Docker Deployment Guide for DocClassifier

This guide explains how to deploy the simplified DocClassifier application using Docker and Docker Compose. The application now uses **RapidOCR** exclusively, which eliminates complex system dependencies like Tesseract or PaddleOCR and significantly reduces container size.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- At least 4GB of RAM allocated to Docker

## Architecture

- **App**: Django 4.2.7 application running on Gunicorn.
- **DB**: Weaviate Vector Database (v1.19.0).
- **OCR Engine**: RapidOCR (ONNX runtime).
- **Embeddings**: Sentence-Transformers (huggingface/all-MiniLM-L6-v2).

## Getting Started

### 1. Configure Environment
Update the environment variables if needed. The `docker-compose.yml` is pre-configured to connect the Django app to the Weaviate container.

### 2. Build and Start
Run the following command in the root directory:

```bash
docker-compose up --build
```

This will:
1. Build the Django application image.
2. Pull the Weaviate image.
3. Start both services.
4. The application will be accessible at `http://localhost:8000`.

### 3. Initialize the Vector Store
Once the containers are running:
1. Navigate to `http://localhost:8000/weaviate/`.
2. Go to the **Rebuild Index** page.
3. Click **Start Rebuild**. 
   *This will process the sample documents in the `samples/` directory and index them into Weaviate.*

## Production Considerations

### Backend Server
The Dockerfile uses `gunicorn` for serving the Django application, which is suitable for production.

### Persistence
Weaviate data is persisted in a Docker volume named `weaviate_data`. source documents for training can be mapped to the `samples_data` volume.

### Port Security
In a production environment, you should:
- Ensure port `8080` (Weaviate) is NOT exposed to the public internet (remove it from `ports` in `docker-compose.yml` if the app and DB are on the same machine/network).
- Use an Nginx reverse proxy or AWS Load Balancer to handle SSL (HTTPS) on port 443 and forward traffic to port 8000.

## Troubleshooting

### Shared Library Errors
If you encounter errors related to `libGL.so.1` or `libgthread-2.0.so.0`, ensure the Dockerfile includes the `libgl1` and `libglib2.0-0` packages.

### Weaviate Connection
If the app cannot connect to Weaviate, verify that the `WEAVIATE_URL` environment variable in `docker-compose.yml` is set to `http://db:8080`.

---
*Simplified and optimized for deployment by Antigravity.*