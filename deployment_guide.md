# AWS Deployment Guide for Document Classifier

## Option 1: Quick Deployment (Keep ChromaDB)

### 1. Prepare for AWS
```bash
# Create requirements.txt for production
pip freeze > requirements.txt

# Add Gunicorn for production server
echo "gunicorn==21.2.0" >> requirements.txt
```

### 2. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy the vector database
COPY ocr_text_db/ ./ocr_text_db/

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "ocr_project.wsgi:application"]
```

### 3. AWS Deployment Options

#### A. AWS ECS (Container Service)
```bash
# Build and push to ECR
aws ecr create-repository --repository-name doc-classifier
docker build -t doc-classifier .
docker tag doc-classifier:latest <account>.dkr.ecr.<region>.amazonaws.com/doc-classifier:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/doc-classifier:latest
```

#### B. AWS Elastic Beanstalk
```bash
# Create application.py for EB
echo "from ocr_project.wsgi import application" > application.py

# Deploy
eb init doc-classifier
eb create production
eb deploy
```

#### C. AWS Lambda (Serverless)
```python
# lambda_handler.py
import json
from django.core.wsgi import get_wsgi_application
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
application = get_wsgi_application()

def lambda_handler(event, context):
    # Handle API Gateway event
    # Convert to Django request and process
    pass
```

---

## Option 2: Production Setup (Recommended)

### 1. Move to AWS RDS + OpenSearch

#### Update settings.py
```python
# ocr_project/settings.py

import os
from pathlib import Path

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'doc_classifier'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Vector Database Configuration
VECTOR_DB_CONFIG = {
    'type': 'opensearch',  # or 'chromadb', 'pinecone'
    'host': os.environ.get('OPENSEARCH_HOST'),
    'port': os.environ.get('OPENSEARCH_PORT', 443),
    'use_ssl': True,
    'index_name': 'document_vectors'
}

# AWS Configuration
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
```

#### Create vector database adapter
```python
# classifier/vector_db.py

from abc import ABC, abstractmethod
import chromadb
from opensearchpy import OpenSearch
import numpy as np

class VectorDB(ABC):
    @abstractmethod
    def query(self, vector, n_results=5):
        pass
    
    @abstractmethod
    def add_documents(self, texts, metadatas, ids):
        pass

class ChromaDBAdapter(VectorDB):
    def __init__(self, path="ocr_text_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("text_docs")
    
    def query(self, vector, n_results=5):
        return self.collection.query(
            query_embeddings=[vector],
            n_results=n_results
        )

class OpenSearchAdapter(VectorDB):
    def __init__(self, host, port=443, use_ssl=True):
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            use_ssl=use_ssl,
            verify_certs=True
        )
        self.index_name = "document_vectors"
    
    def query(self, vector, n_results=5):
        query = {
            "size": n_results,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": vector,
                        "k": n_results
                    }
                }
            }
        }
        
        response = self.client.search(
            index=self.index_name,
            body=query
        )
        
        # Convert to ChromaDB format
        metadatas = [[hit['_source']['metadata'] for hit in response['hits']['hits']]]
        distances = [[1 - hit['_score'] for hit in response['hits']['hits']]]
        
        return {
            'metadatas': metadatas,
            'distances': distances
        }

# Factory function
def get_vector_db():
    from django.conf import settings
    config = settings.VECTOR_DB_CONFIG
    
    if config['type'] == 'chromadb':
        return ChromaDBAdapter()
    elif config['type'] == 'opensearch':
        return OpenSearchAdapter(
            host=config['host'],
            port=config['port'],
            use_ssl=config['use_ssl']
        )
    else:
        raise ValueError(f"Unsupported vector DB type: {config['type']}")
```

#### Update views.py
```python
# classifier/views.py

from .vector_db import get_vector_db

def get_resources():
    # ... existing code ...
    
    # Replace ChromaDB with adapter
    vector_db = get_vector_db()
    
    return vector_db, embedder, easy_ocr, paddle_ocr, surya_ocr

@csrf_exempt
def classify_document(request):
    # ... existing code ...
    
    # For Weaviate version:
    easy_ocr, paddle_ocr, surya_ocr, rapid_ocr, hybrid_classifier = get_resources()
    
    # For Standard version:
    collection, embedder, easy_ocr, paddle_ocr, surya_ocr, rapid_ocr = get_resources()
    
    # ... OCR processing ...
    
    if extracted_text.strip() and not extracted_text.startswith("ERROR"):
        text_vec = embedder.encode([extracted_text])[0].tolist()
        try:
            # Use adapter instead of direct ChromaDB
            query_res = vector_db.query(text_vec, n_results=5)
            
            # ... rest of the code remains same ...
```

### 2. Environment Variables
```bash
# .env for production
DB_NAME=doc_classifier_prod
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432

OPENSEARCH_HOST=your-opensearch-domain.amazonaws.com
OPENSEARCH_PORT=443

AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

DJANGO_SECRET_KEY=your_very_long_secret_key
DEBUG=False
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
```

### 3. AWS Infrastructure (Terraform)
```hcl
# infrastructure/main.tf

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier = "doc-classifier-db"
  engine     = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.micro"
  
  allocated_storage = 20
  storage_type     = "gp2"
  
  db_name  = "doc_classifier"
  username = "postgres"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  skip_final_snapshot = true
}

# OpenSearch Domain
resource "aws_opensearch_domain" "vectors" {
  domain_name    = "doc-classifier-vectors"
  engine_version = "OpenSearch_2.3"
  
  cluster_config {
    instance_type = "t3.small.search"
    instance_count = 1
  }
  
  ebs_options {
    ebs_enabled = true
    volume_type = "gp3"
    volume_size = 20
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "doc-classifier"
}

# ECS Service
resource "aws_ecs_service" "api" {
  name            = "doc-classifier-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "doc-classifier"
    container_port   = 8000
  }
}
```

---

## Quick Start (Option 1)

If you want to deploy quickly without major changes:

1. **Keep the `ocr_text_db` folder**
2. **Use Docker deployment**
3. **Deploy to AWS ECS or Elastic Beanstalk**

```bash
# Quick deployment steps
docker build -t doc-classifier .
aws ecr create-repository --repository-name doc-classifier
# Push to ECR and deploy to ECS
```

## Production Ready (Option 2)

For a scalable production API:

1. **Move vectors to AWS OpenSearch**
2. **Use AWS RDS for metadata**
3. **Deploy with ECS + Load Balancer**
4. **Add API Gateway for rate limiting**

Which option would you prefer? I can help you implement either approach.