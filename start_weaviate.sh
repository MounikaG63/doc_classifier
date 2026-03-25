#!/bin/bash

echo "Starting Weaviate with persistence..."

# Create data directory if it doesn't exist
mkdir -p weaviate_data

# Stop any existing Weaviate container
docker stop weaviate-classifier 2>/dev/null
docker rm weaviate-classifier 2>/dev/null

# Start Weaviate with proper configuration
docker run -d \
  --name weaviate-classifier \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e ENABLE_MODULES= \
  -e CLUSTER_HOSTNAME=node1 \
  -v "$(pwd)/weaviate_data:/var/lib/weaviate" \
  semitechnologies/weaviate:latest

echo "Weaviate is starting..."
echo "Waiting for Weaviate to be ready..."

# Wait for Weaviate to be ready
while ! curl -s http://localhost:8080/v1/meta >/dev/null 2>&1; do
    echo "Still waiting for Weaviate..."
    sleep 2
done

echo ""
echo "✅ Weaviate is ready!"
echo "🌐 Access: http://localhost:8080"
echo "📊 Health: http://localhost:8080/v1/meta"
echo ""
echo "To stop Weaviate: docker stop weaviate-classifier"
echo "To view logs: docker logs weaviate-classifier"
echo ""