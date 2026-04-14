@echo off
echo Starting Weaviate with persistence...


REM Create data directory if it doesn't exist
if not exist "weaviate_data" mkdir weaviate_data

REM Stop any existing Weaviate container
docker stop weaviate-classifier 2>nul
docker rm weaviate-classifier 2>nul

REM Start Weaviate with proper configuration
docker run -d ^
  --name weaviate-classifier ^
  -p 8080:8080 ^
  -p 50051:50051 ^
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true ^
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate ^
  -e DEFAULT_VECTORIZER_MODULE=none ^
  -e ENABLE_MODULES= ^
  -e CLUSTER_HOSTNAME=node1 ^
  -e CLUSTER_ADVERTISE_ADDR=127.0.0.1 ^
  -v "%cd%\weaviate_data:/var/lib/weaviate" ^
  semitechnologies/weaviate:latest

echo Weaviate is starting...
echo Waiting for Weaviate to be ready...

REM Wait for Weaviate to be ready
:wait_loop
timeout /t 2 /nobreak >nul
curl -s http://localhost:8080/v1/meta >nul 2>&1
if %errorlevel% neq 0 (
    echo Still waiting for Weaviate...
    goto wait_loop
)

echo.
echo ✅ Weaviate is ready!
echo 🌐 Access: http://localhost:8080
echo 📊 Health: http://localhost:8080/v1/meta
echo.
echo To stop Weaviate: docker stop weaviate-classifier
echo To view logs: docker logs weaviate-classifier
echo.
pause
