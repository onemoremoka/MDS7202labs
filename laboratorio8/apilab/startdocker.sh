docker run -d -p 8000:8000 \
    -v ${PWD}/data:/app/data \
    --name apilab \
    apilab:latest