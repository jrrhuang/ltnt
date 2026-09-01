FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Node.js 20.x for frontend build
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (torch/torchvision already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (server.py plus optional provider routers).
COPY *.py ./

# Copy the explore_map package (CREATE MAP feature) — COPY *.py only matches top-level files.
COPY explore_map/ ./explore_map/

# Copy and build frontend
COPY ltnt_frontend/ ./ltnt_frontend/
RUN cd ltnt_frontend && npm ci && npm run build

# Create generated images directory
RUN mkdir -p /app/generated

# Environment defaults
ENV HF_HOME=/workspace/huggingface
ENV CORS_ALLOW_ALL=1

EXPOSE 8001

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
