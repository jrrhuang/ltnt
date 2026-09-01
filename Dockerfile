FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ server/
COPY run.sh .
ENV LTNT_MODELS=/models PORT=8001 LTNT_HOST=0.0.0.0
EXPOSE 8001
CMD ["bash", "run.sh"]
