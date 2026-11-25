FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    curl \
    ca-certificates \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY . .

RUN mkdir -p /app/data/models && chmod -R 777 /app/data \
    && curl -fSL "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf" \
       -o /app/data/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf

ENV LOCAL_LLM_MODEL_PATH=/app/data/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
ENV LOCAL_LLM_THREADS=2

EXPOSE 7860

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
