FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data && chmod -R 777 /app/data

EXPOSE 7860

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
