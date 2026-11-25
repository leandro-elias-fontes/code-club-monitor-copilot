FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Precompile <bits/stdc++.h> once at build time: student C++ compiles become ~5-10x faster
RUN H=$(find /usr/include -path '*bits/stdc++.h' | head -n 1) \
    && if [ -n "$H" ]; then g++ -std=c++17 -O0 -x c++-header "$H" -o "$H.gch" || true; fi

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data && chmod -R 777 /app/data

# Low-CPU free tier tuning: skip redundant syntax pass and heavy sanitizer builds
ENV GCC_SYNTAX_CHECK=0
ENV SANDBOX_SANITIZERS=0
ENV SANDBOX_COMPILE_TIMEOUT=25

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-7860}"]
