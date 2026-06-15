FROM python:3.12-slim

ARG REQUIREMENTS_FILE=requirements.txt

# libgl1 replaces deprecated libgl1-mesa-glx on Debian Bookworm+
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

COPY requirements.txt requirements-gpu.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 --retries 10 -r "${REQUIREMENTS_FILE}"

COPY ./worker ./worker

CMD ["python", "worker/main.py"]
