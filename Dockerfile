FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_VERBOSE=False \
    OPENCV_LOG_LEVEL=ERROR

# Встановлюємо залежності ОС
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копіюємо файл залежностей та код проекту
COPY pyproject.toml .
COPY README.md .
COPY config/ ./config/
COPY src/ ./src/
COPY main.py .

# Встановлюємо Python залежності
RUN pip3 install --no-cache-dir -e .

# Порти для мережевих API
EXPOSE 8765
EXPOSE 8080

ENTRYPOINT ["python3", "main.py", "--headless"]
CMD ["--project", "/data/project", "--source", "/data/video.mp4"]
