FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# GPU acceleration support (VAAPI / QSV)
# /dev/dri is mounted at runtime for hardware decode

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p cache/thumbnails images

EXPOSE 5000

ENV PORT=5000
ENV IMAGES_DIR=/app/images
ENV DATA_DIR=/app

CMD ["python3", "app.py"]
