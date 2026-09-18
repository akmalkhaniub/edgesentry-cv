# EdgeSentry — Python + OpenCV edge safety monitor
FROM python:3.12-slim

# OpenCV headless needs a couple of shared libs even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY edgesentry ./edgesentry
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Default: run the offline synthetic demo. Override CMD to point at a stream:
#   docker run edgesentry edgesentry --source rtsp://... --zone "360,320;620,320;620,520;360,520"
ENTRYPOINT ["edgesentry"]
CMD ["--demo"]
