# SECURITY: Python 3.8 is EOL (October 2024) and pins gradio below the CVE fix
# line. Base image bumped to 3.11-slim so pip can resolve gradio>=4.44.1 which
# includes the arbitrary-file-read fixes exercised by demo/gradio_demo.py.
FROM python:3.11-slim

# NOTE: GRADIO_SERVER_NAME defaults to 0.0.0.0 so the container can serve on
# its published port. Only publish that port on a trusted network or behind
# an authenticating reverse proxy — the Gradio demo has no built-in auth.
ARG GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME}

ARG GRADIO_SERVER_PORT="80"
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

RUN apt-get update && \
    apt-get install -y --no-install-recommends  \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN rm -rf /tmp/*

RUN pip install --no-cache-dir PytorchWildlife

EXPOSE 80

