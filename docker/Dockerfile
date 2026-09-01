# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

ARG ACTIVE_MODEL=SeizureTransformer
ENV ACTIVE_MODEL=${ACTIVE_MODEL}

RUN python3 -m pip install --no-cache-dir pyedflib

COPY ExperimentsUtils /app/ExperimentsUtils
COPY Model/${ACTIVE_MODEL}/wu_2025 /app/Model/${ACTIVE_MODEL}/wu_2025
COPY Model/active_model /app/Model/active_model

RUN python3 -m pip install --no-cache-dir \
    /app/ExperimentsUtils \
    /app/Model/${ACTIVE_MODEL}/wu_2025

USER appuser

VOLUME ["/data", "/output"]

ENV INPUT=""
ENV OUTPUT=""

CMD python3 -m experiments_utils "/data/$INPUT" "/output/$OUTPUT"
