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
COPY Model/${ACTIVE_MODEL} /app/Model/${ACTIVE_MODEL}
COPY Model/active_model /app/Model/active_model

# Install the shared pipeline plus every package (folder with a pyproject.toml) of the active model.
RUN python3 -m pip install --no-cache-dir /app/ExperimentsUtils \
    && for p in /app/Model/${ACTIVE_MODEL}/*/pyproject.toml; do \
        python3 -m pip install --no-cache-dir "$(dirname "$p")" || exit 1; \
    done

USER appuser

VOLUME ["/data", "/output"]

ENV INPUT=""
ENV OUTPUT=""

CMD python3 -m experiments_utils "/data/$INPUT" "/output/$OUTPUT"
