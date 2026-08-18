FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/omeify
COPY pyproject.toml README.md LICENSE ./
COPY omeify ./omeify
RUN python -m pip install --upgrade pip && python -m pip install .

ENTRYPOINT ["omeify"]
CMD ["--help"]
