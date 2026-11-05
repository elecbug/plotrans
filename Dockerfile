FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c 'import subprocess, sys, tomllib; \
    config = tomllib.load(open("pyproject.toml", "rb")); \
    dependencies = config["build-system"]["requires"] + ["wheel"] + config["project"]["dependencies"] + config["project"]["optional-dependencies"]["ui"] + config["project"]["optional-dependencies"]["ocr"]; \
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-compile", *dependencies])'
COPY src/ ./src/
RUN pip install --disable-pip-version-check --no-cache-dir --no-compile --no-index --no-deps --no-build-isolation .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/paper_localizer/web.py", "--server.address=0.0.0.0"]
