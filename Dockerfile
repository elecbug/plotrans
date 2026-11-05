FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[ui,ocr]"
COPY src/ ./src/

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/paper_localizer/web.py", "--server.address=0.0.0.0"]
