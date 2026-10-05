# Inference image for BlastoGrade AI (CPU). For GPU serving, swap the base image
# for a CUDA runtime and install the matching torch wheel.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY models_store ./models_store

RUN useradd --create-home grader && chown -R grader /app
USER grader
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "blastograde.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
