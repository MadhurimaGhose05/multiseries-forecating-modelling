FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the foundation-model weights into the image so serving is offline and
# reproducible. Set CHRONOS_REVISION to the exact commit hash you validated.
ARG PRELOAD_CHRONOS=1
ARG CHRONOS_REVISION=main
ENV CHRONOS_REVISION=${CHRONOS_REVISION}
RUN if [ "$PRELOAD_CHRONOS" = "1" ]; then \
      python -c "from chronos import Chronos2Pipeline; Chronos2Pipeline.from_pretrained('amazon/chronos-2', revision='${CHRONOS_REVISION}')"; \
    fi
ENV HF_HUB_OFFLINE=${PRELOAD_CHRONOS}

COPY app app
COPY models models
COPY training training
COPY demo demo
COPY train.py .
COPY artifacts artifacts

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
