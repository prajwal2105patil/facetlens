# Reproducible environment for the preprocessing, retrieval and evaluation
# stages. The scoring model itself is NOT baked in: qwen2.5:7b-instruct is a
# 4.7GB artefact and belongs in an Ollama daemon, not in an application image.
#
#   docker build -t facetlens .
#
#   # Offline: reruns the audit, the tests and the benchmark from the
#   # committed LLM cache. No model and no network needed.
#   docker run --rm facetlens
#
#   # Live scoring against an Ollama daemon on the host:
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     facetlens python -m src.pipeline score --text "I led a team of five engineers."

FROM python:3.12-slim

WORKDIR /app

# Dependencies first so the layer caches across source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so retrieval needs no network at
# run time. This is the same offline path the application uses.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

# Default: prove the repository reproduces without a model.
CMD ["sh", "-c", "python -m pytest tests/ -q && python -m src.pipeline enrich && python -m src.pipeline benchmark"]
