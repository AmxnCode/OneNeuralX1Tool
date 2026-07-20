FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY liquid_foundation_model/ liquid_foundation_model/
COPY runpod/ runpod/
COPY data/combined_training.jsonl /model/combined_training.jsonl

ENV MODEL_PATH=/model/x1_26m_final.pt
ENV DATA_PATH=/model/combined_training.jsonl

CMD ["python", "-u", "runpod/handler.py"]
