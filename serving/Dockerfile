FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY models models
COPY tokenization tokenization
COPY training training
COPY datasets datasets
COPY inference inference
COPY serving serving
ENV VLM_DEVICE=cpu
ENV VLM_CHECKPOINT=/model/model.pt
EXPOSE 8000
CMD ["uvicorn", "serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
