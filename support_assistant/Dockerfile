FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    sentence-transformers \
    chromadb \
    langgraph \
    fastapi \
    uvicorn \
    pydantic

EXPOSE 7860

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]