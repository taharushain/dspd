FROM python:3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./controller/ controller/
COPY ./models/ models/
COPY ./routes/ routes/
COPY config.json .

RUN useradd -m apiuser
USER apiuser

CMD ["uvicorn", "routes.api:app", "--host", "0.0.0.0", "--port", "8080"]