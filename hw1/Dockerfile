FROM python:3.7

COPY ./fastai/ .

RUN pip install -r requirements.txt

RUN useradd -m apiuser
USER apiuser

CMD ["uvicorn", "routes.api:app", "--host", "0.0.0.0", "--port", "8080"]