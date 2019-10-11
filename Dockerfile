FROM python:latest

EXPOSE 8000

ADD ./app /app/app
COPY requirements.txt /app/
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app.server:app"]
