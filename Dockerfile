FROM python:latest
RUN apt-get -y update

EXPOSE 8000

ADD ./app /app/app
COPY requirements.txt /app/
COPY data /app/
COPY config.ini /app/

WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/erikavaris/tokenizer.git

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app.server:app"]
