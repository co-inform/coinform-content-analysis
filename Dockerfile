FROM python:latest

EXPOSE 8000

ADD ./app /app/app
COPY requirements.txt /app/
COPY data /app/
<<<<<<< HEAD
=======
COPY server.py /app/
COPY settings.py /app/
>>>>>>> 78af593899f0e87a15379efd328560c6de832f5a
COPY config.ini /app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "server:app"]
