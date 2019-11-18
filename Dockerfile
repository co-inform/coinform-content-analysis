FROM coinform/docker
#RUN apt-get -y update

EXPOSE 8000

COPY ./app /app/app
COPY data /app/data
COPY config2.ini /app/

COPY ./app/estimators/feature_extractor.py /app/src/

WORKDIR /app

RUN mkdir /tmp-logs

#RUN export LC_ALL=C.UTF-8
#RUN export LANG=C.UTF-8
