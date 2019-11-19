FROM coinform/docker
#RUN apt-get -y update

EXPOSE 8000

COPY src /app
COPY data /app/data
COPY config.ini /app

# COPY src/ /app/src/

WORKDIR /app

RUN mkdir /tmp-logs

#RUN export LC_ALL=C.UTF-8
#RUN export LANG=C.UTF-8
