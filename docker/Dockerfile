FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get install build-essential -y
RUN apt-get install checkinstall -y
RUN apt-get install libssl-dev -y
RUN apt-get install git -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get install python3.6-dev -y
RUN apt-get install python3-pip -y

RUN apt-get install -y locales locales-all
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL C.UTF-8

RUN apt-get install libicu-dev -y
RUN apt-get install libpq-dev -y
RUN apt-get install locales -y
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install wheel

WORKDIR /home/app
COPY . /home/app/

RUN cd /home/app && pip3 install -r requirements.txt
WORKDIR /home/app/api/

ENTRYPOINT ./docker-entrypoint.sh