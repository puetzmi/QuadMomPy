FROM python:3.10.6-slim

COPY . /quadmompy
WORKDIR /quadmompy
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "bash" ]
