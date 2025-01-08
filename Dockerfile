
FROM registry.access.redhat.com/ubi9/python-311

USER root
WORKDIR /app/
COPY vector_db /app/vector_db
COPY Langchain-Redis-Ingest.py /app/
COPY redis_schema.yaml /app/
COPY entrypoint.sh /app/
COPY requirements.txt /app/

RUN chmod -R 777 /app/ && ls -la /app/
RUN dnf install git -y
RUN pip3 install -r requirements.txt
# RUN chown 1001:0 /app/*
# USER 1001

# RUN pip3.11 install -r requirements.txt
