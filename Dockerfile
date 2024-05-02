
FROM registry.access.redhat.com/ubi8/ubi

WORKDIR /app/
COPY Langchain-Redis-Ingest.py /app/
COPY entrypoint.sh /app/
COPY requirements.txt /app/

RUN chmod 777 /app/ && ls -la /app/
RUN dnf install git -y && dnf install python3.11 -y && dnf install python3.11-pip -y && pip3.11 install -r requirements.txt
RUN chown 1001:0 /app/*
USER 1001

# RUN pip3.11 install -r requirements.txt


ENTRYPOINT [ "/usr/bin/bash", "/app/entrypoint.sh" ]
