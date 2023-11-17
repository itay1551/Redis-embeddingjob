
FROM registry.access.redhat.com/ubi9/ubi

WORKDIR /app/
COPY rhods-doc/* /app/rhods-doc/
COPY Langchain-Redis-Ingest.py /app/
COPY requirements.txt /app/

RUN chmod 777 /app/ && ls -la /app/
RUN dnf install python3.11 -y && dnf install python3.11-pip -y && pip3.11 install -r requirements.txt


# RUN pip3.11 install -r requirements.txt

ENTRYPOINT [ "python3.11", "./Langchain-Redis-Ingest.py" ]