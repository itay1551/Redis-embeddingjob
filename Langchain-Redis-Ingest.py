#!/usr/bin/env python
# coding: utf-8

# ## Creating an index and populating it with documents using Redis
# 
# Simple example on how to ingest PDF documents, then web pages content into a Redis VectorStore.
# 
# Requirements:
# - A Redis cluster
# - A Redis database with at least 2GB of memory (to match with the initial index cap)

# ### Base parameters, the Redis info


from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import WebBaseLoader
import os

redis_url = os.getenv('REDIS_URL')
doc_folder = os.getenv('DOC_LOCATION')
temp_folder = os.getenv('TEMP_DIR')
index_name = "docs"


# #### Imports

# ## Initial index creation and document ingestion

# #### Document loading from a folder containing PDFs

pdf_folder_path = temp_folder+'/source_repo/'+doc_folder
print('PDF folder:',pdf_folder_path)
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()
#exit()

# #### Split documents into chunks with some overlap
print(">>>>Document splitting .....")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)

print(">>>>Creating index .....")
# #### Create the index and ingest the documents
embeddings = HuggingFaceEmbeddings()
rds = Redis.from_documents(all_splits,
                           embeddings,
                           redis_url=redis_url,
                           index_name=index_name)


# #### Write the schema to a yaml file to be able to open the index later on
print("Creating schema...")
rds.write_schema("redis_schema.yaml")
# ## Ingesting new documents
# #### Example with Web pages

loader = WebBaseLoader(["https://ai-on-openshift.io/getting-started/openshift/",
                        "https://ai-on-openshift.io/getting-started/opendatahub/",
                        "https://ai-on-openshift.io/getting-started/openshift-data-science/",
                        "https://ai-on-openshift.io/odh-rhods/configuration/",
                        "https://ai-on-openshift.io/odh-rhods/custom-notebooks/",
                        "https://ai-on-openshift.io/odh-rhods/nvidia-gpus/",
                        "https://ai-on-openshift.io/odh-rhods/custom-runtime-triton/",
                        "https://ai-on-openshift.io/odh-rhods/openshift-group-management/",
                        "https://ai-on-openshift.io/tools-and-applications/minio/minio/"
                       ])


print(">>>>Loading Documents .....")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(data)
print(">>>Adding new documents from Web... ")
embeddings = HuggingFaceEmbeddings()
rds = Redis.from_existing_index(embeddings,
                                redis_url=redis_url,
                                index_name=index_name,
                                schema="redis_schema.yaml")

rds.add_documents(all_splits)
