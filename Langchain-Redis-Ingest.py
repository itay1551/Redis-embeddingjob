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


from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import os
from vector_db.db_provider_factory import DBFactory

redis_url = os.getenv('REDIS_URL')
doc_folder = os.getenv('DOC_LOCATION')
temp_folder = os.getenv('TEMP_DIR')
index_name = "docs"

type = os.getenv('DB_TYPE') if os.getenv('DB_TYPE') else "REDIS"
db_provider = DBFactory().create_db_provider(type)

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


db_provider.add_documents(all_splits)

# ## Ingesting new documents
# #### Example with Web pages

loader = WebBaseLoader(["https://ai-on-openshift.io/getting-started/openshift/",
                        "https://ai-on-openshift.io/getting-started/opendatahub/",
                        "https://ai-on-openshift.io/getting-started/openshift-ai/",
                        "https://ai-on-openshift.io/odh-rhoai/configuration/",
                        "https://ai-on-openshift.io/odh-rhoai/custom-notebooks/",
                        "https://ai-on-openshift.io/odh-rhoai/nvidia-gpus/",
                        "https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton/",
                        "https://ai-on-openshift.io/odh-rhoai/openshift-group-management/",
                        "https://ai-on-openshift.io/tools-and-applications/minio/minio/"
                       ])


print(">>>>Loading Documents .....")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(data)
print(">>>Adding new documents from Web... ")

db_provider.add_documents(all_splits)