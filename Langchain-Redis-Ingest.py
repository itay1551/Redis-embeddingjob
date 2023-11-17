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
import os

# In[17]:

redis_url = os.getenv('REDIS_URL')
index_name = "docs"


# #### Imports

# In[18]:

# ## Initial index creation and document ingestion

# #### Document loading from a folder containing PDFs

# In[19]:


pdf_folder_path = 'rhods-doc'

loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()


# #### Split documents into chunks with some overlap



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)


# #### Create the index and ingest the documents

# In[21]:


embeddings = HuggingFaceEmbeddings()
rds = Redis.from_documents(all_splits,
                           embeddings,
                           redis_url=redis_url,
                           index_name=index_name)


# #### Write the schema to a yaml file to be able to open the index later on

# In[22]:

print("Creating schema...")
rds.write_schema("redis_schema.yaml")


# ## Ingesting new documents

# #### Example with Web pages

# In[23]:


from langchain.document_loaders import WebBaseLoader


# In[24]:


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


# In[25]:


data = loader.load()


# In[26]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(data)


# In[27]:


embeddings = HuggingFaceEmbeddings()
rds = Redis.from_existing_index(embeddings,
                                redis_url=redis_url,
                                index_name=index_name,
                                schema="redis_schema.yaml")


# In[28]:


rds.add_documents(all_splits)