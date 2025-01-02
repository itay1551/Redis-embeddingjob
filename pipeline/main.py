from kfp import dsl, compiler
from typing import List
import os
from langchain_core.documents.base import Document
from kfp import kubernetes

RAG_LLM_CM = 'config-pipeline'

def save_docs(docs: List[Document], out_path: str):
    import json
    with open(out_path, 'w') as f:
        json.dump(docs, f)

# S3 bucket
@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311', packages_to_install=['boto3', 'langchain-community'])
def load_data_from_s3(out_data_path: dsl.OutputPath()):
    import boto3
    import os
    from langchain_community.document_loaders import PyPDFLoader
    import json
    from langchain_core.load import dumpd
    
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    bucket_name = os.environ.get('AWS_S3_BUCKET')
    
    # Set up the S3 client
    s3 = boto3.client('s3',
                      endpoint_url=endpoint_url,
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)
    
    tmp_dir = 'tmp_s3'
    os.makedirs(tmp_dir, exist_ok=True)
    
    docs = []
    continuation_token = None
    while True:
        try:
            # check for files in the bucket
            if continuation_token:
                response: dict= s3.list_objects_v2(bucket=bucket_name, ContinuationToken=continuation_token)
            else:
                response: dict= s3.list_objects_v2(bucket=bucket_name)
            
            
            for content in response.get('Contents', []):
                key = content['key']
                if not key.endswith('.pdf'):
                    continue
                
                # download file from the bucket
                local_pdf_path = os.path.join(tmp_dir, key)
                s3.download_file(bucket_name, key, local_pdf_path)
                
                # Load and save the path
                loader = PyPDFLoader(local_pdf_path)
                docs.append(loader.load())
                
                # Remove the local file
                os.remove(local_pdf_path)

            
            # Check for more file
            if response['IsTruncated']:
                continuation_token = response['NextContinuationToken']
            else:
                break
            
        except Exception as e:
            print(e)
    
    with open(out_data_path, 'w') as f:
        json.dump([dumpd(doc) for doc in docs], f)
        
    
# Repository
@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311', packages_to_install=['langchain-community', 'gitpython', 'pypdf'])
def load_data_from_repo(repo_url: str, docs_location: str, out_data_path: dsl.OutputPath()):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from git import Repo
    import os
    import json
    from langchain_core.load import dumpd
    # Clone the repo to tmp_dir
    tmp_directory = 'tmp_repo'
    repo_path = os.path.join(tmp_directory, 'source_repo')
    
    try:
        Repo.clone_from(repo_url, repo_path)
    except Exception as e:
        print(f'Cloning error: {e}')

    docs_dir_path = os.path.join(repo_path, docs_location)
    if not os.path.exists(docs_dir_path):
        raise FileNotFoundError
    
    # Load the files
    loader = PyPDFDirectoryLoader(docs_dir_path)
    docs = loader.load()
    
    # save_docs
    with open(out_data_path, 'w') as f:
            json.dump([dumpd(doc) for doc in docs], f)

# URLs
@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311', packages_to_install=['langchain-community', 'bs4'])
def load_data_from_urls(urls: List[str], out_data_path: dsl.OutputPath()):
    from langchain_community.document_loaders import WebBaseLoader
    import json
    from langchain_core.load import dumpd
        
    # Load URLs
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    # Save docs
    with open(out_data_path, 'w') as f:
            json.dump([dumpd(doc) for doc in docs], f)

@dsl.component(base_image='registry.access.redhat.com/ubi9/python-311', packages_to_install=['langchain-community', 'langchain'])
def split_and_embed(input_docs_path: dsl.InputPath()):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from vector_db.db_provider_factory import DBFactory
    import json
    from langchain_core.load import load
    import os
    
    # Load docs
    with open(input_docs_path, 'r') as f:
        docs = json.load(f)
    docs = [load(doc) for doc in docs]
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=40,
        length_function=len,
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Indexing the VectorDB
    # TODO this is disgusting need to fix
    pgvector_url= os.getenv('PGVECTOR_URL')
    os.environ['PGVECTOR_URL'] = pgvector_url.replace('$(DB_PASS)', os.getenv('DB_PASS'))
    
    db_type = os.getenv('DB_TYPE')
    vector_db = DBFactory().create_db_provider(db_type)
    vector_db.add_documents(all_splits)

def split_embed_pipeline(load_data_task):
    cm_dict = {item: item for item in ['NAMESPACE', 'DB_TYPE', 'PGVECTOR_COLLECTION_NAME', 'PGVECTOR_URL', 'TRANSFORMERS_CACHE']}
    
    # Define component
    split_and_embed_task = split_and_embed(input_docs_path=load_data_task.outputs['out_data_path']).after(load_data_task)
    # Component configurations
    split_and_embed_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        task=split_and_embed_task,
        secret_name='vectordb', 
        secret_key_to_env={'password': 'DB_PASS'}
        )
    kubernetes.use_config_map_as_env(split_and_embed_task, 
                                 config_map_name=RAG_LLM_CM,
                                 config_map_key_to_env=cm_dict)
        
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def rag_llm_pipeline(input_method: str, repo_url: str, docs_location: str, urls: List[str]):
    
    # Load the data
    with dsl.If(input_method == 'repository'):
        load_data_task = load_data_from_repo(repo_url=repo_url, docs_location=docs_location)
        load_data_task.set_caching_options(False)
        # Split embed
        split_embed_pipeline(load_data_task)
    with dsl.Elif(input_method == 's3'):
        load_data_task = load_data_from_s3()
        kubernetes.use_secret_as_env(
        task=load_data_task,
        secret_name='vectordb', # TODO which secret?
        secret_key_to_env={item: item for item in ['AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_S3_ENDPOINT','AWS_S3_BUCKET',]}
        )
        load_data_task.set_caching_options(False)
        # Split embed
        split_embed_pipeline(load_data_task)
    with dsl.Elif(input_method == 'urls'):
        load_data_task = load_data_from_urls(urls=urls)
        load_data_task.set_caching_options(False)
        # Split embed
        split_embed_pipeline(load_data_task)

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=rag_llm_pipeline,
        package_path=__file__.replace('.py', '.yaml'),
        
    )