from kfp import dsl, compiler
from typing import List
import os
from langchain_core.documents.base import Document
from kfp import kubernetes

DB_TYPE = 'DB_TYPE'
IN_METHOD = 'IN_METHOD'
AWS_SECRET = 'aws_secret'
RAG_LLM_CM = 'rag_llm_cm'

def save_docs(docs: List[Document], out_path: dsl.OutputPath):
    import json
    with open(out_path, 'w') as f:
        json.dump(docs, f)

# S3 bucket
@dsl.component(base_image='TODO', packages_to_install=['boto3'])
def load_data_from_s3(out_data_path: dsl.OutputPath):
    import boto3
    import os
    from langchain_community.document_loaders import PyPDFLoader
    
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    bucket_name = os.environ.get('AWS_S3_BUCKET')
    
    # Set up the S3 client
    s3 = boto3.client('s3',
                      endpoint_url=endpoint_url,
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)
    
    # assume tmp not exits
    tmp_dir = 'tmp_s3'
    os.mkdir(tmp_dir)
    
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
                if not key.endwith('.pdf'):
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
                continuation_token = response['ContinuationToken']
            else:
                break
            
        except Exception as e:
            print(e)
    
    save_docs(docs, out_data_path)

    
# Repository
@dsl.component(base_image='TODO')
def load_data_from_repo(out_data_path: dsl.OutputPath):
    import subprocess
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    
    repo_url = os.getenv('REPO_URL')
    docs_location = os.getenv('REPO_DOC_LOCATION', '')
    # Clone the repo to tmp_dir
    repo_path = os.path.join('tmp_repo', 'source_repo')
    try:
        subprocess.run(['git', 'clone', repo_url, repo_path])
    except subprocess.CalledProcessError as e:
        print(f'Cloning error: {e}')
        
    docs_dir_path = os.path.join(repo_path, docs_location)
    if not os.path.exists(docs_dir_path):
        raise FileNotFoundError
    
    # Load the files
    loader = PyPDFDirectoryLoader(docs_dir_path)
    docs = loader.load()
    
    save_docs(docs, out_data_path)

# URLs
@dsl.component(base_image='TODO')
def load_data_from_urls(urls: List[str], out_dir_path: dsl.OutputPath):
    from langchain_community.document_loaders import WebBaseLoader
    
    # Load URLs
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    save_docs(docs, out_dir_path)
        
def load_docs(input_path: dsl.InputPath):
    import json
    with open(input_path, 'r') as f:
        return json.load(f)

@dsl.component(base_image='TODO')
def split_and_embed(input_docs_path: dsl.InputPath):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from vector_db.db_provider_factory import DBFactory
    # Load docs
    docs = load_docs(input_docs_path)
    
    # Split the documents
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=40,
        length_function=len,
    )
    all_splits = text_spliter.split_documents(docs)
    
    # Indexing the VectorDB
    db_type = os.getenv(DB_TYPE)
    vector_db = DBFactory().create_db_provider(db_type)
    vector_db.add_documents(all_splits)

@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def rag_llm_pipeline():
    input_method = os.getenv(IN_METHOD) #TODO Can I get it?
    
    # Load the data
    if input_method == 'repository':
        load_data_task = load_data_from_repo()
    elif input_method == 's3':
        load_data_task = load_data_from_s3()     
        kubernetes.use_secret_as_env(
            task=load_data_task,
            secret_name=AWS_SECRET, # TODO change the secret name 
            secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT'}
            )
    elif input_method == 'urls':
        load_data_task = load_data_from_urls()
    else:
        raise ValueError('IN_METHOD env variable is missing or invalid')
    load_data_task.set_caching_options(False)
    kubernetes.use_config_map_as_env(load_data_task,
                                 config_map_name=RAG_LLM_CM)
                                # TODO I assume that if I dnot spesify below we take all the config args
                                #  config_map_key_to_env={'foo': 'CM_VAR'})
    # Split embed
    split_and_embed_task = split_and_embed(load_data_task.outputs['out_dir_path'])
    split_and_embed_task.set_caching_options(False)

if __name__ == '__main__':
    compiler.Compiler.compile(
        pipeline_func=rag_llm_pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )