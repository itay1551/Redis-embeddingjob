from kfp import dsl, compiler
from typing import List, Dict
import os
from langchain_core.documents.base import Document
from kfp import kubernetes

RAG_LLM_CM = "config-pipeline"
RAG_LLM_SECRET = "ds-pipeline-config-llm-rag"


def save_docs(docs: List[Document], out_path: str):
    import json

    with open(out_path, "w") as f:
        json.dump(docs, f)


# S3 bucket
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=["boto3", "langchain-community", "pypdf"],
)
def load_data_from_s3(out_data_path: dsl.OutputPath()):
    import boto3
    import os
    from langchain_community.document_loaders import PyPDFLoader
    import json
    from langchain_core.load import dumpd
    import ast

    env_dict: dict = ast.literal_eval(os.getenv("S3_CONFIG"))
    for key, value in env_dict.items():
        os.environ[key] = value
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    bucket_name = os.environ.get("AWS_S3_BUCKET")
    folder_path = os.environ.get("AWS_S3_FOLDER_PATH")

    # Set up the S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    tmp_dir = "tmp_s3"
    os.makedirs(tmp_dir, exist_ok=True)

    docs = []
    continuation_token = None
    while True:
        try:
            # check for files in the bucket
            if continuation_token:
                response: dict = s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=folder_path + "/",
                    ContinuationToken=continuation_token,
                )
            else:
                response: dict = s3.list_objects_v2(
                    Bucket=bucket_name, Prefix=folder_path + "/"
                )

            for content in response.get("Contents", []):
                key = content["Key"]
                if not key.endswith(".pdf"):
                    continue
                # download file from the bucket
                local_pdf_path = os.path.join(tmp_dir, key.split("/")[-1])
                s3.download_file(bucket_name, key, local_pdf_path)

                # Load and save the path
                loader = PyPDFLoader(local_pdf_path)
                docs.append(loader.load())

                # Remove the local file
                os.remove(local_pdf_path)

            # Check for more file
            if response["IsTruncated"]:
                continuation_token = response["NextContinuationToken"]
            else:
                break

        except Exception as e:
            print(f"ERROR: {e}")
            break

    with open(out_data_path, "w") as f:
        json.dump([dumpd(doc) for doc in docs], f)


def load_env_var_dict(name: str):
    import ast

    env_dict: dict = ast.literal_eval(os.getenv(name))
    for key, value in env_dict.items():
        os.environ[key] = value


# Repository
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=["langchain-community", "gitpython", "pypdf"],
)
def load_data_from_repo(out_data_path: dsl.OutputPath()):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from git import Repo
    import os
    import json
    from langchain_core.load import dumpd
    import ast

    env_dict: dict = ast.literal_eval(os.getenv("REPO_CONFIG"))
    for key, value in env_dict.items():
        os.environ[key] = value
    # Clone the repo to tmp_dir
    repo_url = os.getenv("REPO_URL")
    docs_location = os.getenv("DOC_LOCATION")
    tmp_directory = "tmp_repo"
    repo_path = os.path.join(tmp_directory, "source_repo")

    try:
        Repo.clone_from(repo_url, repo_path)
    except Exception as e:
        print(f"Cloning error: {e}")

    docs_dir_path = os.path.join(repo_path, docs_location)
    if not os.path.exists(docs_dir_path):
        raise FileNotFoundError

    # Load the files
    loader = PyPDFDirectoryLoader(docs_dir_path)
    docs = loader.load()

    # save_docs
    with open(out_data_path, "w") as f:
        json.dump([dumpd(doc) for doc in docs], f)


# URLs
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311",
    packages_to_install=["langchain-community", "bs4"],
)
def load_data_from_urls(out_data_path: dsl.OutputPath()):
    from langchain_community.document_loaders import WebBaseLoader
    import json
    from langchain_core.load import dumpd
    import ast
    import os

    # Load URLs
    urls: List[str] = ast.literal_eval(os.getenv("URLS_CONFIG"))["URLS_ARRAY"]
    loader = WebBaseLoader(urls)
    docs = loader.load()

    # Save docs
    with open(out_data_path, "w") as f:
        json.dump([dumpd(doc) for doc in docs], f)


# @dsl.component(base_image='quay.io/ecosystem-appeng/embeddingjob:0.0.4',packages_to_install=['langchain-community', 'langchain'])
# @dsl.component(base_image='quay.io/ikatav/redis_embedding:0.0.1',packages_to_install=['langchain-community', 'langchain', 'sentence-transformers'])
# @dsl.container_component
@dsl.component(base_image="quay.io/ikatav/redis_embedding:0.0.2")
def split_and_embed(input_docs_path: dsl.InputPath()):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from vector_db.db_provider_factory import DBFactory
    import json
    from langchain_core.load import load
    import os
    import ast

    for config in ["EMBED_CONF", "DB_CONF"]:
        env_dict: dict = ast.literal_eval(os.getenv(config, "{}"))
        for key, value in env_dict.items():
            os.environ[key] = value
    # Load docs
    with open(input_docs_path, "r") as f:
        docs = json.load(f)
    docs = [load(doc) for doc in docs]
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=40,
        length_function=len,
    )
    all_splits = text_splitter.split_documents(docs)

    db_type = os.getenv("DB_TYPE")
    vector_db = DBFactory().create_db_provider(db_type)
    vector_db.add_documents(all_splits)


def split_embed_pipeline(load_data_task):
    # Define component
    split_and_embed_task = split_and_embed(
        input_docs_path=load_data_task.outputs["out_data_path"]
    ).after(load_data_task)
    # Component configurations
    split_and_embed_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        task=split_and_embed_task,
        secret_name=RAG_LLM_SECRET,
        secret_key_to_env={item: item for item in ["EMBED_CONF", "DB_CONF"]},
    )


def configuration_load_task(task, secret_config):
    kubernetes.use_secret_as_env(
        task=task,
        secret_name=RAG_LLM_SECRET,
        secret_key_to_env={secret_config: secret_config},
    )
    task.set_caching_options(False)


@dsl.pipeline(name=os.path.basename(__file__).replace(".py", ""))
def rag_llm_pipeline(
    is_load_from_repo: bool,
    is_load_from_s3: bool,
    is_load_from_urls: bool,
    TODO_must_update_secret_ds_pipeline_config_llm_rag: str = "",
):

    with dsl.If(is_load_from_repo == True):
        load_data_task = load_data_from_repo()
        configuration_load_task(load_data_task, "REPO_CONFIG")
        split_embed_pipeline(load_data_task)
    with dsl.If(is_load_from_s3 == True):
        load_data_task = load_data_from_s3()
        configuration_load_task(load_data_task, "S3_CONFIG")
        split_embed_pipeline(load_data_task)
    with dsl.If(is_load_from_urls == True):
        load_data_task = load_data_from_urls()
        configuration_load_task(load_data_task, "URLS_CONFIG")
        split_embed_pipeline(load_data_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=rag_llm_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
