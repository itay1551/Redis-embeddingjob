from typing import Optional
from langchain.vectorstores.redis import Redis, RedisVectorStoreRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from vector_db.db_provider import DBProvider
import os
import redis

class RedisProvider(DBProvider):
    type = "Redis"
    url: Optional[str] = None
    index: Optional[str] = None
    schema: Optional[str] = None
    retriever: Optional[any] = None
    db: Optional[any] = None
    retriever: Optional[VectorStoreRetriever] = None
    redis_client: Optional[any] = None

    def __init__(self):
        super().__init__()
        self.url = os.getenv('REDIS_URL')
        self.index =  os.getenv('REDIS_INDEX') if os.getenv('REDIS_INDEX') else "docs"
        self.schema =  os.getenv('REDIS_SCHEMA') if os.getenv('REDIS_SCHEMA') else "redis_schema.yaml"
        if self.url is None:
            raise ValueError("REDIS_URL is not specified")

        pass
  
    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider"""
        return cls.type
    
    def get_redis_client(self):
        # Connect to Redis
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.url)
        return self.redis_client

    
    def index_exists(self) -> bool:
        # Check if index exists
        exists = False
        try:
            self.get_redis_client().ft(self.index).info()
            print("Index already exists")
            exists = True
        except Exception as e:
            print(e)
            # Create RediSearch Index
            exists = False
        return exists
    
    def add_documents(self, docs):
        if self.index_exists():
            self.db = Redis.from_existing_index(self.get_embeddings(),
                                            redis_url=self.url,
                                            index_name=self.index,
                                            schema=self.schema)

            self.db.add_documents(docs)

        else:
            self.db = Redis.from_documents(docs,
                                    self.get_embeddings(),
                                    redis_url=self.url,
                                    index_name=self.index)
            # Write the schema to a yaml file to be able to open the index later on
            print("Creating schema...")
            self.db.write_schema(self.schema)



