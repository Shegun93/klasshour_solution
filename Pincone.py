from pinecone import pinecone
import os
from langchain_pinecone import PineconeSparseVectorStore
from pinecone import Pinecone
from  dotenv import load_dotenv
load_dotenv()

def Pincone_vectorStore(index_name):
    vector_db = os.getenv("Pinecone_api_key")
    pc = Pinecone(api_key=vector_db)
    index = pc.Index(index_name)
    print(f'Successfully connect to {index}')
    return index





