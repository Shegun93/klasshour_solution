from Retrieval import Retrieval
from langchain_ollama.llms import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import torch
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from Pincone import Pincone_vectorStore
from sentence_transformers import SentenceTransformer
import warnings
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


Device = "cuda" if torch.cuda.is_available else "cpu"
index_name="klasshour"
index= Pincone_vectorStore(index_name=index_name)
Embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2", model_kwargs= {"device":Device})
vector_store = PineconeVectorStore(embedding=Embeddings,index=index)
retrieval_obj = Retrieval(device=Device, index=index,Embeddings=Embeddings,vector_store=vector_store)
retrieval, prompt = retrieval_obj.get_retrieval()


parser = StrOutputParser()
model = OllamaLLM(model="llama2:7b",
                  temperature=0.7)
chain = (
    {
        "context": itemgetter("question") | retrieval,
        "question": itemgetter("question"),
    }
    | prompt
    | model
)


result = chain.invoke({"question": "What is FUTA curriculum"})
clean_result = parser.parse(result)
print(clean_result)

