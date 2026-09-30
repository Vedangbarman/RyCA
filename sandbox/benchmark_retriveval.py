import os
import time
import json
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))
in_dir_test = os.path.abspath(os.path.join(script_dir,"..","data","benchmark","data","test_queries.jsonl",))
out_dir_benchmark_results = os.path.abspath(os.path.join(script_dir,"..","data","benchmark","results",".jsonl",))
api_key = os.getenv("GEMINI")
if not api_key:
    raise RuntimeError("GEMINI key not found - check your .env file")
os.environ["GOOGLE_API_KEY"] = api_key

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# def vector_store_distance(type):
#     if type == "cosine":
#         vector_store = Chroma(
#                 collection_name="master_directories",
#                 embedding_function=embeddings,
#                 persist_directory=in_dir_data,
#                 collection_metadata={"hnsw:space": "cosine"}
#         )
#         results = vector_store.similarity_search(query)
#         return results
        
#     elif type == "l2": #Euclidean
#         vector_store = Chroma(
#                         collection_name="master_directories",
#                         embedding_function=embeddings,
#                         persist_directory=in_dir_data,
#                         collection_metadata={"hnsw:space": "lr2"}
#                 )
#         results = vector_store.similarity_search(query)
#         return results
        
#     elif type == "ip": #dot product
#         vector_store = Chroma(
#                                 collection_name="master_directories",
#                                 embedding_function=embeddings,
#                                 persist_directory=in_dir_data,
#                                 collection_metadata={"hnsw:space": "lr2"}
#                 )
#         results = vector_store.similarity_search(query)
#         return results
    
#     elif type == "mmr":
#         vector_store = Chroma(
#                                         collection_name="master_directories",
#                                         embedding_function=embeddings,
#                                         persist_directory=in_dir_data,
#                                         collection_metadata={"hnsw:space": "lr2"}
#                         )
#         results = vector_store.similarity_search(query)
#         return results
        

    
    
#goal is bench mark dense vector search (cosine similarity, Euclidean distance), lexical (BM25,TF-IDF), Hybrid, MMR 

# with open("data.jsonl", "r", encoding="utf-8") as file:
#     for index, line in enumerate(file):
#         query = json.loads(line)
query = "What is the defination of ‘qualifying assets’ of NBFC-MFIs"
vector = embeddings.embed_query(query)
magnitude = np.linalg.norm(vector)

print(f"Vector length (dimensions): {len(vector)}")
print(f"Vector magnitude: {magnitude:.6f}")








