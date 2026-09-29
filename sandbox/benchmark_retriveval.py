import os
import time
import json
import argparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))

api_key = os.getenv("GEMINI")
if not api_key:
    raise RuntimeError("GEMINI key not found - check your .env file")
os.environ["GOOGLE_API_KEY"] = api_key

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(
        collection_name="master_directories",
        embedding_function=embeddings,
        persist_directory=in_dir_data,
)


query = "What is the defination of ‘qualifying assets’ of NBFC-MFIs"

results = vector_store.similarity_search(query)

print(results)