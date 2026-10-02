import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
API_KEY = os.getenv('GEMINI')
os.environ["GOOGLE_API_KEY"] = API_KEY
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))

def load_store(): # load chroma database
    return Chroma(
        collection_name="master_directories",
        embedding_function=embeddings,
        persist_directory=in_dir_data,
    )

if __name__ == "__main__":
    db = load_store()
    
    # Access the underlying Chroma collection or use the LangChain `.get()` method
    sample = db._collection.get(limit=5, include=["metadatas"])

    print(sample["metadatas"])