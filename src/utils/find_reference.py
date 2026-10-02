import os
import json
import re
import unicodedata
import jsonlines
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
API_KEY = os.getenv('GEMINI')
os.environ["GOOGLE_API_KEY"] = API_KEY
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))


UPDATED_RE = re.compile(r"\s*\(updated as on [^)]*\)", re.IGNORECASE)

def clean(title: str) -> str:
    return UPDATED_RE.sub("", title).strip()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)   
    s = re.sub(r"\s*-\s*", "-", s)                 
    s = re.sub(r"\s+", " ", s)                     
    return s.lower().strip()


def load_titles(path: str) -> dict[str, str]:
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                t = clean(obj["title"] if isinstance(obj, dict) else obj)
                titles[norm(t)] = t
    return titles


def find_mds(text: str, titles: dict[str, str]) -> list[str]:
    t = norm(text)
    return [title for key, title in titles.items() if key in t]


def load_store(): # load chroma database
    return Chroma(
        collection_name="master_directories",
        embedding_function=embeddings,
        persist_directory=in_dir_data,
    )

def build_title_filter(title_keywords: list[str]) -> dict:
    if not title_keywords:
        return {}
    if len(title_keywords) == 1:
        return {"doc_title": {"$contains": title_keywords[0]}}
    
    return {
        "$or": [
            {"doc_title": {"$contains": kw}} for kw in title_keywords
        ]
    }


def double_hop(query,titles):
    db = load_store()
    results = db.similarity_search(
        query=query,
        k=5,
        filter=build_title_filter(titles)
    )
    return results 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(script_dir, "..", "data", "master_directory", "master_directory.jsonl")
    titles = load_titles(path)                 

    with jsonlines.open(path, mode="r") as r:
        for obj in r:
            query = obj["text"]
            for t in find_mds(query, titles):
                print(f"{obj["id"]}"," -", t)
                

