import os
import json
import jsonlines
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
API_KEY = os.getenv('GEMINI')
os.environ["GOOGLE_API_KEY"] = API_KEY

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(script_dir,"..","data","master_directory","master_directory_html.jsonl"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

out_dir_chroma = os.path.abspath(os.path.join(script_dir,"..","chroma_db"))

vector_store = Chroma(
    collection_name="master_directories",
    embedding_function=embeddings,
    persist_directory= out_dir_chroma,  # Where to save data locally, remove if not necessary
)

def flush(cur, chunks, item):
    if cur["text"].strip():
        chunks.append({
            "doc_id": item["id"], "doc_title": item["title"], "url": item["url"],
            "chapter": cur["chapter"], "section": cur["section"],
            "para_no": cur["para_no"], "text": cur["text"].strip(),
        })
    cur["para_no"] = ""
    cur["text"] = ""


def create_embeddings():
    
    data = []
    with open(in_dir_data,"r") as file:
        for line in file:
            item = json.loads(line)
            data.append(item)
            

    soup = BeautifulSoup(item["text"], "html.parser")
    inner = soup.find("table", class_="td").find("td")
    

    chunks = []
    cur = {"chapter": "Preamble", "section": "", "para_no": "", "text": ""}

    for child in inner.find_all(recursive=False):
        if child.name == "hr":
            break
        text = child.get_text(" ", strip=True)
        if not text:
            continue
        first_word = text.split(" ")[0]

        span = child.find(class_="head")
        own_head = "head" in (child.get("class") or [])
        span_head = span is not None and len(span.get_text(strip=True)) > 0.8 * len(text)
        is_head = own_head or span_head
        heading_text = span.get_text(" ", strip=True) if span_head else text

        if child.name == "table":
            if text.startswith("Table of Contents"):
                continue
            rows = [" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"]))
                    for tr in child.find_all("tr")]
            cur["text"] += "\n" + "\n".join(rows)
        elif is_head and text.lower().startswith("chapter"):
            flush(cur, chunks, item)
            cur["chapter"] = text
            cur["section"] = ""
        elif is_head:
            if cur["chapter"] == "Preamble":
                continue                      # document title, skip
            flush(cur, chunks, item)
            cur["section"] = heading_text
        elif first_word[0].isdigit() and first_word.endswith("."):
            flush(cur, chunks, item)
            cur["para_no"] = first_word.rstrip(".")
            cur["text"] = text
        elif first_word.startswith("(") and not first_word.endswith(")"):
            continue                          # signature line
        else:
            cur["text"] += "\n" + text        # sub-clauses, notes, lists

    flush(cur, chunks, item)                  # don't forget the last one

    for c in chunks:
        print(c["chapter"][:18], "|", c["section"][:22], "|", c["para_no"], "|", len(c["text"]))
        print(len(chunks))
        
    
        
                
if __name__ =="__main__"  :
    create_embeddings()
        
        
        
        