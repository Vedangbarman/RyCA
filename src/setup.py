import os 
import re
import json
import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(os.path.dirname(script_dir,"..","data","setup.txt")))
out_dir_notifications = in_dir_data = os.path.abspath(os.path.join(os.path.dirname(script_dir,"..","data","notification","notification.jsonl")))
os.makedirs(out_dir_notifications,exist_ok = True)
out_dir_md = out_dir_notifications = in_dir_data = os.path.abspath(os.path.join(script_dir,"..","data","md_historical","md.jsonl"))
os.makedirs(out_dir_md,exist_ok = True)
out_dir_db = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))

#env variables ( kind of of ) and also gatorade

SOURCE_DIRECTORIES = "directories"
SOURCE_NOTIFICATIONS = "notifications"

max_char = 1500
overlap = 150
batch_size = 10
retries = 8
max_len = 1024    
encode_batch_size = 4    
notification_batch_size = 2  

CHAPTER_RE = re.compile(r"^(chapter|part)\s*[-–]?\s*([ivxlc]+|[a-z]|\d+)\b", re.I)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_char,
    chunk_overlap=overlap,
    separators=["\n", ". ", " ", ""],
)

model_path = os.path.abspath(os.path.join(script_dir, "..", "model", "embeddinggemma-2"))

def get_vector_store(device="cuda"):
    # Pass the absolute path directly to HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},  # "cuda" or "cpu"
        encode_kwargs={"normalize_embeddings": True, "batch_size": encode_batch_size},
    )
    # Without this cap the model pads every batch to its longest text, which is what ran out of memory
    embeddings._client.max_seq_length = max_len
    print(f"Embedding with max_seq_length={max_len}, batch_size={encode_batch_size}")

    return Chroma(
        collection_name="master_directories",
        embedding_function=embeddings,
        persist_directory=out_dir_db,
    )

def flush(cur,chunks, item):
    text = cur["text"].strip()
    if text:
        chunks.append({
            "doc_id":str(item["id"]),
            "doc_title":item.get("title") or"",
            "url": item.get("url") or "",
            "chapter" : cur["chapter"],
            "section" : cur["section"],
            "para_no" : cur["para_no"],
            "text": text,
        })
    cur["para_no"] = ""
    cur["text"] = ""

def parse_document(item):
    soup = BeautifulSoup(item["text"],"html.parser")
    table = soup.find("table", class_="td")
    if table:
        inner = table.find("td")
    else:
        None
    
    if inner is None:
        print(f"[skip] no <table class='td'><td> found in document {item.get('id')}")
        return []

    chunks = []
    cur = {"chapter" : "Preamble", "section" : "", "para_no" : "", "text" : ""}
    
    kids = inner.find_all(recursive = False)
    hrs = [i for i, k in enumerate(kids) if k.name == "hr"]
    
    if hrs:
        kids = kids[:hrs[-1]]
    
    for child in kids:
        if child.name =="hr":
            continue
        
        text = child.get_text("",strip=True)
        if not text:
            continue
        first_word = text.split(" ")[0]
        
        span = child.find(class_="head")
        own_head = "head" in (child.get("class") or [])
        span_head = span is not None and len(span.get_text(strip=True)) > 0.8 * len(text)
        is_head = own_head or span_head
        heading_text = span.get_text(" ", strip=True) if span_head else text
        
        
            

def setup():
    with open(in_dir_data, "r", encoding="utf-8") as file:
        for i,line in enumerate(file):
            if i<45:
                url = f"https://rbi.org.in/scripts/BS_ViewMasDirections.aspx?id={line}"
                resp = requests.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, features="html.parser")                  
                title = soup.find('b').get_text(strip=True)
                content = soup.find('tr', class_='tablecontent2')
                for junk in content.find_all(["script", "style", "input"]):
                    junk.decompose()
                clean_text = str(content).replace("\r", "")
                clean_data = {"id": id, "title": title, "category": "NBFC", "url": url, "text": clean_text}
                with open(out_dir_notifications,"a",encoding = "utf-8") as f:
                    f.write(json.dumps(clean_data) + "\n")
            
            else:
                url = f"https://rbi.org.in/scripts/NotificationUser.aspx?Id={id}"
                resp = requests.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, features="html.parser")
                title = soup.find('b').get_text(strip=True)
                content = soup.find('tr', class_='tablecontent2')
                for junk in content.find_all(["script", "style", "input"]):
                    junk.decompose()
                clean_text = content.get_text(separator="\n", strip=True).replace("\r", "")
                               
                clean_data = {"id": id, "title": title, "category": "NBFC", "url": url, "text": clean_text}
                               
                file_path = os.path.abspath(os.path.join(out_dir_md,f"notification.jsonl"))
                with open(file_path,"a",encoding = "utf-8") as f:
                    f.write(json.dumps(clean_data) + "\n")
            