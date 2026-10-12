import os
import re
import time
import json
import argparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

script_dir = os.path.dirname(os.path.realpath(__file__))
IN_DIR_DATA = os.path.abspath(
    os.path.join(script_dir, "..", "data", "master_directory", "master_directory_html.jsonl")
)
OUT_DIR_CHROMA = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))

MAX_CHARS = 1500    
OVERLAP = 150       
BATCH_SIZE = 10               
PAUSE_BETWEEN_BATCHES = 0
MAX_RETRIES = 8               

CHAPTER_RE = re.compile(r"^(chapter|part)\s*[-–]?\s*([ivxlc]+|[a-z]|\d+)\b", re.I)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHARS,
    chunk_overlap=OVERLAP,
    separators=["\n", ". ", " ", ""],
)

model_path = os.path.abspath(os.path.join(script_dir, "..", "model","embeddinggemma-2"))

def get_vector_store():
    # Pass the absolute path directly to HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cuda"},  # or "cpu"
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return Chroma(
        collection_name="master_directories",
        embedding_function=embeddings,
        persist_directory=OUT_DIR_CHROMA,
    )


def flush(cur, chunks, item):
    """Save the paragraph being built (if any) and reset it."""
    text = cur["text"].strip()
    if text:
        chunks.append({
            "doc_id": str(item["id"]),
            "doc_title": item.get("title") or "",
            "url": item.get("url") or "",
            "chapter": cur["chapter"],
            "section": cur["section"],
            "para_no": cur["para_no"],
            "text": text,
        })
    cur["para_no"] = ""
    cur["text"] = ""


def parse_document(item):
    """Split one HTML document into paragraph-level chunks."""
    soup = BeautifulSoup(item["text"], "html.parser")
    table = soup.find("table", class_="td")
    inner = table.find("td") if table else None
    if inner is None:
        print(f"[skip] no <table class='td'><td> found in document {item.get('id')}")
        return []

    chunks = []
    cur = {"chapter": "Preamble", "section": "", "para_no": "", "text": ""}

    kids = inner.find_all(recursive=False)
    hrs = [i for i, k in enumerate(kids) if k.name == "hr"]
    if hrs:
        kids = kids[:hrs[-1]]  # stop at the LAST <hr> (footer / superseded-circulars annex)

    for child in kids:
        if child.name == "hr":
            continue  # <hr> in the middle, e.g. between cover circular and the Directions

        # `text` must be defined BEFORE anything below uses it
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
            if text.lower().startswith(("table of contents", "index")):
                continue
            rows = [
                " | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"]))
                for tr in child.find_all("tr")
            ]
            cur["text"] += "\n" + "\n".join(rows)

        elif CHAPTER_RE.match(text) and (is_head or len(text) < 80):
            flush(cur, chunks, item)
            cur["chapter"] = text
            cur["section"] = ""

        elif is_head:
            if cur["chapter"] == "Preamble":
                continue  # document title, skip
            flush(cur, chunks, item)
            cur["section"] = heading_text

        elif first_word[0].isdigit() and first_word.endswith("."):
            flush(cur, chunks, item)  # new numbered paragraph
            cur["para_no"] = first_word.rstrip(".")
            cur["text"] = text

        elif first_word.startswith("(") and not first_word.endswith(")"):
            continue  # signature line

        else:
            cur["text"] += "\n" + text  # sub-clauses, notes, lists

    flush(cur, chunks, item)  # don't forget the last one
    return chunks


def chunks_to_documents(chunks):
    """Turn parsed chunks into LangChain Documents (+ stable ids), splitting long ones."""
    docs, ids = [], []
    for n, c in enumerate(chunks):
        header_parts = [
            c["doc_title"],
            c["chapter"],
            c["section"],
            f"Para {c['para_no']}" if c["para_no"] else "",
        ]
        header = " > ".join(p for p in header_parts if p)

        for k, piece in enumerate(splitter.split_text(c["text"])):
            docs.append(Document(
                page_content=f"{header}\n\n{piece}",
                metadata={  # Chroma metadata must be str / int / float / bool
                    "doc_id": c["doc_id"],
                    "doc_title": c["doc_title"],
                    "url": c["url"],
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "para_no": c["para_no"],
                    "part": k,
                },
            ))
            ids.append(f"{c['doc_id']}::{n}::{k}")  # same id on re-run -> upsert, no duplicates
    return docs, ids


def add_in_batches(vector_store, docs, ids):
    # Resume support: skip chunks that are already in Chroma
    existing = set(vector_store.get(include=[])["ids"])
    todo = [(d, i) for d, i in zip(docs, ids) if i not in existing]
    print(f"{len(existing)} chunks already stored, {len(todo)} left to embed")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        batch_docs = [d for d, _ in batch]
        batch_ids = [i for _, i in batch]

        for attempt in range(MAX_RETRIES):
            try:
                vector_store.add_documents(batch_docs, ids=batch_ids)
                break
            except Exception as e:
                if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                    raise  # a real error, not rate limiting - don't hide it
                wait = min(15 * 2 ** attempt, 120)
                print(f"  rate limited; waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
        else:
            raise RuntimeError(
                "Still rate limited after all retries. Progress so far is saved - "
                "re-run the script later to resume. If this keeps happening, you've "
                "probably hit the daily quota."
            )

        print(f"  embedded {start + len(batch)}/{len(todo)}")
        time.sleep(PAUSE_BETWEEN_BATCHES)


def create_embeddings(dry_run=False):
    with open(IN_DIR_DATA, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data)} documents")

    all_docs, all_ids = [], []
    for item in data:  # note: same variable used for parsing and looping
        chunks = parse_document(item)
        print(f"\n{item.get('title') or item.get('id')}: {len(chunks)} paragraph chunks")

        if dry_run:
            for c in chunks:
                print(f"  {c['chapter'][:18]:18} | {c['section'][:22]:22} | "
                      f"{c['para_no']:>5} | {len(c['text'])} chars")

        docs, ids = chunks_to_documents(chunks)
        all_docs.extend(docs)
        all_ids.extend(ids)

    print(f"\nTotal: {len(all_docs)} chunks")
    if dry_run:
        return

    vector_store = get_vector_store()
    add_in_batches(vector_store, all_docs, all_ids)
    print("Done - saved to", OUT_DIR_CHROMA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and print chunks without calling the embedding API")
    args = parser.parse_args()
    create_embeddings(dry_run=args.dry_run)