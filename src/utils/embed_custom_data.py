import os
import re
import sys
import time
import json
import argparse
import requests
import chromadb
import traceback
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

script_dir = os.path.dirname(os.path.realpath(__file__))

# make the project root importable so `utils` resolves when this file is run directly
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.error_store import error_store

# paths and settings below must match setup.py, otherwise the updated chunks will not
# line up with the ones that are already in the database
master_directory_file = os.path.abspath(os.path.join(script_dir,"..", "..", "data", "master_directory", "master_directory_html.jsonl"))
chroma_dir = os.path.abspath(os.path.join(script_dir,"..", "..", "chroma_db"))
chroma_state_file = os.path.join(chroma_dir,"..", "setup_state.json")

config = os.path.abspath(os.path.join(script_dir, "..", "..","config.json"))
with open(config, "r", encoding="utf-8") as f:
        config = json.load(f)


model_path = config["Model_Details"]["Embedding_Model_path"]

chroma_collection = "master_directories"
master_directory_url = "https://rbi.org.in/scripts/BS_ViewMasDirections.aspx?id={}"

request_timeout = 30
request_delay = 1
request_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

max_chars = 1500
overlap = 150
batch_size = 10
max_seq_length = 1024
encode_batch_size = 4
write_batch_size = 500

source_directories = "directories"
source_notifications = "notifications"

error_file_name = "ai_inference"

embedding_required_files = [
    "config.json",
    "modules.json",
    "model.safetensors",
    "tokenizer.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
]

chapter_pattern = re.compile(r"^(chapter|part)\s*[-–]?\s*([ivxlc]+|[a-z]|\d+)\b", re.I)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_chars,
    chunk_overlap=overlap,
    separators=["\n", ". ", " ", ""],
)

embedding_model = None


def log_error(error, where=""):
    """Send an error to error_store. Never raises, so logging cannot hide the real problem.

    Call it inside an `except` block to capture the real traceback. When it is called
    for a health-check failure (no exception), the traceback field says so instead.
    """
    message = f"[{where}] {error}" if where else str(error)

    trace = traceback.format_exc()
    if trace.strip() == "NoneType: None":
        trace = "no exception raised (health check or validation failure)"

    try:
        error_store(
            error_message=message,
            trace_back=trace,
            time=str(datetime.now(timezone.utc)),
            error_count="Null",
            error_file=error_file_name,
        )
    except Exception as store_error:
        print(f"error_store failed ({store_error}), original error: {message}")


def embedding_model_exists():
    if not os.path.isdir(model_path):
        return False

    for name in embedding_required_files:
        if not os.path.isfile(os.path.join(model_path, name)):
            return False
    return True


def open_chroma_client():
    return chromadb.PersistentClient(path=chroma_dir)


def count_by_source(collection, source_name):
    found = collection.get(where={"source": source_name}, include=[])
    return len(found["ids"])


def directory_filter(record_id):
    """Chroma filter that matches every chunk of one master directory."""
    return {"$and": [{"source": source_directories}, {"doc_id": str(record_id)}]}


def read_chroma_state():
    if not os.path.exists(chroma_state_file):
        return None

    try:
        with open(chroma_state_file, "r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        log_error(error, "read_chroma_state: setup_state.json is corrupted")
        return None

    if not isinstance(state, dict):
        log_error(f"expected a JSON object, got {type(state).__name__}", "read_chroma_state")
        return None
    return state


def update_chroma_state(collection):
    """Refreshes the chunk counts in setup_state.json so setup.py still sees the database as healthy.

    setup.py compares the real chunk counts with this record, and an updated directory
    can have a different number of chunks than before. If setup was never completed
    there is no record, and it is left alone.
    """
    state = read_chroma_state()
    if state is None:
        print("chromadb: no setup record found, setup_state.json left alone")
        return False

    counts = {}
    for source_name in (source_directories, source_notifications):
        counts[source_name] = count_by_source(collection, source_name)
    state["counts"] = counts

    with open(chroma_state_file, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)
    print(f"chromadb: setup record refreshed ({sum(counts.values())} chunks)")
    return True


def download_record(session, url, record_id, keep_html):
    response = session.get(url, timeout=request_timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, features="html.parser")

    title_tag = soup.find("b")
    content = soup.find("tr", class_="tablecontent2")
    if title_tag is None or content is None:
        return None

    for junk in content.find_all(["script", "style", "input"]):
        junk.decompose()

    if keep_html:
        text = str(content)
    else:
        text = content.get_text(separator="\n", strip=True)

    return {
        "id": record_id,
        "title": title_tag.get_text(strip=True),
        "category": "NBFC",
        "url": url,
        "text": text.replace("\r", ""),
    }


def save_record(record):
    """Replaces this id's line in master_directory_html.jsonl, or adds it if it is new.

    setup.py reads this file instead of downloading again, so if it kept the old page
    a later rebuild of the database would quietly bring the old version back.
    """
    os.makedirs(os.path.dirname(master_directory_file), exist_ok=True)
    record_id = str(record["id"])

    lines = []
    replaced = False

    if os.path.exists(master_directory_file):
        with open(master_directory_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue

                if str(existing.get("id")) == record_id:
                    if not replaced:
                        lines.append(json.dumps(record))
                        replaced = True
                    continue
                lines.append(line)

    if not replaced:
        lines.append(json.dumps(record))

    temp_file = master_directory_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    os.replace(temp_file, master_directory_file)


def get_embeddings(device):
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": encode_batch_size},
        )
        embedding_model._client.max_seq_length = max_seq_length
        print(f"Embedding with max_seq_length={max_seq_length}, batch_size={encode_batch_size}")
    return embedding_model


# ---- parsing and chunking: copied from setup.py, keep the two in sync ----

def save_paragraph(paragraph, chunks, item):
    text = paragraph["text"].strip()
    if text:
        chunks.append({
            "doc_id": str(item["id"]),
            "doc_title": item.get("title") or "",
            "url": item.get("url") or "",
            "chapter": paragraph["chapter"],
            "section": paragraph["section"],
            "para_no": paragraph["para_no"],
            "text": text,
        })
    paragraph["para_no"] = ""
    paragraph["text"] = ""


def parse_document(item):
    soup = BeautifulSoup(item["text"], "html.parser")

    inner = None
    table = soup.find("table", class_="td")
    if table is not None:
        inner = table.find("td")

    if inner is None:
        message = f"no <table class='td'><td> found in document {item.get('id')}"
        print(f"[skip] {message}")
        log_error(message, "parse_document")
        return []

    chunks = []
    paragraph = {"chapter": "Preamble", "section": "", "para_no": "", "text": ""}

    children = inner.find_all(recursive=False)
    last_hr = None
    for index, child in enumerate(children):
        if child.name == "hr":
            last_hr = index
    if last_hr is not None:
        children = children[:last_hr]

    for child in children:
        if child.name == "hr":
            continue

        text = child.get_text(" ", strip=True)
        if not text:
            continue
        first_word = text.split(" ")[0]

        head_span = child.find(class_="head")
        child_classes = child.get("class")
        if child_classes is None:
            child_classes = []

        own_head = "head" in child_classes
        span_head = head_span is not None and len(head_span.get_text(strip=True)) > 0.8 * len(text)
        is_head = own_head or span_head

        if span_head:
            heading_text = head_span.get_text(" ", strip=True)
        else:
            heading_text = text

        if child.name == "table":
            if text.lower().startswith(("table of contents", "index")):
                continue
            rows = []
            for table_row in child.find_all("tr"):
                cells = []
                for cell in table_row.find_all(["td", "th"]):
                    cells.append(cell.get_text(" ", strip=True))
                rows.append(" | ".join(cells))
            paragraph["text"] += "\n" + "\n".join(rows)

        elif chapter_pattern.match(text) and (is_head or len(text) < 80):
            save_paragraph(paragraph, chunks, item)
            paragraph["chapter"] = text
            paragraph["section"] = ""

        elif is_head:
            if paragraph["chapter"] == "Preamble":
                continue
            save_paragraph(paragraph, chunks, item)
            paragraph["section"] = heading_text

        elif first_word[0].isdigit() and first_word.endswith("."):
            save_paragraph(paragraph, chunks, item)
            paragraph["para_no"] = first_word.rstrip(".")
            paragraph["text"] = text

        elif first_word.startswith("(") and not first_word.endswith(")"):
            continue

        else:
            paragraph["text"] += "\n" + text

    save_paragraph(paragraph, chunks, item)
    return chunks


def chunks_to_documents(chunks):
    documents = []
    document_ids = []

    for chunk_number, chunk in enumerate(chunks):
        header_parts = [chunk["doc_title"], chunk["chapter"], chunk["section"]]
        if chunk["para_no"]:
            header_parts.append(f"Para {chunk['para_no']}")

        header_items = []
        for part in header_parts:
            if part:
                header_items.append(part)
        header = " > ".join(header_items)

        pieces = splitter.split_text(chunk["text"])
        for piece_number, piece in enumerate(pieces):
            documents.append(Document(
                page_content=f"{header}\n\n{piece}",
                metadata={
                    "source": source_directories,
                    "doc_id": chunk["doc_id"],
                    "doc_title": chunk["doc_title"],
                    "url": chunk["url"],
                    "chapter": chunk["chapter"],
                    "section": chunk["section"],
                    "para_no": chunk["para_no"],
                    "part": piece_number,
                },
            ))
            document_ids.append(f"{chunk['doc_id']}::{chunk_number}::{piece_number}")

    return documents, document_ids


# ---- the update itself ----

def find_record_id(collection, value):
    """Turns what the user typed into a record id.

    A number is used as the id directly. Anything else is treated as the exact title of
    a master directory that is already in the database, and its id is looked up there.
    """
    value = str(value).strip()
    if value.isdigit():
        return value

    found = collection.get(
        where={"$and": [{"source": source_directories}, {"doc_title": value}]},
        limit=1,
        include=["metadatas"],
    )
    if found["ids"]:
        return str(found["metadatas"][0]["doc_id"])
    return None


def embed_documents(documents, device):
    """Embeds every chunk and keeps the vectors in memory, nothing is written to the database here."""
    embeddings = get_embeddings(device)

    vectors = []
    started_at = time.time()
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]

        texts = []
        for document in batch:
            texts.append(document.page_content)

        vectors.extend(embeddings.embed_documents(texts))

        done = start + len(batch)
        print(f"  embedded {done}/{len(documents)} | {time.time() - started_at:.0f}s elapsed", flush=True)

    return vectors


def replace_in_database(collection, record_id, documents, new_ids, vectors):
    """Stores the new chunks, then removes old chunks that no longer belong to the document.

    The new chunks are written first, so the old ones are never deleted before the new
    ones are safely stored. Chunk ids are deterministic (doc_id::chunk::piece): a chunk
    at the same position is simply overwritten, and only leftover trailing ids are deleted.
    Returns (chunks that existed before, stale chunks removed).
    """
    old_ids = collection.get(where=directory_filter(record_id), include=[])["ids"]

    for start in range(0, len(new_ids), write_batch_size):
        end = start + write_batch_size

        texts = []
        metadatas = []
        for document in documents[start:end]:
            texts.append(document.page_content)
            metadatas.append(document.metadata)

        collection.upsert(
            ids=new_ids[start:end],
            embeddings=vectors[start:end],
            documents=texts,
            metadatas=metadatas,
        )

    new_id_set = set(new_ids)
    stale_ids = []
    for old_id in old_ids:
        if old_id not in new_id_set:
            stale_ids.append(old_id)

    for start in range(0, len(stale_ids), write_batch_size):
        collection.delete(ids=stale_ids[start:start + write_batch_size])

    return len(old_ids), len(stale_ids)


def update_directory(session, collection, record_id, device):
    """Downloads one master directory and replaces it in the database. Returns True on success."""
    print(f"\n=== updating master directory {record_id} ===")
    url = master_directory_url.format(record_id)

    try:
        record = download_record(session, url, record_id, True)
    except requests.RequestException as error:
        print(f"  download failed: {error}")
        log_error(error, f"update_directory: download id {record_id}")
        return False

    if record is None:
        message = f"id {record_id} has no content, database left unchanged"
        print(f"  {message}")
        log_error(message, "update_directory")
        return False
    print(f"  downloaded: {record['title'][:60]}")

    chunks = parse_document(record)
    documents, new_ids = chunks_to_documents(chunks)
    if not documents:
        message = f"id {record_id} produced no chunks, database left unchanged"
        print(f"  {message}")
        log_error(message, "update_directory")
        return False
    print(f"  parsed into {len(chunks)} paragraph chunks, {len(documents)} stored chunks")

    # everything slow or likely to fail happens before the database is touched
    vectors = embed_documents(documents, device)

    old_count, stale_count = replace_in_database(collection, record_id, documents, new_ids, vectors)
    print(f"  database: {old_count} old chunks, {len(documents)} written, {stale_count} stale removed")

    try:
        save_record(record)
    except Exception as error:
        print(f"  warning: database is updated but {os.path.basename(master_directory_file)} "
              f"could not be: {error}")
        log_error(error, f"update_directory: save_record id {record_id}")

    return True


def embed(master_directories, device="cuda"):
    """Re-downloads master directories and replaces them in the RAG database.

    Takes one master directory or a list of them, each either its numeric id (the number
    in the rbi.org.in url) or the exact title of one already in the database.
    Returns True only if every one of them was updated.
    """
    if not isinstance(master_directories, (list, tuple)):
        master_directories = [master_directories]

    if not embedding_model_exists():
        message = f"embedding model is missing or incomplete: {model_path} (run setup.py first)"
        print(f"error: {message}")
        log_error(message, "embed")
        return False

    try:
        collection = open_chroma_client().get_collection(name=chroma_collection)
    except Exception as error:
        print(f"error: database '{chroma_collection}' not found in {chroma_dir} (run setup.py first)")
        log_error(error, "embed: open collection")
        return False

    record_ids = []
    failed = []
    for value in master_directories:
        record_id = find_record_id(collection, value)
        if record_id is None:
            message = f"could not find a master directory titled '{value}', use its id instead"
            print(f"error: {message}")
            log_error(message, "embed")
            failed.append(str(value))
        elif record_id not in record_ids:
            record_ids.append(record_id)

    session = requests.Session()
    session.headers.update(request_headers)

    updated = []
    for number, record_id in enumerate(record_ids):
        if number > 0:
            time.sleep(request_delay)

        try:
            ok = update_directory(session, collection, record_id, device)
        except Exception as error:
            print(f"  update failed: {error}")
            log_error(error, f"embed: id {record_id}")
            ok = False

        if ok:
            updated.append(record_id)
        else:
            failed.append(record_id)

    if updated:
        try:
            update_chroma_state(collection)
        except Exception as error:
            print(f"chromadb: could not refresh the setup record ({error})")
            log_error(error, "embed: update_chroma_state")
            failed.append("setup_state.json")

    print(f"\nupdated {len(updated)} of {len(updated) + len(failed)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-download master directories and replace them in the RAG database."
    )
    parser.add_argument("master_directories", nargs="+",
                        help="master directory ids, or exact titles already in the database")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="where to run the embedding model (default: cuda)")
    args = parser.parse_args()
    sys.exit(0 if embed(args.master_directories, args.device) else 1)