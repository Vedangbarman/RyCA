import os
import re
import sys
import copy
import time
import json
import argparse
import traceback
import requests
import chromadb
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

script_dir = os.path.dirname(os.path.realpath(__file__))

# make the project root importable so `utils` resolves when this file is run directly
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.error_store import error_store  # noqa: E402

ids_file = os.path.abspath(os.path.join(script_dir, "..", "data", "setup.txt"))
master_directory_file = os.path.abspath(os.path.join(script_dir, "..", "data", "master_directory", "master_directory_html.jsonl"))
notification_file = os.path.abspath(os.path.join(script_dir, "..", "data", "notification_historical", "filtered_notifications_deduped.jsonl"))
chroma_dir = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))
chroma_state_file = os.path.join(chroma_dir, "setup_state.json")
model_path = os.path.abspath(os.path.join(script_dir, "..", "model", "embeddinggemma-2"))
config_file = os.path.abspath(os.path.join(script_dir, "..", "config.json"))

chroma_collection = "master_directories"

master_directory_url = "https://rbi.org.in/scripts/BS_ViewMasDirections.aspx?id={}"
notification_url = "https://rbi.org.in/scripts/NotificationUser.aspx?Id={}"
master_directory_count = 45

request_timeout = 30
request_delay = 1
request_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

max_chars = 1500
overlap = 150
batch_size = 10
notification_batch_size = 2
max_seq_length = 1024
encode_batch_size = 4

source_directories = "directories"
source_notifications = "notifications"

error_file_name = "ai_inference"

chapter_pattern = re.compile(r"^(chapter|part)\s*[-–]?\s*([ivxlc]+|[a-z]|\d+)\b", re.I)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_chars,
    chunk_overlap=overlap,
    separators=["\n", ". ", " ", ""],
)

config_template = {
    "scraper_settings": {
        "time_scraper": "",
        "run_count": ""
    },
    "data_check": {
        "ref_date": "",
        "delta_ref_date": "",
        "ref_article": "",
        "ref_file": "",
        "clean_ref_file": "",
        "matched_ref_file": ""
    },
    "Model_Details": {
        "Embedding_Model": "",
        "Embedding_Model_path": "",
        "Main_Model": "",
        "Main_Model_Path": ""
    },
    "Hyper_Param": {
        "temperature": "",
        "top_p": "",
        "max_tokens": "",
        "frequency_penalty": "",
        "presence_penalty": "",
        "seed": "",
        "top_k": "",
        "min_p": "",
        "repeat_penalty": ""
    }
}


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


def write_config(data):
    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def fill_missing_keys(template, current):
    changed = False
    for key, value in template.items():
        if key not in current:
            current[key] = copy.deepcopy(value)
            changed = True
            continue

        if isinstance(value, dict):
            if isinstance(current[key], dict):
                if fill_missing_keys(value, current[key]):
                    changed = True
            else:
                current[key] = copy.deepcopy(value)
                changed = True

    return changed


def find_structure_problems(template, current, path=""):
    """Lists every place where `current` differs from the structure of `template`."""
    problems = []
    for key, value in template.items():
        where = f"{path}.{key}" if path else key
        if key not in current:
            problems.append(f"missing key: {where}")
        elif isinstance(value, dict):
            if not isinstance(current[key], dict):
                problems.append(f"{where} should be an object")
            else:
                problems.extend(find_structure_problems(value, current[key], where))
    return problems


def check_config():
    """Makes sure config.json exists and has the right structure, building or fixing it if not.

    Existing values are always kept. Returns True once the config is healthy. If the
    repair itself fails (for example a permissions error) the exception propagates to
    setup(), which logs it and reports failure.
    """
    if not os.path.exists(config_file):
        write_config(config_template)
        print(f"config: not found, created it at {config_file}")
        return True

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            current = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        log_error(error, "check_config: config.json could not be parsed")
        current = None

    if current is not None and not isinstance(current, dict):
        log_error(f"expected a JSON object, got {type(current).__name__}", "check_config")

    if not isinstance(current, dict):
        broken_file = config_file + ".broken"
        os.replace(config_file, broken_file)
        write_config(config_template)
        print(f"config: unreadable, moved to {broken_file} and created a fresh one")
        return True

    problems = find_structure_problems(config_template, current)
    if problems:
        for problem in problems:
            print(f"config: {problem}")
        fill_missing_keys(config_template, current)
        write_config(current)
        print("config: structure fixed, existing values kept")
        return True

    print("config: found, structure is correct")
    return True


def open_chroma_client():
    return chromadb.PersistentClient(path=chroma_dir)


def count_by_source(collection, source_name):
    found = collection.get(where={"source": source_name}, include=[])
    return len(found["ids"])


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


def chromadb_exists():
    """True if the chroma folder exists and the collection holds at least one chunk."""
    if not os.path.isdir(chroma_dir):
        return False
    try:
        collection = open_chroma_client().get_collection(name=chroma_collection)
        return collection.count() > 0
    except Exception:
        return False


def check_chromadb():
    """Returns True if the database exists and matches the setup record, False otherwise."""
    if not chromadb_exists():
        print("chromadb: not found")
        return False

    state = read_chroma_state()
    if state is None:
        print("chromadb: found, but setup was never completed")
        return False

    counts = state.get("counts")
    if not isinstance(counts, dict) or not counts:
        print("chromadb: setup record is invalid")
        log_error("setup record has no valid 'counts'", "check_chromadb")
        return False

    try:
        collection = open_chroma_client().get_collection(name=chroma_collection)
        for source_name, expected in counts.items():
            found = count_by_source(collection, source_name)
            if found != expected:
                message = f"{source_name} has {found} chunks, expected {expected}"
                print(f"chromadb: {message}")
                log_error(message, "check_chromadb")
                return False
        sample = collection.get(limit=1, include=["embeddings"])
    except Exception as error:
        print(f"chromadb: could not be read ({error})")
        log_error(error, "check_chromadb")
        return False

    embeddings = sample["embeddings"]
    if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
        print("chromadb: stored vectors are unreadable")
        log_error("stored vectors are empty or unreadable", "check_chromadb")
        return False

    print(f"chromadb: healthy ({sum(counts.values())} chunks)")
    return True


def write_chroma_state():
    """Marks setup as complete. Returns True if the record was written."""
    collection = open_chroma_client().get_collection(name=chroma_collection)

    counts = {}
    for source_name in (source_directories, source_notifications):
        counts[source_name] = count_by_source(collection, source_name)

    for source_name, count in counts.items():
        if count == 0:
            message = f"no {source_name} chunks stored, setup not marked as complete"
            print(f"chromadb: {message}")
            log_error(message, "write_chroma_state")
            return False

    with open(chroma_state_file, "w", encoding="utf-8") as file:
        json.dump({"collection": chroma_collection, "counts": counts}, file, indent=4)
    print(f"chromadb: setup marked as complete ({sum(counts.values())} chunks)")
    return True


def read_ids():
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"ids file not found: {ids_file}")

    record_ids = []
    with open(ids_file, "r", encoding="utf-8") as file:
        for line in file:
            record_id = line.strip()
            if record_id:
                record_ids.append(record_id)
    return record_ids


def load_saved_ids(path):
    saved_ids = set()
    if not os.path.exists(path):
        return saved_ids

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                log_error(error, f"load_saved_ids: bad line in {os.path.basename(path)}")
                continue
            saved_ids.add(str(record.get("id")))
    return saved_ids


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


def download_records(record_ids, url_template, output_file, keep_html, label):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    saved_ids = load_saved_ids(output_file)

    pending = []
    for record_id in record_ids:
        if record_id in saved_ids:
            continue
        if record_id in pending:
            continue
        pending.append(record_id)

    print(f"\n=== downloading {label} ===")
    print(f"{len(saved_ids)} already saved, {len(pending)} to download")

    session = requests.Session()
    session.headers.update(request_headers)

    failed_ids = []
    for number, record_id in enumerate(pending, start=1):
        url = url_template.format(record_id)
        try:
            record = download_record(session, url, record_id, keep_html)
        except requests.RequestException as error:
            print(f"  [{number}/{len(pending)}] id {record_id} failed: {error}")
            log_error(error, f"download_records({label}): id {record_id}")
            failed_ids.append(record_id)
            continue

        if record is None:
            print(f"  [{number}/{len(pending)}] id {record_id} has no content")
            failed_ids.append(record_id)
            continue

        with open(output_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print(f"  [{number}/{len(pending)}] id {record_id} saved - {record['title'][:60]}")
        time.sleep(request_delay)

    return failed_ids


def get_vector_store(device):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": encode_batch_size},
    )
    embeddings._client.max_seq_length = max_seq_length
    print(f"Embedding with max_seq_length={max_seq_length}, batch_size={encode_batch_size}")

    return Chroma(
        client=open_chroma_client(),
        collection_name=chroma_collection,
        embedding_function=embeddings,
    )


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


def build_directory_docs():
    if not os.path.exists(master_directory_file):
        print("no master directory file found, skipping directories")
        return [], []

    items = []
    with open(master_directory_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    print(f"\nLoaded {len(items)} master directory documents")

    all_documents = []
    all_ids = []
    for item in items:
        chunks = parse_document(item)
        print(f"{item.get('title') or item.get('id')}: {len(chunks)} paragraph chunks")

        documents, document_ids = chunks_to_documents(chunks)
        all_documents.extend(documents)
        all_ids.extend(document_ids)

    print(f"Total directory chunks: {len(all_documents)}")
    return all_documents, all_ids


def build_notification_docs():
    if not os.path.exists(notification_file):
        print("no notification file found, skipping notifications")
        return [], []

    records = {}
    skipped = 0

    with open(notification_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[warn] notifications line {line_number}: invalid JSON ({error}) - skipped")
                log_error(error, f"build_notification_docs: line {line_number}")
                skipped += 1
                continue

            record_id = record.get("id")
            text = (record.get("text") or "").strip()
            if record_id is None or not text:
                print(f"[warn] notifications line {line_number}: missing id or empty text - skipped")
                skipped += 1
                continue

            record_id = str(record_id)
            records[f"notif::{record_id}"] = Document(
                page_content=text,
                metadata={
                    "source": source_notifications,
                    "doc_id": record_id,
                    "doc_title": str(record.get("title") or ""),
                    "url": str(record.get("url") or ""),
                },
            )

    documents = list(records.values())
    document_ids = list(records.keys())

    longest = 0
    for document in documents:
        if len(document.page_content) > longest:
            longest = len(document.page_content)

    print(f"\nTotal notifications: {len(documents)} ({skipped} skipped, longest {longest} chars)")
    return documents, document_ids


def add_in_batches(vector_store, documents, document_ids, size):
    stored = vector_store.get(include=["metadatas"])
    known_ids = set(stored["ids"])

    pending = []
    for document, document_id in zip(documents, document_ids):
        if document_id in known_ids:
            continue
        known_ids.add(document_id)
        pending.append((document, document_id))

    already_saved = len(documents) - len(pending)
    print(f"{already_saved} already stored, {len(pending)} left to embed")

    started_at = time.time()
    for start in range(0, len(pending), size):
        batch = pending[start:start + size]

        batch_documents = []
        batch_ids = []
        longest = 0
        for document, document_id in batch:
            batch_documents.append(document)
            batch_ids.append(document_id)
            if len(document.page_content) > longest:
                longest = len(document.page_content)

        print(f"  embedding {start + 1}-{start + len(batch)} of {len(pending)} "
              f"(longest in batch: {longest:,} chars) ...", flush=True)

        vector_store.add_documents(batch_documents, ids=batch_ids)

        done = start + len(batch)
        print(f"  saved {already_saved + done}/{len(documents)} | {len(pending) - done} remaining | "
              f"{time.time() - started_at:.0f}s elapsed", flush=True)


def embed_data(device):
    directory_documents, directory_ids = build_directory_docs()
    notification_documents, notification_ids = build_notification_docs()

    if not directory_documents and not notification_documents:
        print("nothing to embed")
        return False

    vector_store = get_vector_store(device)

    print(f"\n=== embedding {source_directories} ===")
    add_in_batches(vector_store, directory_documents, directory_ids, batch_size)

    print(f"\n=== embedding {source_notifications} ===")
    add_in_batches(vector_store, notification_documents, notification_ids, notification_batch_size)

    return True


def build_chromadb(device):
    """Downloads and embeds everything. Returns True only if setup was completed and marked."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"embedding model folder not found: {model_path}")

    record_ids = read_ids()
    master_record_ids = record_ids[:master_directory_count]
    notification_record_ids = record_ids[master_directory_count:]

    failed_master = download_records(
        master_record_ids, master_directory_url, master_directory_file, True, "master directories"
    )
    failed_notifications = download_records(
        notification_record_ids, notification_url, notification_file, False, "notifications"
    )

    if failed_master:
        print(f"\nmaster directory ids that failed: {', '.join(failed_master)}")
    if failed_notifications:
        print(f"\nnotification ids that failed: {', '.join(failed_notifications)}")

    embedded = embed_data(device)
    if not embedded:
        message = "nothing was embedded, setup not marked as complete"
        print(f"\nchromadb: {message}")
        log_error(message, "build_chromadb")
        return False

    if failed_master or failed_notifications:
        message = (f"some downloads failed ({len(failed_master)} master, "
                   f"{len(failed_notifications)} notification ids), setup not marked as complete")
        print("\nchromadb: some downloads failed, setup not marked as complete")
        print("run the script again to retry, finished work is skipped")
        log_error(message, "build_chromadb")
        return False

    return write_chroma_state()


def setup(device="cuda"):
    """Runs the full setup. Returns True only if the database and config are both healthy at the end."""
    db_ok = False
    config_ok = False

    try:
        if check_chromadb():
            print("chromadb is healthy, skipping download and embedding")
            db_ok = True
        else:
            print("chromadb: building it now (anything already downloaded or embedded is skipped)")
            db_ok = build_chromadb(device)
    except Exception as error:
        log_error(error, "setup: chromadb")
        print(f"\nchromadb setup failed: {error}")

    try:
        config_ok = check_config()
    except Exception as error:
        log_error(error, "setup: config")
        print(f"\nconfig check failed: {error}")

    if db_ok and config_ok:
        print("\nsetup finished")
        return True

    print("\nsetup finished with problems, see messages above")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="where to run the embedding model (default: cuda)")
    args = parser.parse_args()
    sys.exit(0 if setup(args.device) else 1)