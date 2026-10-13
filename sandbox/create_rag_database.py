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
IN_FILE_DIRECTORIES = os.path.abspath(
    os.path.join(script_dir, "..", "data", "master_directory", "master_directory_html.jsonl")
)
IN_FILE_NOTIFICATIONS = os.path.abspath(
    os.path.join(script_dir, "..", "data", "notification_historical", "filtered_notifications_deduped.jsonl")
)
OUT_DIR_CHROMA = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))

# Value stored in the `source` metadata field - use it to filter searches
SOURCE_DIRECTORIES = "directories"
SOURCE_NOTIFICATIONS = "notifications"

MAX_CHARS = 1500
OVERLAP = 150
BATCH_SIZE = 10
PAUSE_BETWEEN_BATCHES = 0
MAX_RETRIES = 8

# GPU memory controls (the notifications are embedded whole, so they can be very long)
MAX_SEQ_LENGTH = 1024    # tokens; longer text is cut off for the embedding only (full text is still stored)
ENCODE_BATCH_SIZE = 4    # texts per GPU forward pass; lower this if you still run out of memory
NOTIFICATION_BATCH_SIZE = 2   # notifications are long, so save + report progress every 2 documents

CHAPTER_RE = re.compile(r"^(chapter|part)\s*[-–]?\s*([ivxlc]+|[a-z]|\d+)\b", re.I)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHARS,
    chunk_overlap=OVERLAP,
    separators=["\n", ". ", " ", ""],
)

model_path = os.path.abspath(os.path.join(script_dir, "..", "model", "embeddinggemma-2"))


def get_vector_store(device="cuda"):
    # Pass the absolute path directly to HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},  # "cuda" or "cpu"
        encode_kwargs={"normalize_embeddings": True, "batch_size": ENCODE_BATCH_SIZE},
    )
    # Without this cap the model pads every batch to its longest text, which is what ran out of memory
    embeddings._client.max_seq_length = MAX_SEQ_LENGTH
    print(f"Embedding with max_seq_length={MAX_SEQ_LENGTH}, batch_size={ENCODE_BATCH_SIZE}")

    return Chroma(
        collection_name="master_directories",
        embedding_function=embeddings,
        persist_directory=OUT_DIR_CHROMA,
    )


# ======================================================================
# MASTER DIRECTORIES: parsed from HTML, split into paragraph-level chunks
# ======================================================================
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
                    "source": SOURCE_DIRECTORIES,
                    "doc_id": c["doc_id"],
                    "doc_title": c["doc_title"],
                    "url": c["url"],
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "para_no": c["para_no"],
                    "part": k,
                },
            ))
            ids.append(f"{c['doc_id']}::{n}::{k}")  # same id on re-run -> no duplicates
    return docs, ids


def build_directory_docs(dry_run=False):
    with open(IN_FILE_DIRECTORIES, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data)} master directory documents")

    all_docs, all_ids = [], []
    for item in data:
        chunks = parse_document(item)
        print(f"\n{item.get('title') or item.get('id')}: {len(chunks)} paragraph chunks")

        if dry_run:
            for c in chunks:
                print(f"  {c['chapter'][:18]:18} | {c['section'][:22]:22} | "
                      f"{c['para_no']:>5} | {len(c['text'])} chars")

        docs, ids = chunks_to_documents(chunks)
        all_docs.extend(docs)
        all_ids.extend(ids)

    print(f"\nTotal directory chunks: {len(all_docs)}")
    return all_docs, all_ids


# ======================================================================
# NOTIFICATIONS: one record = one document, never split
# Input fields: id, title, url, text
# ======================================================================
def build_notification_docs():
    records = {}  # duplicate ids keep the last occurrence (Chroma rejects dupes in a batch)
    skipped = 0

    with open(IN_FILE_NOTIFICATIONS, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] notifications line {line_no}: invalid JSON ({e}) - skipped")
                skipped += 1
                continue

            doc_id = rec.get("id")
            text = (rec.get("text") or "").strip()
            if doc_id is None or not text:
                print(f"[warn] notifications line {line_no}: missing id or empty text - skipped")
                skipped += 1
                continue

            doc_id = str(doc_id)
            # "notif::" prefix keeps these ids from ever colliding with directory chunk ids
            records[f"notif::{doc_id}"] = Document(
                page_content=text,  # full text, embedded as a single document
                metadata={
                    "source": SOURCE_NOTIFICATIONS,
                    "doc_id": doc_id,
                    "doc_title": str(rec.get("title") or ""),
                    "url": str(rec.get("url") or ""),
                },
            )

    docs, ids = list(records.values()), list(records.keys())
    longest = max((len(d.page_content) for d in docs), default=0)
    print(f"\nTotal notifications: {len(docs)} ({skipped} skipped, longest {longest} chars)")
    return docs, ids


# ======================================================================
# Shared: write to Chroma
# ======================================================================
def add_in_batches(vector_store, docs, ids, batch_size=BATCH_SIZE):
    stored = vector_store.get(include=["metadatas"])
    existing = set(stored["ids"])

    # Chunks stored by an earlier run (before `source` existed) are tagged in place - no re-embedding
    by_id = dict(zip(ids, docs))
    untagged = [
        i for i, m in zip(stored["ids"], stored["metadatas"])
        if i in by_id and not (m or {}).get("source")
    ]
    for start in range(0, len(untagged), 1000):
        part = untagged[start:start + 1000]
        vector_store._collection.update(
            ids=part,
            metadatas=[{"source": by_id[i].metadata["source"]} for i in part],
        )
    if untagged:
        print(f"Added `source` metadata to {len(untagged)} previously stored chunks")

    # Resume support: skip chunks that are already in Chroma
    todo = [(d, i) for d, i in zip(docs, ids) if i not in existing]
    print(f"{len(docs) - len(todo)} already stored, {len(todo)} left to embed")

    already_saved = len(docs) - len(todo)
    t0 = time.time()

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        batch_docs = [d for d, _ in batch]
        batch_ids = [i for _, i in batch]

        longest = max(len(d.page_content) for d in batch_docs)
        print(f"  embedding {start + 1}-{start + len(batch)} of {len(todo)} "
              f"(longest in batch: {longest:,} chars) ...", flush=True)

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

        done = start + len(batch)
        print(f"  saved {already_saved + done}/{len(docs)} | {len(todo) - done} remaining | "
              f"{time.time() - t0:.0f}s elapsed", flush=True)
        time.sleep(PAUSE_BETWEEN_BATCHES)


def create_embeddings(source="all", dry_run=False, device="cuda"):
    jobs = []  # (name, docs, ids)
    if source in ("all", SOURCE_DIRECTORIES):
        jobs.append((SOURCE_DIRECTORIES, *build_directory_docs(dry_run)))
    if source in ("all", SOURCE_NOTIFICATIONS):
        jobs.append((SOURCE_NOTIFICATIONS, *build_notification_docs()))

    if dry_run:
        return

    vector_store = get_vector_store(device)  # load the model once, shared by both sources
    for name, docs, ids in jobs:
        print(f"\n=== Embedding {name} ===")
        batch_size = NOTIFICATION_BATCH_SIZE if name == SOURCE_NOTIFICATIONS else BATCH_SIZE
        add_in_batches(vector_store, docs, ids, batch_size)
    print("\nDone - saved to", OUT_DIR_CHROMA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and print chunks without calling the embedding model")
    parser.add_argument("--source", choices=["all", SOURCE_DIRECTORIES, SOURCE_NOTIFICATIONS],
                        default="all", help="which dataset to process (default: all)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="where to run the embedding model (default: cuda)")
    args = parser.parse_args()
    create_embeddings(source=args.source, dry_run=args.dry_run, device=args.device)