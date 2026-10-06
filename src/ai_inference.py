import os
import re 
import glob
import json
import jsonlines
import traceback
from typing import Optional
from dotenv import load_dotenv
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from utils.error_store import error_store
from langchain_core.documents import Document
from utils.week_file_save import current_week_file
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.find_reference import find_mds, load_titles, double_hop
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from llm_queue import submit, NOTIFY
load_dotenv()
API_KEY = os.getenv('GEMINI')
os.environ["GOOGLE_API_KEY"] = API_KEY


script_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(script_dir, "..", "data", "master_directory", "master_directory.jsonl")
out_dir_output = os.path.abspath(os.path.join(script_dir, "..", "data", "notifications_output"))
_cache = {}
out_dir_chat = os.path.abspath(os.path.join(script_dir, "..", "data", "chat"))

class ComplianceEvaluation(BaseModel):
    """ Confirm schema and datatype of model output
    """
    title: str
    link: str
    pubdate : str = Field(description="The date string formatted exactly as 'ddd, DD MMM YYYY HH:MM:SS' (e.g. 'Mon, 21 Sep 2026 17:25:00')")
    is_applicable : bool
    is_master_direction_matching : bool
    reason : Optional[str] = None
    correct_master_dir : Optional[str] = None
    diff : str
    explanation : str
    actions_required : str
    action_date: str = Field(description="Either a date formatted as 'ddd, DD MMM YYYY HH:MM:SS', "
                                "or the exact text from the notification describing when "
                                "the change takes effect, if no specific date is given."
                    )
    effective_date: str = Field(description="The date string formatted exactly as 'ddd, DD MMM YYYY HH:MM:SS' (e.g. 'Mon, 21 Sep 2026 17:25:00')")
    
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

embeddings = get_embeddings()


in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))


class ChatEvaluation(BaseModel):
    answer : str


def load_store(): # load chroma database
    return Chroma(
            collection_name=f"master_directories",
            embedding_function=embeddings,
            persist_directory=in_dir_data,
        )


def build_retriever(): 
    """ get raw documetns for bm25 and build retriever functions 
    because if I am calling retrieve function inside loop so having this logic in there means wastage of compute"""
    cos_store = load_store()
    data = cos_store.get()
    corpus = [
        Document(page_content=t, metadata={**(m or {}), "chunk_id": i})
        for t, m, i in zip(data["documents"], data["metadatas"], data["ids"])
    ]
    K = 4
    bm25 = BM25Retriever.from_documents(corpus, k=K)
    dense = cos_store.as_retriever(search_kwargs={"k": K})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.2, 0.8])

_hybrid, _titles = None, None

def get_retriever():
    if "hybrid" not in _cache:
        _cache["hybrid"] = build_retriever()
        _cache["titles"] = load_titles(path)
    return _cache["hybrid"], _cache["titles"]

def retrieve(query, hybrid, k=4):
    return hybrid.invoke(query)[:k]


def run_batch():
    """ Match incoming notifications against rag database check for other master directory mention and if yes then retrieve 
    relevant data from that particular document """
    
    in_dir_config = os.path.abspath(os.path.join(script_dir,"..","config.json"))
    out_dir_output = os.path.abspath(os.path.join(script_dir,"..","data","notifications_output"))
    os.makedirs(out_dir_output,exist_ok = True)

    with open(in_dir_config) as f:
        data = json.load(f)

    debug = False
    if debug:
        in_dir_notifications = os.path.abspath(os.path.join(script_dir,"..","data","notifications_matched","test.jsonl"))

    if not debug:
        in_dir_notifications = data["data_check"]["matched_ref_file"]
        
    notifications = []
    with open(in_dir_notifications, "r", encoding="utf-8") as file_notifications:
        for line in file_notifications:
            notifications.append(json.loads(line))

    hybrid, titles = get_retriever()
    
    in_dir_profile = os.path.abspath(os.path.join(script_dir,"..","profile.json"))
    with open(in_dir_profile, "r", encoding="utf-8") as file_profile:
        profile = json.load(file_profile)
        output_path = current_week_file(out_dir_output, format="jsonl")
        
        for i, json_line in enumerate(notifications):
            
            try : 
                    llm = get_llm()
                    structured_llm = llm.with_structured_output(ComplianceEvaluation)
                
                    results = []
                
                    retrieved_data = retrieve(json_line["clean_description"],hybrid) # retireve data 
                    combined_text = "\n\n".join([doc.page_content for doc in retrieved_data])
                    recorvered_mds = find_mds(combined_text,titles)
                    
                    if recorvered_mds:
                        double_hop_data = double_hop(combined_text,titles)
                        final = retrieved_data + double_hop_data
                        combined_text = "\n\n---\n\n".join([
                                f"Document Title: {doc.metadata.get('doc_title', 'Unknown')}\nContent:\n{doc.page_content}"
                                for doc in final])
                        
   
                    promt = f"""Evaluate compliance applicability. 
                            "title"(title of notification), "link" (link of notification), "pubdate" (as mentioned in data),
                            "is_applicable" (boolean),"is_master_direction_matching" (boolean),"reason"(if master directory doesn't match to the given data return reason if no master directory to match return none),
                            "correct_master_dir"(return not applicable if no master dir needed, mention correct one if master directory mentioned is wrong with full title as mentioned in circular ,otherwise write correct), 
                            "diff","mention direct proposed ( if not return not applicable ) changes in the text 
                            example : Reserve Bank of India (Non-Banking Financial Companies – Concentration Risk Management) Fourth Amendment Directions, 2026

                            The Reserve Bank has issued the Reserve Bank of India (Non-Banking Financial Companies – Concentration Risk Management) Directions, 2025 dated November 28, 2025 (hereinafter referred to as ‘Directions’). On a review, it has been decided to revise the large exposure framework for Infrastructure Debt Fund-Non-Banking Financial Company (IDF-NBFC) in the Upper Layer.

                            2. Accordingly, in exercise of the powers conferred by Chapter III B of the Reserve Bank of India Act, 1934, and all other provisions / laws enabling the Reserve Bank of India (‘RBI’) in this regard, RBI being satisfied that it is necessary and expedient in the public interest so to do, hereby, issues the Amendment Directions hereinafter specified.

                            3. These Amendment Directions shall be called the Reserve Bank of India (Non-Banking Financial Companies – Concentration Risk Management) Fourth Amendment Directions, 2026.

                            4. These Amendment Directions shall come into force with immediate effect.

                            5. These Amendment Directions shall modify the Directions as under:

                            (1) After paragraph 39 in ‘Chapter IV - Guidelines Applicable to NBFC – Upper Layer’, a new paragraph 39A shall be inserted as under:

                            39A. The large exposure limits applicable to NBFC-IFC shall also be applicable to IDF-NBFC that are subject to Upper Layer regulations in terms of paragraph 60A of the Reserve Bank of India (Non-Banking Financial Companies – Undertaking of Financial Services) Directions, 2025 read together with paragraph 18 (4) (i) of the Reserve Bank of India (Commercial Banks – Undertaking of Financial Services) Directions, 2025.

                            mentions "applicable to Upper Layer" " After paragraph 39 in ‘Chapter IV - Guidelines Applicable to NBFC – Upper Layer’, a new paragraph 39A shall be inserted as under:

                            39A. The large exposure limits applicable to NBFC-IFC shall also be applicable to IDF-NBFC that are subject to Upper Layer regulations in terms of paragraph 60A of the Reserve Bank of India (Non-Banking Financial Companies – Undertaking of Financial Services) Directions, 2025 read together with paragraph 18 (4) (i) of the Reserve Bank of India (Commercial Banks – Undertaking of Financial Services) Directions, 2025."
                            return this with the old text
                            "explanation" (explane new changes and old changes and how it affect things in proper words as described in notification), "actions_required" (actions required by the company to make).
                            "action_date" ( when to take action mention date or return exact text when the changes is proposed)

                            Profile:
                            {json.dumps(profile,indent = 2 )}
                            
                            Data : 
                            {json.dumps(json_line,indent = 2 )}
                            
                            Retrieved_Data
                            {combined_text}
                            """
                            
                    evaluation : ComplianceEvaluation = structured_llm.invoke(promt)
                    
                    results.append(evaluation.model_dump())

                    output_path = current_week_file(out_dir_output,format = "jsonl")
                
                    with jsonlines.open(output_path, mode="a") as writer:
                        writer.write(evaluation.model_dump())
            
            except Exception as e:
                error_store(error_message=str(e),trace_back = traceback.format_exc(),time = str(datetime.now(timezone.utc)),error_count="Null",error_file="ai_inference")

def invoke_ai():
    submit(NOTIFY, run_batch).result()

 
SYSTEM_PROMPT = """You are a compliance assistant for RBI master directions.
Answer using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}"""

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Document Title: {d.metadata.get('doc_title', 'Unknown')}\nContent:\n{d.page_content}"
        for d in docs
    )
        
_histories = {}     # chat name -> list of {"role", "content"}


def safe_chat_name(name):
    """The chat name becomes a folder name, so keep only letters, numbers, space, - and _."""
    return re.sub(r"[^\w\- ]", "", name).strip()[:80] or "gatorade"


def save_chat(name, messages):
    folder = os.path.join(out_dir_chat, name)
    os.makedirs(folder, exist_ok=True)
    with jsonlines.open(current_week_file(folder, format="jsonl"), mode="a") as w:
        for m in messages:
            w.write({"ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"), **m})


def load_chat(name):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir_chat, name, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:      # half-written last line
                    pass
    rows.sort(key=lambda r: r.get("ts", ""))      # works across week files
    return rows


def list_chats():
    """Every saved chat, most recently used first."""
    chats = []
    if os.path.isdir(out_dir_chat):
        for name in os.listdir(out_dir_chat):
            rows = load_chat(name)
            if rows:
                chats.append({"name": name, "last": rows[-1].get("ts", ""),
                              "messages": [{"role": r["role"], "content": r["content"]} for r in rows]})
    chats.sort(key=lambda c: c["last"], reverse=True)
    return [{"name": c["name"], "messages": c["messages"]} for c in chats]


def chat_job(sid, question, max_turns=6):          # runs on the queue worker
    sid = safe_chat_name(sid)
    if sid not in _histories:                      # first message since a restart: pick the saved chat back up
        _histories[sid] = load_chat(sid)
    hist = _histories[sid]
    hybrid, titles = get_retriever()

    recent_user = [m["content"] for m in hist if m["role"] == "user"][-2:]
    chunks = retrieve("\n".join(recent_user + [question]), hybrid)
    chunks_text = "\n\n".join(d.page_content for d in chunks)

    if find_mds(chunks_text, titles):
        chunks_text = format_docs(chunks + double_hop(chunks_text, titles)[:4])

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=chunks_text))]
    for m in hist[-2 * max_turns:]:
        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=question))

    answer = get_llm().invoke(messages).text
    new = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    save_chat(sid, new)                            # disk first: if this fails, memory stays unchanged
    hist += new
    return answer

def read_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir_output, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:   # half-written last line
                    pass
    return rows