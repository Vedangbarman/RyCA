import os
import re 
import glob
import json
import jsonlines
import traceback
from typing import Literal
from dotenv import load_dotenv
from functools import lru_cache
from langchain_chroma import Chroma
from utils.timings import timer, log
from llm_queue import submit, NOTIFY
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone
from utils.error_store import error_store
from langchain_core.documents import Document
from utils.model import get_embeddings, get_store
from utils.week_file_save import current_week_file
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from utils.find_reference import find_mds, load_titles, double_hop
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()
API_KEY = os.getenv('GEMINI')
os.environ["GOOGLE_API_KEY"] = API_KEY


script_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(script_dir, "..", "data", "master_directory", "master_directory.jsonl")
out_dir_output = os.path.abspath(os.path.join(script_dir, "..", "data", "notifications_output"))
config = os.path.abspath(os.path.join(script_dir, "..","config.json"))
with open(config, "r", encoding="utf-8") as f:
        config = json.load(f)
embedding_path = config["Model_Details"]["Embedding_Model_path"]

out_dir_chat = os.path.abspath(os.path.join(script_dir, "..", "data", "chat"))

_cache = {}
    
    
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=embedding_path,
        model_kwargs={"device": "cpu"},  # keeps VRAM free for the LLM
        encode_kwargs={"normalize_embeddings": True},
    )



@lru_cache(maxsize=1)
def get_llm():
    config_path = os.path.abspath(os.path.join(script_dir, "..", "config.json"))
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    temperature = config["Hyper_Param"]["temperature"]
    top_p = config["Hyper_Param"]["top_p"]
    max_tokens = config["Hyper_Param"]["max_tokens"]
    frequency_penalty = config["Hyper_Param"]["frequency_penalty"]
    presence_penalty = config["Hyper_Param"]["presence_penalty"]
    seed = config["Hyper_Param"]["seed"]
    top_k = config["Hyper_Param"]["top_k"]
    min_p = config["Hyper_Param"]["min_p"]
    repeat_penalty = config["Hyper_Param"]["repeat_penalty"]

    return ChatOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="none",
        model="local",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        extra_body={
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
        },
    )

embeddings = get_embeddings()
in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))

class ChatEvaluation(BaseModel):
    answer : str




def build_retriever(type):
    """ get raw documents for bm25 and build retriever functions
    because if I am calling retrieve function inside loop so having this logic in there means wastage of compute
    type here is either "chat" or "watcher" (watcher = run_batch), it decides which settings to read from config.json"""
    cos_store = get_store()
    K = 2

    # bm25 is the same for chat and watcher so build it only once and reuse it
    if "bm25" not in _cache:
        data = cos_store.get()
        corpus = [
            Document(page_content=t, metadata={**(m or {}), "chunk_id": i})
            for t, m, i in zip(data["documents"], data["metadatas"], data["ids"])
        ]
        _cache["bm25"] = BM25Retriever.from_documents(corpus, k=K)
    bm25 = _cache["bm25"]

    in_dir_config = os.path.abspath(os.path.join(script_dir, "..", "config.json"))
    with open(in_dir_config) as f:
        config = json.load(f)

    # pick the settings for chat or watcher
    if type == "chat":
        settings = config["retrieve_chat"]
    elif type == "watcher":
        settings = config["retrieve_watcher"]
    else:
        error_store(error_message="Invalid type", trace_back="Not Applicable", time=str(datetime.now(timezone.utc)), error_count="Not Applicable", error_file="ai_inference")
        raise ValueError(f"Invalid type: {type}")

    filter = settings["filter"]
    profile = settings["profile"]

    # sources the dense search is allowed to look in
    if filter == "master_directories":
        sources = ["directories"]
    elif filter == "notifications":
        sources = ["notifications"]
    elif filter == "both":
        sources = ["notifications", "directories"]
    elif filter == "company" or filter == "all":
        sources = ["notifications", "directories", "company"]
    else:
        error_store(error_message="Invalid filter", trace_back="Not Applicable", time=str(datetime.now(timezone.utc)), error_count="Not Applicable", error_file="ai_inference")
        raise ValueError(f"Invalid filter: {filter}")

    # custom profile also searches the profile chunks
    if profile == "custom":
        sources.append("profile")
    elif profile != "default":
        error_store(error_message="Invalid profile", trace_back="Not Applicable", time=str(datetime.now(timezone.utc)), error_count="Not Applicable", error_file="ai_inference")
        raise ValueError(f"Invalid profile: {profile}")

    dense = cos_store.as_retriever(search_kwargs={"k": K, "filter": {"source": {"$in": sources}}})

    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.2, 0.8])

chat_retriever = build_retriever("chat")
watcher_retriever = build_retriever("watcher")

_titles = load_titles(path)

TITLES = Literal[(*_titles.values(), "none")]
class ComplianceEvaluation(BaseModel,):
    """ Confirm schema and datatype of model output
    """
    title: str
    link: str
    is_relevant : str
    notification_date : str = Field(description="The date string formatted exactly as 'ddd, DD MMM YYYY HH:MM:SS' (e.g. 'Mon, 21 Sep 2026 17:25:00')")
    explanation : str
    actions_required : str
    effective_date: str = Field(description="The date string formatted exactly as 'ddd, DD MMM YYYY HH:MM:SS' (e.g. 'Mon, 21 Sep 2026 17:25:00')")
    retrieval_status : Literal[-1, 0, 1, 2]
    correct_master_dir : TITLES # pyright: ignore[reportInvalidTypeForm]
    master_dir_update : TITLES # pyright: ignore[reportInvalidTypeForm]


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

    
    in_dir_profile = os.path.abspath(os.path.join(script_dir,"..","profile.json"))
    with open(in_dir_profile, "r", encoding="utf-8") as file_profile:
        profile = json.load(file_profile)
        output_path = current_week_file(out_dir_output, format="jsonl")
        
        for i, json_line in enumerate(notifications):
            
            try : 
                    llm = get_llm()
                    structured_llm = llm.with_structured_output(ComplianceEvaluation)
                
                    results = []
                
                    retrieved_data = retrieve(json_line["clean_description"],watcher_retriever) # retireve data 
                    combined_text = "\n\n".join([doc.page_content for doc in retrieved_data])
                    recorvered_mds = find_mds(combined_text,_titles)
                    
                    if recorvered_mds:
                        double_hop_data = double_hop(combined_text, recorvered_mds)
                        final = retrieved_data + double_hop_data
                        combined_text = "\n\n---\n\n".join([
                                f"Document Title: {doc.metadata.get('doc_title', 'Unknown')}\nContent:\n{doc.page_content}"
                                for doc in final])
                        
                    if data["misc"]["tool_calling"] == "on":
                             
                        promt = f"""You are Regulatory compliance agent your task is to take company profile tagged with Profile : and Data : (Which is reserve bank of India Notifications)
                                and retrieved_data which rag retrieves on basis of notification from vector database made up of rbi master directories your job is to look at the data
                                and see if the notification is relevant to the company based on the profile if yes then look at the retrieved data to check the context and what changes 
                                the notification proposes to the master directory based on the below format and if retrieved data is wrong return actual master directory name from 
                                which data must be retrieved and only return it when it's needed like some notification are general rules that don't need retirveal but the architecture is 
                                two step retrieval so data always retrieves. 
                                "title" ( as mentioned in "data")
                                "link" ( as mentioned in "data")
                                "is relevant" (is notification relevant to profile and nbfc or not)
                                "explanation" (explain new changes and old changes and how it affect things in proper words as described in notification),
                                "actions_required" (actions required by the company to make).
                                "notification_date" (when the notification was issused)
                                "effective_date" ( when does the notification takes effect in only return when date is mentioned in notification otherwise return not mentioned in notification) 
                                "retrieval_status" : return -1 ( If incorrect ) or 0  ( if retrieval was not needed ) or 1 (Correct) and numbers only return 2 if you can't determine the master directory
                                "correct_master_dir" : Return actual master dir name if retrieval_status is -1 from the notification
                                "master_dir_update" : Return name of master directory that the notification ammends if nothing ammeneds return none

                                Profile:
                                {json.dumps(profile,indent = 2 )}
                                
                                Data : 
                                {json.dumps(json_line,indent = 2 )}
                                
                                Retrieved_Data
                                {combined_text}
                                """
                    else:
                        promt = f"""You are Regulatory compliance agent your task is to take company profile tagged with Profile : and Data : (Which is reserve bank of India Notifications)
                                and retrieved_data which rag retrieves on basis of notification from vector database made up of rbi master directories your job is to look at the data
                                and see if the notification is relevant to the company based on the profile if yes then look at the retrieved data to check the context and what changes 
                                the notification proposes to the master directory based on the below format and if retrieved data is wrong return retrieved data is wrong and changes can be explicatly explained please refer to the link
                                and also return this statement under "explanation", return effective date as not applicable and return correct_master_dir as tool_calling_off
                                only return it when it's needed like some notification are general rules that don't need retirveal but the architecture is 
                                two step retrieval so data always retrieves. 
                                "title" ( as mentioned in "data")
                                "link" ( as mentioned in "data")
                                "is relevant" (is notification relevant to profile and nbfc or not)
                                "explanation" (explane new changes and old changes and how it affect things in proper words as described in notification),
                                "actions_required" (actions required by the company to make).
                                "notification_date" (when the notification was issused)
                                "effective_date" ( when does the notification takes effect in only return when date is mentioned in notification otherwise return not mentioned in notification) 
                                "retrieval_status" : return -1 ( If incorrect ) or 0  ( if retrieval was not needed ) or 1 (Correct) and numbers only
                                "correct_master_dir" : Return actual master dir name if retrieval_status is -1 from the notification
                                "master_dir_update" : Return name of master directory that the notification ammends if nothing ameneds return none
                        
                                Profile:
                                {json.dumps(profile,indent = 2 )}                      
                                
                                Data : 
                                {json.dumps(json_line,indent = 2 )}
                                                        
                                Retrieved_Data
                                {combined_text}
                                """
                            
                    evaluation : ComplianceEvaluation = structured_llm.invoke(promt)
                    
                    tool_calling = data["misc"]["tool_calling"]
                    if tool_calling == "on":
                        tool_calling_retries = data["misc"]["tool_calling_retires"]
                        previous = None
                        for i in range(tool_calling_retries):
                            if evaluation.retrieval_status == -1:
                                if evaluation.correct_master_dir == previous:
                                    break
                                previous = evaluation.correct_master_dir
                                correct_master_dir = []
                                correct_master_dir.append(evaluation.correct_master_dir)
                                tool_called_data = double_hop(json_line["clean_description"],correct_master_dir)
                                correct_text = "\n\n".join([doc.page_content for doc in tool_called_data])
                                promt = f"""You are Regulatory compliance agent your task is to take company profile tagged with Profile : and Data : (Which is reserve bank of India Notifications)
                                        and retrieved_data which rag retrieves on basis of notification from vector database made up of rbi master directories your job is to look at the data
                                        and see if the notification is relevant to the company based on the profile if yes then look at the retrieved data to check the context and what changes 
                                        the notification proposes to the master directory based on the below format and if retrieved data is wrong return actual master directory name from 
                                        which data must be retrieved and only return it when it's needed like some notification are general rules that don't need retirveal but the architecture is 
                                        two step retrieval so data always retrieves. 
                                        "title" ( as mentioned in "data")
                                        "link" ( as mentioned in "data")
                                        "is relevant" (is notification relevant to profile and nbfc or not)
                                        "explanation" (explane new changes and old changes and how it affect things in proper words as described in notification),
                                        "actions_required" (actions required by the company to make).
                                        "notification_date" (when the notification was issused)
                                        "effective_date" ( when does the notification takes effect in only return when date is mentioned in notification otherwise return not mentioned in notification) 
                                        "retrieval_status" : return -1 ( If incorrect ) or 0  ( if retrieval was not needed ) or 1 (Correct) and numbers only return 2 if you can't determine the master directory
                                        "correct_master_dir" : Return actual master dir name if retrieval_status is -1 from the notification otherwise return none
                                        "master_dir_update" : Return name of master directory that the notification ammends if nothing ameneds return none
                                        Profile:
                                        {json.dumps(profile,indent = 2 )}                       
                                        Data : 
                                        {json.dumps(json_line,indent = 2 )}
                                                                
                                        Retrieved_Data
                                        {correct_text}
                                        """
                                if not tool_called_data:
                                    break
                                try:
                                    evaluation = structured_llm.invoke(promt)
                                except Exception as e:
                                    error_store(error_message=str(e), trace_back=traceback.format_exc(),
                                                time=str(datetime.now(timezone.utc)), error_count="Null",
                                                error_file="ai_inference")
                                    break
                            else:
                                break
                        if evaluation.retrieval_status == -1:
                            evaluation.explanation = f"The model could not retrieve the correct master direction after {tool_calling_retries} retries, so the data could not be retrieved. Please refer to the link for the full details of this notification."
                            evaluation.actions_required = "Not available. Please refer to the link."
                            evaluation.effective_date = "Not available"   
                    
                    else:
                        if evaluation.retrieval_status in (-1,2):
                            evaluation.explanation = f"The model could not retrieve the correct master direction, the correct could not be generated . Please refer to the link for the full details of this notification."
                            evaluation.actions_required = "Not available. Please refer to the link."
                            evaluation.effective_date = "Not available"   
                                 
                    if  evaluation.correct_master_dir != "none":
                        print("Hi")
                    
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
    return re.sub(r"[^\w\- ]", "", name).strip()[:80] or "default"


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


def chat_job(sid, question, max_turns=6):
    log(f"chat_job called, file={__file__}")
    sid = safe_chat_name(sid)
    if sid not in _histories:
        _histories[sid] = load_chat(sid)
    hist = _histories[sid]

    recent_user = [m["content"] for m in hist if m["role"] == "user"][-2:]
    with timer("chat/retrieve"):
        chunks = retrieve("\n".join(recent_user + [question]), chat_retriever)
    chunks_text = "\n\n".join(d.page_content for d in chunks)
    
    mds = find_mds(chunks_text,_titles)
    if mds:
        with timer("chat/double_hop"):
            chunks_text = format_docs(chunks + double_hop(question, mds)[:4])

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=chunks_text))]
    for m in hist[-2 * max_turns:]:
        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=question))
    log(f"[timing] chat prompt chars: {sum(len(m.content) for m in messages)}")

    with timer("chat/llm"):
        answer = get_llm().invoke(messages).text

    new = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    save_chat(sid, new)
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