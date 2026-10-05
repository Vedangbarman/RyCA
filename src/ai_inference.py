import os
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

load_dotenv()
API_KEY = os.getenv('GEMINI')
os.environ["GOOGLE_API_KEY"] = API_KEY


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
    
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

script_dir = os.path.dirname(os.path.realpath(__file__))
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


def retrieve(query, hybrid, k=4):
    return hybrid.invoke(query)[:k]


def invoke_ai():
    """ Match incoming notifications against rag database check for other master directory mention and if yes then retrieve 
    relevant data from that particular document """
    script_dir = os.path.dirname(os.path.realpath(__file__))
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
    path = os.path.join(script_dir, "..", "data", "master_directory", "master_directory.jsonl")
    with open(in_dir_profile, "r", encoding="utf-8") as file_profile:
        profile = json.load(file_profile)
        output_path = current_week_file(out_dir_output, format="jsonl")
        
        hybrid = build_retriever()
        titles = load_titles(path)
        for i, json_line in enumerate(notifications):
            
            try : 
                
                    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
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
    
SYSTEM_PROMPT = """You are a compliance assistant for RBI master directions.
Answer using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}"""

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"Document Title: {d.metadata.get('doc_title', 'Unknown')}\nContent:\n{d.page_content}"
        for d in docs
    )

def chat_ai(max_turns=6):
    """Function to chat with AI takes user query like "What is meant by Upper Layer in NBFC" and then find the relevant chunk 
    in rag database, it then checks the retrieved data for other master directory reference and then retrive the relevant 
    from those specific documents"""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    hybrid = build_retriever()   # build once, outside the loop
    history = []                 # HumanMessage / AIMessage objects
    path = os.path.join(script_dir, "..", "data", "master_directory", "master_directory.jsonl")
    titles = load_titles(path) # load titles 
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        # retrieval query: last 2 user turns + current question + double hop data 
        
        recent_user = [m.content for m in history if isinstance(m, HumanMessage)][-2:]
        
        chunks = retrieve("\n".join(recent_user + [question]), hybrid)
        chunks_text = "\n\n".join([doc.page_content for doc in chunks])
        recorvered_mds = find_mds(chunks_text,titles)
        
        if recorvered_mds:
            double_hop_data = double_hop(chunks_text,titles)
            final = chunks + double_hop_data
            chunks_text = "\n\n---\n\n".join([f"Document Title: {doc.metadata.get('doc_title', 'Unknown')}\nContent:\n{doc.page_content}"for doc in final])
            
            
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(chunks_text))]
        messages += history[-2 * max_turns:]
        messages.append(HumanMessage(content=question))

        response = llm.invoke(messages)
        answer = response.text   # str; .content may be a list of blocks
        print(answer)

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
if __name__ == "__main__":
    chat_ai()