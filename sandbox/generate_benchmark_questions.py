import os
import json
import random
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL = "gemini-3.8-flash"              # use the Gemini model you already use
script_dir = os.path.dirname(os.path.abspath(__file__))
notifications = os.path.dirname(os.path.abspath(script_dir,"filtered_notifications_deduped"))
config = os.path.abspath(os.path.join(script_dir, "..","config.json"))
with open(config, "r", encoding="utf-8") as f:
        config = json.load(f)
embedding_path = config["Model_Details"]["Embedding_Model_path"]


OUT_FILE = os.path.join(os.path.abspath(script_dir,"..","data","benchmark","data"))


llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=os.getenv("GEMINI"), temperature=0.7)

TYPES = {
    "factual": "a simple factual question answered directly by the passage",
    "numeric": "a question whose answer is a specific number, date or threshold from the passage",
    "reworded": "a question a busy compliance officer would ask in casual words, without using the passage's own phrases",
    "multi_chunk": "a question that needs information from BOTH passages to answer fully",
    "unanswerable": "a realistic question on the same topic that the passage does NOT answer",
}

# 100 questions in total
COUNTS = {"factual": 30, "numeric": 20, "reworded": 20, "multi_chunk": 20, "unanswerable": 10}

chunks = [json.loads(line) for line in open(notifications, encoding="utf-8")]
random.seed(1)

# one list of question types, shuffled, and a different starting chunk for each question
plan = [t for t, n in COUNTS.items() for _ in range(n)]
random.shuffle(plan)
starts = random.sample(range(len(chunks) - 1), len(plan))

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for q_type, start in zip(plan, starts):

        # multi_chunk uses two neighbouring chunks, the others use one
        if q_type == "multi_chunk":
            used = chunks[start:start + 2]
        else:
            used = [chunks[start]]

        passages = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in used)

        prompt = f"""You are writing test questions for a RAG system over RBI regulations for NBFCs.

Passage(s):
{passages}

Write {TYPES[q_type]}.
Answer using only the passage(s). For an unanswerable question, the answer is "Not covered in the documents."

Reply with JSON only, in this format:
{{"question": "...", "reference_answer": "..."}}"""

        try:
            reply = llm.invoke(prompt).content
            reply = reply.replace("```json", "").replace("```", "").strip()
            data = json.loads(reply)
        except Exception as e:
            print("skipped:", e)
            time.sleep(5)
            continue

        data["type"] = q_type
        data["source_titles"] = [c["title"] for c in used]
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        time.sleep(1)  # avoid hitting rate limits
        
# with open(notifications,"r") as f:
#     for i,line in enumerate:
#         data = json.load(f)
#         data = 