import os
import time
import json
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_data = os.path.abspath(os.path.join(script_dir, "..", "chroma_db"))
in_dir_test = os.path.abspath(os.path.join(script_dir, "..", "data", "benchmark", "data", "test_queries.jsonl"))
out_dir_results = os.path.abspath(os.path.join(script_dir, "..", "data", "benchmark", "results", "results.jsonl"))
os.makedirs(os.path.dirname(out_dir_results), exist_ok=True)

api_key = os.getenv("GEMINI")
os.environ["GOOGLE_API_KEY"] = api_key

K = 4
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def load_store():
    return Chroma(
        collection_name=f"master_directories",
        embedding_function=embeddings,
        persist_directory=in_dir_data,
    )


cos_store = load_store()

data = cos_store.get()   # includes "ids"
corpus = [
    Document(page_content=t, metadata={**(m or {}), "chunk_id": i})
    for t, m, i in zip(data["documents"], data["metadatas"], data["ids"])
]

bm25 = BM25Retriever.from_documents(corpus, k=K)
dense = cos_store.as_retriever(search_kwargs={"k": K})
hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

dense_weights = [0.5, 0.6, 0.7, 0.8]
hybrids = {
    f"hybrid{int(w*100)}": EnsembleRetriever(retrievers=[bm25, dense], weights=[1 - w, w])
    for w in dense_weights
}

def search(query, method):
    if method == "cosine":
        return cos_store.similarity_search(query, k=K)
    if method == "mmr":
        return cos_store.max_marginal_relevance_search(query, k=K, fetch_k=20)
    if method == "bm25":
        return bm25.invoke(query)
    if method in hybrids:
        return hybrids[method].invoke(query)[:K]
    raise ValueError(method)


# ---------- relevance + scoring ----------
def chunk_id(doc):
    return doc.metadata.get("chunk_id") or doc.metadata.get("id")


def is_relevant(doc, item):
    if "relevant_ids" in item and chunk_id(doc) is not None:
        return chunk_id(doc) in item["relevant_ids"]
    text = doc.page_content.lower()
    return any(s.lower() in text for s in item.get("relevant_snippets", []))


def score(docs, item):
    rel = [is_relevant(d, item) for d in docs]

    if item.get("relevant_ids"):
        targets = set(item["relevant_ids"])
        found = {chunk_id(d) for d in docs if chunk_id(d) in targets}
        n_total = len(targets)
        # gain only for the first time a target is seen
        seen, gains = set(), []
        for d in docs:
            cid = chunk_id(d)
            gains.append(1 if cid in targets and cid not in seen else 0)
            seen.add(cid)
    else:
        snippets = [s.lower() for s in item.get("relevant_snippets", [])]
        n_total = len(snippets) or 1
        found, gains = set(), []
        for d in docs:
            text = d.page_content.lower()
            new = [s for s in snippets if s in text and s not in found]
            gains.append(1 if new else 0)
            found.update(new)

    hit = float(any(rel))
    precision = sum(rel) / K
    recall = len(found) / n_total
    mrr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(n_total, K)))
    ndcg = dcg / idcg if idcg else 0.0

    return {"hit": hit, "precision": precision, "recall": recall, "mrr": mrr, "ndcg": ndcg}



methods = ["cosine", "mmr", "bm25"] + list(hybrids)
agg = {m: {} for m in methods}

with open(in_dir_test, "r", encoding="utf-8") as fin, \
     open(out_dir_results, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        query = item["query"]

        for method in methods:
            t0 = time.perf_counter()
            docs = search(query, method)
            latency = time.perf_counter() - t0

            s = {**score(docs, item), "latency": latency}
            for k, v in s.items():
                agg[method].setdefault(k, []).append(v)

            fout.write(json.dumps({
                "query": query,
                "method": method,
                **s,
                "retrieved": [
                    {"chunk_id": chunk_id(d), "content": d.page_content[:200]}
                    for d in docs
                ],
            }, default=str) + "\n")


print(f"{'method':<8}{'hit':>6}{'prec':>7}{'recall':>8}{'mrr':>7}{'ndcg':>7}{'ms':>9}")
for m, vals in agg.items():
    print(f"{m:<8}{np.mean(vals['hit']):>6.2f}{np.mean(vals['precision']):>7.2f}"
          f"{np.mean(vals['recall']):>8.2f}{np.mean(vals['mrr']):>7.2f}"
          f"{np.mean(vals['ndcg']):>7.2f}{np.mean(vals['latency'])*1000:>9.1f}")
        
# query = "What is the defination of ‘qualifying assets’ of NBFC-MFIs"
# vector = embeddings.embed_query(query)
# magnitude = np.linalg.norm(vector)

# print(f"Vector length (dimensions): {len(vector)}")
# print(f"Vector magnitude: {magnitude:.6f}")








