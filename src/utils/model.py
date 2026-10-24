# src/utils/models.py
import json, os
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

with open(os.path.join(_root, "config.json"), encoding="utf-8") as f:
    _embedding_path = json.load(f)["Model_Details"]["Embedding_Model_path"]

CHROMA_DIR = os.path.join(_root, "chroma_db")


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=_embedding_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_store():
    return Chroma(
        collection_name="master_directories",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )