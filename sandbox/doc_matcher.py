import os
import json
import re
import unicodedata
import jsonlines
UPDATED_RE = re.compile(r"\s*\(updated as on [^)]*\)", re.IGNORECASE)


def clean(title: str) -> str:
    return UPDATED_RE.sub("", title).strip()

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)   
    s = re.sub(r"\s*-\s*", "-", s)                 
    s = re.sub(r"\s+", " ", s)                     
    return s.lower().strip()

def load_titles(path: str) -> dict[str, str]:
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                t = clean(obj["title"] if isinstance(obj, dict) else obj)
                titles[norm(t)] = t
    return titles

def find_mds(text: str, titles: dict[str, str]) -> list[str]:
    t = norm(text)
    return [title for key, title in titles.items() if key in t]


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(script_dir, "..", "data", "master_directory", "master_directory.jsonl")
    titles = load_titles(path)                 

with jsonlines.open(path, mode="r") as r:
    for obj in r:
        query = obj["text"]

        for t in find_mds(query, titles):
            print(f"{obj["id"]}"," -", t)