import os
import time
import json
import requests
from find_reference import load_titles

script_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(script_dir,"..", "..", "data", "master_directory", "master_directory.jsonl")
titles = list(load_titles(path).values())
notification = os.path.abspath(os.path.join(script_dir,"..", "..", "data", "notifications_matched", "test.jsonl"))
profile = os.path.join(script_dir,"..","profile.json")


import time
t = time.perf_counter()

def jev_the_builder(description):
    criteria = {t: "" for t in titles}
    criteria["none"] = "None of these"
    
    questions = {
        "applies_to_nbfc": {
            "type": "noul",
            "instructions": (
                "An NBFC is a non-banking financial company registered with the RBI: it lends or invests "
                "but is not a bank. Commercial banks, co-operative banks, small finance banks and payments banks "
                "are NOT NBFCs. Is this notification addressed to or applicable to NBFCs?"
            )
        }
    }
    key_to_title = {}
    for i, title in enumerate(titles):
        key = f"md_{i}"
        key_to_title[key] = title
        questions[key] = {
            "type": "noul",
            "instructions": f"Does this notification amend or relate to the RBI master direction '{title}'?",
        }
    
    body = {
        "model": "jev-latest",
        "state": description,
        "questions": questions,
    }
    r = requests.post("http://127.0.0.1:8017/v1/systemone", json=body, timeout=60)
    print(f"{r.status_code} | {time.perf_counter() - t:.2f}s | server: {r.headers.get('Server-Timing')}")
    return r.json(), key_to_title
    
   
    

if __name__ == "__main__":

    print("reading:", notification, "| exists:", os.path.exists(notification))
    rows = 0
    with open(notification, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows += 1
            item = json.loads(line)
            description = item.get("title", "") + "\n\n" + (item.get("clean_description") or item.get("description") or item.get("text", ""))
            if not description.strip():
                print(i, "skipped, empty description | keys:", list(item.keys()))
                continue
            data, key_to_title = jev_the_builder(description)
            answers = data["answers"]
            print(item.get("title", ""))
            print("applies_to_nbfc:", round(answers["applies_to_nbfc"]["noul"], 3))
            ranked = sorted(key_to_title, key=lambda k: answers[k]["noul"], reverse=True)
            for k in ranked[:5]:
                print(round(answers[k]["noul"], 3), key_to_title[k])
            print()
    print("rows read:", rows)