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
    questions = {
        "applies_to_nbfc": {
            "type": "noul",
            "instructions": (
                "Does this RBI notification apply to non-banking financial companies (NBFCs)? "
                "Answer yes if it is addressed to, or places obligations on, any of these: "
                "NBFCs of any kind (investment and credit companies, infrastructure finance companies, "
                "microfinance institutions, factors, peer-to-peer lending platforms, account aggregators, "
                "infrastructure debt funds), Housing Finance Companies, Core Investment Companies, "
                "Mortgage Guarantee Companies, or Standalone Primary Dealers. "
                "Also answer yes if it applies to all RBI-regulated entities or all financial institutions "
                "without naming NBFCs, such as KYC, AML or sanctions-list (UAPA) notices. "
                "Answer no if it is addressed only to banks (commercial banks, small finance banks, payments banks, "
                "regional rural banks, local area banks, urban or rural co-operative banks), All India Financial "
                "Institutions, Credit Information Companies or Asset Reconstruction Companies, "
                "even if the text mentions NBFCs in passing, for example as a borrower, counterparty or investee."
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