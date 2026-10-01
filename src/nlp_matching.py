import os 
import json
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Pattern: the word "master" followed by "direction(s)"
matcher.add("MD", [[
    {"LOWER": "master"},
    {"LOWER": {"IN": ["direction", "directions"]}},
]])



script_dir = os.path.dirname(os.path.realpath(__file__))
in_dir_config = os.path.abspath(os.path.join(script_dir, "..", "config.json"))
with open(in_dir_config) as file:
    data = json.load(file)


file_path = data["data_check"]["matched_ref_file"]
with open(file_path,"r",encoding = "utf-8") as file:
    lines = file.readlines()
    
for i in range(0,1):
    query = json.loads(lines[1])

# sent = "The NBFC shall follow Master Direction on Know Your Customer, and the provisions thereof."
query = """RBI/2026-27/237
DOR.CRE.REC.210/07-03-008/2026-27

August 25, 2026

Reserve Bank of India (Non-Banking Financial Companies – Concentration Risk Management) Fourth Amendment Directions, 2026

The Reserve Bank has issued the Reserve Bank of India (Non-Banking Financial Companies – Concentration Risk Management) Directions, 2025 dated November 28, 2025 (hereinafter referred to as ‘Directions’). On a review, it has been decided to revise the large exposure framework for Infrastructure Debt Fund-Non-Banking Financial Company (IDF-NBFC) in the Upper Layer.

2. Accordingly, in exercise of the powers conferred by Chapter III B of the Reserve Bank of India Act, 1934, and all other provisions / laws enabling the Reserve Bank of India (‘RBI’) in this regard, RBI being satisfied that it is necessary and expedient in the public interest so to do, hereby, issues the Amendment Directions hereinafter specified.

3. These Amendment Directions shall be called the Reserve Bank of India (Non-Banking Financial Companies – Concentration Risk Management) Fourth Amendment Directions, 2026.

4. These Amendment Directions shall come into force with immediate effect.

5. These Amendment Directions shall modify the Directions as under:

(1) After paragraph 39 in ‘Chapter IV - Guidelines Applicable to NBFC – Upper Layer’, a new paragraph 39A shall be inserted as under:

39A. The large exposure limits applicable to NBFC-IFC shall also be applicable to IDF-NBFC that are subject to Upper Layer regulations in terms of paragraph 60A of the Reserve Bank of India (Non-Banking Financial Companies – Undertaking of Financial Services) Directions, 2025 read together with paragraph 18 (4) (i) of the Reserve Bank of India (Commercial Banks – Undertaking of Financial Services) Directions, 2025.

(Dr. Sudarsana Sahoo)
Chief General Manager"""
doc = nlp(query)                      # spaCy splits the text into tokens

for _, start, end in matcher(doc):   # start/end are token positions of "Master Direction"
    print("Matched:", doc[start:end].text)

    # Walk forward from the match until we hit a stop token
    stop = end
    for i in range(end, len(doc)):
        if doc[i].text in {",", ";", "."} or doc[i].text.lower() in {"and", "shall", "which"}:
            stop = i
            break

    print("Title guess:", doc[end:stop].text)