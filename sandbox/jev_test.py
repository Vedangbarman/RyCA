import os
import io
import re
import sys
import time
import json
import random
import contextlib

script_dir = os.path.dirname(os.path.realpath(__file__))
root = os.path.abspath(os.path.join(script_dir, ".."))      # sandbox sits directly under the project root
sys.path.insert(0, os.path.join(root, "src", "utils"))      # lets "import jev" find src/utils/jev.py

import jev
from jev import jev_the_builder

in_dir = os.path.join(root, "data", "notification_historical")
notification_path = os.path.join(in_dir, "notification.jsonl")
filtered_path = os.path.join(in_dir, "filtered_notifications_deduped.jsonl")
master_path = os.path.join(root, "data", "master_directory", "master_directory_html.jsonl")   # set this to the master file you switched to
out_path = os.path.join(script_dir,"..","output","" "jev_scores.jsonl")

MAX_CHARS = 6000        # about 1500 tokens, the server limit is 8192
SAMPLE_REST = 90        # how many notifications outside the filtered file to test as well
MASTER_RE = re.compile(r"Directions, 20\d\d(\s*\(Updated as on [^)]*\))?\s*$", re.IGNORECASE)


def read_jsonl(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_master_title(title):
    return bool(MASTER_RE.search(title)) and "amendment" not in title.lower()


if __name__ == "__main__":

    print("using:", jev.__file__)
    print("files exist:", os.path.exists(notification_path), os.path.exists(filtered_path), os.path.exists(master_path))

    notifications = read_jsonl(notification_path)
    filtered = read_jsonl(filtered_path)
    master_ids = {str(row["id"]) for row in read_jsonl(master_path)}
    print("notification.jsonl:", len(notifications), "| filtered:", len(filtered), "| master ids:", len(master_ids))

    for k in ("is_nbfc_in_title", "names_other_entity", "discard", "gemini_yes"):
        print(k, {v: sum(1 for r in filtered if r.get(k) == v) for v in {r.get(k) for r in filtered}})

    # 1. remove the master directions that are in the master file
    removed = [row for row in notifications if str(row.get("id")) in master_ids]
    notifications = [row for row in notifications if str(row.get("id")) not in master_ids]
    print("removed by master id:", len(removed), "| left:", len(notifications))

    # 2. remove full master directions of other entity types by title (never touches the filtered rows)
    filtered_urls = {row.get("url") for row in filtered}
    kept = []
    dropped = []
    for row in notifications:
        if is_master_title(row.get("title", "")) and row.get("url") not in filtered_urls:
            dropped.append(row)
        else:
            kept.append(row)
    notifications = kept
    print("dropped by title:", len(dropped), "| left:", len(notifications))
    for row in dropped[:8]:
        print("  dropped:", row.get("title", ""))

    # 3. test every filtered row plus a fixed random sample of the rest
    in_filtered = [row for row in notifications if row.get("url") in filtered_urls]
    rest = [row for row in notifications if row.get("url") not in filtered_urls]
    rows = in_filtered + rest
    print("testing:", len(in_filtered), "filtered +", len(rest), "from the rest")

    results = []
    failed = 0
    t_start = time.perf_counter()

    for i, item in enumerate(rows):
        text = item.get("clean_description") or item.get("description") or item.get("text", "")
        description = (item.get("title", "") + "\n\n" + text)[:MAX_CHARS]

        with contextlib.redirect_stdout(io.StringIO()):         # hides the per-call print inside jev.py
            data, key_to_title = jev_the_builder(description)

        if "answers" not in data:
            failed += 1
            if failed <= 3:
                print("failed:", item.get("title", "")[:80], str(data)[:200])
            continue

        answers = data["answers"]
        ranked = sorted(key_to_title, key=lambda k: answers[k]["noul"], reverse=True)
        results.append({
            "url": item.get("url"),
            "title": item.get("title", ""),
            "is_nbfc": item.get("url") in filtered_urls,
            "nbfc": answers["applies_to_nbfc"]["noul"],
            "top": [[key_to_title[k], answers[k]["noul"]] for k in ranked[:3]],
        })

        if (i + 1) % 25 == 0:
            print(i + 1, "/", len(rows), f"{time.perf_counter() - t_start:.0f}s")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    in_set = [r for r in results if r["is_nbfc"]]
    not_in_set = [r for r in results if not r["is_nbfc"]]
    print()
    print("tested:", len(results), "| failed:", failed, f"| time: {time.perf_counter() - t_start:.0f}s")
    print("in filtered file:", len(in_set), "| not in filtered file:", len(not_in_set))
    for cut in (0.5, 0.8, 0.9):
        a = sum(1 for r in in_set if r["nbfc"] >= cut)
        b = sum(1 for r in not_in_set if r["nbfc"] >= cut)
        print(f">= {cut}: yes on filtered {a}/{len(in_set)} | yes on the rest {b}/{len(not_in_set)}")

    print("lowest 10 among filtered (these would be wrongly skipped):")
    for r in sorted(in_set, key=lambda r: r["nbfc"])[:10]:
        print(round(r["nbfc"], 3), "|", r["title"])

    print("highest 10 among the rest:")
    for r in sorted(not_in_set, key=lambda r: r["nbfc"], reverse=True)[:10]:
        print(round(r["nbfc"], 3), "|", r["title"])