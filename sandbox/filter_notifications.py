import os
import re
import json
import time
import traceback
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI api key from the .env file

script_dir = os.path.dirname(os.path.realpath(__file__))

# Input: .../data/notification_historical/notification.jsonl
in_file = os.path.abspath(os.path.join(script_dir, "..", "data", "notification_historical", "notification.jsonl"))
# Output + cache: same folder as the input
out_file = os.path.join(os.path.dirname(in_file), "filtered_notifications.jsonl")
cache_file = os.path.join(os.path.dirname(in_file), "gemini_cache.json")

REQUIRED_COLUMNS = ["title", "url"]  # the filter only uses these two

GEMINI_PROMPT = (
    "Below is the title and text of an RBI notification. Answer only 'yes' or 'no'. "
    "Answer 'yes' if it is relevant to NBFCs (non-banking financial companies, housing finance companies, "
    "core investment companies, standalone primary dealers, mortgage guarantee companies, NOFHCs) "
    "OR applicable to everyone (all regulated entities, all banks and NBFCs). "
    "Answer 'no' if it is only for other entity types.\n\n"
)

NBFC_PATTERN = re.compile(
    r"non[\s-]?banking financial compan|nbfc"
    r"|core investment compan(?:y|ies)"
    r"|standalone primary dealer|\bspd\b"
    r"|mortgage guarantee compan(?:y|ies)|\bmgc\b"
    r"|non-?operative financial holding compan(?:y|ies)|\bnofhc\b"
    r"|housing finance compan(?:y|ies)|\bhfc\b",
    re.IGNORECASE
)

# Regex patterns for non-NBFC entities
OTHER_ENTITY_PATTERN = re.compile(
    r"regional rural bank"
    r"|urban co-?operative bank"
    r"|rural co-?operative bank"
    r"|state co-?operative bank"
    r"|district central co-?operative bank"
    r"|scheduled commercial bank"
    r"|commercial bank"
    r"|payments? bank"
    r"|small finance bank"
    r"|local area bank"
    r"|co-?operative bank"
    r"|banker and debt manager to government"
    r"|banker to governments? and banks"
    r"|consumer education and protection"
    r"|all india financial institutions?"
    r"|asset reconstruction compan(?:y|ies)"
    r"|credit information compan(?:y|ies)"
    r"|financial inclusion and development"
    r"|financial market"
    r"|issuer of currency"
    r"|payments? and settlement systems?",
    re.IGNORECASE
)

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]  # tried in order
GEMINI_ROUNDS = 2       # full passes over the models before giving up on a row
GEMINI_DELAY = 1        # seconds between calls, set to 60 / your requests-per-minute limit
GEMINI_BACKOFF = 10     # seconds, multiplied by the round number after a failed round
GEMINI_TIMEOUT_MS = 30_000  # per-request timeout in milliseconds
GEMINI_GIVE_UP = 5      # stop calling Gemini after this many rows in a row fail

unread_streak = 0       # rows in a row that Gemini didn't read
active_models = list(GEMINI_MODELS)
cache = {}              # url -> True/False


def load_cache():
    global cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached Gemini answers from {cache_file}", flush=True)
        except Exception as e:
            print(f"Could not read cache ({e}), starting fresh", flush=True)
            cache = {}


def save_cache():
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def check_models(client):
    """Keep only the models your key can actually use, and print them."""
    global active_models
    try:
        available = {m.name.replace("models/", "") for m in client.models.list()}
    except Exception as e:
        print(f"Could not list models ({e}); using the configured list as is", flush=True)
        return
    print("Gemini models available to your key:", flush=True)
    for name in sorted(n for n in available if n.startswith("gemini")):
        print(f"  {name}", flush=True)
    valid = [m for m in GEMINI_MODELS if m in available]
    invalid = [m for m in GEMINI_MODELS if m not in available]
    if invalid:
        print(f"WARNING: these configured models do not exist for your key and will be skipped: {invalid}", flush=True)
    if not valid:
        raise RuntimeError("None of GEMINI_MODELS are available. Edit GEMINI_MODELS using the list above.")
    active_models = valid
    print(f"Using models: {active_models}", flush=True)


def ask_gemini(client, url, title, text):
    """True/False from Gemini, or "didn't read" if no model could answer"""
    global unread_streak

    if url in cache:
        return cache[url]

    if unread_streak >= GEMINI_GIVE_UP:  # quota is gone, don't waste minutes per row
        return "didn't read"

    text = text if isinstance(text, str) else ""
    contents = GEMINI_PROMPT + f"Title: {title}\n\nText: {text[:3000]}"

    for attempt in range(GEMINI_ROUNDS):
        for model in list(active_models):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                answer = (resp.text or "").strip().lower().startswith("yes")
                unread_streak = 0
                cache[url] = answer
                save_cache()
                time.sleep(GEMINI_DELAY)
                return answer
            except Exception as e:
                msg = str(e)
                print(f"    {model} failed: {msg[:150]}", flush=True)
                if "404" in msg or "NOT_FOUND" in msg:
                    active_models.remove(model)  # model doesn't exist, don't retry it
                    print(f"    removed {model} from active models", flush=True)
        if not active_models:
            break
        wait = GEMINI_BACKOFF * (attempt + 1)
        print(f"    all models failed, round {attempt + 1}/{GEMINI_ROUNDS}, waiting {wait}s", flush=True)
        time.sleep(wait)

    unread_streak += 1
    print(f"    row not read ({unread_streak}/{GEMINI_GIVE_UP} in a row)", flush=True)
    return "didn't read"


def filter_historical_notifications(in_path=in_file, out_path=out_file):
    try:
        if not os.path.exists(in_path) or os.path.getsize(in_path) == 0:
            print(f"Input file missing or empty: {in_path}")
            return False

        # dtype/convert_dates off so values are written back exactly as they were read
        df = pd.read_json(in_path, lines=True, dtype=False, convert_dates=False)
        if df.empty:
            print("No notification data in input file")
            return False

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{in_path} is missing columns: {missing}")

        total = len(df)
        df = df.drop_duplicates(subset="url").reset_index(drop=True)
        deduped = len(df)

        # Same filter logic as match_data()
        df["is_nbfc_in_title"] = df["title"].str.contains(NBFC_PATTERN, na=False)
        df["names_other_entity"] = df["title"].str.contains(OTHER_ENTITY_PATTERN, na=False)
        df["discard"] = df["names_other_entity"] & ~df["is_nbfc_in_title"]
        discarded = int(df["discard"].sum())

        kept = df[~df["discard"]].copy()
        regex_kept = len(kept)
        print(f"Read {total} | after de-dup {deduped} | regex discarded {discarded} | regex kept {regex_kept}", flush=True)

        # Gemini only checks what the regex kept: NBFC related or applicable to everyone?
        client = genai.Client(
            api_key=os.getenv("GEMINI"),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        check_models(client)
        load_cache()

        n = len(kept)
        answers = []
        start = time.time()
        print(f"\nAsking Gemini about {n} notifications...\n", flush=True)
        for i, (_, r) in enumerate(kept.iterrows(), 1):
            t0 = time.time()
            was_cached = r["url"] in cache
            ans = ask_gemini(client, r["url"], r["title"], r.get("text"))
            answers.append(ans)
            label = {True: "YES", False: "NO"}.get(ans, "UNREAD")
            src = "cache" if was_cached else f"{time.time() - t0:.1f}s"
            print(f"[{i}/{n}] {label:6} ({src}) | {str(r['title'])[:80]}", flush=True)
        print(f"\nGemini pass finished in {time.time() - start:.0f}s\n", flush=True)

        kept["gemini_yes"] = pd.Series(answers, index=kept.index, dtype="object")
        yes_count = sum(a is True for a in answers)
        no_count = sum(a is False for a in answers)
        unread = sum(a == "didn't read" for a in answers)

        kept = kept[kept["gemini_yes"] != False]  # drops only a real "no", "didn't read" rows stay

        print(
            f"Summary: read {total} | after de-dup {deduped} | regex discarded {discarded} | "
            f"regex kept {regex_kept} | gemini yes {yes_count} | gemini no {no_count} | "
            f"didn't read {unread} | final kept {len(kept)}",
            flush=True,
        )

        if kept.empty:
            print("None Saved")
            return False

        payload = kept.to_json(orient="records", lines=True, force_ascii=False)
        if not payload.endswith("\n"):
            payload += "\n"
        with open(out_path, "w", encoding="utf-8") as f:   # "w": rerunning rebuilds the file, no duplicates
            f.write(payload)

        print(f"Saved {len(kept)} rows -> {out_path}")
        return True

    except Exception:
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    filter_historical_notifications()