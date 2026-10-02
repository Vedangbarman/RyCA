import re
import os
import json
import traceback
import pandas as pd
from datetime import datetime, timezone
from utils.error_store import error_store
from utils.week_file_save import current_week_file


REQUIRED_COLUMNS = ["title", "clean_description", "link"]

NBFC_PATTERN = re.compile(
    r"non[\s-]?banking financial compan|nbfc"
    r"|core investment compan(?:y|ies)"
    r"|standalone primary dealer|\bspd\b"
    r"|mortgage guarantee compan(?:y|ies)|\bmgc\b"
    r"|non-?operative financial holding compan(?:y|ies)|\bnofhc\b"
    r"|housing finance compan(?:y|ies)|\bhfc\b",
    re.IGNORECASE
)

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


def match_data(notifications_data):
    try:
        if not isinstance(notifications_data, pd.DataFrame) or notifications_data.empty:
            print("No notification data received")
            return False

        missing = [c for c in REQUIRED_COLUMNS if c not in notifications_data.columns]
        if missing:
            raise ValueError(f"notifications_data is missing columns: {missing}")

        file_path = os.path.dirname(os.path.realpath(__file__))
        in_dir_config_file = os.path.abspath(os.path.join(file_path, "..","..", "config.json"))

        out_path = os.path.join(file_path, "..", "data", "notifications_matched")
        os.makedirs(out_path, exist_ok=True)
        out_path_json = current_week_file(out_path, format="json")

        notifications = notifications_data.drop_duplicates(subset="link").copy()

        if os.path.exists(out_path_json) and os.path.getsize(out_path_json) > 0:
            done = set(pd.read_json(out_path_json, lines=True)["link"])
            notifications = notifications[~notifications["link"].isin(done)]

        if notifications.empty:
            print("Nothing new to match")
            return False

        notifications = notifications.reset_index(drop=True)

        notifications["is_nbfc_in_title"] = notifications["title"].str.contains(NBFC_PATTERN, na=False)
        notifications["names_other_entity"] = notifications["title"].str.contains(OTHER_ENTITY_PATTERN, na=False)
        notifications["discard"] = notifications["names_other_entity"] & ~notifications["is_nbfc_in_title"]

        kept = notifications[~notifications["discard"]].copy()

        if len(kept) > 0:
            payload = kept.to_json(orient="records", lines=True, force_ascii=False)
            if not payload.endswith("\n"):
                payload += "\n"
            with open(out_path_json, "a", encoding="utf-8") as f:
                f.write(payload)
            print(f"{len(kept)} : Candidates Saved")

            with open(in_dir_config_file) as config_file:
                config = json.load(config_file)
            config["data_check"]["matched_ref_file"] = str(out_path_json)
            with open(in_dir_config_file, "w") as config_file:
                json.dump(config, config_file, indent=4)
            return True

        else:
            print("None Saved")
            return False

    except Exception as e:
        error_store(str(e), traceback.format_exc(), str(datetime.now(timezone.utc)), "Not Applicable", "Match_data")
        return False


if __name__ == "__main__":
    import sys
    match_data(pd.read_csv(sys.argv[1]))