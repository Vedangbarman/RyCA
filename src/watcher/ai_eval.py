import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from utils.error_store import error_store
from utils.week_file_save import current_week_file
from langchain_google_genai import ChatGoogleGenerativeAI

debug = True

if debug :
    script_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir_output = os.path.abspath(os.path.join(script_dir,"..","..","data","notifications_output","2026-09-21.json"))
    df_results = pd.read_json(out_dir_output, lines=True)
    
def check_json(df_results):
    
    false_rows = df_results[df_results['is_master_direction_matching'] == False]
    if not false_rows.empty:
        for row in df_results.itertuples():
            if row.correct_master_dir != "not aplicable":
                print(f"{row.Index}:{row.correct_master_dir}")
            elif row.correct_matsr_dir == "not aplicable":
                print(f"{row.Index}: is not applicable")
    
        
        
if __name__ == "__main__":
    check_json(df_results)