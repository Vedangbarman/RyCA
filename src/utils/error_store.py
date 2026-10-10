import os 
import json
from utils.week_file_save import current_week_file
from datetime import datetime, timezone
script_dir = os.path.dirname(os.path.realpath(__file__))
out_dir_error_logs = os.path.abspath(os.path.join(script_dir,"..","..","data","error_logs"))


def check():
    time =  str(datetime.now(timezone.utc))
    error_store("Debug",time,"Not Applicable","Debug")


def error_store(error_message,trace_back,time,error_count,error_file):
    
    errors_ds = {}
    errors_ds['Error_Message'] = error_message
    errors_ds['Time'] = time
    errors_ds['Trace_back'] = trace_back
    errors_ds['Error Count'] = error_count
    errors_ds['Error_File'] = error_file
    format_errors = "jsonl"
    current_path_error_log = current_week_file(out_dir_error_logs,format_errors)
    data = json.dumps(errors_ds)
    with open (current_path_error_log, "a") as file:
        file.write(data + "\n")
    print(f"Data saved to {current_path_error_log}")
    
if __name__ == "__main__":
    time =  str(datetime.now(timezone.utc))
    error_store("Debug",time,"Not Applicable","Debug")


if __name__ == "__main__":
    error_store(error_message="test, You can safely Ignore this",trace_back = "test",time = str(datetime.now(timezone.utc)),error_count="Null",error_file="error_test")