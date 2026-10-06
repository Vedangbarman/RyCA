import os
import json
import time
import traceback
from ai_inference import invoke_ai
from datetime import datetime, timezone
from utils.error_store import error_store
from watcher.match_data import match_data
from watcher.clean_data import clean_data
from watcher.scraper import rbi_webscraper
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler

script_dir = os.path.dirname(os.path.realpath(__file__))
out_dir_error_logs = os.path.abspath(os.path.join(script_dir,"..","data","error_logs"))
os.makedirs(out_dir_error_logs,exist_ok = True)

in_dir_config_file = os.path.abspath(os.path.join(script_dir,"..","config.json"))

with open(in_dir_config_file) as file:
    data = json.load(file)
print("File Loaded!")

def ring():
    count = 0
    while count < 5:
        try:
            flag = rbi_webscraper()
            print("Scraper loaded")
            if flag is not False:
                flag = clean_data(flag)
                print("Data Cleaned")
                print("Clean file laoded")
                if flag is not False:
                    flag = match_data(flag)
                    print("Match file loaded")
                    if flag == True:
                        print("Data Matched")
                        invoke_ai()
                    else:
                        print("Matched returned false")
                else:
                    print("Cleaned returned false")
            else:
                print("Scraper return false")
            break         
                
        except Exception as e:
                print(f"Error {e}")
                error_message = str(e)
                count +=1
                timestamp = str(datetime.now(timezone.utc))
                
                Error_Message = error_message
                Time = timestamp
                Error_Count = count
                Error_File = "Ring_file"
                trace_back = traceback.format_exc()
                error_store(Error_Message,trace_back,Time,Error_Count,Error_File)
                time.sleep(count*5)
        
_scheduler = None

def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    scheduler = BackgroundScheduler()

    run_count = data["scraper_settings"]["run_count"]
    start_time = data["scraper_settings"]["time_scraper"]
    if run_count <= 1:
        scheduler.add_job(ring, 'cron', hour=start_time, minute=0,
                          id='daily_fixed_run', max_instances=1)
    else:
        interval = int(24 / run_count)
        for i in range(run_count):
            hour = (int(start_time) + i * interval) % 24
            scheduler.add_job(ring, 'cron', hour=str(hour), minute=0,
                              id=f'{i}th_interval_runs', max_instances=1,
                              replace_existing=True)

    scheduler.add_job(ring, id='startup_run')     # replaces the bare ring() call, runs in background
    scheduler.start()
    _scheduler = scheduler
    return scheduler


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, workers=1)