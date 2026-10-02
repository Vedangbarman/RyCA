import os
import lxml
import json 
import requests
import time
import traceback
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from utils.data_check import check_data
from utils.error_store import error_store
from utils.week_file_save import current_week_file



script_dir = os.path.dirname(os.path.realpath(__file__))

out_dir_notifications = os.path.abspath(os.path.join(script_dir,"..","..","data","notifications"))
os.makedirs(out_dir_notifications,exist_ok = True)

out_dir_error_logs = os.path.abspath(os.path.join(script_dir,"..","..","data","error_logs"))
os.makedirs(out_dir_error_logs,exist_ok = True)

in_dir_config_file = os.path.abspath(os.path.join(script_dir,"..","..","config.json"))
        
def isFileEmpty(filename): 
    try:
        if os.stat(filename).st_size > 0:
               return False
        else:
            return True
    except OSError:
        flag = "os_error"
        return flag



def rbi_webscraper():
    count = 0
    while count < 5:
        try: 
            url = "https://rbi.org.in/notifications_rss.xml"
            resp = requests.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, features="xml")

            items = soup.find_all('item')
            with open (in_dir_config_file) as config_file:
                config_data = json.load(config_file)
            pr_items = [] 
            format_notifications = "csv"
            current_path_notifications = current_week_file(out_dir_notifications,format_notifications)
            for item in items:
                pr_item = {}
                pr_item['title'] = item.title.text
                pr_item['description'] = item.description.text
                pr_item['link'] = item.link.text
                pr_item['pubDate'] = item.pubDate.text
                pr_items.append(pr_item)
            
            if pr_items:
                str_current_path_notification  = str(current_path_notifications)
                checked_pr_items = check_data(pr_items,str_current_path_notification)
                    
                if checked_pr_items:
                    pr_dataframe = pd.DataFrame(checked_pr_items,columns=['title','description','link','pubDate'])
                    path_check = os.path.exists(current_path_notifications)
                        
                    if path_check == True:
                        file_empty_status = isFileEmpty(current_path_notifications)
                        if file_empty_status == True:
                            
                            return pr_dataframe
                                
                        elif file_empty_status == False:
                            
                            return pr_dataframe
                            
                        elif  file_empty_status == "os_error":
                            timestamp = str(datetime.now(timezone.utc))
                            Error_Message = "OS Error in scraper.py while checking for file empty status"
                            error_count =  "Not Applicable"
                            Error_File = "Scraper"
                            trace_back = traceback.format_exc()
                            error_store(Error_Message,trace_back,timestamp,error_count,Error_File)
                            return False
                           
                        else:
                            timestamp = str(datetime.now(timezone.utc))
                            Error_Message = "Unknow Error in scraper.py while checking for file empty status"
                            error_count =  "Not Applicable"
                            Error_File = "Scraper"
                            trace_back = traceback.format_exc()
                            error_store(Error_Message,trace_back,timestamp,error_count,Error_File)
                            return False                          
                            
                    else:
                        
                        return pr_dataframe
                
                else:
                    print("nothing Found")
                    return False
            else:
                print("nothing Found")
                return False
            break
            
        except Exception as e:
            
            print(f"Error {e}")
            error_message = str(e)
            count +=1
            timestamp = str(datetime.now(timezone.utc))
            Error_File = "Scraper"
            trace_back = traceback.format_exc()
            error_store(error_message,trace_back,timestamp,error_count,Error_File)

        sleep_count = count*5
        print(f"Retrying in {sleep_count} seconds.......")
        time.sleep(sleep_count)
                
if __name__ == "__main__":
    rbi_webscraper()