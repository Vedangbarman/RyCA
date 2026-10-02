import os
import json 
import traceback
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from utils.error_store import error_store
from utils.week_file_save import current_week_file


script_dir = os.path.dirname(os.path.realpath(__file__))
out_dir_notifications = os.path.abspath(os.path.join(script_dir,"..","..","data","notifications"))

out_dir_notifications_clean = os.path.abspath(os.path.join(script_dir,"..","..","data","notifications_clean"))
os.makedirs(out_dir_notifications_clean,exist_ok = True)

in_dir_config_file = os.path.abspath(os.path.join(script_dir,"..","..","config.json"))

import pandas as pd
from bs4 import BeautifulSoup


def isFileEmpty(filename): 
    try:
        if os.stat(filename).st_size > 0:
               return False
        else:
            return True
    except OSError:
        flag = "os_error"
        return flag


def clean_xml(xml_data):
    if pd.isna(xml_data) or not isinstance(xml_data, str):
        return xml_data
        
    soup = BeautifulSoup(xml_data, 'xml')
    clean_text = soup.get_text(separator=' ', strip=True)
    return clean_text


def clean_data(notification_data):
    try : 
        with open (in_dir_config_file) as file:
            config_data = json.load(file)
        file_path = config_data["data_check"]["ref_file"]
        
        date_format = "%a, %d %b %Y %H:%M:%S"
        ref_date = config_data["data_check"]["ref_date"]
        delta_ref_date = config_data["data_check"]["delta_ref_date"]
        
        if ref_date != "zero" and delta_ref_date != "zero":
            ref_date_formatted = datetime.strptime(ref_date,date_format)
            delta_ref_date_formatted = datetime.strptime(delta_ref_date,date_format)
        
        
        format = "csv"
        clean_xml_path = current_week_file(out_dir_notifications_clean,format)
        file_empty_status = isFileEmpty(out_dir_notifications)
        
        if ref_date == "zero":
            notification_data['clean_description'] = notification_data['description'].apply(clean_xml)
            notification_data = notification_data.drop("description",axis='columns')
            notification_data.to_csv(clean_xml_path,mode = 'a',header = True,index = False, encoding = 'utf-8')
            print(f"Data Saved to {clean_xml_path}")
            config_data["data_check"]["clean_ref_file"] = str(clean_xml_path)
            with open(in_dir_config_file, 'w') as file:
                json.dump(config_data, file, indent=4)
            return notification_data
            
        elif ref_date_formatted == delta_ref_date_formatted:
                notification_data['clean_description'] = notification_data['description'].apply(clean_xml)
                notification_data = notification_data.drop("description",axis='columns')
                notification_data.to_csv(clean_xml_path,mode = 'a',header = True,index = False, encoding = 'utf-8')
                print(f"Data Saved to {clean_xml_path}")
                config_data["data_check"]["clean_ref_file"] = str(clean_xml_path)
                with open(in_dir_config_file, 'w') as file:
                    json.dump(config_data, file, indent=4)
                return notification_data
                
        elif ref_date_formatted > delta_ref_date_formatted:
            if file_empty_status == True:
                notification_data['clean_description'] = notification_data['description'].apply(clean_xml)       
                notification_data = notification_data.drop("description",axis='columns')
                notification_data.to_csv(clean_xml_path,mode = 'a',header = True,index = False, encoding = 'utf-8')
                print(f"Data Saved to {clean_xml_path}")
                config_data["data_check"]["clean_ref_file"] = str(clean_xml_path)
                with open(in_dir_config_file, 'w') as file:
                    json.dump(config_data, file, indent=4)
                return notification_data
                    
            elif file_empty_status == False:
                notification_data['clean_description'] = notification_data['description'].apply(clean_xml)
                notification_data = notification_data.drop("description",axis='columns')
                notification_data.to_csv(clean_xml_path,mode = 'a',header = False,index = False, encoding = 'utf-8')
                print(f"Data Saved to {clean_xml_path}")
                config_data["data_check"]["clean_ref_file"] = str(clean_xml_path)
                with open(in_dir_config_file, 'w') as file:
                    json.dump(config_data, file, indent=4)
                return notification_data
                    
            elif file_empty_status == "os_error":
                time = str(datetime.now(timezone.utc))
                Error_Message = "File not found in clean_data.py while checking for file empty status"
                error_count =  "Not Applicable"
                Error_File = "clean_data"
                error_store(Error_Message,time,error_count,Error_File)
                return False
                    
            else :
                
                return False
            
        else:
            print("date error")
            return False

    except Exception as e:
        time = str(datetime.now(timezone.utc))
        Error_Message = str(e)
        error_count =  "Not Applicable"
        Error_File = "clean_data"
        trace_back = traceback.format_exc()
        error_store(Error_Message,trace_back,time,error_count,Error_File)
        return False
        
        
if __name__ == "__main__":
    clean_data()