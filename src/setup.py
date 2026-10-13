import os 
import json
import requests
from bs4 import BeautifulSoup
in_dir_data = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","data","setup.txt"))

out_dir_notifications = in_dir_data = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","data","notification","notification.jsonl"))
os.makedirs(out_dir_notifications,exist_ok = True)
out_dir_md = out_dir_notifications = in_dir_data = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","data","md_historical","md.jsonl"))
os.makedirs(out_dir_md,exist_ok = True)

def setup():
    with open(in_dir_data, "r", encoding="utf-8") as file:
        for i,line in enumerate(file):
            if i<45:
                url = f"https://rbi.org.in/scripts/BS_ViewMasDirections.aspx?id={line}"
                resp = requests.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, features="html.parser")                  
                title = soup.find('b').get_text(strip=True)
                content = soup.find('tr', class_='tablecontent2')
                for junk in content.find_all(["script", "style", "input"]):
                    junk.decompose()
                clean_text = str(content).replace("\r", "")
                clean_data = {"id": id, "title": title, "category": "NBFC", "url": url, "text": clean_text}
                with open(out_dir_notifications,"a",encoding = "utf-8") as f:
                    f.write(json.dumps(clean_data) + "\n")
            
            else:
                url = f"https://rbi.org.in/scripts/NotificationUser.aspx?Id={id}"
                resp = requests.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, features="html.parser")
                title = soup.find('b').get_text(strip=True)
                content = soup.find('tr', class_='tablecontent2')
                for junk in content.find_all(["script", "style", "input"]):
                    junk.decompose()
                clean_text = content.get_text(separator="\n", strip=True).replace("\r", "")
                               
                clean_data = {"id": id, "title": title, "category": "NBFC", "url": url, "text": clean_text}
                               
                file_path = os.path.abspath(os.path.join(out_dir_md,f"notification.jsonl"))
                with open(file_path,"a",encoding = "utf-8") as f:
                    f.write(json.dumps(clean_data) + "\n")
            