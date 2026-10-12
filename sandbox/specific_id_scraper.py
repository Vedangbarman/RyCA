import os 
import json
import asyncio
import requests
import traceback
from bs4 import BeautifulSoup

script_dir = os.path.dirname(os.path.realpath(__file__))

out_dir_master_directory = os.path.abspath(os.path.join(script_dir,"..","data","master_directory"))
os.makedirs(out_dir_master_directory,exist_ok = True)


async def scrape_master_directory():

    file_path = os.path.abspath(os.path.join(out_dir_master_directory,f"master_directory_html.jsonl"))
    count = 0 
    list = [13274, 12943, 10485, 11510, 11200, 12381]
    try:
            for i in list:
                if i:
                    id = i
                    url = f"https://rbi.org.in/scripts/BS_ViewMasDirections.aspx?id={id}"
                    resp = requests.get(url)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.content, features="html.parser")
                    
                    title = soup.find('b').get_text(strip=True)
                    
                    
                    content = soup.find('tr', class_='tablecontent2')
                    
                    clean_data = {"id": id, "title": title, "category": "NBFC", "url": url, "text": str(content)}
                    with open(file_path,"a",encoding = "utf-8") as f:
                        f.write(json.dumps(clean_data) + "\n")
                
                
            
    except Exception as e:
        traceback.print_exc()
           
            
if __name__ == "__main__": 
    asyncio.run(scrape_master_directory())