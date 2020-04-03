import requests
import json
import zipfile
from slugify import slugify
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0) Gecko/20100101 Firefox/73.0'}

def download(url, dlDictionary):
    with open("All Songs/temp.zip", "wb") as outFile:
        response = requests.get(url, headers=headers)
        # Response when file is not found when downloaded
        if(response.content == "Not Found"):
            print("Not Found: "+url)
            return dlDictionary
        outFile.write(response.content)

    with zipfile.ZipFile("All Songs/temp.zip", 'r') as zip_ref:
        zip_ref.extractall("All Songs/temp")

    os.remove("All Songs/temp.zip")
    with open("All Songs/temp/info.dat","r") as jsonFile:
        songName = json.load(jsonFile)['_songName']

    songName = slugify(songName)
    print(songName)
    if songName in dlDictionary:
        keyName = songName
        songName+="("+str(dlDictionary[songName])+")"
        dlDictionary[keyName]+=1
    else:
        dlDictionary[songName] = 1

    os.rename("./All Songs/temp",'./All Songs/'+songName)

    return dlDictionary

if __name__ == "__main__":
    fulldictionary = {}
    with open("links","r+") as downloadLinks:
        for link in downloadLinks:
            fulldictionary = download(link[:-1], fulldictionary)