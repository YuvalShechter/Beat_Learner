import requests
import json
import zipfile
from zipfile import BadZipfile
from slugify import slugify
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0) Gecko/20100101 Firefox/73.0'}

def download(url, dlDictionary):
    with open("All Songs/temp.zip", "wb") as outFile:
        response = requests.get(url, headers=headers)
        outFile.write(response.content)
    
    # Sometimes download is not found (and so isn't a zip)
    try:
        with zipfile.ZipFile("All Songs/temp.zip", 'r') as zip_ref:
            zip_ref.extractall("All Songs/temp")
    except BadZipfile:
        print("Not found: "+url)
        return dlDictionary

    os.remove("All Songs/temp.zip")

    # Checks for info.dat in folder and if doesn't exist skips it
    infoCap = ""
    for fname in os.listdir("All Songs/temp"):
        if fname.lower() == "info.dat":
            infoCap = fname
            break
    if not infoCap:
        print("No Info: "+url)
        return dlDictionary
    
    with open("All Songs/temp/"+infoCap,"r") as jsonFile:
        songName = json.load(jsonFile)['_songName']

    songName = slugify(songName)
    if not songName:
        songName = "_"
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
    toSkip = len(os.listdir("./All Songs/"))
    with open("links","r+") as downloadLinks:
        for link in downloadLinks:
            if not toSkip:
                try:
                    fulldictionary = download(link[:-1], fulldictionary)
                except OSError:
                    with open("./restartDict","w+") as restartF:
                        json.dump(fulldictionary, restartF)
                    print("Uncaught Error: Restart With Dictionary")
            else:
                toSkip-=1