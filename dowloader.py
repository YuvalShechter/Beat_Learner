import requests
import json
from pyquery import PyQuery as pq
import zipfile
from slugify import slugify
import os

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0) Gecko/20100101 Firefox/73.0'}

def download(url, dlDictionary):
    with open("All Songs/temp.zip", "wb") as outFile:
        response = requests.get(url, headers=headers)
        outFile.write(response.content)

    with zipfile.ZipFile("All Songs/temp.zip", 'r') as zip_ref:
        zip_ref.extractall("All Songs/temp")

    os.remove("All Songs/temp.zip")
    with open("All Songs/temp/info.dat","r") as jsonFile:
        songName = json.load(jsonFile)['_songName']

    songName = slugify(songName)
    if songName in dlDictionary:
        keyName = songName
        songName+="("+str(dlDictionary[songName])+")"
        dlDictionary[keyName]+=1
    else:
        dlDictionary[songName] = 1

    os.rename("./All Songs/temp",'./All Songs/'+songName)

    return dlDictionary

# Script for scraping first 5000 pages of songs on bsaber.com
def scrapeLinks(url):
    with open("links", "a+") as linkFile:
        htmlText = requests.get(url, headers=headers).text
        parsedhtmlText = pq(htmlText)
        linkAHrefs = parsedhtmlText(".-download-zip").items()
        for link in linkAHrefs:
            linkFile.write(link.attr['href']+"\n")

if __name__ == "__main__":
    # for i in range(1,11):
    #     scrapeLinks("https://bsaber.com/songs/top/page/{0}/?time=all".format(i))

    fulldictionary = {}
    with open("links","r") as downloadLinks:
        for link in downloadLinks:
            fulldictionary = download(link[:-1], fulldictionary)
    pass