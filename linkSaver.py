import requests
from pyquery import PyQuery as pq

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0) Gecko/20100101 Firefox/73.0'}

# Script for scraping first 5000 pages of songs on bsaber.com
def scrapeLinks(url):
    with open("links", "a+") as linkFile:
        htmlText = requests.get(url, headers=headers).text
        parsedhtmlText = pq(htmlText)
        linkAHrefs = parsedhtmlText(".-download-zip").items()
        for link in linkAHrefs:
            linkFile.write(link.attr['href']+"\n")

if __name__ == "__main__":
    for i in range(1,692):
        scrapeLinks("https://bsaber.com/songs/top/page/{0}/?time=all".format(i))