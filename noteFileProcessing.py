import numpy as np
import json

if __name__ == "__main__":
    with open("./sample_song/Expert.dat","r") as jsonFile:
        loadedFile = json.load(jsonFile)
    maxLayer = loadedFile["_notes"][0]["_time"]
    for note in loadedFile["_notes"]:
        maxLayer = max(maxLayer, note["_time"])
    
    print(maxLayer)

def parseNotes(fpath):
    with open(fpath,"r") as jsonFile:
        loadedFile = json.load(jsonFile)

    outputDict = {"time": [], "vector": []}
    
    # First Key "_notes" -> [array] -> Each element
    # Element format: {'_time': 3, '_lineIndex': 2, '_lineLayer': 0, '_type': 1, '_cutDirection': 1}
    # index = columns with 4
    # layer = rows with 3
    # cut direction = 8 different directions
    # Convert to 96 hot vector
    # In each element _time = beat offset from start of song
    # time in tenth-seconds = (_time / bpm) * 60 * 10
    # 0 - 6.4 nps is within the realm of tenth-seconds