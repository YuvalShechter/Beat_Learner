import numpy as np
import json
import os, sys
import pickle
import subprocess

# hard-coded 0.005 second time step
# creating array of zeros: np.zeros((4,3,3,9), dtype=int)
# index, layer, type, direction
if __name__ == "__main__":
    # Sample paths:
    # ./All Songs/24k-magic
    # ./sample_song
    with open("./sample_song/Expert.dat","r") as jsonFile:
        loadedFile = json.load(jsonFile)
    maxLayer = loadedFile["_notes"][0]["_type"]
    for note in loadedFile["_notes"]:
        maxLayer = min(maxLayer, note["_type"])
    
    print(maxLayer)

# https://stackoverflow.com/questions/31024968/using-ffmpeg-to-obtain-video-durations-in-python
def get_song_length(input_song):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_song], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def note_placement_helper(current, i, j, k, l):
    if current == None:
        temp = np.zeros((4,3,3,9), dtype=int)
    else:
        temp = current
    temp[i][j][k][l] = 1
    return temp

def parseNotes(notedatapath, noteinfopath, songpath):
    with open(notedatapath,"r") as dataFile:
        notedata = json.load(dataFile)
    allnotes = notedata["_notes"]

    with open(noteinfopath,"r") as infoFile:
        noteinfo = json.load(infoFile)
    bpm = noteinfo["_beatsPerMinute"]

    songlength = get_song_length(songpath)
    sparsenotearray = {"length": int((songlength/bpm) * 12000)+1, "notes":{}}

    for singlenote in allnotes:
        timestamp = int((float(singlenote["_time"])/bpm) * 12000)
        i = int(singlenote["_lineIndex"])
        j = int(singlenote["_lineLayer"])
        k = int(singlenote["_type"]) if int(singlenote["_type"]) != 3 else 2
        l = int(singlenote["_cutDirection"])
        sparsenotearray["notes"][timestamp] = note_placement_helper(sparsenotearray["notes"].get(timestamp),i,j,k,l)

    return sparsenotearray

    
    # First Key "_notes" -> [array] -> Each element
    # Element format: {'_time': 3, '_lineIndex': 2, '_lineLayer': 0, '_type': 1, '_cutDirection': 1}
    # index = columns with 4 (0-3)
    # layer = rows with 3 (0-2)
    # cut direction = different directions (0-8) [8 is the direction agnostic one]
    # type = red/blue/something else (0,1,3) [use 0-2 and convert later]
    # Convert to 96 hot vector
    # In each element _time = beat offset from start of song
    # time in 0.005-seconds = (_time / bpm) * 12000
    # 0 - 6.4 nps is within the realm of tenth-seconds

"""
    1. Read in spectrogram pickle-file and get length
    2. Align notes vector arrays with 0.005 second time steps
    3. Output: time (variable) x notes state per 0.005 second alignment (92-hot vector)
"""