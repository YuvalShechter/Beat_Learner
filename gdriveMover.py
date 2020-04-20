import os
import subprocess
import time
import shutil

def gdriveMove(folderpath, movepath):
    for folder in os.listdir(folderpath):
        shutil.move(folderpath+folder, movepath+folder)

if __name__ == "__main__":
    try:
        gdriveMove("All Songs/", "/home/yuval/GoogleDrive/")
    except:
        subprocess.run(["fusermount","-u","/home/yuval/GoogleDrive"])
        time.sleep(5)
        subprocess.run(["google-drive-ocamlfuse","/home/yuval/GoogleDrive"])
        time.sleep(5)