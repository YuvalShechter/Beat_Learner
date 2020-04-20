import os
import subprocess
import time
import shutil

def gdriveMove(folderpath, mountpath, movepath):
        for folder in os.listdir(folderpath):
            if(os.path.ismount(mountpath)):
                shutil.move(folderpath+folder, mountpath+movepath+folder)
            else:
                subprocess.run(["fusermount","-u",mountpath])
                time.sleep(5)
                subprocess.run(["google-drive-ocamlfuse",mountpath])
                time.sleep(5)
                shutil.move(folderpath+folder, mountpath+movepath+folder)

if __name__ == "__main__":
        gdriveMove("All Songs/", "/home/yuval/GoogleDrive/", "Beat_Saber_Dataset/")