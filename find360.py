import os
import shutil

count = 0
folders = []
for folder in os.listdir("All Songs/"):
	found = False
	for currfile in os.listdir("All Songs/"+folder):
		if ("360" in currfile) and (".dat" in currfile):
			if not found:
				folders.append("All Songs/"+folder)
				count+=1
				print("All Songs/"+folder)
				print(" - "+currfile)
				found = True
			else:
				print(" - "+currfile)
		if ("90" in currfile) and (".dat" in currfile):
			if not found:
				folders.append("All Songs/"+folder)
				count+=1
				print("All Songs/"+folder)
				print(" - "+currfile)
				found = True
			else:
				print(" - "+currfile)
for folder in folders:
	shutil.rmtree(folder)
print(count)
