import os
import shutil

path = "images/"

for x in range(1, 16):
    if os.path.exists(path + str(x)):
        shutil.rmtree(path + str(x))
    os.mkdir(path + str(x))

flist = os.listdir(path)
for f in flist:
    if os.path.isdir(path + f):
        continue
    print(f)
    classId = f.split(".")[0].split("_")[1]
    os.rename(path + f, path + classId + "/" + f)
