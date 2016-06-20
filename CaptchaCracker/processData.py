import numpy as np
import os
from scipy.misc import imread, imresize

cwd = os.getcwd()
path = "images/"
imgsize = [125, 30]
data_name = "processedData"
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
imgcnt = 0
nclass = 36
letters = 6

flist = os.listdir(path)
curImgCnt = 0
for f in flist:
    if os.path.splitext(f)[1].lower() not in valid_exts:
        continue
    imgcnt = imgcnt + 1

print (str(imgcnt) + " images loaded")

totalimg   = np.ndarray((imgcnt, imgsize[0], imgsize[1], 3))
totallabel = np.ndarray((imgcnt, letters, nclass))

flist = os.listdir(path)
curImgCnt = 0
for f in flist:
    if os.path.splitext(f)[1].lower() not in valid_exts:
        continue
    fullpath = os.path.join(path, f)
    currimg  = imread(fullpath)

    grayimg  = currimg
    # Reshape
    graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255.
    grayvec   = graysmall
    # Save
    totalimg[curImgCnt, :] = grayvec
    label = f.split("-")[0]
    empty = np.zeros([letters, 36])
    i = 0
    for x in list(label):
        charID = 0
        if (ord(x) >= 65 and ord(x) <= 90):
            charID = ord(x) - 65
        elif (ord(x) >= 48 and ord(x) <= 57):
            charID = ord(x) - 23
        empty[i][charID] = 1
        i = i + 1
    totallabel[curImgCnt, :] = empty
    curImgCnt = curImgCnt + 1

# Divide total data into training and test set
randidx  = np.random.randint(imgcnt, size=imgcnt)
trainidx = randidx[0:int(4*imgcnt/5)]
testidx  = randidx[int(4*imgcnt/5):imgcnt]

trainimg   = totalimg[trainidx, :]
trainlabel = totallabel[trainidx, :]
testimg    = totalimg[testidx, :]
testlabel  = totallabel[testidx, :]

savepath = cwd + "/" + data_name + ".npz"
np.savez(savepath, trainimg=trainimg, trainlabel=trainlabel
         , testimg=testimg, testlabel=testlabel)
print("Saved to %s" % (savepath))
