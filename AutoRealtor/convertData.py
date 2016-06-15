import numpy as np
import os
from scipy.misc import imread, imresize

cwd = os.getcwd()
# Training set folder
paths = {"images/1", "images/2", "images/3", "images/4", "images/5", "images/6", "images/7", "images/8", "images/9", "images/10", "images/11", "images/12", "images/13",
 "images/14", "images/15", "images/16", }
# The reshape size
imgsize = [36, 36]
# Save name
data_name = "processedData"


valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
imgcnt = 0
nclass = len(paths)
for relpath in paths:
    fullpath = cwd + "/" + relpath
    flist = os.listdir(fullpath)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(fullpath, f)
        imgcnt = imgcnt + 1

print ("Number of total images is %d" % (imgcnt))

# Then, let's save them!
totalimg   = np.ndarray((imgcnt, imgsize[0], imgsize[1], 3))
totallabel = np.ndarray((imgcnt, nclass))
imgcnt     = 0
previousGV = None
for i, relpath in zip(range(nclass), paths):
    path = cwd + "/" + relpath
    flist = os.listdir(path)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(path, f)
        try:
            currimg  = imread(fullpath)
        except IOError as e:
            totalimg[imgcnt, :] = previousGV
            totallabel[imgcnt, :] = 16
            imgcnt    = imgcnt + 1
            print (imgcnt)
            print ('error, continuing')
            continue
        grayimg  = currimg
        # Reshape
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec   = graysmall
        # Save
        totalimg[imgcnt, :] = grayvec
        totallabel[imgcnt, :] = np.eye(nclass, nclass)[i]
        imgcnt    = imgcnt + 1
        print(imgcnt)
        previousGV = grayvec

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
