import numpy as np
import os
#import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

cwd = os.getcwd()
# Training set folder
paths = {"images/"}
# The reshape size
imgsize = [72, 72]
# Grayscale
use_gray = 0
# Save name
data_name = "processedData"


# First, check the total number of training data
"""
 The reason for doing this a priori is that
 it is better to pre-allocate memory rather than
 dynamically adjust the size of the matrix.
"""
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
# Grayscale
def rgb2gray(rgb):
    if len(currimg.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        print ("Current Image if GRAY!")
        return rgb
if use_gray:
    totalimg   = np.ndarray((imgcnt, imgsize[0]*imgsize[1]))
else:
    totalimg   = np.ndarray((imgcnt, imgsize[0]*imgsize[1]*3))
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
            print (imgcnt + ' error')
            continue
        # Convert to grayscale
        if use_gray:
            grayimg  = rgb2gray(currimg)
        else:
            grayimg  = currimg
        # Reshape
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec   = np.reshape(graysmall, (1, -1))
        # Save
        totalimg[imgcnt, :] = grayvec
        totallabel[imgcnt, :] = f.split(".")[0].split("_")[1]
        imgcnt    = imgcnt + 1
        print (grayvec)
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
print ("Saved to %s" % (savepath))
