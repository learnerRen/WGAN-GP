import numpy as np
from scipy.misc import imread
import os
path="resizeimages"
a=os.listdir(path)
b = imread(os.path.join(path, a[0]))
b=b[np.newaxis, :, :, :]
c = imread(os.path.join(path,a[1]))
c=c[np.newaxis, :, :, :]
for i in range(2, len(a)):
    if i % 100 == 0:
        print(i)
    d = imread(os.path.join(path, a[i]))
    try:
        d = d[np.newaxis,:,:,:]
    except:
        print(os.path.join(path, a[i]))
    if i % 1000 == 0:
        b = np.concatenate([b,c],axis=0)
        c = d
    else:
        c = np.concatenate((c,d),axis=0)
b=np.concatenate([b,c],axis=0)
print(b.shape)
np.save("image2", b)

