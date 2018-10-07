def topK(arr,k):
    """
    finding top k index in an array arr
    """
    c=np.copy(arr)
    value=[]
    idxs=[]
    for i in range(k):
        idx=np.argmax(c)
        idxs.append(idx)
        val=float("{0:.5f}".format(c[idx]))
        value.append(val)
        c[idx]=0
    return idxs,value
#############################################################
def top_Obj(path,idxs):
    """
    finding the relative objects for the top k indexes
    """
    objects={}
    object=[]
    with open(path, 'r') as f:
        for i, x in enumerate(f):
            objects[i]=x
    for i in idxs:
        object.append(objects[i])
    return object


# import os
# from numpy.linalg import inv
import pandas as pd
import scipy.io as spio
import numpy as np


#################################   # TOPIC Modeling ############################
"""
Nonnegative matrix factorization:
The data to be used for this problem consists of :
8447 documents from The New York Times.--the j_th document
The vocabulary size is 3012 words. --the i_th word
Use this data to construct the matrix X: 3012×8447 --Xij
"""
#################################
# loading the data
X=np.zeros((3012,8447))
with open('nyt_data.txt', 'r') as f:
    j=0
    for line in f:
        words_count = line.split(",")
        for word in words_count:
            i,count= word.split(":")
            i=int(i)
            count=int(count)
            X[i-1][j]=count
        j=j+1

np.savetxt("data_q2.csv", X, delimiter=",")

#################################
"""
To get the most 10 frequent words in each topic.
In the other word, for each column of W, list the 10 words having the largest weight.
"""
# read data from matlab
w0 = spio.loadmat('q2_w.mat', squeeze_me=True)
w = w0['w2']

# process the result
out=[]
for i in range(25):
    idxs,value=topK(w[:,i],10)
    object=top_Obj("nyt_vocab.dat",idxs)
    out.append(dict(zip(value,object)))

# output the result
with open('q2_output.txt', 'w') as f:
    for i in range(25):
        f.write(str(out[i]))
        f.write("\n")
