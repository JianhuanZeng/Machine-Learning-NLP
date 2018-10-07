import os
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# root mean square error metric between predictions and the ground truth
def rmse(pred, true):
    assert pred.shape == true.shape
    n, = true.shape
    return np.sqrt(np.sum((pred - true) ** 2) / n)

def prepo(X):
    """
    For easier updatig, preposessing
    Create a full rating matrix M: N1*N2
    Create two dictionaries mapping users to movies and the reverse
    """
    N1=max(X[0]) # N1 users
    N2=max(X[1]) # N2 movies
    M=np.zeros((N1, N2))
    uv={} # the user i to all movies rated
    vu={} # the movie j to all users rating
    for i in range(len(X[0])):
        # rating matrix M
        M[X[0][i]-1][X[1][i]-1]=X[2][i]
        # a dictionary mapping users ui to movies
        if X[0][i]-1 in uv:
            uv[X[0][i]-1].append(X[1][i]-1)
        else:
            uv[X[0][i]-1]=[X[1][i]-1]
        # another dictionary mapping movies vj to users
        if X[1][i]-1 in vu:
            vu[X[1][i]-1].append(X[0][i]-1)
        else:
            vu[X[1][i]-1]=[X[0][i]-1]
    return M,uv,vu

# update the user i location ui or the object j location vj: d*1 dimension
def update(a,r,list,const,d):
    """
    update a vector b using some vectors a[i] in matrix a and a index list containing {i}.
    r: a rating array
    d: dimension
    const: constant
    :return: updated b
    """
    s1=np.zeros((d,d))
    s2=np.zeros(d)
    for i in list:
        s1=s1+(a[i][np.newaxis, :].T)*a[i]
        s2=s2+r[i]*a[i]
    return sum((inv(const+s1)*s2).T)

# the objective function
def objfunc(X,u,v,M_pred,var,numbda):
    s1=0
    for i in range(len(X[0])):
        s1=s1+(X[2][i]-M_pred[X[0][i]-1,X[1][i]-1])**2
    return -(s1/2/var+sum(sum(v**2))*numbda/2+sum(sum(u**2))*numbda/2)

def MatrixfFactorization(X,M,uv,vu,var,d,numbda,T):
    """
    To recomender user ui for object vj given Gaussian Model Assumption by MatrixfFactorization
    T: Iteration
    :return: predicted missing rates
    """
    # Constants
    N1=max(X[0]) # N1 users
    N2=max(X[1]) # N2 movies

    # Initialze vj from a Guassian distribution
    mean=np.zeros((d,), dtype=np.int)
    var0=1/numbda*np.eye(d)
    v = np.random.multivariate_normal(mean, var0, N2)
    u = np.random.multivariate_normal(mean, var0, N1)

    # Updating
    L=[]
    const=numbda*var*np.eye(d)
    for t in range(T):
        # update the user i location ui
        for i in range(N1):
            if i in vu:
                u[i]=update(v,M[i],uv[i],const,d)

        # update the object j location vj
        for j in range(N2):
            if j in vu:
                v[j]=update(u,M.T[j],vu[j],const,d)

        # predict ratings
        u1=np.asmatrix(u)
        v1=np.asmatrix(v)
        M_pred=u1*(v1.T)

        # the objective function
        Lt=objfunc(X,u,v,var,numbda)
        # make a list for every value of the objective function
        L.append(Lt)
    return L, M_pred, v

#############################################################
# for question 2
# loading the set indexes an incomplete ratings matrix M
X = pd.read_csv(os.path.join("/Users/cengjianhuan/Documents/ML/HW4", 'ratings.csv'), header=None)
# loading the testing set
M_ture = pd.read_csv(os.path.join("/Users/cengjianhuan/Documents/ML/HW4", 'ratings_test.csv'), header=None)
# constants
var = 0.25
d = 10
numbda = 1.0
T=100
# create a full rating matrix M: N1*N2 and two dictionaries mapping users to movies and the reverse
M,uv,vu=prepo(X)
# create lists to store results
L=[]
rmse_list=[]
Lmax=-5000
# run 10 times
for k in range(10):
    # get the objective value and prediction at the k time
    Lk, M_pred, vk=MatrixfFactorization(X,M,uv,vu,var,d,numbda,100)
    L.append(Lk[-1])
    # pick the relative predicted value M_test for rmse
    M_test=np.zeros(len(M_ture[2]))
    for i in range(len(M_ture[2])):
        M_test[i]=M_pred[M_ture[0][i]-1,M_ture[1][i]-1]
    # find the rmse at k time
    rmse_list.append(rmse(M_test, M_ture[2]))

    # find the optimal L
    if L[-1]>Lmax:
        Lmax=L
        vmax=vk

# for question a
for i in range(10):
    plt.plot(L[i][1:],label="L"+str(i))
plt.legend(loc='best')
plt.show()

# for question b
N2,_=vmax.shape
xt=[vmax[49],vmax[484],vmax[181]]
d=[]
minv=[]
idxv=[]
for k in range(len(xt)):
    minv.append([])
    idxv.append([])
    for i in range(1682):
        d.append(np.linalg.norm(xt[k]-vmax[i]))
    M_test=np.zeros(len(M_ture[2]))
    for i in range(len(M_ture[2])):
        M_test[i]=M_pred[M_ture[0][i]-1,M_ture[1][i]-1]
    for t in range(11):
        mn,idx = min((d[i],i) for i in range(len(d)))
        minv[k].append(mn)
        idxv[k].append(idx)
        d[idx]=10
