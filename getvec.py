import numpy as np
import pickle
import torch
import random
import torch.nn.functional as F
from numpy import linalg as la
from torch.autograd import Variable
import scipy.sparse as sparse
from torch import nn
from gensim.models import word2vec
from collections import Counter
import gensim
#Load Data
with open('data/semat.pickle', 'rb') as file:#SE mat
    semat = pickle.load(file)
with open('data/smat.pickle', 'rb') as file:#ST mat
    smat = pickle.load(file)
with open('data/emat.pickle', 'rb') as file:#ET mat
    emat = pickle.load(file)

def getvocab(mat):
    m = []
    for k in mat.keys():
        m.append(k)
        m+=list(mat[k].keys())
    m = list(set(m))
    vocab = []
    for k in m:
        if (k[0] == 'a' and 'b'+k[1:] in m) or (k[0] == 'b' and 'a'+k[1:] in m):
                vocab.append(k[1:])
    return list(set(vocab))
vocab = getvocab(semat)
vocab2dig = dict(zip(vocab,range(len(vocab))))
dig2vocab = dict(zip(range(len(vocab)),vocab))



tarmat = np.zeros((len(vocab), len(vocab)))
attsmat = np.zeros((len(vocab), 8))
attemat = np.zeros((len(vocab), 8))
for k1 in semat.keys():
    if k1[1:] not in vocab:
        continue
    for k2 in semat[k1].keys():
        if k2[1:] not in vocab:
            continue
        t1 = vocab2dig[k1[1:]]
        t2 = vocab2dig[k2[1:]]
        if k1[0] == 'a':
            tarmat[t1][t2] = int(semat[k1][k2]/60)
        else:
            tarmat[t2][t1] = int(semat[k1][k2]/60)
for i in range(attsmat.shape[0]):
    attsmat[i] = smat['a' + dig2vocab[i]]
    attemat[i] = emat['b' + dig2vocab[i]]


def gettarmat(mat):
    def getpar(mat, x):
        result = np.sum(mat,x)
        for i in range(len(result)):
            if result[i]!=0:
                result[i] = 1/result[i]
        return result[i]
    row = getpar(mat, 1)
    col = getpar(mat, 0)
    tarmat = np.sum(mat) * np.dot(np.dot(row,mat),col)
    e = np.log(5)
    for i in range(tarmat.shape[0]):
        for j in range(tarmat.shape[1]):
            if tarmat[i][j] <= 5:
                tarmat[i][j] = 0
            else:
                tarmat[i][j] = np.log(tarmat[i][j]) - e
    return tarmat
tarmat = gettarmat(tarmat)
attsmat = gettarmat(attsmat)
attemat = gettarmat(attemat)

bigmat = np.eye(tarmat.shape[0] + 8)
bigmat[:tarmat.shape[0],:tarmat.shape[0]] = tarmat
bigmat[:tarmat.shape[0],tarmat.shape[0]:] = 100*attsmat
bigmat[tarmat.shape[0]:,:tarmat.shape[0]] = 100*attemat.T



def datasvd(data, dim):
    print(data.shape)
    u,sigma,vt = la.svd(data)
    S = np.zeros((dim,dim))
    for i in range(dim):
        S[i][i] = np.sqrt(sigma[i])
    sve = np.dot(u[:,:dim],S)
    eve = np.dot(S,vt[:dim,:])
    return (sve[:tarmat.shape[0]],(eve.T)[:tarmat.shape[0]])
(sve,eve) = datasvd(bigmat.astype('float32'), 64)

poive = {}
for i in range(sve.shape[0]):
    poive[dig2vocab[i]] = np.hstack((sve[i], eve[i]))

