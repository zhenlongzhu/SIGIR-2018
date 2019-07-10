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
#Load data
with open('data/binarygraph.pickle', 'rb') as file:
    binarygraph = pickle.load(file)
with open('data/myvec.pickle', 'rb') as file:
    vecmat = pickle.load(file)



def isnan(num):
    return num != num
flag = np.zeros(24)
flag[6:12] = 1
flag[12:18] = 2
flag[18:] = 3
def trans(ttstr):
    m = int(ttstr.split(' ')[1].split(':')[0])
    return str(int(flag[m]))
def get_accnum(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return (num_correct , total)


vocab = []
for m in vecmat.keys():
    vocab.append('a'+m)
    vocab.append('b'+m)
vocab2dig = dict(zip(vocab,range(len(vocab))))
dig2vocab = dict(zip(range(len(vocab)),vocab))
vocabvec = np.zeros((len(vocab),64))
for key in vocab:
    if key[0] == 'a':
        vocabvec[vocab2dig[key]] = vecmat[key[1:]][:64]
    else:
        vocabvec[vocab2dig[key]] = vecmat[key[1:]][64:]
vocabvec = torch.tensor(vocabvec).float().cuda()
time2dig = {'a0':0,'a1':1,'a2':2,'a3':3}

timevec = torch.rand(4,64).cuda()
timevec.requires_grad = True





posdata = {}#Positive Sample Data
negdata = {}#Negative sample data
relabel = {}
for eg in binarygraph.keys():
    eglist = eg.split(' ')
    if 'a'+eglist[0] not in vocab or 'b'+eglist[1] not in vocab:
        continue
    s = vocab2dig['a'+eglist[0]]
    e = vocab2dig['b'+eglist[1]]
    t = time2dig[eglist[2]]
    if t not in posdata.keys():
        posdata[t] = [[],[]]
        relabel[t] = []
    for i in range(int(binarygraph[eg]/60)):
        posdata[t][0].append(s)
        posdata[t][1].append(e)
        m = min(10,int(binarygraph[eg]/120))
        relabel[t].append(m)

    if t not in negdata.keys():
        negdata[t] = []
    for i in range(int(binarygraph[eg]/60)):
        negdata[t].append(e)
for key in posdata.keys():
    posdata[key][0] = np.array(posdata[key][0])
    posdata[key][1] = np.array(posdata[key][1])
    relabel[key] = torch.tensor(np.array(relabel[key])).long().cuda()
    negdata[key] = np.array(negdata[key])



class predictnet(nn.Module):
    def __init__(self):
        super(predictnet, self).__init__()
        self.block1 = nn.Linear(192,400)
        self.block2 = nn.Linear(400,400)
        self.classifier1 = nn.Linear(400, 2)
        self.classifier2 = nn.Linear(400, 11)
    def forward(self, x,y,t):
        input1 = torch.cat((x,y,t),1)
        hidden2 = F.relu(self.block1(input1))
        hidden3 = F.relu(self.block2(hidden2))
        out1 = self.classifier1(hidden3)
        out2 = self.classifier2(hidden3)
        return (out1,out2)
    
net = predictnet().cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(net.parameters())+[timevec], lr=3e-3,betas=(0.5, 0.999))



import copy
epoch = 800
num = 10000

posid= {}
negid = {}
for key in posdata.keys():
    posid[key] = list(range(posdata[key][0].shape[0]))
    negid[key] = list(range(negdata[key].shape[0]))

for i in range(epoch):
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for key in posid.keys():
        random.shuffle(posid[key])
        random.shuffle(negid[key])
    for time in posid.keys():
        startid = 0
        while(startid < len(posid[time])):
            endid = min(len(posid[time]) , num + startid)
            psdata = posdata[time][0][posid[time][startid:endid]]
            pedata = posdata[time][1][posid[time][startid:endid]]
            nedata = negdata[time][negid[time][startid:endid]]
            psve = vocabvec[psdata]
            peve = vocabvec[pedata]
            neve = vocabvec[nedata]
            timeve = timevec[time].expand(endid-startid,64)
            (pout1,pout2) = net(psve,peve,timeve)
            (nout1,nout2) = net(psve,neve,timeve)
            plabel1 = torch.ones(endid-startid).long().cuda()
            plabel2 = relabel[time][posid[time][startid:endid]]
            nlabel = torch.zeros(plabel1.shape).long().cuda() * 1
            data = torch.cat((pout1,nout1),0)
            label = torch.cat((plabel1,nlabel),0)
            (num_correct , total) = get_accnum(data, label)
            s1 += num_correct
            s2 += total
            (num_correct , total) = get_accnum(pout2,plabel2)
            s3 += num_correct
            s4 += total
            loss1 = loss_fn(data,label)
            loss2 = loss_fn(pout2,plabel2)
            loss = 4*loss1 + loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            startid = endid
    if (i+1)%100 == 0:
        print((i+1),loss1.item(),loss2.item(),loss.item(),s1,s2,s1/s2,s3,s4,s3/s4)
    if (i+1)%100 == 0 and s3/s4 >= 0.98 and s1/s2>=0.93:
        Net = copy.deepcopy(net)
        tv = copy.deepcopy(timevec)
        torch.save(Net.cpu(), 'data/muti_model_'+str(i+1)+'.pkl')
        np.save('data/timevec_'+str(i+1) + '.npy', tv.cpu().detach().numpy())