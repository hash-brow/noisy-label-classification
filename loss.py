import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q 
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
    
    def forward(self, logits, targets, indexes):
        # targets = torch.tensor(targets, dtype=torch.int64)
        p = F.softmax(logits, dim=1)
        # print(targets.dtype, targets.get_device())
        # print(p)
        # print(targets.shape)
        # print(targets)
        # print(torch.argmax(targets, dim=1))
        # print(torch.unsqueeze(targets, 1).type(torch.IntTensor).dtype)
        # print(torch.unsqueeze(targets, 1).type(torch.IntTensor).get_device())
        Yg = torch.gather(p, 1, torch.unsqueeze(torch.argmax(targets, dim=1), 1))
        # print(Yg)
        # print(indexes)
        # print(self.weight[indexes])
        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss
    
    def updateWeight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1, dtype=torch.int64))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)