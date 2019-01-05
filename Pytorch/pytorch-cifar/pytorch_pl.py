import torch
import torch.nn.functional as F
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ParametricLog(torch.nn.Module):
    def __init__(self):
        super(ParametricLog, self).__init__()
        self.c1 = None #Variable(torch.randn(shape)) # TODO : change this
        self.c2 = None #Variable(torch.randn(shape)
        self.start = 0

    def forward(self, x):
        if(self.start == 0):
            self.c1 = Variable(torch.randn(x.shape))
            self.c2 = Variable(torch.randn(x.shape))
            self.c1 = self.c1.to(device)
            self.c2 = self.c2.to(device)
            self.start = 1

        pos_v, neg_v = F.relu(x), (-(x - torch.abs(x)) * 0.5)
        pos = torch.log(pos_v + torch.abs(self.c1) + 1) #* (pos_v / x)
        #neg = - torch.log(neg_v + torch.abs(self.c2) + 1) #* (neg_v / x)
        return pos #+ neg