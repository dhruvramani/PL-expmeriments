import torch
import torch.nn.functional as F
from torch.autograd import Variable

class ParametricLog(torch.nn.Module):
    def __init__(self, shape):
        super(ParametricLog, self).__init__()
        self.c1 = Variable(torch.random_normal(shape)) # TODO : change this
        self.c2 = Variable(torch.random_normal(shape))

    def forward(self, x):
        pos = torch.log(F.relu(x + self.c1) + 1)
        neg = - torch.log(F.relu(-(x - torch.abs(x)) * 0.5 + self.c2) + 1)
        return pos + neg