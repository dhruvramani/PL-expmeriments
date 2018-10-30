import torch
import torch.nn.functional as F
from torch.autograd import Variable

class ParametricLog(torch.nn.Module):
    def __init__(self, shape):
        super(ParametricLog, self).__init__()
        self.c1 = Variable(torch.randn(shape)) # TODO : change this
        self.c2 = Variable(torch.randn(shape))
        self.c1 = self.c1.to(device)
        self.c2 = self.c2.to(device)

    def forward(self, x):
        pos = torch.log(F.relu(x + self.c1) + 1)
        neg = - torch.log(F.relu(-(x - torch.abs(x)) * 0.5 + self.c2) + 1)
        return pos + neg