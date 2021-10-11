import ecgraph
import torch.nn.functional as F


class Function(object):
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            orig = super(Function, cls)
            cls._instance = orig.__new__(cls)
        cls._instance.compGraph = None
        return cls._instance

    def mm(self, x1, w1):
        x2 = x1.tensor.mm(w1.tensor)
        x2 = ecgraph.ECTensor(x2, x1, w1, 'mm', None)
        self.compGraph = x2
        return x2

    def leaky_relu(self, x1, leaky_alpha):
        x2 = F.leaky_relu(x1.tensor, leaky_alpha)
        x2 = ecgraph.ECTensor(x2, x1, None, 'leaky_relu', leaky_alpha)
        self.compGraph = x2
        return x2

    def relu(self, x1):
        x2 = F.relu(x1.tensor)
        x2 = ecgraph.ECTensor(x2, x1, None, 'relu', None)
        self.compGraph = x2
        return x2

    def log_softmax(self, x1, dim):
        x2 = F.softmax(x1.tensor, dim=dim).detach()
        x3 = F.log_softmax(x1.tensor, dim=dim)
        x1.tensor.data=x2
        x3 = ecgraph.ECTensor(x3, x1, None, 'log_softmax', None)
        self.compGraph = x3
        return x3

    def nll_loss(self, x1, lab):
        loss = F.nll_loss(x1.tensor, lab.tensor)
        loss = ecgraph.ECTensor(loss, x1, lab, 'nllloss', None)
        self.compGraph = loss
        return loss
