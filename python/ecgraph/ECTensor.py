from ecgraph import BackWard


class ECTensor(object):
    def __init__(self, tensor=None, left=None, right=None, operator=None,leaky_alpha=None):
        self.tensor = tensor
        self.left = left
        self.right = right
        self.operator = operator
        self.isLeaf=False
        self.grad=None
        self.backwardF=None
        if leaky_alpha is not None:
            self.leaky_alpha=leaky_alpha
        if operator == 'mm':
            self.backwardF = BackWard.MmBackward
        elif operator == 'nllloss':
            self.backwardF = BackWard.NllLossBackward
        elif operator == 'log_softmax':
            self.backwardF = BackWard.LogSoftmaxBackward
        elif operator == 'relu':
            self.backwardF = BackWard.ReluBackward0
        elif operator == 'leaky_relu':
            self.backwardF = BackWard.LeakyReluBackward0
        elif operator is None:
            self.isLeaf=True

    def backward(self):
        BackWard.preOrderTraversal(self)