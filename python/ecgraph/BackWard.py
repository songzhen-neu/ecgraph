import torch
import numpy as np


# f means former, l means latter;
# e.g., mm: xl=xf.mm(wf)

def MmBackward(xf, wf, xl_g, flag):
    if flag == 'x':
        return xl_g.mm(wf.t())
    elif flag == 'w':
        return xf.t().mm(xl_g)


def NllLossBackward(xf, lab,idx_train):
    # dim,index,src
    # g_x = torch.zeros_like(xf).scatter(1, lab.reshape(lab.size()[0], 1), -1)
    g_x=torch.zeros_like(xf)
    for i in idx_train:
        g_x[i][lab[i]]=-1



    # g_x = g_x / xf.size()[0]
    g_x = g_x / len(idx_train)
    return g_x
    # print(g_x)
    # print(lab)


def LogSoftmaxBackward(xf_softmax, xl_g):
    xf_softmax = xf_softmax.detach().tolist()
    x1_g = np.zeros_like(xf_softmax)
    size_0 = len(xf_softmax)
    size_1 = len(xf_softmax[0])
    index = torch.nonzero(xl_g)
    index = index.detach().tolist()
    for m in range(len(index)):
        i = index[m][0]
        j = index[m][1]
        for k in range(size_1):
            if j == k:
                x1_g[i][k] += 1 - xf_softmax[i][k]
            else:
                x1_g[i][k] += -xf_softmax[i][k]

    x1_g = torch.FloatTensor(x1_g)
    x2_g = xl_g.mm(torch.ones(xl_g.size()[1], 1))
    x1_g = x1_g * x2_g
    return x1_g


def ReluBackward0(xf, xl_g):
    zero = torch.zeros_like(xf)
    one = torch.ones_like(xf)
    x_g = torch.where(xf > 0, one, zero)
    return x_g * xl_g

def LeakyReluBackward0(xf,xl_g,alpha):
    alpha_X=0
    if alpha!=0:
        alpha_X = torch.ones_like(xf)*alpha
    else :
        print("error: alpha cannot equal to 0")
    one = torch.ones_like(xf)
    x_g = torch.where(xf > 0, one, alpha_X)
    return x_g * xl_g

def preOrderTraversal(root):
    if root.operator is None:
        return
    else:
        if root.operator == 'mm':
            if root.left is not None:
                root.left.grad=root.backwardF(root.left.tensor,root.right.tensor,root.grad,'x')
            if root.right is not None:
                root.right.grad=root.backwardF(root.left.tensor,root.right.tensor,root.grad,'w')
        elif root.operator == 'nllloss':
            if root.left is not None:
                root.left.grad=root.backwardF(root.left.tensor,root.right.tensor)
            if root.right is not None:
                root.right.grad=root.backwardF(root.left.tensor,root.right.tensor)
        elif root.operator == 'log_softmax':
            if root.left is not None:
                root.left.grad=root.backwardF(root.left.tensor,root.grad)
            if root.right is not None:
                root.right.grad=root.backwardF(root.right.tensor,root.grad)
        elif root.operator == 'relu':
            if root.left is not None:
                root.left.grad=root.backwardF(root.left.tensor,root.grad)
            if root.right is not None:
                root.right.grad=root.backwardF(root.right.tensor,root.grad)
        elif root.operator == 'leaky_relu':
            if root.left is not None:
                root.left.grad=root.backwardF(root.left.tensor,root.grad,root.leaky_alpha)
            if root.right is not None:
                root.right.grad=root.backwardF(root.right.tensor,root.grad,root.leaky_alpha)
        if root.left is not None:
            preOrderTraversal(root.left)
    if root.right is not None:
        preOrderTraversal(root.right)

