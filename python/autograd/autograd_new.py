import torch.nn.functional as F
#from sympy.core import singleton

import util_python.remote_access as ra

import torch
import numpy as np
import sys, os
from context import context


class AutoGrad(object):
    # G1-4 H1-4,Z1-4,Loss, Y0-3,A_X_H0-3,sigma_z(1-3)_grad,B0-3,errorH1,errorAgg1

    G = []
    H = []
    Z = []
    Y = []
    A_X_H = []
    sigma_z_grad = []
    B = []
    weight = []
    bias = []
    activation = []
    layer_num = 0

    def __init__(self):
        self.layer_num = context.glContext.config['layerNum']
        self.G = [None for i in range(self.layer_num+2)]
        self.Z = [None for i in range(self.layer_num+2)]
        self.Y = [None for i in range(self.layer_num+2)]
        self.H = [None for i in range(self.layer_num+2)]
        self.B = [None for i in range(self.layer_num+2)]
        self.A_X_H = [None for i in range(self.layer_num+2)]
        self.sigma_z_grad = [None for i in range(self.layer_num+2)]
        self.weight = [None for i in range(self.layer_num+2)]
        self.bias = [None for i in range(self.layer_num+2)]
        self.activation = [None for i in range(self.layer_num+2)]

    def set_A_X_H(self, layer_id, aggregate):
        self.A_X_H[layer_id] = aggregate

    def set_activation(self, activation):
        self.activation = activation

    def forward_detail_layer(self, model, x, adj, nodes, epoch, weight, bias, layer):
        x = model.gc[layer](x, adj, nodes, epoch, weight[layer-1], bias[layer-1], self)
        if layer == self.layer_num:
            self.Z[layer] = x
            return x

        x = F.normalize(x, p=2, dim=1) # 是否每层都需要
        self.Z[layer] = x
        x = self.Active(x, layer)
        self.H[layer] = x
        return x

    def Active(self, x, layer):
        act = self.activation[layer]
        if act!=None:
            x = act(x)
        return x

    def de_activation(self, layer):
        act = self.activation[layer]
        if(act == F.relu):
            self.sigma_z_grad[layer] = torch.tensor(np.where(self.Z[layer].data > 0, 1, 0))
        elif(act == F.sigmoid):
            # sigmod' = x(1-x)
            self.sigma_z_grad[layer] = self.Z[layer].data
            self.sigma_z_grad[layer] = self.sigma_z_grad[layer] *(1 - self.sigma_z_grad[layer])
            self.sigma_z_grad[layer] = torch.tensor(self.sigma_z_grad[layer])
        elif(act == F.tanh):
            # tanh' = 1 - x*x
            self.sigma_z_grad[layer] = self.Z[layer].data
            self.sigma_z_grad[layer] = 1 - self.sigma_z_grad[layer]*self.sigma_z_grad[layer]
        elif(act == None):
            return

    def forward_detail(self, model, x, adj, nodes, epoch, weight, bias):
        for i in range(1, self.layer_num+1):
            x = self.forward_detail_layer(model, x, adj, nodes, epoch, weight, bias, i)

        # 是否所有的层都要经过 x = F.normalize(x, p=2, dim=1)?
        # x = model.gc[self.layer_num](x, adj, nodes, epoch, weight[self.layer_num-1], bias[self.layer_num-1],self)
        # self.Z[self.layer_num] = x
        return x

    def back_prop_detail_layer(self, lay_id, model, epoch, nodes, adjs , dgnnClient, id_new2old_map):
        if lay_id == self.layer_num:
            self.G[self.layer_num] = self.Z[self.layer_num].grad
            self.Y[self.layer_num-1] = torch.mm(self.A_X_H[self.layer_num-1].t(), self.G[self.layer_num])
            self.B[self.layer_num-1] = self.G[self.layer_num].detach().numpy().sum(axis=0)
        else:
            self.de_activation(lay_id)
            context.glContext.dgnnServerRouter[0].server_Barrier(0)
            # pullNeighborG 代码应该进行修改
            ra.pullNeighborG(self, nodes, epoch, lay_id+1)
            a = torch.spmm(adjs, self.G[lay_id+1])
            b = torch.mm(a, model.gc[lay_id+1].weight.t())
            self.G[lay_id] = torch.mul(b, self.sigma_z_grad[lay_id])
            self.Y[lay_id-1] = torch.mm(self.A_X_H[lay_id-1].t(), self.G[lay_id])
            self.B[lay_id-1] = self.G[lay_id].detach().numpy().sum(axis=0)

        if lay_id != 1:
            G_list = self.G[lay_id].detach().numpy().tolist()
            G_dict = {}
            for j in range(len(G_list)):
                G_dict[id_new2old_map[j]] = G_list[j]
            dgnnClient.setG(G_dict, lay_id)

    def back_prop_detail(self, dgnnClient, model, id_new2old_map, nodes, epoch, adjs):
        for i in range(self.layer_num, 0, -1):
            self.back_prop_detail_layer(i, model, epoch, nodes, adjs , dgnnClient, id_new2old_map)

        np.set_printoptions(threshold=sys.maxsize)
        for i in range(1, self.layer_num+1):
            model.gc[i].weight.grad.data = self.Y[i-1]
            model.gc[i].bias.grad.data = torch.FloatTensor(self.B[i-1])

    def set_HZ(self, output, required_grad, if_retain_grad, x):
        self.H[x] = output
        self.Z[x].required_grad = required_grad
        if if_retain_grad:
            self.Z[x].retain_grad()
