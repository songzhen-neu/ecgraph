from ecgraph.context import context
import torch
import numpy as np
from cmake.build.lib.pb11_ec import *

def changeCompressBit(comp_percent):
    if comp_percent!=0 and context.glContext.config['isChangeRate']:
        if comp_percent<0.5 and context.glContext.config['bitNum'] !=8:
            context.glContext.config['bitNum']= context.glContext.config['bitNum'] * 2
        elif comp_percent>0.8 and context.glContext.config['bitNum'] !=1:
            context.glContext.config['bitNum']= context.glContext.config['bitNum'] / 2


def pullNeighborG(autograd, nodes, epoch, layerId,graph):
    # 这个函数主要是补全atg.G2
    # 去指定worker获取一阶邻居
    # needed_G_map_from_workers=context.glContext.dgnnClientRouterForCpp.getG(layerId,
    #                                                                         context.glContext.config['emb_dims'][layerId],epoch)
    needed_G_map_from_workers= context.glContext.dgnnClientRouterForCpp.getG(
        graph.fsthop_for_worker,layerId,
        context.glContext.config['id'],
        context.glContext.config['worker_num'],
        context.glContext.config['ifBackPropCompress'],
        context.glContext.config['ifBackPropCompensate'],
        context.glContext.config['bitNum_backProp'],
        graph.id_old2new_dict, len(nodes), context.glContext.config['emb_dims'][layerId],
        epoch
    )



    # 将needed_embs转化为tensor
    needed_G = np.array(needed_G_map_from_workers)
    needed_G = torch.FloatTensor(needed_G)

    #Yu change
    autograd.G[layerId] = torch.cat((autograd.G[layerId], needed_G), 0)
