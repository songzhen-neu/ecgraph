from context import context
import context.store as store
import torch
import numpy as np

# def pullNeighborG(nodes, epoch, layerId):
#     # 这个函数主要是补全atg.G2
#     # 去指定worker获取一阶邻居
#     needed_G_map_from_workers = {}
#
#     # start = time.time()
#     for i in range(context.glContext.config['worker_num']):
#         if i != context.glContext.worker_id:
#             try:
#                 # t=threading.Thread(target=accessG_backProp,args=(needed_G_map_from_workers, i, layerId, epoch,))
#                 # t.start()
#                 # _thread.start_new_thread(accessG_backProp, (needed_G_map_from_workers, i, layerId, epoch))
#                 accessG_backProp(needed_G_map_from_workers, i, layerId, epoch)
#             except:
#                 print("Error:无法启动线程")
#
#     needed_G = [None] * (len(context.glContext.newToOldMap) - len(nodes))
#     # for循环遍历每个从远端获取的特征
#     for wid in range(context.glContext.config['worker_num']):
#         if wid != context.glContext.worker_id:
#             for i, nid in enumerate(context.glContext.config['firstHopForWorkers'][wid]):
#                 new_id = context.glContext.oldToNewMap[nid] - len(nodes)
#                 needed_G[new_id] = needed_G_map_from_workers[wid][i]
#
#     # 将needed_embs转化为tensor
#     needed_G = np.array(needed_G)
#     needed_G = torch.FloatTensor(needed_G)
#
#     #Yu change
#     autoG.atg.G[layerId] = torch.cat((autoG.atg.G[layerId], needed_G), 0)
#     if layerId == 2:
#         atg.G2 = torch.cat((atg.G2, needed_G), 0)
#     elif layerId == 3:
#         atg.G3 = torch.cat((atg.G3, needed_G), 0)
#     elif layerId == 4:
#         atg.G4 = torch.cat((atg.G4, needed_G), 0)

def accessG_backProp(needed_G_map_from_workers, i, layerId, epoch):
    if not context.glContext.config['ifBackPropCompress']:
        needed_G_map_from_workers[i] = \
            context.glContext.dgnnWorkerRouter[i].worker_pull_needed_G(
                context.glContext.config['firstHopForWorkers'][i], layerId)
    else:
        needed_G_map_from_workers[i] = \
            context.glContext.dgnnWorkerRouter[i].worker_pull_needed_G_compress(
                context.glContext.config['firstHopForWorkers'][i],
                context.glContext.config['ifBackPropCompensate'], layerId, epoch,
                context.glContext.config['bucketNum_backProp'],
                context.glContext.config['bitNum_backProp']
            )
    store.threadCountList_backProp[layerId - 2] += 1

def changeCompressBit(comp_percent):
    if comp_percent!=0 and context.glContext.config['isChangeRate']:
        if comp_percent<0.5 and context.glContext.config['bitNum'] !=8:
            context.glContext.config['bitNum']=context.glContext.config['bitNum']*2
        elif comp_percent>0.8 and context.glContext.config['bitNum'] !=1:
            context.glContext.config['bitNum']=context.glContext.config['bitNum']/2

def pullNeighborG(autograd, nodes, epoch, layerId):
    # 这个函数主要是补全atg.G2
    # 去指定worker获取一阶邻居
    needed_G_map_from_workers = {}

    # start = time.time()
    for i in range(context.glContext.config['worker_num']):
        if i != context.glContext.worker_id:
            try:
                # t=threading.Thread(target=accessG_backProp,args=(needed_G_map_from_workers, i, layerId, epoch,))
                # t.start()
                # _thread.start_new_thread(accessG_backProp, (needed_G_map_from_workers, i, layerId, epoch))
                accessG_backProp(needed_G_map_from_workers, i, layerId, epoch)
            except:
                print("Error:无法启动线程")

    needed_G = [None] * (len(context.glContext.newToOldMap) - len(nodes))
    # for循环遍历每个从远端获取的特征
    for wid in range(context.glContext.config['worker_num']):
        if wid != context.glContext.worker_id:
            for i, nid in enumerate(context.glContext.config['firstHopForWorkers'][wid]):
                new_id = context.glContext.oldToNewMap[nid] - len(nodes)
                needed_G[new_id] = needed_G_map_from_workers[wid][i]

    # 将needed_embs转化为tensor
    needed_G = np.array(needed_G)
    needed_G = torch.FloatTensor(needed_G)

    #Yu change
    autograd.G[layerId] = torch.cat((autograd.G[layerId], needed_G), 0)
