import argparse
from ecgraph.context import context


def parserInit():
    parser = argparse.ArgumentParser(description="Pytorch argument parser")
    parser.add_argument('--role_id', type=str, help='machine role and id')
    parser.add_argument('--ifctx_mode', type=str, help='context from file or arguments delivering')
    parser.add_argument('--worker_server_num', type=str, help='the number of worker and server')
    parser.add_argument('--vtx_edge_feat_class_train_val_test', type=str, help='vtx_edge_feat_class_train_val_test')

    parser.add_argument('--hidden', type=str, help='hidden')
    parser.add_argument('--data_path', type=str, help='data_path')

    parser.add_argument('--if_cprs_trend_backcprs_backcpst_changeBit', type=str, help='if_cprs_trend_backcprs_backcpst_changeBit')

    parser.add_argument('--bit_backbit_trend_printepoch', type=str, help='bit_backbit_trend_printepoch')

    parser.add_argument('--iter_lr_pttMethod', type=str, help='iter_lr_pttMethod')

    parser.add_argument('--servers', type=str, help='server ip')
    parser.add_argument('--workers', type=str, help='worker ip')
    parser.add_argument('--master', type=str, help='master ip')

    parser.add_argument('--distgnnr',type=int,default=1,help='r for distgnn')


    # parameter for GAT
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--prune_layer', type=int, default=0, help='prune.')
    parser.add_argument('--neigh_sam', type=str, help='neigh_sam.')

    args = parser.parse_args()
    ifctx_mode=str.split(args.ifctx_mode,',')


    if ifctx_mode[0] == 'true':
        context.Context.localCodeMode = True
    else:
        context.Context.localCodeMode = False

    if context.Context.localCodeMode:
        print("setting mode as code")
        role_id=str.split(args.role_id,',')
        context.glContext.config['role'] = role_id[0]
        context.glContext.config['id']=int(role_id[1])

        context.glContext.config['mode'] = 'code'
        context.glContext.config['layerNum'] = len(context.glContext.config['hidden']) + 1
        context.glContext.config['emb_dims'].append(context.glContext.config['feature_dim'])
        context.glContext.config['emb_dims'].extend(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['class_num'])
    else:
        print("setting mode as test")
        role_id=str.split(args.role_id,',')
        context.glContext.config['role'] = role_id[0]
        context.glContext.config['id']=int(role_id[1])

        context.glContext.config['mode'] = ifctx_mode[1]
        context.glContext.config['hidden'] = list(map(int, args.hidden.split(',')))
        context.glContext.config['layerNum'] = len(context.glContext.config['hidden']) + 1

        worker_server_num=str.split(args.worker_server_num,',')
        context.glContext.config['worker_num'] = int(worker_server_num[0])
        context.glContext.config['server_num'] = int(worker_server_num[1])

        vtx_edge_feat_class_train_val_test=str.split(args.vtx_edge_feat_class_train_val_test,',')
        context.glContext.config['data_num'] = int(vtx_edge_feat_class_train_val_test[0])
        context.glContext.config['edge_num']=int(vtx_edge_feat_class_train_val_test[1])
        context.glContext.config['feature_dim'] = int(vtx_edge_feat_class_train_val_test[2])
        context.glContext.config['class_num'] = int(vtx_edge_feat_class_train_val_test[3])
        context.glContext.config['train_num'] = int(vtx_edge_feat_class_train_val_test[4])
        context.glContext.config['val_num'] = int(vtx_edge_feat_class_train_val_test[5])
        context.glContext.config['test_num'] = int(vtx_edge_feat_class_train_val_test[6])

        context.glContext.config['emb_dims'].append(context.glContext.config['feature_dim'])
        context.glContext.config['emb_dims'].extend(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['class_num'])

        context.glContext.config['prune_layer']=int(args.prune_layer)


        # GAT
        context.glContext.config['nb_heads']=int(args.nb_heads)
        context.glContext.config['alpha']=float(args.alpha)

        if_array=str.split(args.if_cprs_trend_backcprs_backcpst_changeBit,',')
        if if_array[0] == 'false':
            context.glContext.config['ifCompress'] = False
        elif if_array[0] == 'true':
            context.glContext.config['ifCompress'] = True

        if if_array[1] == 'false':
            context.glContext.config['isChangeRate'] = False
        else:
            context.glContext.config['isChangeRate'] = True

        if if_array[2] == 'false':
            context.glContext.config['ifBackPropCompress'] = False
        elif if_array[2] == 'true':
            context.glContext.config['ifBackPropCompress'] = True

        if if_array[3] == 'false':
            context.glContext.config['ifBackPropCompensate'] = False
        elif if_array[3] == 'true':
            context.glContext.config['ifBackPropCompensate'] = True

        if if_array[4] == 'false':
            context.glContext.config['isChangeBitNum'] = False
        elif if_array[4]  == 'true':
            context.glContext.config['isChangeBitNum'] = True

        bit_backbit_trend_printepoch=str.split(args.bit_backbit_trend_printepoch,',')

        context.glContext.config['neigh_sam']=str.split(args.neigh_sam, ',')
        context.glContext.config['neigh_sam']=[int(context.glContext.config['neigh_sam'][i]) for i in range(len(
            context.glContext.config['neigh_sam']))]
        context.glContext.config['bitNum'] = int(bit_backbit_trend_printepoch[0])
        context.glContext.config['bitNum_backProp'] = int(bit_backbit_trend_printepoch[1])
        context.glContext.config['trend'] = int(bit_backbit_trend_printepoch[2])
        context.glContext.config['print_result_interval'] = int(bit_backbit_trend_printepoch[3])

        context.glContext.config['data_path'] = args.data_path

        iter_lr_pttMethod=str.split(args.iter_lr_pttMethod,',')
        context.glContext.config['iterNum'] = int(iter_lr_pttMethod[0])
        context.glContext.config['lr'] = float(iter_lr_pttMethod[1])
        context.glContext.config['partitionMethod'] = iter_lr_pttMethod[2]

        context.glContext.config['distgnn_r']=args.distgnnr

    context.glContext.ipInit(args.servers, args.workers, args.master)
    # store.init()
    print('server:{0}'.format(context.glContext.config['server_address']))
    print('master:{0}'.format(context.glContext.config['master_address']))
    print('worker:{0}'.format(context.glContext.config['worker_address']))


def printContext():
    print("role={0},id={1},mode={2},worker_num={3},data_num={4},feature_dim={5},"
          "class_num={6},hidden={7},ifCompress={8},iterNum={9},ifBackPropCompress={10},"
          "ifBackPropCompensate={11}"
          .format(context.glContext.config['role'], context.glContext.config['id'], context.glContext.config['mode'],
                  context.glContext.config['worker_num'],
                  context.glContext.config['data_num'],
                  context.glContext.config['feature_dim'],
                  context.glContext.config['class_num'], context.glContext.config['hidden'],
                  context.glContext.config['ifCompress'],
                  context.glContext.config['iterNum'], context.glContext.config['ifBackPropCompress'],
                  context.glContext.config['ifBackPropCompensate']))