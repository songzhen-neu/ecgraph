import argparse
import context.store as store
from context import context

def parserInit():
    parser = argparse.ArgumentParser(description="Pytorch argument parser")
    parser.add_argument('--role', type=str, help='machine role')
    parser.add_argument('--id', type=int, help='the id of role')
    parser.add_argument('--mode', type=str, help='mode')
    parser.add_argument('--worker_num', type=int, help='the number of worker')
    parser.add_argument('--server_num', type=int, help='the number of server')
    parser.add_argument('--data_num', type=int, help='the number of data_raw')
    parser.add_argument('--feature_dim', type=int, help='the dim size of feature')
    parser.add_argument('--class_num', type=int, help='the number of data_raw')
    parser.add_argument('--hidden', type=str, help='hidden')
    parser.add_argument('--ifCompress', type=str, help='ifCompress')
    parser.add_argument('--ifCompensate', type=str, help='ifCompensate')
    parser.add_argument('--data_path', type=str, help='data_path')

    parser.add_argument('--isNeededExactBackProp', type=str, help='isNeededExactBackProp')
    parser.add_argument('--bucketNum', type=int, help='bucketNum')
    parser.add_argument('--IterNum', type=int, help='IterNum')
    parser.add_argument('--ifBackPropCompress', type=str, help='ifBackPropCompress')
    parser.add_argument('--ifBackPropCompensate', type=str, help='ifBackPropCompensate')

    parser.add_argument('--bucketNum_backProp', type=int, help='bucketNum_backProp')
    parser.add_argument('--changeToIter', type=int, help='changeToIter')
    parser.add_argument('--compensateMethod', type=str, help='compensateMethod')
    parser.add_argument('--isChangeRate', type=str, help='isChangeRate')
    parser.add_argument('--bitNum', type=int, help='bitNum')
    parser.add_argument('--trend', type=int, help='trend')
    parser.add_argument('--bitNum_backProp', type=int, help='bitNum_backProp')
    parser.add_argument('--localCodeMode', type=str, help='localCodeMode')

    parser.add_argument('--partitionMethod', type=str, help='partitionMethod')
    parser.add_argument('--raw_data_path', type=str, help='raw_data_path')
    parser.add_argument('--edge_num', type=int, help='edge_num')
    parser.add_argument('--lr', type=float, help='learning rate')

    parser.add_argument('--train_num', type=int, help='train_num')
    parser.add_argument('--val_num', type=int, help='val_num')
    parser.add_argument('--test_num', type=int, help='test_num')

    parser.add_argument('--servers', type=str, help='server ip')
    parser.add_argument('--workers', type=str, help='worker ip')
    parser.add_argument('--master', type=str, help='master ip')

    args = parser.parse_args()
    if args.localCodeMode == 'true':
        context.Context.localCodeMode = True
    else:
        context.Context.localCodeMode = False

    if context.Context.localCodeMode:
        print("setting mode as code")
        context.glContext.config['role'] = args.role
        context.glContext.config['id'] = args.id
        context.glContext.config['mode'] = 'code'
        context.glContext.config['layerNum'] = len(context.glContext.config['hidden']) + 1
    else:
        print("setting mode as test")
        context.glContext.config['role'] = args.role
        context.glContext.config['id'] = args.id
        context.glContext.config['mode'] = args.mode
        context.glContext.config['hidden'] = list(map(int, args.hidden.split(',')))
        context.glContext.config['layerNum'] = len(context.glContext.config['hidden']) + 1
        context.glContext.config['worker_num'] = args.worker_num
        context.glContext.config['server_num'] = args.server_num
        context.glContext.config['data_num'] = args.data_num
        context.glContext.config['feature_dim'] = args.feature_dim
        context.glContext.config['class_num'] = args.class_num

        context.glContext.config['ifCompensate'] = args.ifCompensate

        context.glContext.config['isNeededExactBackProp'] = args.isNeededExactBackProp
        context.glContext.config['bucketNum'] = args.bucketNum
        context.glContext.config['IterNum'] = args.IterNum
        context.glContext.config['ifBackPropCompress'] = args.ifBackPropCompress
        context.glContext.config['ifBackPropCompensate'] = args.ifBackPropCompensate

        context.glContext.config['bucketNum_backProp'] = args.bucketNum_backProp
        context.glContext.config['changeToIter'] = args.changeToIter
        context.glContext.config['compensateMethod'] = args.compensateMethod
        context.glContext.config['data_path'] = args.data_path
        context.glContext.config['bitNum'] = args.bitNum
        context.glContext.config['trend'] = args.trend
        context.glContext.config['bitNum_backProp'] = args.bitNum_backProp
        context.glContext.config['raw_data_path'] = args.raw_data_path
        context.glContext.config['edge_num'] = args.edge_num
        context.glContext.config['partitionMethod'] = args.partitionMethod
        context.glContext.config['lr'] = args.lr
        context.glContext.config['train_num'] = args.train_num
        context.glContext.config['val_num'] = args.val_num
        context.glContext.config['test_num'] = args.test_num

        if args.isChangeRate == 'false':
            context.glContext.config['isChangeRate'] = False
        else:
            context.glContext.config['isChangeRate'] = True

        if args.ifCompress == 'false':
            context.glContext.config['ifCompress'] = False
        elif args.ifCompress == 'true':
            context.glContext.config['ifCompress'] = True

        if args.ifCompensate == 'false':
            context.glContext.config['ifCompensate'] = False
        elif args.ifCompensate == 'true':
            context.glContext.config['ifCompensate'] = True

        if args.isNeededExactBackProp == 'false':
            context.glContext.config['isNeededExactBackProp'] = False
        elif args.isNeededExactBackProp == 'true':
            context.glContext.config['isNeededExactBackProp'] = True

        if args.ifBackPropCompress == 'false':
            context.glContext.config['ifBackPropCompress'] = False
        elif args.ifBackPropCompress == 'true':
            context.glContext.config['ifBackPropCompress'] = True

        if args.ifBackPropCompensate == 'false':
            context.glContext.config['ifBackPropCompensate'] = False
        elif args.ifBackPropCompensate == 'true':
            context.glContext.config['ifBackPropCompensate'] = True

    context.glContext.ipInit(args.servers, args.workers, args.master)
    store.init()
    print('server:{0}'.format(context.glContext.config['server_address']))
    print('master:{0}'.format(context.glContext.config['master_address']))
    print('worker:{0}'.format(context.glContext.config['worker_address']))


def printContext():
    print("role={0},id={1},mode={2},worker_num={3},data_num={4},isNeededExactBackProp={5},feature_dim={6},"
          "class_num={7},hidden={8},ifCompress={9},ifCompensate={10},bucketNum={11},IterNum={12},ifBackPropCompress={13},"
          "ifBackPropCompensate={14},bucketNum_backProp={15},changeToIter={16},compensateMethod={17}"
          .format(context.glContext.config['role'], context.glContext.config['id'], context.glContext.config['mode'],
                  context.glContext.config['worker_num'],
                  context.glContext.config['data_num'], context.glContext.config['isNeededExactBackProp'],
                  context.glContext.config['feature_dim'],
                  context.glContext.config['class_num'], context.glContext.config['hidden'],
                  context.glContext.config['ifCompress'],
                  context.glContext.config['ifCompensate'], context.glContext.config['bucketNum'],
                  context.glContext.config['IterNum'], context.glContext.config['ifBackPropCompress'],
                  context.glContext.config['ifBackPropCompensate'],
                  context.glContext.config['bucketNum_backProp'], context.glContext.config['changeToIter'],
                  context.glContext.config['compensateMethod']))