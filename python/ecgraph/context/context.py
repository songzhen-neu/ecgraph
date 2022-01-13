from cmake.build.lib.pb11_ec import *


class Context(object):
    def foo(self):
        pass

    server_ip = None
    master_ip = None

    worker_address = {}
    server_address = {}
    code_ip = "127.0.0.1"
    localCodeMode = True
    ifShowInfo = False
    isTrain=False

    config = {
        'mode': 'code',  # test or code
        'server_address': server_address,
        'role': 'default',
        'id': -1,
        'worker_address': worker_address,
        'master_address': None,
        'server_num': 2,
        'worker_num': 2,
        'lr': 0.01,

        'print_result_interval': 1,

        # 'data_path': '/mnt/data/reddit-small',
        # 'raw_data_path': '/mnt/data/reddit-small',
        # 'data_num': 232965,  # 231443
        # 'hidden': [16],
        # 'feature_dim': 602,
        # 'class_num': 41,
        # 'edge_num': 57307946,  # 11606919
        # 'train_num':15393, #153932
        # 'val_num':23699,
        # 'test_num':5533, #55334

        # 'data_path': '/mnt/data/cora/',
        # 'raw_data_path': '/mnt/data_raw/cora/',
        # 'hidden': [16],
        # 'data_num': 2708,
        # 'feature_dim': 1433,
        # 'class_num': 7,
        # 'edge_num': 5278,
        # 'train_num': 140,  # 140
        # 'val_num': 300,
        # 'test_num': 1000,
        # 'neigh_sam':[1,5],

        # 'data_path': '/mnt/data/ogbn-papers100M',
        # 'raw_data_path':'/mnt/data_raw/ogbn-papers100M',
        # 'hidden': [16,16],
        # 'data_num': 1546782,
        # 'feature_dim': 128,
        # 'class_num': 172,
        # 'edge_num':13649351,
        # 'train_num':1207179, #140
        # 'val_num':125265,
        # 'test_num':214338,
        # 'neigh_sam':[1,5],

        # 'data_path': '/mnt/data/ogbn-products',
        # 'raw_data_path':'/mnt/data_raw/ogbn-products',
        # 'hidden': [16],
        # 'data_num': 2449029,
        # 'feature_dim': 100,
        # 'class_num': 47,
        # 'edge_num':61859012,
        # 'train_num':196615,
        # 'val_num':39323,
        # 'test_num':2213091,
        # 'neigh_sam':[1,5],

        'data_path': '/mnt/data/pubmed',
        'raw_data_path': '/mnt/data/pubmed',
        'hidden': [16],
        'data_num': 19717,
        'feature_dim': 500,
        'class_num': 3,
        'edge_num': 44324,
        'train_num': 60,  # 12816,
        'val_num': 500,  # 1971,
        'test_num': 1000,  # 4930,
        'neigh_sam': [1000, 100],
        'prune_layer':2,

        # 'data_path': '/mnt/data/test',
        # 'raw_data_path':'/mnt/data/test',
        # 'hidden': [16],
        # 'data_num': 10,
        # 'feature_dim': 2,
        # 'class_num': 2,
        # 'edge_num':20,
        # 'train_num':2,
        # 'val_num':6,
        # 'test_num':2,

        'master_id': 0,
        'ifCompress': False,
        'isChangeRate': False,
        'isChangeBitNum': False,
        'trend': 10,
        'bitNum': 2,  # 2,4,8,16bits分别对应桶数2,14,254,65534
        'layerNum': 2,
        'emb_dims': [],
        'iterNum': 200,
        'ifBackPropCompress': False,
        'ifBackPropCompensate': False,
        'bitNum_backProp': 4,
        'changeRateMode': 'select',  # select or normal
        'partitionMethod': 'hash',  # hash,metis
    }
    global worker_id
    global dgnnServerRouter
    global dgnnWorkerRouter
    global dgnnClient
    global dgnnMasterRouter
    global dgnnClientRouterForCpp
    global newToOldMap
    global oldToNewMap
    time_epoch = {}
    weights = {}
    bias = {}
    weightForServer = {}
    parameters = {}
    parametersForServer = {}
    gradients = {}
    gradientsForServer = {}
    firstHopFeature = None

    # server 2001 worker 3001 master 4001
    def ipInit(self, servers, workers, master):
        worker_num = glContext.config['worker_num']
        server_num = glContext.config['server_num']
        if self.config['mode'] == 'code':
            self.server_ip = self.code_ip
            self.master_ip = self.code_ip
            for i in range(worker_num):
                self.worker_address[i] = self.code_ip + ":300" + str(i + 1)
            for i in range(server_num):
                self.server_address[i] = self.code_ip + ":200" + str(i + 1)
            self.config['master_address'] = self.master_ip + ":4001"
        elif self.config['mode'] == 'test':
            workers = str.split(workers, ',')
            servers = str.split(servers, ',')
            for i in range(worker_num):
                self.worker_address[i] = workers[i]
                self.server_address[i] = servers[i]
                self.config['master_address'] = master
        self.time_epoch['set_embs'] = 0
        self.time_epoch['get_embs'] = 0
        self.time_epoch['trans_embs_dict'] = 0
        self.time_epoch['forward'] = 0
        self.time_epoch['backward'] = 0
        self.time_epoch['backward_m'] = 0
        self.time_epoch['update'] = 0
        self.time_epoch['set_g'] = 0
        self.time_epoch['get_g'] = 0


    def initCluster(self):
        self.dgnnServerRouter = []
        self.dgnnWorkerRouter = []
        self.dgnnClient = DGNNClient()
        self.dgnnMasterRouter = DGNNClient()
        self.dgnnClientRouterForCpp = Router()
        self.worker_id = self.config['id']
        id = self.config['id']
        # 当前机器的客户端，需要启动server，以保证不同机器间中间表征向量传输

        self.dgnnClient.serverAddress = self.config['worker_address'][id]

        self.dgnnClient.startClientServer()
        for i in range(self.config['server_num']):
            self.dgnnServerRouter.insert(i, DGNNClient())
            self.dgnnServerRouter[i].init_by_address(self.config['server_address'][i])
        for i in range(self.config['worker_num']):
            self.dgnnWorkerRouter.insert(i, DGNNClient())
            self.dgnnWorkerRouter[i].init_by_address(self.config['worker_address'][i])

        self.dgnnMasterRouter.init_by_address(self.config['master_address'])

        # 在c++端初始化dgnnWorkerRouter
        self.dgnnClientRouterForCpp.initWorkerRouter(self.config['worker_address'])

        # 所有创建的类都在一个进程里，通过c++对静态变量操作，在所有类中都可见
        # print(dgnnClient.testString)
        # print(dgnnMasterRouter.testString)
        # print(dgnnServerRouter[0].testString)

        self.dgnnMasterRouter.pullDataFromMasterGeneral(
            id, self.config['worker_num'],
            self.config['data_num'],
            self.config['data_path'],
            self.config['feature_dim'],
            self.config['class_num'],
            self.config['partitionMethod'],
            self.config['edge_num'])


glContext = Context()
