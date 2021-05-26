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

    config = {
        'mode': 'code',  # test or code
        'server_address': server_address,
        'role': 'default',
        'id': -1,
        'worker_address': worker_address,
        'master_address': None,
        'server_num': 1,
        'worker_num': 1,
        'lr': 0.1,

        # 'data_path': '/mnt/data/reddit-small',
        # 'raw_data_path': '/mnt/data/reddit-small',
        # 'data_num': 232965,  # 231443
        # 'hidden': [64,64,64],
        # 'feature_dim': 602,
        # 'class_num': 41,
        # 'edge_num': 57307946,  # 11606919
        # 'train_num':153932,
        # 'val_num':23699,
        # 'test_num':55334,

        'data_path': '/mnt/data/cora/',
        'raw_data_path':'/mnt/data_raw/cora/',
        'hidden': [16],
        'data_num': 2708,
        'feature_dim': 1433,
        'class_num': 7,
        'edge_num':5278,
        'train_num':140, #140
        'val_num':300,
        'test_num':1000,

        # 'data_path': '../../data/ogbn-products',
        # 'raw_data_path':'../../data_raw/ogbn-products',
        # 'hidden': [16,4,4],
        # 'data_num': 2449029,
        # 'feature_dim': 100,
        # 'class_num': 47,
        # 'edge_num':61859012,
        # 'train_num':196615,
        # 'val_num':39323,
        # 'test_num':2213091,

        # 'data_path': '/mnt/data/pubmed',
        # 'raw_data_path':'/mnt/data/pubmed',
        # 'hidden': [16,16],
        # 'data_num': 19717,
        # 'feature_dim': 500,
        # 'class_num': 3,
        # 'edge_num':44324,
        # 'train_num':12816,
        # 'val_num':1971,
        # 'test_num':4930,


        # 'data_path': '../../data/test',
        # 'raw_data_path':'../../data_raw/test',
        # 'hidden': [16,16],
        # 'data_num': 36,
        # 'feature_dim': 34,
        # 'class_num': 2,
        # 'edge_num':1,
        # 'train_num':36,
        # 'val_num':0,
        # 'test_num':0,

        'master_id': 0,
        'firstHopForWorkers': [],
        'ifCompress': False,
        'ifMomentum': False,
        'layerNum': 2,
        'bitNum': 1,  # 2,4,8,16bits分别对应桶数2,14,254,65534
        'iterNum': 3000,
        'trend':10,
        'firstProp': True,
        'ifBackPropCompress': False,
        'ifBackPropCompensate': False,
        'bitNum_backProp': 4,
        'isChangeRate': False,
        'isChangeBitNum':False,
        'changeRateMode':'select', # select or normal
        'partitionMethod': 'metis',  # hash,metis
        # accorMix 前k-1层 compensates by Layer, the last layer compensates by Iteration;
        # accorMix2 the first k-1 compensates by Iteration, the last layer pass the complete data_raw
        # accorMix3 前n轮Mix2，从第n+1轮开始按迭代轮
    }
    global worker_id
    global dgnnServerRouter
    global dgnnWorkerRouter
    global dgnnClient
    global dgnnMasterRouter
    global dgnnClientRouterForCpp
    global newToOldMap
    global oldToNewMap
    weights={}
    bias={}
    weightForServer={}

    # server 2001 worker 3001 master 4001
    def ipInit(self,servers,workers,master):
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
            workers=str.split(workers,',')
            servers=str.split(servers,',')
            for i in range(worker_num):
                self.worker_address[i]=workers[i]
                self.server_address[i]=servers[i]
                self.config['master_address']=master
            # self.server_ip = "219.216.64.103"
            # self.master_ip = "219.216.64.103"
            # for i in range(worker_num):
            #     if i == 0:
            #         self.worker_address[i] = "219.216.64.103:3001"
            #     elif i + 1 < 10:
            #         self.worker_address[i] = "192.168.111.10" + str(i + 1) + ":3001"
            #     else:
            #         self.worker_address[i] = "192.168.111.1" + str(i + 1) + ":3001"
            # for i in range(server_num):
            #     if i == 0:
            #         self.server_address[i] = "219.216.64.103:2001"
            #     elif i + 1 < 10:
            #         self.server_address[i] = "192.168.111.10" + str(i + 1) + ":2001"
            #     else:
            #         self.server_address[i] = "192.168.111.1" + str(i + 1) + ":2001"



glContext = Context()
