//
// Created by songzhen on 2020/10/6.
//

#include "router.h"
#include <cstring>

Router::Router() {}

vector<DGNNClient *> Router::dgnnWorkerRouter;


float Router::get_comp_percent(int dataNum,int layerNum){
    float tmp=WorkerStore::comp_percent/(float)(dataNum*(layerNum-1));
    WorkerStore::comp_percent=0;
    return tmp;
};


void Router::initWorkerRouter(map<int, string> &dgnnWorkerAddress) {

    for (auto &address:dgnnWorkerAddress) {
        DGNNClient *dgnnClient = new DGNNClient();
        dgnnClient->init_by_address(address.second);
        dgnnWorkerRouter.push_back(dgnnClient);
        cout << address.second << endl;
    }
//    cout<<dgnnWorkerRouter[1]->add1()<<endl;
}


static void parseDenseEmb(int workerNum, int localId, vector<EmbMessage> replyVec, vector<vector<int>> nodes,
                          map<int, int> oldToNewMap, int localNodeSize, int feat_num, float *ptr_result) {
    for (int i = 0; i < workerNum; i++) {
        if (i != localId) {
            auto &denseMessage = replyVec[i].denseembmessage();
            auto &node_worker = nodes[i];
            int len = node_worker.size();
            for (int j = 0; j < len; j++) {
                int nid = node_worker[j];
                int new_id = oldToNewMap[nid] - localNodeSize;
                const auto &denseMessage_node = denseMessage.embs(j);

                for (int k = 0; k < feat_num; k++) {
                    ptr_result[new_id * feat_num + k] = (float) denseMessage_node.tensor(k);

                }
            }
        }
    }


}

py::array_t<float> Router::getNeededEmb(vector<vector<int>> &nodes,
                                        int epoch, int layerId, int localId,
                                        map<int, int> &oldToNewMap, int workerNum, int localNodeSize,
                                        bool ifCompress, int layerNum, int bitNum, bool isChangeRate, bool isTrain,
                                        int trend, int feat_num, string changeRateMode) {
//    for (int i = 0; i < nodes.size(); i++) {
//        cout << "request nodes from worker  " << i << " num:" << nodes[i].size() << endl;
//    }
//    cout << "oldToNewMap size:" << oldToNewMap.size() << endl;

    vector<EmbMessage> replyVec(workerNum);

    int totalNodeNum = 0;
    if(isTrain){
        totalNodeNum=oldToNewMap.size()-localNodeSize;
//        cout<<"totalNodeNum:"<<totalNodeNum<<endl;
    }else{
        for (int i = 0; i < nodes.size(); i++) {
            if (i != localId) {
                totalNodeNum += nodes[i].size();
            }
        }
    }


//    cout<<"local node size:  "<<localNodeSize<<endl;
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    auto result = py::array_t<float>(totalNodeNum * feat_num);
    result.resize({totalNodeNum, feat_num});
    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;
//    memset(ptr_result,0,totalNodeNum * feat_num);
    for(int i=0;i<totalNodeNum*feat_num;i++){
        if(ptr_result[i]!=0){
            ptr_result[i]=0;
        }
    }


    //  从远端异步获取
    for (int i = 0; i < workerNum; i++) {
        if (i != localId) {
            pthread_t p;
            auto *metaData = new ReqEmbsMetaData;
            metaData->reply = &replyVec[i];
            metaData->serverId = i;
            metaData->workerId = localId;
            metaData->epoch = epoch;
            metaData->nodes = &nodes[i];
            metaData->layerId = layerId;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->ifCompress = ifCompress;
            metaData->layerNum = layerNum;
            metaData->bitNum = bitNum;
            metaData->trend = trend;
            metaData->ptr_result = ptr_result;
            metaData->oldToNewMap = &oldToNewMap;
            metaData->localNodeSize = localNodeSize;
            metaData->feat_num = feat_num;


            // multiple threads for requesting
//            DGNNClient::worker_pull_needed_emb_parallel((void *) metaData);

            if (isChangeRate && isTrain) {
                 if (changeRateMode=="select"){
                    pthread_create(&p, NULL, DGNNClient::worker_pull_emb_trend_parallel_select, (void *) metaData);
                }

            } else {
                if (ifCompress && isTrain) {
                    pthread_create(&p, NULL, DGNNClient::worker_pull_emb_compress_parallel, (void *) metaData);
                } else {
                    pthread_create(&p, NULL, DGNNClient::worker_pull_needed_emb_parallel, (void *) metaData);
                }
            }

        }
    }


    while (ThreadUtil::count_respWorkerNumForEmbs != workerNum - 1) {}
    ThreadUtil::count_respWorkerNumForEmbs = 0;



    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
//    cout << "reply responsing time:" << timeuse << "s" << endl;
    return result;
}



py::array_t<float> Router::getG(vector<vector<int>> &nodes,int layerId,
                                int localId, int workerNum,
                                bool ifCompress, bool ifcompensate,
                                int bitNum, map<int, int> &oldToNewMap, int localNodeSize,int emb_dim,int epoch) {

    vector<EmbMessage> replyVec(workerNum);
    int totalNodeNum = 0;
    totalNodeNum=oldToNewMap.size()-localNodeSize;


//    cout<<"local node size:  "<<localNodeSize<<endl;
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);


    auto result = py::array_t<float>(totalNodeNum * emb_dim);
    result.resize({totalNodeNum, emb_dim});
    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;
//    memset(ptr_result,0,totalNodeNum * feat_num);
    for(int i=0;i<totalNodeNum*emb_dim;i++){
        if(ptr_result[i]!=0){
            ptr_result[i]=0;
        }
    }

    //  从远端异步获取
    for (int i = 0; i < workerNum; i++) {
        if (i != localId) {
            pthread_t p;
            auto *metaData = new ReqEmbsMetaData;
            metaData->reply = &replyVec[i];
            metaData->serverId = i;
            metaData->workerId = localId;
            metaData->nodes = &nodes[i];
            metaData->layerId = layerId;
            metaData->dgnnClient = dgnnWorkerRouter[i];
            metaData->ifCompress = ifCompress;
            metaData->bitNum = bitNum;
            metaData->ptr_result = ptr_result;
            metaData->oldToNewMap = &oldToNewMap;
            metaData->localNodeSize = localNodeSize;
            metaData->feat_num=emb_dim;
            metaData->ifCompensate=ifcompensate;
            metaData->epoch=epoch;

            // multiple threads for requesting
//            DGNNClient::worker_pull_needed_emb_parallel((void *) metaData);
            if(!ifCompress){
                pthread_create(&p, NULL, DGNNClient::worker_pull_g_parallel, (void *) metaData);
            }
            else{
                pthread_create(&p, NULL, DGNNClient::worker_pull_g_compress_parallel, (void *) metaData);
            }


        }
    }


    while (ThreadUtil::count_respWorkerNumForEmbs != workerNum - 1) {}
    ThreadUtil::count_respWorkerNumForEmbs = 0;



    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
//    cout << "reply responsing time:" << timeuse << "s" << endl;
    return result;
}
