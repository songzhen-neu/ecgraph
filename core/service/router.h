//
// Created by songzhen on 2020/10/6.
//

#ifndef DGNN_TEST_ROUTER_H
#define DGNN_TEST_ROUTER_H
//#include <vector>
#include "dgnn_client.h"
#include <map>
#include <string>
#include <vector>
#include <time.h>


using namespace std;

class Router {
public:
    static vector<DGNNClient*> dgnnWorkerRouter;
    Router();
    void initWorkerRouter(map<int,string> &dgnnWorkerAddress);
    py::array_t<float> getNeededEmb(
             vector<vector<int>> &nodes,
            int epoch, int layerId, int localId,
             map<int,int> &oldToNewMap,int workerNum,int localNodeSize,
             bool ifCompress,int layerNum,int bitNum,bool isChangeRate,bool isTrain,int trend,int feat_num,string changeRateMode);
    py::array_t<float> getNeededEmb_train(int epoch, int layerId,  bool isTrain,int feat_num,int bitNum);

    float get_comp_percent(int dataNum,int layerNum);
    py::array_t<float> getG(vector<vector<int>> &nodes,int layerId,
                            int localId, int workerNum,
                            bool ifCompress, bool ifcompensate,
                            int bitNum, map<int, int> &oldToNewMap, int localNodeSize,int g_size,int epoch);

//    py::array_t<float> getG(int layerId,
//                           int g_size,int epoch);




};


#endif //DGNN_TEST_ROUTER_H
