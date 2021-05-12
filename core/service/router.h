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
    float get_comp_percent(int dataNum,int layerNum);

};


#endif //DGNN_TEST_ROUTER_H
