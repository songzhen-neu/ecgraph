//
// Created by songzhen on 2020/10/8.
//

#ifndef DGNN_TEST_WORKERSTORE_H
#define DGNN_TEST_WORKERSTORE_H

#include <vector>
#include <map>
#include <set>
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../util/check.h"

// 不要导入<>这种grpc编译的库，不然生不成新的message
using namespace std;
using dgnn_test::NodeMessage;
using dgnn_test::DataMessage;
using dgnn_test::DataMessage_FeatureMessage;
using dgnn_test::DataMessage_LabelMessage;
using dgnn_test::DataMessage_AdjMessage;
using dgnn_test::DataMessage_FeatureMessage_FeatureItem;
using dgnn_test::DataMessage_LabelMessage_LabelItem;
using dgnn_test::DataMessage_AdjMessage_AdjItem;
using dgnn_test::ContextMessage;

// vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
// node;feature;label;adj
class WorkerStore {
public:
    static vector<int> nodes;
    static map<int, vector<float>> features;
    static map<int, int> labels;
    static map<int, set<int>> adjs;
    static vector<vector<int>> nodesForEachWorker;
    static string testString;
    static map<int,vector<float>> embs;
    static map<int,map<int,map<int,vector<float>>>> embs_from_remote;
    static map<int,map<int,vector<float>>>  embs_momument;
    // worker layer node
    static map<int,map<int,map<int,vector<float>>>>  embs_last;
    // 1st map (workerId) 2nd map(layerId) 3rd map (vertexId)
    static map<int,map<int,map<int,vector<float>>>>  embs_change_rate;
    static map<int,map<int,map<int,vector<float>>>>  embs_change_rate_worker;

    static map<int,map<int,vector<float>>> embs_compensate;
    static map<int,map<int,vector<float>>> G_compensate;
    // 用来记录每轮迭代的神经网络权重
    static map<int,vector<vector<float>>> weights;
    static int layer_num;
    static vector<bool> compFlag;

    // 存储各种G，假设两层，反向传播最后一层是G2，倒数第二层是G1
    static map<int,map<int,vector<float>>> G_map;

    static vector<vector<uint>> bucketPositionBitMap;

    static float comp_percent;
    static double embs_max;
    static double embs_min;




    static void set_nodes(const NodeMessage *nodeMessage);
    static void set_nodes_for_each_worker(const DataMessage *reply);

    static void set_features(const DataMessage_FeatureMessage *featureMessage);

    static void set_labels(const DataMessage_LabelMessage *labelMessage);

    static void set_adjs(const DataMessage_AdjMessage *adjMessage);

    static void getData(DataMessage *reply);




};


#endif //DGNN_TEST_WORKERSTORE_H
