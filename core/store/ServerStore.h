//
// Created by songzhen on 2020/10/15.
//

#ifndef DGNN_TEST_SERVERSTORE_H
#define DGNN_TEST_SERVERSTORE_H
#include <iostream>
#include <vector>
#include <map>
#include "../util/check.h"
#include "../compress/compress.h"
using namespace std;
class ServerStore {
public:
    // 神经网络每层参数
    static int feat_dim;
    static int worker_num;
    static int server_num;
    static vector<int> hid_dims;
    static int class_dim;
    static map<string,vector<float>> params;
    static map<string,vector<float>> grads_agg;
    static map<string,vector<float>> m_grads_t;
    static map<string,vector<float>> v_grads_t;



    static map<int,vector<vector<float>>> weights;
    static map<int,vector<float>> bias;
    // 用完记得清空
    static map<int,vector<vector<float>>> weights_grad_agg;
    static map<int,vector<float>> bias_grad_agg;

    // adam更新的m_t,v_t,t
    static map<int,vector<vector<float>>> m_weight_t;
    static map<int,vector<vector<float>>> v_weight_t;
    static map<int,vector<float>> m_bias_t;
    static map<int,vector<float>> v_bias_t;

    static int t;
    static int serverId;
    static float val_accuracy;
    static float train_accuracy;
    static float test_accuracy;
    static float val_f1_accuracy;
    static float test_f1_accuracy;
    static vector<int> train_nodes;
    static vector<int> val_nodes;
    static vector<int> test_nodes;
    static void initParams(const int &workerN,const int &serverN,const int & feat_dim,const int & class_dim,const vector<int> &hid_dims);
};


#endif //DGNN_TEST_SERVERSTORE_H
