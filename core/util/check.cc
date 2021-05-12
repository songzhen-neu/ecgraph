//
// Created by songzhen on 2020/10/8.
//

#include "check.h"


void Check::check_features(map<int, vector<float>> &features) {
    cout << "server: feature size:" << features.size() << endl;
    cout << "server: feature dims:" << features.begin()->second.size() << endl;
}

void Check::check_labels(map<int, int> &labels) {
    cout << "server: label size:" << labels.size() << endl;
}

void Check::check_adjs(map<int, set<int>> &adjs) {
    cout << "server: adjs size:" << adjs.size() << endl;
}

void Check::check_partition_pass(
        const int &workerNum, const int &dataNum, const string &dataPath, const int &feature_dim,
        const int &class_num) {
    cout << "worker number:" << workerNum << endl;
    cout << "dataNum:" << dataNum << endl;
    cout << "dataPath:" << dataPath << endl;
    cout << "feature_dim:" << feature_dim << endl;
    cout << "class_num:" << class_num << endl;
}

void Check::check_initParameter_ServerStore() {
    // 输出初始化的weights和bias，看层数和维度是否正确
    int weights_num=ServerStore::weights.size();
    int bias_num=ServerStore::bias.size();
    cout<< "weight layer number:"<< weights_num<<", bias layer number:" <<bias_num<<endl;
    vector<int> weights_size;
    for(const auto& weight:ServerStore::weights){
        int weight_size_x=weight.second.size();
        int weight_size_y=weight.second[0].size();
        cout<< "weight layer "<<weight.first<<" size: " << weight_size_x <<"*"<<weight_size_y<<endl;
    }

    for(const auto& bias:ServerStore::bias){
        int bias_size_x=bias.second.size();
        cout<< "bias layer "<< bias.first<< " size: " <<bias_size_x <<endl;
    }
}