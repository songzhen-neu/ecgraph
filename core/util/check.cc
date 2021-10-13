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
    cout << "*******check initParameter start***********" << endl;
    map<string, vector<float>>::iterator it;
    for (it = ServerStore::params.begin(); it != ServerStore::params.end(); it++) {
        cout<<"layer "<<it->first<<" : "<<it->second.size()<<endl;
    }
    cout << "*******check initParameter end***********" << endl;
}