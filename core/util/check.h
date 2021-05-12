//
// Created by songzhen on 2020/10/15.
//

#ifndef DGNN_TEST_CHECK_H
#define DGNN_TEST_CHECK_H

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include "../store/ServerStore.h"

using namespace std;

class Check {
public:
    static void check_features(map<int, vector<float>> &features);

    static void check_labels(map<int, int> &labels);

    static void check_adjs(map<int, set<int>> &adjs);

    static void check_partition_pass(
            const int &workerNum, const int &dataNum,
            const string &dataPath, const int &feature_dim,
            const int &class_num);

    static void check_initParameter_ServerStore();
};

#endif //DGNN_TEST_CHECK_H
