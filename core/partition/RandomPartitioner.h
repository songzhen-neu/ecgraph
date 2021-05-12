//
// Created by songzhen on 2020/10/3.
//

#ifndef DGNN_TEST_RANDOMPARTITIONER_H
#define DGNN_TEST_RANDOMPARTITIONER_H

#include "Partitioner.h"
#include <map>
#include<set>
#include<fstream>
#include<vector>
#include<iostream>
#include<string>
#include<sstream>
#include "../util/string_split.h"
#include "../structure/Array.cc"
#include "../util/Length.cc"



class RandomPartitioner :public Partitioner{
    // hash 划分
public:
    RandomPartitioner();
    void init (const int & data_num, const int & worker_num,const string &filename,const int & feature_size,const int & label_size);
    int startPartition(int worker_num,string partitionMethod,int nodeNum,int edgeNum) override;
    // vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
    vector<int> nodes[13];
    map<int,vector<float>>  features[13];
    map<int,int> labels[13];
    map<int,set<int>> adjs[13];



};

#endif //DGNN_TEST_RANDOMPARTITIONER_H
