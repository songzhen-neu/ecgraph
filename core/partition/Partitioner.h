//
// Created by songzhen on 2020/10/2.
//

#ifndef DGNN_TEST_PARTITIONER_H
#define DGNN_TEST_PARTITIONER_H

#include<iostream>
#include <vector>

using namespace std;

class Partitioner {

protected:
    int data_num;
    int worker_num;
    int feature_size;
    string filename;
    int label_size;
public:
    Partitioner();
    // avoid the case Partitioner(A) where it transforms 'A' to Partitioner
    Partitioner(int data_num, int worker_num, string filename,int feature_size,int label_size);
    virtual int startPartition(int worker_num,string partitionMethod,int nodeNum,int edgeNum)=0;
};


#endif //DGNN_TEST_PARTITIONER_H
