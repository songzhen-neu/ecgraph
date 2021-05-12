//
// Created by songzhen on 2020/10/2.
//

#include "Partitioner.h"
Partitioner::Partitioner() {}
Partitioner::Partitioner(int data_num,int worker_num,string filename,int feature_size,int label_size) {
    this->data_num=data_num;
    this->worker_num=worker_num;
    this->filename=filename;
    this->feature_size=feature_size;
    this->label_size=label_size;
}


//int Partitioner::startPartition(DGNNClient clients[])  {
//
//}


