//
// Created by songzhen on 2020/10/13.
//

#ifndef DGNN_TEST_DGNN_SERVER_H
#define DGNN_TEST_DGNN_SERVER_H

#include <iostream>
#include <grpcpp/grpcpp.h>
#include<grpcpp/health_check_service_interface.h>
#include<grpcpp/ext/proto_server_reflection_plugin.h>
#include "core/store/WorkerStore.h"
#include "cmake/build/dgnn_test.grpc.pb.h"
#include "cmake/build/dgnn_test.pb.h"
#include "core/partition/RandomPartitioner.h"
#include "core/partition/GeneralPartition.h"
#include "core/util/threadUtil.h"
#include "core/store/ServerStore.h"
#include <cmath>
#include <unistd.h>
#include "core/compress/compress.h"
#include <math.h>
#include <time.h>
#include <google/protobuf/repeated_field.h>
using grpc::Server;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerBuilder;


// here is proto defining package name. and the classes belong to this package
using dgnn_test::DgnnProtoService;
using dgnn_test::intM;
using dgnn_test::DataMessage;
using dgnn_test::BoolMessage;
using dgnn_test::ContextMessage;
using dgnn_test::PartitionMessage;
using dgnn_test::NetInfoMessage;
using dgnn_test::IntMessage;
using dgnn_test::WeightsAndBiasMessage;
using dgnn_test::TensorMessage;
using dgnn_test::ReqEmbMessage;
using dgnn_test::GradientMessage;
using dgnn_test::TestVMessage;
using dgnn_test::EmbMessage;
using dgnn_test::IntTensorMessage;
using dgnn_test::BitArrayMessage;
using dgnn_test::LargeMessage;
using dgnn_test::SmallMessage;
using dgnn_test::RespEmbSparseMessage;
using dgnn_test::ByteTensorMessage;
using dgnn_test::ChangeRateMessage;
using dgnn_test::AccuracyMessage;


class ServiceImpl final:public DgnnProtoService::Service{
public:
    Status add1(ServerContext* context,const intM* request,
                intM* reply) override;
    Status sendDataToEachWorker(
            ServerContext* context,const DataMessage* request,
            BoolMessage* reply) override;
    Status pullDataFromServer(
            ServerContext* context,const intM* request,
            DataMessage* reply) override;
    Status pullDataFromMaster(
            ServerContext* context,const ContextMessage* request,
            DataMessage* reply) override;
    Status pullDataFromMasterGeneral(
            ServerContext* context,const ContextMessage* request,
            DataMessage* reply) override;
    Status initParameter(
            ServerContext* context,const NetInfoMessage* request,
            BoolMessage* reply) override;
    Status pullWeights(
            ServerContext* context,const IntMessage* request,
            WeightsAndBiasMessage* reply) override;
    Status pullBias(
            ServerContext* context,const IntMessage* request,
            WeightsAndBiasMessage* reply) override;
    Status barrier(
            ServerContext* context,const BoolMessage* request,
            BoolMessage* reply) override;
    Status workerPullEmb(
            ServerContext* context,const EmbMessage* request,
            EmbMessage* reply) override;

    Status workerPullG(
            ServerContext* context,const EmbMessage* request,
            EmbMessage* reply) override;

    Status workerPullGCompress(ServerContext* context,const EmbMessage* request,
                               EmbMessage* reply) override;

    Status Server_SendAndUpdateModels(
            ServerContext* context,const GradientMessage* request,
            BoolMessage* reply) override;

    Status TestVariant(
            ServerContext* context,const TestVMessage* request,
            BoolMessage* reply) override;

    Status workerPullEmbCompress(
            ServerContext* context,const EmbMessage* request,
            EmbMessage* reply) override;
    Status Test1Bit(ServerContext* context,const BitArrayMessage* request,
                    BitArrayMessage* reply) override;

    Status testLargeSize(ServerContext* context,const LargeMessage* request,
                         LargeMessage* reply) override;
    Status testSmallSize(ServerContext* context,const SmallMessage* request,
                         SmallMessage* reply) override;
//    Status workerPullEmbCompress_iter(ServerContext *context, const ReqEmbSparseMessage *request,
//                                                   ReqEmbSparseMessage *reply) override;
//    Status workerPullEmbTrend(ServerContext *context, const EmbMessage *request,
//                              EmbMessage *reply) override;

    Status workerPullEmbTrendSelect(ServerContext *context, const EmbMessage *request,
                              EmbMessage *reply) override;

    Status sendAccuracy(ServerContext *context,const AccuracyMessage *request,
                        AccuracyMessage *reply) override;
    Status freeMaster(ServerContext *context,const BoolMessage *request,BoolMessage *reply) override;

    //    static void* RunServer(void* address_tmp);
    static void RunServerByPy(const string& address,int serverId);

    //Yu
    Status workerSendTrainNode(ServerContext *context, const NodeMessage *request, BoolMessage *reply) override;
    Status serverSendTrainNode(ServerContext *context, const ContextMessage *request, NodeMessage *reply) override;

    Status workerSendValNode(ServerContext *context, const NodeMessage *request, BoolMessage *reply) override;
    Status serverSendValNode(ServerContext *context, const ContextMessage *request, NodeMessage *reply) override;

    Status workerSendTestNode(ServerContext *context, const NodeMessage *request, BoolMessage *reply) override;
    Status serverSendTestNode(ServerContext *context, const ContextMessage *request, NodeMessage *reply) override;
};




#endif //DGNN_TEST_DGNN_SERVER_H







