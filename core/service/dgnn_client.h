//
// Created by songzhen on 2020/10/15.
//

#ifndef DGNN_TEST_DGNN_CLIENT_H
#define DGNN_TEST_DGNN_CLIENT_H
//
// Created by songzhen on 2020/9/22.
//



#include <iostream>
#include <grpcpp/grpcpp.h>
//#include "../../cmake/build/dgnn_test.grpc.pb.h"

#include <vector>
#include "../store/WorkerStore.h"
#include <pthread.h>
#include "../../dgnn_server.h"
#include "../../cmake/build/dgnn_test.grpc.pb.h"
#include "../../cmake/build/dgnn_test.pb.h"
#include <time.h>
#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"
namespace py=pybind11;

using namespace std;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;

using dgnn_test::intM;
using dgnn_test::DgnnProtoService;
using dgnn_test::DataMessage;
using dgnn_test::BoolMessage;
using dgnn_test::NodeMessage;
using dgnn_test::ContextMessage;
using dgnn_test::PartitionMessage;
using dgnn_test::NetInfoMessage;
using dgnn_test::ReqEmbMessage;
using dgnn_test::GradientMessage;
using dgnn_test::TestVMessage;
using dgnn_test::BitArrayMessage;
using dgnn_test::EmbMessage;
using dgnn_test::IntTensorMessage;
using dgnn_test::LargeMessage;
using dgnn_test::SmallMessage;
using dgnn_test::RespEmbSparseMessage;
using dgnn_test::ByteTensorMessage;
using dgnn_test::TensorMessage;
using dgnn_test::StringM;
using dgnn_test::Param;
using dgnn_test::GradMessage;




class DGNNClient {
//private:

public:
    std::unique_ptr<DgnnProtoService::Stub> stub_;
    ServiceImpl serverImpl;
    string serverAddress;
    explicit DGNNClient(std::shared_ptr<Channel> channel);

    DGNNClient() ;

    void init(std::shared_ptr<Channel> channel);

    static void *RunServer(void* address_tmp);

    void initParameter(const int &worker_num,const int &server_num,const int &feat_dim,
                       const vector<int> &hid_dims,const int &class_dim,const int &wid,
                       map<string, vector<float>> &weights);

    void startClientServer();

    void freeSpace();

    void init_by_address(std::string address) ;

    void set_testString();

    string get_testString();

    void freeMaster();

    vector<int> get_nodes();

    void set_nodes();

    map<int, vector<float>> get_features() ;

    void set_features() ;

    map<int, int> get_labels() ;

    void set_labels() ;

    map<int, set<int>> get_adjs() ;

    string get_serverAddress();

    void set_nodesForEachWorker();
    vector<vector<int>> get_nodesForEachWorker();

    void set_serverAddress(string serverAddress);

    void set_adjs() ;

    void set_layerNum(int layerNum);
    void get_layerNum();


    int add1() ;

    int add(int a, int b);

    // vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
    void sendDataToEachWorker(
            const vector<int> &nodes, const map<int, vector<float>> &features,
            const map<int, int> &labels, const map<int, set<int>> &adjs) ;

    // client解析读取的data消息
    void pullDataFromServer();
    void pullDataFromMaster( int worker_id,int worker_num,int data_num,
                             string data_path,int feature_dim,int class_num);

    void pullDataFromMasterGeneral( int worker_id,int worker_num,int data_num,
                             const string& data_path,int feature_dim,int class_num,const string& partitionMethod,int edgeNum);


    vector<vector<float>> server_PullWeights(int layer_id);
    vector<float> server_PullBias(int layer_id);
    void server_Barrier(int layer_id);

//    map<int,vector<float>> worker_pull_needed_emb(const set<int> &needed_emb_set);
//    map<int,vector<float>> worker_pull_needed_emb(py::array_t<int>& nodes);
    py::array_t<float> worker_pull_needed_emb(py::array_t<int>& nodes,int epoch,int layerId,int workerId,int serverId);



    static void* worker_pull_needed_emb_parallel(void* metaData);
    static void* worker_pull_needed_emb_parallel_fb(void* metaData);
    py::array_t<float> worker_pull_emb_compress(
            py::array_t<int>& needed_emb_set,bool ifcompensate,int layerId,int epoch,
            const string& compensateMethod,int bucketNum,int changeToIter,int workerId,int layerNum,int bitNum);

    static void* worker_pull_emb_compress_parallel(void* metaData_void);

    py::array_t<float> getChangeRate(int workerId,int layerId);

//    py::array_t<float> worker_pull_emb_trend(
//            py::array_t<int>& needed_emb_set,int layerId,int epoch,
//            int bucketNum,int workerId,int serverId,int layerNum,int trend,int bitNum);

    void initCompressBitMap(int bitNum);

    void worker_setEmbs(const map<int,vector<float>> &embMap);

    void sendAndUpdateModels( int worker_id, int server_id,map<int,vector<vector<float>>> &weight_grads, map<int,vector<float>> &bia_grads,float lr);
    void server_updateModels( int worker_id, int server_id,float lr,const string& key, py::array_t<float>& grad);


    void testVariant();
    void test1Bit();
    void test_workerPullEmbCompress();
    static void deCompress(int shape0, int shape1, int shape_dim, int bitNum,
                    EmbMessage &reply, float *ptr_result);
    static void deCompressConcat(int shape0, int shape1, int shape_dim, int bitNum,
                           EmbMessage &reply, float *ptr_result);

    static void deCompressTotal( int bitNum,int localId,int localNodeSize,
                           vector<EmbMessage> &reply, float *ptr_result,map<int,int>& oldToNewMap,vector<vector<int>> &nodes);

    void setG(const map<int,vector<float>> &g,int id,double max_v,double min_v);
    py::array_t<float> worker_pull_needed_G(py::array_t<int>& needed_G_set,int layerId);
    py::array_t<float> worker_pull_needed_G_compress(py::array_t<int>& needed_G_set,
                                                     bool ifCompensate,int layerId,int epoch,int bitNum);

//    static void* worker_pull_emb_trend_parallel(void* metaData_void);
    static void* worker_pull_emb_trend_parallel_select(void* metaData_void);
//    py::array_t<float> worker_pull_needed_emb_compress_iter(py::array_t<int>& needed_G_set,bool ifCompensate,int layerId,int epoch,int bucketNum);
    static void* worker_pull_g_parallel(void* metaData_void);
    static void* worker_pull_g_compress_parallel(void* metaData_void);
    void set_embs(const map<int, vector<float>> &embMap);
    map<int,vector<float>> get_embs();

    map<string,float> sendAccuracy(float val_acc,float train_acc,float acc_test,float val_f1,float test_f1);

    void test_large();
    void test_small();

// Yu
    void sendTrainNode(int worker_id, py::array_t<int> list);
    py::array_t<int> pullTrainNode();

    void sendValNode(int worker_id, py::array_t<int> list);
    py::array_t<int> pullValNode();

    void sendTestNode(int worker_id, py::array_t<int> list);
    py::array_t<int> pullTestNode();

    py::array_t<float> server_PullParams(const string& param_id);

    py::array_t<float> server_aggGrad(int worker_id, int server_id,float lr,const string& key, py::array_t<float>& grad);

    void sendNode(int layid,py::array_t<int> list);
    py::array_t<int> pullNode(int layid);
//    void set_embs_byNumply()

    void setCtxForCpp(vector<vector<int>> &request_nodes, int worker_id, const map<int,int> &oldToNewMap, int worker_num,
                    bool iscompress,bool ischangerate,int bits,int local_node_size,int trend,const vector<int> &emb_nodes,int laynum,
                      bool iscompress_bp,bool ischangerate_bp,int bits_bp);
};

struct ReqEmbsMetaData{
    vector<int>* nodes{};
    int epoch{};
    int layerId{};
    int workerId{};
    int serverId{};
    int layerNum{};
    bool ifCompress{};
    int bitNum{};
    int trend{};
    int localNodeSize{};
    int feat_num{};
    int ifCompensate{};
    float* ptr_result{};
    map<int,int>* oldToNewMap{};
    EmbMessage* reply{};
    DGNNClient* dgnnClient{};



};


#endif //DGNN_TEST_DGNN_CLIENT_H
