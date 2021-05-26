//
// Created by songzhen on 2020/9/22.
//



#include "dgnn_client.h"


DGNNClient::DGNNClient(std::shared_ptr<Channel> channel) : stub_(DgnnProtoService::NewStub(channel)) {}

DGNNClient::DGNNClient() = default;

void DGNNClient::init(std::shared_ptr<Channel> channel) {
    stub_ = (DgnnProtoService::NewStub(channel));
}

void DGNNClient::initCompressBitMap(int bitNum) {
    int oneIntDimNum = 32 / bitNum;
    int bucketNum = pow(2, bitNum);
    // compress bucket num in each bit position
    cout << "bucket position map:" << endl;
    for (uint i = 0; i < bucketNum; i++) {
        vector<uint> bidOfEachPosition(oneIntDimNum);
        for (int j = 0; j < oneIntDimNum; j++) {
            bidOfEachPosition[j] = (i << (32 - (j + 1) * 2));
            cout << bidOfEachPosition[j] << ",";
        }
        cout << endl;
        WorkerStore::bucketPositionBitMap.push_back(bidOfEachPosition);
    }
    cout << endl;

}

void *DGNNClient::RunServer(void *address_tmp) {
//    std::string server_address("192.168.184.136:2001");
    string address = *((string *) address_tmp);
//        string address = "192.168.184.138:3001";
//        cout<<"address_tmp:"<<address_tmp<<endl;
    ServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(2147483647);
    builder.SetMaxSendMessageSize(2147483647);
    builder.SetMaxMessageSize(2147483647);


    std::unique_ptr<Server> server(builder.BuildAndStart());


    std::cout << "Server Listening on" << address << std::endl;
    server->Wait();
//    cout << "aaaa" << endl;
}


//    static void* hello(void *str){
//        cout<<"hello"<<endl;
//    }

void DGNNClient::startClientServer() {

    pthread_t serverThread;
    pthread_create(&serverThread, NULL, DGNNClient::RunServer, (void *) &this->serverAddress);
//        ServiceImpl::RunServerByPy(address);

}

void DGNNClient::deCompress(int shape0, int shape1, int shape_dim, int bitNum,
                            EmbMessage &reply, float *ptr_result) {

    vector<float> bucket;
    for (auto value:reply.values()) {
        bucket.push_back(value);
    }
    if (bitNum == 2) {
        for (int i = 0; i < shape0; i++) {
            const IntTensorMessage &tm = reply.embs(i);
            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                if (j * 16 + 15 < shape_dim) {
                    uint dim = tm.tensor(j);
                    for (int l = 0; l < 16; l++) {
                        int a = dim >> (30 - l * 2) & 0x00000003;
                        ptr_result[i * shape_dim + j * 16 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 16;
                    uint dim = tm.tensor(j);
                    for (int k = 0; k < num; k++) {
                        int a = dim >> (30 - k * 2) & 0x00000003;
                        ptr_result[i * shape_dim + j * 16 + k] = bucket[a];
                    }

                }
            }
        }
    } else if (bitNum == 4) {
        for (int i = 0; i < shape0; i++) {
            const IntTensorMessage &tm = reply.embs(i);
            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                if (j * 8 + 7 < shape_dim) {
                    uint dim = tm.tensor(j);
                    for (int l = 0; l < 8; l++) {
                        int a = dim >> (28 - l * 4) & 0x0000000f;
                        ptr_result[i * shape_dim + j * 8 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 8;
                    uint dim = tm.tensor(j);
                    for (int k = 0; k < num; k++) {
                        int a = dim >> (28 - k * 4) & 0x0000000f;
                        ptr_result[i * shape_dim + j * 8 + k] = bucket[a];
                    }

                }
            }
        }
    } else if (bitNum == 8) {
        for (int i = 0; i < shape0; i++) {
            const IntTensorMessage &tm = reply.embs(i);
            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                if (j * 4 + 3 < shape_dim) {
                    uint dim = tm.tensor(j);
                    for (int l = 0; l < 4; l++) {
                        int a = dim >> (24 - l * 8) & 0x000000ff;
                        ptr_result[i * shape_dim + j * 4 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 4;
                    uint dim = tm.tensor(j);
                    for (int k = 0; k < num; k++) {
                        int a = dim >> (24 - k * 8) & 0x000000ff;
                        ptr_result[i * shape_dim + j * 4 + k] = bucket[a];
                    }

                }
            }
        }
    } else if (bitNum == 16) {
        for (int i = 0; i < shape0; i++) {
            const IntTensorMessage &tm = reply.embs(i);
            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                if (j * 2 + 1 < shape_dim) {
                    uint dim = tm.tensor(j);
                    for (int l = 0; l < 2; l++) {
                        int a = dim >> (16 - l * 16) & 0x0000ffff;
                        ptr_result[i * shape_dim + j * 2 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 2;
                    uint dim = tm.tensor(j);
                    for (int k = 0; k < num; k++) {
                        int a = dim >> (16 - k * 16) & 0x0000ffff;
                        ptr_result[i * shape_dim + j * 2 + k] = bucket[a];
                    }

                }
            }
        }
    }


}

void DGNNClient::deCompressConcat(int shape0, int shape1, int shape_dim, int bitNum,
                                  EmbMessage &reply, float *ptr_result) {

    vector<float> bucket;
    for (auto value:reply.values()) {
        bucket.push_back(value);
    }
    auto &reply_concat = reply.resp_compress_emb_concat();

    if (bitNum == 2) {
        for (int i = 0; i < shape0; i++) {
            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                uint dim = reply_concat.Get(i * shape1 + j);
                if (j * 16 + 15 < shape_dim) {
                    for (int l = 0; l < 16; l++) {
                        int a = dim >> (30 - l * 2) & 0x00000003;
                        ptr_result[i * shape_dim + j * 16 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 16;
                    for (int k = 0; k < num; k++) {
                        int a = dim >> (30 - k * 2) & 0x00000003;
                        ptr_result[i * shape_dim + j * 16 + k] = bucket[a];
                    }

                }
            }
        }
    } else if (bitNum == 4) {
        for (int i = 0; i < shape0; i++) {
            for (int j = 0; j < shape1; j++) {
                uint dim = reply_concat.Get(i * shape1 + j);
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                if (j * 8 + 7 < shape_dim) {

                    for (int l = 0; l < 8; l++) {
                        int a = dim >> (28 - l * 4) & 0x0000000f;
                        ptr_result[i * shape_dim + j * 8 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 8;

                    for (int k = 0; k < num; k++) {
                        int a = dim >> (28 - k * 4) & 0x0000000f;
                        ptr_result[i * shape_dim + j * 8 + k] = bucket[a];
                    }

                }
            }
        }
    } else if (bitNum == 8) {
        for (int i = 0; i < shape0; i++) {

            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
                uint dim = reply_concat.Get(i * shape1 + j);
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                if (j * 4 + 3 < shape_dim) {

                    for (int l = 0; l < 4; l++) {
                        int a = dim >> (24 - l * 8) & 0x000000ff;
                        ptr_result[i * shape_dim + j * 4 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 4;

                    for (int k = 0; k < num; k++) {
                        int a = dim >> (24 - k * 8) & 0x000000ff;
                        ptr_result[i * shape_dim + j * 4 + k] = bucket[a];
                    }

                }
            }
        }
    } else if (bitNum == 16) {
        for (int i = 0; i < shape0; i++) {

            for (int j = 0; j < shape1; j++) {
                // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                uint dim = reply_concat.Get(i * shape1 + j);
                if (j * 2 + 1 < shape_dim) {

                    for (int l = 0; l < 2; l++) {
                        int a = dim >> (16 - l * 16) & 0x0000ffff;
                        ptr_result[i * shape_dim + j * 2 + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * 2;

                    for (int k = 0; k < num; k++) {
                        int a = dim >> (16 - k * 16) & 0x0000ffff;
                        ptr_result[i * shape_dim + j * 2 + k] = bucket[a];
                    }

                }
            }
        }
    }


}


void DGNNClient::deCompressTotal(int bitNum, int localId, int localNodeSize,
                                 vector<EmbMessage> &replyVec, float *ptr_result, map<int, int> &oldToNewMap,
                                 vector<vector<int>> &nodes) {
    for (int w = 0; w < replyVec.size(); w++) {
        if (w != localId) {
            auto &node_worker = nodes[w];
            vector<float> bucket;
            EmbMessage &reply = replyVec[w];

            int shape0 = 0;
            int shape1 = 0;
            int shape_dim = reply.shapedim();
            if (reply.embs_size() != 0) {
                shape0 = reply.embs_size();
                shape1 = reply.embs().begin()->tensor_size();
            } else if (reply.denseembmessage().embs_size() != 0) {
                shape0 = reply.denseembmessage().embs_size();
                shape1 = reply.denseembmessage().embs().begin()->tensor_size();
            } else {
                cout << "shape0 == 0, pull embeddings error!!!!" << endl;
                exit(1);
            }

            for (auto value:reply.values()) {
                bucket.push_back(value);
            }


            if (bitNum == 2) {
                for (int i = 0; i < shape0; i++) {
                    int nid = node_worker[i];
                    int new_id = oldToNewMap[nid] - localNodeSize;
                    const IntTensorMessage &tm = reply.embs(i);
                    auto &node_worker = nodes[i];
                    for (int j = 0; j < shape1; j++) {
                        // transform to 4 int data_raw
                        if (j * 16 + 15 < shape_dim) {
                            uint dim = tm.tensor(j);
                            for (int l = 0; l < 16; l++) {
                                int a = dim >> (30 - l * 2) & 0x00000003;
                                ptr_result[new_id * shape_dim + j * 16 + l] = bucket[a];
                            }
                        } else {
                            int num = shape_dim - j * 16;
                            uint dim = tm.tensor(j);
                            for (int k = 0; k < num; k++) {
                                int a = dim >> (30 - k * 2) & 0x00000003;
                                ptr_result[new_id * shape_dim + j * 16 + k] = bucket[a];
                            }

                        }
                    }
                }
            } else if (bitNum == 4) {
                for (int i = 0; i < shape0; i++) {
                    int nid = node_worker[i];
                    int new_id = oldToNewMap[nid] - localNodeSize;
                    const IntTensorMessage &tm = reply.embs(i);
                    for (int j = 0; j < shape1; j++) {
                        // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                        if (j * 8 + 7 < shape_dim) {
                            uint dim = tm.tensor(j);
                            for (int l = 0; l < 8; l++) {
                                int a = dim >> (28 - l * 4) & 0x0000000f;
                                ptr_result[new_id * shape_dim + j * 8 + l] = bucket[a];
                            }
                        } else {
                            int num = shape_dim - j * 8;
                            uint dim = tm.tensor(j);
                            for (int k = 0; k < num; k++) {
                                int a = dim >> (28 - k * 4) & 0x0000000f;
                                ptr_result[new_id * shape_dim + j * 8 + k] = bucket[a];
                            }

                        }
                    }
                }
            } else if (bitNum == 8) {
                for (int i = 0; i < shape0; i++) {
                    int nid = node_worker[i];
                    int new_id = oldToNewMap[nid] - localNodeSize;
                    const IntTensorMessage &tm = reply.embs(i);
                    for (int j = 0; j < shape1; j++) {
                        // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                        if (j * 4 + 3 < shape_dim) {
                            uint dim = tm.tensor(j);
                            for (int l = 0; l < 4; l++) {
                                int a = dim >> (24 - l * 8) & 0x000000ff;
                                ptr_result[new_id * shape_dim + j * 4 + l] = bucket[a];
                            }
                        } else {
                            int num = shape_dim - j * 4;
                            uint dim = tm.tensor(j);
                            for (int k = 0; k < num; k++) {
                                int a = dim >> (24 - k * 8) & 0x000000ff;
                                ptr_result[new_id * shape_dim + j * 4 + k] = bucket[a];
                            }

                        }
                    }
                }
            } else if (bitNum == 16) {
                for (int i = 0; i < shape0; i++) {
                    int nid = node_worker[i];
                    int new_id = oldToNewMap[nid] - localNodeSize;
                    const IntTensorMessage &tm = reply.embs(i);
                    for (int j = 0; j < shape1; j++) {
                        // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                        if (j * 2 + 1 < shape_dim) {
                            uint dim = tm.tensor(j);
                            for (int l = 0; l < 2; l++) {
                                int a = dim >> (16 - l * 16) & 0x0000ffff;
                                ptr_result[new_id * shape_dim + j * 2 + l] = bucket[a];
                            }
                        } else {
                            int num = shape_dim - j * 2;
                            uint dim = tm.tensor(j);
                            for (int k = 0; k < num; k++) {
                                int a = dim >> (16 - k * 16) & 0x0000ffff;
                                ptr_result[new_id * shape_dim + j * 2 + k] = bucket[a];
                            }

                        }
                    }
                }
            }
        }
    }


}

void DGNNClient::init_by_address(std::string address) {
    grpc::ChannelArguments channel_args;
    channel_args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 2147483647);
    std::shared_ptr<Channel> channel = (grpc::CreateCustomChannel(
            address, grpc::InsecureChannelCredentials(), channel_args));
    WorkerStore::testString = address;
    stub_ = DgnnProtoService::NewStub(channel);
}

void DGNNClient::set_testString() {};

void
DGNNClient::initParameter(const int &worker_num, const int &server_num, const int &feat_dim,
                          const vector<int> &hid_dims, const int &class_dim,
                          const int &wid, map<int, vector<vector<float>>> &weights,
                          map<int, vector<vector<float>>> &bias) {
    // init workerstore some object
    for (int i = 0; i < worker_num; i++) {
        WorkerStore::compFlag.push_back(false);
    }
    // 发送到server中
    NetInfoMessage request;
    request.set_featuredim(feat_dim);
    request.set_classdim(class_dim);
    request.set_servernum(server_num);
    for (auto hid_dim:hid_dims) {
        request.add_hiddendim(hid_dim);
    }
    request.set_workernum(worker_num);
    request.set_wid(wid);
    for (int i = 0; i < weights.size(); i++) {
        auto *weight = request.add_weights();
        auto *bia = request.add_bias();
        bia->mutable_tensor()->Add(bias[i][0].begin(), bias[i][0].end());
        for (int j = 0; j < weights[i].size(); j++) {
//            auto* tensor=weight->add_weight();
            weight->add_weight()->mutable_tensor()->Add(weights[i][j].begin(), weights[i][j].end());
//            for(int k=0;k<weights[i][j].size();k++){
//                tensor->add_tensor(weights[i][j][k]);
//            }
        }
    }
//    for(int i=0;i<request.weights_size();i++){
//        cout<<"initParameter weights size:"<<request.weights().size()
//            <<"*"<<request.weights(i).weight_size()<<"*"<<request.weights(i).weight().begin()->tensor_size()<<endl;
//        cout<<"initParameter bias size:"<<request.bias_size()
//            <<"*"<<request.bias(i).tensor_size()<<endl;
//    }



    ClientContext context;
    BoolMessage reply;
    stub_->initParameter(&context, request, &reply);
}

string DGNNClient::get_testString() {
    return WorkerStore::testString;
}

//    void startServer(std::string address){
//        pthread_t serverThread;
//        pthread_create(&serverThread, NULL, clientServerRun, (void *)&address);
//    }

vector<int> DGNNClient::get_nodes() {
    return WorkerStore::nodes;
}

void DGNNClient::set_nodes() {};

map<int, vector<float>> DGNNClient::get_features() {
    return WorkerStore::features;
}

void DGNNClient::set_nodesForEachWorker() {};

vector<vector<int>> DGNNClient::get_nodesForEachWorker() {
    return WorkerStore::nodesForEachWorker;
}

void DGNNClient::set_features() {};

map<int, int> DGNNClient::get_labels() {
    return WorkerStore::labels;
}

void DGNNClient::set_labels() {}

map<int, set<int>> DGNNClient::get_adjs() {
    return WorkerStore::adjs;
}

string DGNNClient::get_serverAddress() {
    return this->serverAddress;
}

void DGNNClient::set_serverAddress(string serverAddress) {
    cout << serverAddress << endl;
    this->serverAddress = serverAddress;
}

void DGNNClient::set_adjs() {};

void DGNNClient::set_layerNum(int layerNum) {
    WorkerStore::layer_num = layerNum;
//    cout<<"layer_num"<<layerNum<<endl;
}


void DGNNClient::get_layerNum() {}

int DGNNClient::add1() {
    intM request;
    request.set_value(105);
    intM reply;
    ClientContext context;

    Status status = stub_->add1(&context, request, &reply);

    if (status.ok()) {
//            cout<<reply.value()<<endl;
        return reply.value();
    } else {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
        return -1;
    }
}

int DGNNClient::add(int a, int b) {
    return a + b;
}

// vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
void DGNNClient::sendDataToEachWorker(
        const vector<int> &nodes, const map<int, vector<float>> &features,
        const map<int, int> &labels, const map<int, set<int>> &adjs) {
    DataMessage dataMessage;
//        int cur=0;
    // 将各类数据转化成protobuf类型
    NodeMessage *nodeMessage = dataMessage.nodelist().New();
    for (auto node:nodes) {
        nodeMessage->add_nodes(node);
    }

    dataMessage.set_allocated_nodelist(nodeMessage);


    DataMessage::FeatureMessage *featureMessage = dataMessage.featurelist().New();
    for (const auto &feature : features) {
        DataMessage::FeatureMessage::FeatureItem *featureItem = featureMessage->add_features();
        int vid = feature.first;
        vector<float> vec_temp = feature.second;
        featureItem->set_vid(vid);
        for (float i:vec_temp) {
            featureItem->add_feature(i);
        }
    }


    dataMessage.set_allocated_featurelist(featureMessage);


    DataMessage::LabelMessage *labelMessage = dataMessage.labellist().New();
    for (auto label : labels) {
        DataMessage::LabelMessage::LabelItem *labelItem = labelMessage->add_labels();
        labelItem->set_vid(label.first);
        labelItem->set_label(label.second);

    }
    dataMessage.set_allocated_labellist(labelMessage);
//
    DataMessage::AdjMessage *adjMessage = dataMessage.adjlist().New();
    for (const auto &adj : adjs) {
        int vid = adj.first;
        set<int> neibors = adj.second;
        DataMessage::AdjMessage::AdjItem *adjItem = adjMessage->add_adjs();
        adjItem->set_vid(vid);
        for (int i:neibors) {
            adjItem->add_neibors(i);
        }
    }
    dataMessage.set_allocated_adjlist(adjMessage);

    BoolMessage response;
    ClientContext context;
    Status status = stub_->sendDataToEachWorker(&context, dataMessage, &response);
    if (status.ok()) {
        cout << "sendDataToEachWorker completed" << endl;
    } else {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
    }


}

void DGNNClient::worker_setEmbs(const map<int, vector<float>> &embMap) {
    WorkerStore::embs.clear();
    WorkerStore::embs = embMap;
//    cout<<"worker_setEmbs check:"<<"size: "<<WorkerStore::embs.size()<<endl;
}

void DGNNClient::set_embs(const map<int, vector<float>> &embMap) {
    WorkerStore::embs = embMap;
}

// set_embs_byNumpy
//void DGNNClient::set_embs_byNumpy

map<int, vector<float>> DGNNClient::get_embs() {
    return WorkerStore::embs;
}


void DGNNClient::sendAndUpdateModels(int worker_id, int server_id, map<int, vector<vector<float>>> &weights,
                                     map<int, vector<float>> &bias, float lr) {
    // 发送收到的梯度到参数服务器中
    ClientContext context;
    GradientMessage request;
    BoolMessage reply;

    request.set_worker_id(worker_id);
    request.set_lr(lr);
    // 按层构建
    int layerNum = weights.size();
    for (int i = 0; i < layerNum; i++) {
        WeightsAndBiasMessage *weightsAndBiasMessage = request.add_grads();
        // 构建第i层weights
        for (const auto &row:weights[i]) {
            TensorMessage *tensor = weightsAndBiasMessage->add_weights();
            for (auto dim:row) {
                tensor->add_tensor(dim);
            }
        }
//        cout<<"send layer "<<i<<" to server"<<server_id<<" weight size:"<<request.grads(i).weights_size() <<"*"
//            <<request.grads(i).weights().begin()->tensor_size()<<endl;
        // 构建第i层bias
        if (server_id == 0) {
            TensorMessage *tensor = weightsAndBiasMessage->bias().New();
            for (const auto &bia:bias[i]) {
                tensor->add_tensor(bia);
            }
            weightsAndBiasMessage->set_allocated_bias(tensor);
//            cout<<"send layer "<<i<<" to server"<<server_id<<" bias size:"<<request.grads(i).bias().tensor_size()<<endl;
        }


    }

    Status status = stub_->Server_SendAndUpdateModels(&context, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "update parameters false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }


}


// client解析读取的data消息
void DGNNClient::pullDataFromServer() {
    ClientContext context;
    intM request;
    DataMessage response;
    request.set_value(0);
    Status status = stub_->pullDataFromServer(&context, request, &response);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "pull false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
//        cout<<response.nodelist().nodes(0);
    // 将response解析成pybind11需要的内容返回
    // 赋值workerContext

    WorkerStore::set_nodes(&response.nodelist());
    WorkerStore::set_adjs(&response.adjlist());
    WorkerStore::set_labels(&response.labellist());

//        for(int i=0;i<20;i++){
//            cout<<this->workerStore.labels[i]<<endl;
//        }
    WorkerStore::set_features(&response.featurelist());

}

// request master
void DGNNClient::pullDataFromMaster(
        int worker_id, int worker_num, int data_num,
        string data_path, int feature_dim, int class_num) {
    ContextMessage m;
    ClientContext context;
    DataMessage response;
    m.set_workerid(worker_id);
    m.set_workernum(worker_num);
    PartitionMessage *partitionMessage = m.partition().New();
    partitionMessage->set_workernum(worker_num);
    partitionMessage->set_datanum(data_num);
    partitionMessage->set_datapath(data_path);
    partitionMessage->set_classnum(class_num);
    partitionMessage->set_featuredim(feature_dim);
    m.set_allocated_partition(partitionMessage);

    Status status = stub_->pullDataFromMaster(&context, m, &response);
    if (status.ok()) {
        cout << "pullDataFromMaster completed" << endl;
    } else {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
    }
    // 将获取到的数据set到自己本地的store中
    cout << "***************local worker store****************" << endl;
    WorkerStore::set_adjs(&response.adjlist());
    WorkerStore::set_labels(&response.labellist());
    WorkerStore::set_features(&response.featurelist());
    WorkerStore::set_nodes(&response.nodelist());

}

void DGNNClient::freeMaster() {
    BoolMessage req;
    ClientContext context;
    BoolMessage reply;
    stub_->freeMaster(&context, req, &reply);
}

void DGNNClient::freeSpace() {
    map<int, vector<float>>().swap(WorkerStore::features);
    map<int, set<int>>().swap(WorkerStore::adjs);
}

void DGNNClient::pullDataFromMasterGeneral(
        int worker_id, int worker_num, int data_num,
        const string &data_path, int feature_dim, int class_num, const string &partitionMethod, int edgeNum) {
    ContextMessage m;
    ClientContext context;
    DataMessage reply;
    m.set_workerid(worker_id);
    m.set_workernum(worker_num);
    PartitionMessage *partitionMessage = m.partition().New();
    partitionMessage->set_workernum(worker_num);
    partitionMessage->set_datanum(data_num);
    partitionMessage->set_datapath(data_path);
    partitionMessage->set_classnum(class_num);
    partitionMessage->set_featuredim(feature_dim);
    partitionMessage->set_partitionmethod(partitionMethod);
    partitionMessage->set_edgenum(edgeNum);

    m.set_allocated_partition(partitionMessage);

    Status status = stub_->pullDataFromMasterGeneral(&context, m, &reply);
    if (status.ok()) {
        cout << "pullDataFromMaster completed" << endl;
    } else {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
    }
    // 将获取到的数据set到自己本地的store中
    cout << "***************local worker store****************" << endl;
    WorkerStore::set_adjs(&reply.adjlist());
    WorkerStore::set_labels(&reply.labellist());
    WorkerStore::set_features(&reply.featurelist());
    WorkerStore::set_nodes(&reply.nodelist());
    WorkerStore::set_nodes_for_each_worker(&reply);


}

py::array_t<float> DGNNClient::worker_pull_needed_G_compress(py::array_t<int> &needed_G_set, bool ifCompensate,
                                                             int layerId, int epoch, int bucketNum, int bitNum) {
    ClientContext context;
    EmbMessage request;
    EmbMessage reply;

    request.set_layerid(layerId);
    request.set_ifcompensate(ifCompensate);
    request.set_iterround(epoch);
    request.set_bucketnum(bucketNum);
    request.set_bitnum(bitNum);

    py::buffer_info nodes_buf = needed_G_set.request();
    if (nodes_buf.ndim != 1) {
        throw std::runtime_error("numpy.ndarray dims must be 1!");
    }
    int *ptr1 = (int *) nodes_buf.ptr;
    int nodeNum = nodes_buf.shape[0];
    request.mutable_nodes()->Reserve(nodeNum);
    for (int i = 0; i < nodeNum; i++) {
        request.mutable_nodes()->Add(ptr1[i]);
    }

    Status status = stub_->workerPullGCompress(&context, request, &reply);

    // 开始解析压缩后的G
    vector<float> bucket;
    for (auto value:reply.values()) {
        bucket.push_back(value);
    }

    int shape0 = reply.resp_node_size();
    int shape1 = reply.resp_featdim_size();
    int shape_dim = reply.shapedim();
    auto result = py::array_t<float>(shape0 * shape_dim);

//    cout<<"G result size:"<<shape0<<"*"<<shape1<<endl;
    result.resize({shape0, shape_dim});

    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;


    deCompressConcat(shape0, shape1, shape_dim, bitNum, reply, ptr_result);

    return result;
}


//py::array_t<float> DGNNClient::worker_pull_needed_emb_compress_iter(py::array_t<int> &needed_G_set, bool ifCompensate,
//                                                                    int layerId, int epoch, int bucketNum) {
//    ClientContext context;
//    EmbMessage request;
//    EmbMessage reply;
//
//    request.set_layerid(layerId);
//    request.set_ifcompensate(ifCompensate);
//    request.set_iterround(epoch);
//    request.set_bucketnum(bucketNum);
//
//    py::buffer_info nodes_buf = needed_G_set.request();
//    if (nodes_buf.ndim != 1) {
//        throw std::runtime_error("numpy.ndarray dims must be 1!");
//    }
//    int *ptr1 = (int *) nodes_buf.ptr;
//
//    for (int i = 0; i < nodes_buf.shape[0]; i++) {
//        request.add_nodes(ptr1[i]);
//    }
//
//    Status status = stub_->workerPullEmbCompress_iter(&context, request, &reply);
//
//    // 开始解析压缩后的G
//    vector<float> bucket;
//    for (auto value:reply.values()) {
//        bucket.push_back(value);
//    }
//
//    int shape0 = reply.embs_size();
//    int shape1 = reply.embs().begin()->tensor_size();
//    int shape_dim = reply.shapedim();
//    auto result = py::array_t<float>(shape0 * shape_dim);
//
////    cout<<"G result size:"<<shape0<<"*"<<shape1<<endl;
//    result.resize({shape0, shape_dim});
//
//    py::buffer_info buf_result = result.request();
//    float *ptr_result = (float *) buf_result.ptr;
//
//    for (int i = 0; i < shape0; i++) {
//        const IntTensorMessage &tm = reply.embs(i);
//        for (int j = 0; j < shape1; j++) {
//            // transform to 4 int data_raw
////                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
//            if (j * 4 + 3 < shape_dim) {
//                uint dim = tm.tensor(j);
//                int a_1 = dim >> 24;
//                int a_2 = dim >> 16 & 0x000000ff;
//                int a_3 = dim >> 8 & 0x000000ff;
//                int a_4 = dim & 0x000000ff;
//                ptr_result[i * shape_dim + j * 4] = bucket[a_1];
//                ptr_result[i * shape_dim + j * 4 + 1] = bucket[a_2];
//                ptr_result[i * shape_dim + j * 4 + 2] = bucket[a_3];
//                ptr_result[i * shape_dim + j * 4 + 3] = bucket[a_4];
//            } else {
//                int num = shape_dim - j * 4;
//                uint dim = tm.tensor(j);
//                vector<int> a;
//                a.push_back(dim >> 24);
//                a.push_back(dim >> 16 & 0x000000ff);
//                a.push_back(dim >> 8 & 0x000000ff);
//                a.push_back(dim & 0x000000ff);
//                for (int k = 0; k < num; k++) {
//                    ptr_result[i * shape_dim + j * 4 + k] = bucket[a[k]];
//                }
//
//            }
//        }
//    }
//
//    return result;
//}


py::array_t<float> DGNNClient::worker_pull_emb_compress(
        py::array_t<int> &needed_emb_set, bool ifCompensate, int layerId, int epoch,
        const string &compensateMethod, int bucketNum, int changeToIter, int workerId, int layerNum, int bitNum) {
    ClientContext context;
    EmbMessage request;
    EmbMessage reply;

    request.set_layerid(layerId);
    request.set_ifcompensate(ifCompensate);
    request.set_iterround(epoch);
    request.set_compensatemethod(compensateMethod);
    request.set_bucketnum(bucketNum);
    request.set_changetoiter(changeToIter);
    request.set_layernum(layerNum);
    request.set_bitnum(bitNum);

//    cout<<"workerId:"<<workerId<<endl;
    request.set_workerid(workerId);


    py::buffer_info nodes_buf = needed_emb_set.request();
//    cout<<"node size:"<<nodes.size()<<",buf1 size:"<<nodes_buf.size<<endl;
    if (nodes_buf.ndim != 1) {
        throw std::runtime_error("numpy.ndarray dims must be 1!");
    }
    int *ptr1 = (int *) nodes_buf.ptr;

    // 去服务器中获取嵌入,这里建立的是每个worker的channel

    // 构建request
    for (int i = 0; i < nodes_buf.shape[0]; i++) {
        request.add_nodes(ptr1[i]);
    }


//    clock_t start=clock();
    Status status = stub_->workerPullEmbCompress(&context, request, &reply);
//    clock_t end=clock();
//    cout<<"pull emb compress total time1111111111111111:"<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//    cout<<"byte size:"<<request.ByteSizeLong()<<endl;


    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "worker_pull_emb_compress false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }

//    start=clock();
    // de compress


    int shape0 = 0;
    int shape1 = 0;
    int shape_dim = reply.shapedim();
    if (reply.embs_size() != 0) {
        shape0 = reply.embs_size();
        shape1 = reply.embs().begin()->tensor_size();
    } else if (reply.denseembmessage().embs_size() != 0) {
        shape0 = reply.denseembmessage().embs_size();
        shape1 = reply.denseembmessage().embs().begin()->tensor_size();
    } else {
        cout << "shape0 == 0, pull embeddings error!!!!" << endl;
        exit(1);
    }


    auto result = py::array_t<float>(shape0 * shape_dim);
    result.resize({shape0, shape_dim});

    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;


    if (compensateMethod == "accorLayer" && ifCompensate) {

        if (layerId == WorkerStore::layer_num - 1) {
            for (int i = 0; i < shape0; i++) {
                const auto &tm = reply.denseembmessage().embs(i);
                for (int j = 0; j < shape1; j++) {
                    ptr_result[i * shape1 + j] = tm.tensor(j);
                }
            }
        } else {
            vector<float> bucket;
            for (auto value:reply.values()) {
                bucket.push_back(value);
            }

            for (int i = 0; i < shape0; i++) {
                const IntTensorMessage &tm = reply.embs(i);
                for (int j = 0; j < shape1; j++) {
                    // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
                    if (j * 4 + 3 < shape_dim) {
                        uint dim = tm.tensor(j);
                        int a_1 = dim >> 24;
                        int a_2 = dim >> 16 & 0x000000ff;
                        int a_3 = dim >> 8 & 0x000000ff;
                        int a_4 = dim & 0x000000ff;
                        ptr_result[i * shape_dim + j * 4] = bucket[a_1];
                        ptr_result[i * shape_dim + j * 4 + 1] = bucket[a_2];
                        ptr_result[i * shape_dim + j * 4 + 2] = bucket[a_3];
                        ptr_result[i * shape_dim + j * 4 + 3] = bucket[a_4];
                    } else {
                        int num = shape_dim - j * 4;
                        uint dim = tm.tensor(j);
                        vector<int> a;
                        a.push_back(dim >> 24);
                        a.push_back(dim >> 16 & 0x000000ff);
                        a.push_back(dim >> 8 & 0x000000ff);
                        a.push_back(dim & 0x000000ff);
                        for (int k = 0; k < num; k++) {
                            ptr_result[i * shape_dim + j * 4 + k] = bucket[a[k]];
                        }

                    }
                }
            }
//            cout<<"4"<<endl;
        }
    } else if (compensateMethod == "accorMix2" && ifCompensate) {
        if (layerId == WorkerStore::layer_num - 1) {
//        cout<<"************WorkerStore layer_num in worker"<<WorkerStore::layer_num<<endl;
//            cout<<"haha1"<<endl;
            for (int i = 0; i < shape0; i++) {
                const auto &tm = reply.denseembmessage().embs(i);
                for (int j = 0; j < shape1; j++) {
                    ptr_result[i * shape1 + j] = tm.tensor(j);
                }
            }

        } else {
//            cout<<"haha3"<<endl;
            vector<float> bucket;
            for (auto value:reply.values()) {
                bucket.push_back(value);
            }

            for (int i = 0; i < shape0; i++) {
                const IntTensorMessage &tm = reply.embs(i);
                for (int j = 0; j < shape1; j++) {
                    ptr_result[i * shape1 + j] = bucket[tm.tensor(j)];
                }
            }
        }
    } else {
        // 返回的是压缩的后的值

        deCompress(shape0, shape1, shape_dim, bitNum, reply, ptr_result);


    }
//    end=clock();
//    cout<<"bucket process time22222222222:"<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//    delete &request;
//    delete &reply;

    return result;

}

map<string, float>
DGNNClient::sendAccuracy(float val_acc, float train_acc, float test_acc, float val_f1, float test_f1) {
    ClientContext context;
    AccuracyMessage request;
    AccuracyMessage reply;

    request.set_val_acc(val_acc);
    request.set_train_acc(train_acc);
    request.set_test_acc(test_acc);
    request.set_val_f1(val_f1);
    request.set_test_f1(test_f1);


    Status status = stub_->sendAccuracy(&context, request, &reply);
    map<string, float> map_acc;
    map_acc.insert(pair<string, float>("val", reply.val_acc_entire()));
    map_acc.insert(pair<string, float>("train", reply.train_acc_entire()));
    map_acc.insert(pair<string, float>("test", reply.test_acc_entire()));
    map_acc.insert(pair<string, float>("val_f1", reply.val_f1_entire()));
    map_acc.insert(pair<string, float>("test_f1", reply.test_f1_entire()));

    return map_acc;
}

void *DGNNClient::worker_pull_emb_compress_parallel(void *metaData_void) {
    ClientContext context;
    EmbMessage request;

    auto metaData = (ReqEmbsMetaData *) metaData_void;
    vector<int> &nodes = *metaData->nodes;
    int epoch = metaData->epoch;
    int layerId = metaData->layerId;
    int workerId = metaData->workerId;
    int serverId = metaData->serverId;
    int layerNum = metaData->layerNum;
    int bitNum = metaData->bitNum;
    EmbMessage &reply = *metaData->reply;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    float *ptr_result = metaData->ptr_result;

    auto oldToNewMap = *metaData->oldToNewMap;
    int localNodeSize = metaData->localNodeSize;
    int feat_num = metaData->feat_num;
    int oneIntDimNum = 32 / bitNum;

    request.set_layerid(layerId);
    request.set_iterround(epoch);
    request.set_layernum(layerNum);
    request.set_bitnum(bitNum);

//    cout<<"workerId:"<<workerId<<endl;
    request.set_workerid(workerId);

    // 去服务器中获取嵌入,这里建立的是每个worker的channel

    // 构建request
    int nodeNum = nodes.size();
//    cout<<"node size:"<<nodes.size()<<endl;
    for (int i = 0; i < nodeNum; i++) {
        request.add_nodes(nodes[i]);
    }


    Status status = dgnnClient->stub_->workerPullEmbCompress(&context, request, &reply);
    uint mask = 0;
    if (bitNum == 1) {
        mask = 0x00000001;
    } else if (bitNum == 2) {
        mask = 0x00000003;
    } else if (bitNum == 4) {
        mask = 0x0000000f;
    } else if (bitNum == 8) {
        mask = 0x000000ff;
    } else if (bitNum == 16) {
        mask = 0x0000ffff;
    }

    if (status.ok()) {

        vector<float> bucket;

        int shape0 = reply.resp_node_size();
        int shape1 = reply.resp_featdim_size();
        int shape_dim = reply.shapedim();

//        cout<<"bucket size:!!!"<<reply.values_size()<<endl;
        for (auto value:reply.values()) {
            bucket.push_back(value);
//            cout<<value<<endl;

        }

        auto &embReply = reply.resp_compress_emb_concat();

        for (int i = 0; i < shape0; i++) {
            int nid = nodes[i];
            int new_id = oldToNewMap[nid] - localNodeSize;
            auto &node_worker = nodes[i];
            for (int j = 0; j < shape1; j++) {
                uint dim = embReply.Get(i * shape1 + j);
                // transform to 4 int data_raw
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = bucket[a];
//                        if(bucket[a]!=0){
//                            cout<<"bucket a:"<<bucket[a]<<endl;
//                        }

                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = bucket[a];
                    }

                }

            }


        }


        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
        ThreadUtil::count_respWorkerNumForEmbs++;
        lck.unlock();
    } else {
        cout << "worker_pull_emb_compress false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }

}


void DGNNClient::test_workerPullEmbCompress() {
    ClientContext context;
    EmbMessage request;
    EmbMessage reply;
    request.set_layerid(0);
    request.set_bucketnum(10);
    request.set_compensatemethod("accorIter");
    request.set_ifcompensate(true);
    stub_->workerPullEmbCompress(&context, request, &reply);
    request.set_iterround(1);
    stub_->workerPullEmbCompress(&context, request, &reply);
}

py::array_t<float>
DGNNClient::worker_pull_needed_emb(py::array_t<int> &nodes, int epoch, int layerId, int workerId, int serverId) {
    py::buffer_info nodes_buf = nodes.request();
//    cout<<"node size:"<<nodes.size()<<",buf1 size:"<<nodes_buf.size<<endl;
    if (nodes_buf.ndim != 1) {
        throw std::runtime_error("numpy.ndarray dims must be 1!");
    }
    int *ptr1 = (int *) nodes_buf.ptr;

    // 去服务器中获取嵌入,这里建立的是每个worker的channel
    clock_t start_totle = clock();
    ClientContext context;
    EmbMessage request;
    EmbMessage reply;
    request.set_epoch(epoch);
    request.set_layerid(layerId);
    request.set_workerid(workerId);

    int nodeNum = nodes_buf.shape[0];
    // 构建request
    for (int i = 0; i < nodeNum; i++) {
        request.add_nodes(ptr1[i]);
    }


//    clock_t start = clock();
//    cout <<"aaaaaaaaaaaaa"<<endl;
//    cout<<WorkerStore::testString<<endl;

//    Status status = stub_->workerPullEmb(&context, request, &reply);
    Status status;
    CompletionQueue cq;
    void *got_tag;
    bool ok = false;
    unique_ptr<ClientAsyncResponseReader<EmbMessage>> rpc(
            stub_->PrepareAsyncworkerPullEmb(&context, request, &cq));
    rpc->StartCall();

    rpc->Finish(&reply, &status, (void *) 1);
    //
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(got_tag == (void *) 1);
    GPR_ASSERT(ok);

//    clock_t end = clock();
//    cout<<"bbbbbbbbbbbbbbbbb"<<endl;
//    cout<<"pull emb time from server "<<serverId<<" :"<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//    cout<<"byte size:"<<request.ByteSizeLong()<<endl;
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "pull needed embeddings false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }


//    int count=0;
    int shape0 = reply.denseembmessage().embs_size();
    int shape1 = reply.denseembmessage().embs().begin()->tensor_size();
    auto result = py::array_t<float>(shape0 * shape1);

//    cout<<"[worker_pull_needed_emb] result size:"<<shape0<<"*"<<shape1<<endl;
    result.resize({shape0, shape1});

    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;


    for (int i = 0; i < shape0; i++) {
        const TensorMessage &tm = reply.denseembmessage().embs(i);
        for (int j = 0; j < shape1; j++) {
            ptr_result[i * shape1 + j] = tm.tensor(j);
        }
    }

    clock_t end_totle = clock();
//    cout<<"pull emb time2:"<<(double)(end_totle-start_totle)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    return result;

}


void *DGNNClient::worker_pull_needed_emb_parallel(void *metaData_void) {
    auto metaData = (ReqEmbsMetaData *) metaData_void;
    vector<int> &nodes = *metaData->nodes;
    int epoch = metaData->epoch;
    int layerId = metaData->layerId;
    int workerId = metaData->workerId;
    int serverId = metaData->serverId;
    EmbMessage &reply = *metaData->reply;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    float *ptr_result = metaData->ptr_result;

    auto oldToNewMap = *metaData->oldToNewMap;
    int localNodeSize = metaData->localNodeSize;
    int feat_num = metaData->feat_num;

    //    cout<<"node size:"<<nodes.size()<<",buf1 size:"<<nodes_buf.size<<endl;

    // 去服务器中获取嵌入,这里建立的是每个worker的channel

    ClientContext context;
    EmbMessage request;
//    RespEmbSparseMessage reply;
    request.set_epoch(epoch);
    request.set_layerid(layerId);
    request.set_workerid(workerId);

    int nodeNum = nodes.size();
    // 构建request
    for (int i = 0; i < nodeNum; i++) {
        request.add_nodes(nodes[i]);
    }


    Status status = dgnnClient->stub_->workerPullEmb(&context, request, &reply);

    if (status.ok()) {
        cout<<"resp_none_compress_emb_concat size:"<<reply.resp_none_compress_emb_concat().size()<<",mutable size:"<<
            reply.mutable_resp_none_compress_emb_concat()->size()<<endl;
        cout<<nodeNum<<","<<feat_num<<endl;
        auto &embConcat = reply.resp_none_compress_emb_concat();
        for (int j = 0; j < nodeNum; j++) {
            int nid = nodes[j];
            int new_id = oldToNewMap[nid] - localNodeSize;
//            if(j==5){
//                cout<<"+***************"<<embConcat.Get(j * feat_num+10)<<endl;
//            }
            for (int k = 0; k < feat_num; k++) {
                ptr_result[new_id * feat_num + k] = embConcat.Get(j * feat_num + k);

            }
        }


        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
        ThreadUtil::count_respWorkerNumForEmbs++;
        lck.unlock();
    } else {
        cout << "pull needed embeddings false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
        exit(-1);
    }


//    if (status.ok()) {
//
//        auto &denseMessage = reply.denseembmessage();
//        int len = nodes.size();
//        for (int j = 0; j < len; j++) {
//            int nid = nodes[j];
//            int new_id = oldToNewMap[nid] - localNodeSize;
//            const auto &denseMessage_node = denseMessage.embs(j);
//            for (int k = 0; k < feat_num; k++) {
//                ptr_result[new_id * feat_num + k] = (float) denseMessage_node.tensor(k);
//
//            }
//        }
//
//
//        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
//        ThreadUtil::count_respWorkerNumForEmbs++;
//        lck.unlock();
//    } else {
//        cout << "pull needed embeddings false" << endl;
//        cout << "error detail:" << status.error_details() << endl;
//        cout << "error message:" << status.error_message() << endl;
//        cout << "error code:" << status.error_code() << endl;
//        exit(-1);
//    }


}


py::array_t<float> DGNNClient::worker_pull_needed_G(py::array_t<int> &needed_G_set, int layerId) {
    // 这里的stub就是索引到指定的worker server中
    // 去服务器中获取嵌入,这里建立的是每个worker的channel
    py::buffer_info nodes_buf = needed_G_set.request();
    if (nodes_buf.ndim != 1) {
        throw std::runtime_error("numpy.ndarray dims must be 1!");
    }
    int *ptr1 = (int *) nodes_buf.ptr;


    ClientContext context;
    EmbMessage request;
    EmbMessage reply;
    // 构建request
    // 构建request
    request.mutable_nodes()->Reserve(needed_G_set.size());


    for (int i = 0; i < nodes_buf.shape[0]; i++) {
//        request.add_nodes(ptr1[i]);
        request.mutable_nodes()->Add(ptr1[i]);
    }

//    cout<<"g2 nodes size:"<<request.nodes_size()<<endl;
    request.set_layerid(layerId);
    Status status = stub_->workerPullG(&context, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "pull needed embeddings false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }


    int shape0 = reply.resp_node_size();
    int shape1 = reply.resp_featdim_size();
    auto result = py::array_t<float>(shape0 * shape1);

//    cout<<"G result size:"<<shape0<<"*"<<shape1<<endl;
    result.resize({shape0, shape1});

    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;

//    for (int i = 0; i < shape0; i++) {
//        const TensorMessage &tm = reply.embs(i);
//        for (int j = 0; j < shape1; j++) {
//            ptr_result[i * shape1 + j] = tm.tensor(j);
//        }
//    }
    int totalSize = reply.resp_none_compress_emb_concat_size();
    for (int i = 0; i < totalSize; i++) {
        ptr_result[i] = reply.resp_none_compress_emb_concat(i);
    }

    return result;


}


vector<vector<float>> DGNNClient::server_PullWeights(int layer_id) {
    vector<vector<float>> weights;
    // 获取权重
    ClientContext clientContext;
    IntMessage request;
    request.set_id(layer_id);
    WeightsAndBiasMessage reply;
    Status status = stub_->pullWeights(&clientContext, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "pull weights false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }


    for (const auto &row:reply.weights()) {
        vector<float> vec;
        for (auto item:row.tensor()) {
            vec.push_back(item);
        }
        weights.push_back(vec);
    }

    WorkerStore::weights[layer_id] = weights;

    return weights;
}


vector<float> DGNNClient::server_PullBias(int layer_id) {
    vector<float> bias;
    // 获取权重
    ClientContext clientContext;
    IntMessage request;
    request.set_id(layer_id);
    WeightsAndBiasMessage reply;
    Status status = stub_->pullBias(&clientContext, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "pull bias false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }

    for (const auto &row:reply.bias().tensor()) {
        bias.push_back(row);
    }

    return bias;
}

void DGNNClient::server_Barrier(int layer_id) {
    ClientContext clientContext;
    BoolMessage request;
    BoolMessage reply;
    Status status = stub_->barrier(&clientContext, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "pull bias false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}

void DGNNClient::testVariant() {

    ClientContext clientContext;
    TestVMessage request;
    BoolMessage reply;
    cout << "size:" << request.ByteSizeLong() << endl;
    // 10个5 12个字节
    // 100个5 102字节
    // 10000个5 10003个字节
    for (int i = 0; i < 120; i++) {
        request.add_values(127);
    }
    cout << "size:" << request.ByteSizeLong() << endl;

    stub_->TestVariant(&clientContext, request, &reply);
}

void DGNNClient::test1Bit() {
    // 使用protobuf传不了1bit的数据
    ClientContext clientContext;
    BitArrayMessage request;
    BitArrayMessage reply;

    for (int i = 0; i < 10; i++) {
        cout << "size:" << request.ByteSizeLong() << endl;
        request.add_array(true);
        request.add_array(false);
    }

    cout << "size:" << request.ByteSizeLong() << endl;


}

void DGNNClient::setG(const map<int, vector<float>> &g, int id) {
    if (WorkerStore::G_map.count(id) != 0) {
        WorkerStore::G_map[id] = g;
    } else {
        WorkerStore::G_map.insert(pair<int, map<int, vector<float>>>(id, g));
    }

}

void DGNNClient::test_large() {
    ClientContext clientContext;
    LargeMessage request;
    LargeMessage reply;

    stub_->testLargeSize(&clientContext, request, &reply);
}

//py::array_t<float> DGNNClient::worker_pull_emb_trend(py::array_t<int> &needed_emb_set, int layerId,
//                                                     int epoch, int bucketNum, int workerId, int serverId, int layerNum,
//                                                     int trend,
//                                                     int bitNum) {
//    ClientContext context;
//    EmbMessage request;
//    EmbMessage reply;
//
//    request.set_layerid(layerId);
//    request.set_iterround(epoch);
//    request.set_epoch(epoch);
//    request.set_bucketnum(bucketNum);
//    request.set_layernum(layerNum);
//    request.set_workerid(workerId);
//    request.set_trend(trend);
//    request.set_bitnum(bitNum);
//
//
//    py::buffer_info nodes_buf = needed_emb_set.request();
////    cout<<"node size:"<<nodes.size()<<",buf1 size:"<<nodes_buf.size<<endl;
//    if (nodes_buf.ndim != 1) {
//        throw std::runtime_error("numpy.ndarray dims must be 1!");
//    }
//    int *ptr1 = (int *) nodes_buf.ptr;
//
//    // 去服务器中获取嵌入,这里建立的是每个worker的channel
//
//    // 构建request
//    for (int i = 0; i < nodes_buf.shape[0]; i++) {
//        request.add_nodes(ptr1[i]);
//    }
//
//
//    Status status = stub_->workerPullEmbTrend(&context, request, &reply);
//
//
//    if (status.ok()) {
////        cout << "okokokokok" << endl;
//    } else {
//        cout << "worker_pull_emb_compress false" << endl;
//        cout << "error detail:" << status.error_details() << endl;
//        cout << "error message:" << status.error_message() << endl;
//        cout << "error code:" << status.error_code() << endl;
//    }
//
////    start=clock();
//    // de compress
//
//
//    int shape0 = 0;
//    int shape1 = 0;
//    int shape_dim = reply.shapedim();
//
//
//    if ((epoch + 1) % trend == 0) {
//        shape0 = reply.denseembmessage().embs_size();
//        shape1 = reply.denseembmessage().embs().begin()->tensor_size();
//    } else {
//        shape0 = reply.embs_size();
//        shape1 = reply.embs().begin()->tensor_size();
//    }
//    auto result = py::array_t<float>(shape0 * shape_dim);
//    result.resize({shape0, shape_dim});
//
//    py::buffer_info buf_result = result.request();
//    float *ptr_result = (float *) buf_result.ptr;
//
////    cout<<"shape0,shape1,shapedim:"<<shape0<<","<<shape1<<","<<shape_dim<<endl;
//
//    // 两种情况，1是请求压缩的，2是请求完整的加变化率
//    if ((epoch + 1) % trend == 0) {
//        if ((epoch + 1) / trend == 1) {
//            auto &changeMatrix = reply.changerate().changematrix();
//            map<int, map<int, vector<float>>> map_map_tmp;
//            WorkerStore::embs_change_rate_worker.insert(
//                    pair<int, map<int, map<int, vector<float>>>>(serverId, map_map_tmp));
//            map<int, vector<float>> map_tmp;
//            WorkerStore::embs_change_rate_worker[serverId].insert(pair<int, map<int, vector<float>>>(layerId, map_tmp));
//            for (int i = 0; i < shape0; i++) {
//                vector<float> vec_tmp;
//                const auto &tm = reply.denseembmessage().embs(i);
//                auto &changeVector = changeMatrix.Get(i);
//                for (int j = 0; j < shape1; j++) {
//                    vec_tmp.push_back(changeVector.tensor(j));
//                    ptr_result[i * shape1 + j] = tm.tensor(j);
//                }
//                WorkerStore::embs_change_rate_worker[serverId][layerId].insert(pair<int, vector<float>>(i, vec_tmp));
//            }
//        } else {
//            auto &changeMatrix = reply.changerate().changematrix();
//            map<int, vector<float>> map_tmp;
//            auto &changeRateLayer_ws = WorkerStore::embs_change_rate_worker[serverId][layerId];
//            for (int i = 0; i < shape0; i++) {
//                auto &changeRateNode_ws = changeRateLayer_ws[i];
//                const auto &tm = reply.denseembmessage().embs(i);
//                auto &changeVector = changeMatrix.Get(i);
//                for (int j = 0; j < shape1; j++) {
//                    changeRateNode_ws[j] = changeVector.tensor(j);
//                    ptr_result[i * shape1 + j] = tm.tensor(j);
//                }
//
//            }
//        }
//
////        cout<<"WorkerStore::embs_change_rate_worker shape"<<WorkerStore::embs_change_rate_worker.size()<<
////            "*"<<WorkerStore::embs_change_rate_worker.begin()->second.size()<<endl;
//
//    } else {
//
//        // 返回的是压缩的后的值
//
//        deCompress(shape0, shape1, shape_dim, bitNum, reply, ptr_result);
//
//
//    }
//
//    return result;
//
//}

void deCompressNew(EmbMessage &reply, int shape0, int shape1, int shape_dim,
                   int bitNum, vector<int> nodes, int localNodeSize, map<int, int> oldToNewMap, float *ptr_result,
                   int serverId, int layerId,
                   vector<uint> &vec_flags) {
    vector<float> bucket;
    for (auto value:reply.values()) {
        bucket.push_back(value);
    }

    auto &embs_from_remote = WorkerStore::embs_from_remote[serverId][layerId];

    // comp_error,pred_error,mix_error

    auto &embReply = reply.resp_compress_emb_concat();
    int count_non_remove_comp = 0;
    int oneIntDimNum = 32 / bitNum;
    uint mask = 0;
    if (bitNum == 1) {
        mask = 0x00000001;
    } else if (bitNum == 2) {
        mask = 0x00000003;
    } else if (bitNum == 4) {
        mask = 0x0000000f;
    } else if (bitNum == 8) {
        mask = 0x000000ff;
    } else if (bitNum == 16) {
        mask = 0x0000ffff;
    }


    for (int i = 0; i < shape0; i++) {
        int nid = nodes[i];
        int new_id = oldToNewMap[nid] - localNodeSize;
        auto &embs_node = embs_from_remote[nid];
        int flag = vec_flags[i];

        if (flag == 0) {
            for (int j = 0; j < shape1; j++) {
                uint dim = embReply.Get(count_non_remove_comp * shape1 + j);
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = bucket[a];
                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        uint dim = embReply.Get(count_non_remove_comp * shape1 + j);
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = bucket[a];
                    }
                }

            }
            count_non_remove_comp++;

        } else if (flag == 1) {
            for (int j = 0; j < shape1; j++) {
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = embs_node[j * oneIntDimNum + l];
                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = embs_node[j * oneIntDimNum + l];
                    }
                }
            }
        } else {
            for (int j = 0; j < shape1; j++) {
                uint dim = embReply.Get(count_non_remove_comp * shape1 + j);
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] =
                                (bucket[a] + embs_node[j * oneIntDimNum + l]) / 2;
                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] =
                                (bucket[a] + embs_node[j * oneIntDimNum + l]) / 2;
                    }
                }

            }
            count_non_remove_comp++;
        }

    }
}


void
deCompressTrend(int epoch, int trend, const vector<int> &nodes, EmbMessage &reply, int bitNum, int shape0, int shape1,
                int shape_dim,
                float *ptr_result, int layerId, int feat_num, int localNodeSize, map<int, int> oldToNewMap,
                int serverId, vector<uint> &vec_flags) {
    int round = (epoch + 1) % trend;
    auto &changeRate_layer = WorkerStore::embs_change_rate_worker[serverId][layerId];
    auto &emb_layer = WorkerStore::embs_from_remote[serverId][layerId];

//    cout<<"round,serverid,layerid:"<<round<<","<<serverId<<","<<layerId<<endl;
//    cout<<"changeRate_layer size:"<<changeRate_layer.size()<<"*"<<changeRate_layer.begin()->second.size()<<endl;
//    cout<<"emb_layer size:"<<emb_layer.size()<<"*"<<emb_layer.begin()->second.size()<<endl;

    vector<float> bucket;
    for (auto value:reply.values()) {
        bucket.push_back(value);
//        cout<<"bucket value:"<<value<<endl;
    }

    auto &embReply = reply.resp_compress_emb_concat();
    int count_non_remove_comp = 0;
    int oneIntDimNum = 32 / bitNum;
    uint mask = 0;
    if (bitNum == 1) {
        mask = 0x00000001;
    } else if (bitNum == 2) {
        mask = 0x00000003;
    } else if (bitNum == 4) {
        mask = 0x0000000f;
    } else if (bitNum == 8) {
        mask = 0x000000ff;
    } else if (bitNum == 16) {
        mask = 0x0000ffff;
    }


    for (int i = 0; i < shape0; i++) {

        int nid = nodes[i];
        int new_id = oldToNewMap[nid] - localNodeSize;
        auto &embs_node = emb_layer[nid];
        auto &changeRate_node = changeRate_layer[nid];
        int flag = vec_flags[i];
        // @test 3
//        if ( nid== 0 && layerId != 0) {
//            cout<< new_id <<","<< "client:";
//        }
        if (flag == 0) {
            for (int j = 0; j < shape1; j++) {
                uint dim = embReply.Get(count_non_remove_comp * shape1 + j);
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = bucket[a];
                        // @test 3
//                        if(nid==0 && layerId!=0){
//                            cout<<bucket[a]<<" ";
//                        }
                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = bucket[a];
                        // @test 3
//                        if(nid==0 && layerId!=0){
//                            cout<<bucket[a]<<" ";
//                        }
                    }
                }

            }
            count_non_remove_comp++;

        } else if (flag == 1) {
            for (int j = 0; j < shape1; j++) {
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        float embs_trend =
                                round * changeRate_node[j * oneIntDimNum + l] + embs_node[j * oneIntDimNum + l];
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = embs_trend;
                        // @test 3
//                        if(nid==0 && layerId!=0){
//                            cout<<embs_trend<<" ";
//                        }
                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        float embs_trend =
                                round * changeRate_node[j * oneIntDimNum + l] + embs_node[j * oneIntDimNum + l];
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = embs_trend;
                    }
                }
            }
        } else {
            for (int j = 0; j < shape1; j++) {
                uint dim = embReply.Get(count_non_remove_comp * shape1 + j);
                if (j * oneIntDimNum + (oneIntDimNum - 1) < shape_dim) {
                    for (int l = 0; l < oneIntDimNum; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        float embs_trend =
                                round * changeRate_node[j * oneIntDimNum + l] + embs_node[j * oneIntDimNum + l];
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] =(bucket[a] + embs_trend) / 2;
                        // @test 3
//                        if(nid==0 && layerId!=0){
//                            cout<<(bucket[a] + embs_trend) / 2<<" ";
//                        }
                    }
                } else {
                    int num = shape_dim - j * oneIntDimNum;
                    for (int l = 0; l < num; l++) {
                        int a = dim >> (32 - (l + 1) * bitNum) & mask;
                        float embs_trend =
                                round * changeRate_node[j * oneIntDimNum + l] + embs_node[j * oneIntDimNum + l];
                        ptr_result[new_id * shape_dim + j * oneIntDimNum + l] = (bucket[a] + embs_trend) / 2;
                        // @test 3
//                        if(nid==0 && layerId!=0){
//                            cout<< (bucket[a] + embs_trend) / 2<<" ";
//                        }
                    }
                }

            }

            count_non_remove_comp++;
        }
    }
    // @test 1
//    cout<<endl;


}

void *DGNNClient::worker_pull_emb_trend_parallel_select(void *metaData_void) {
    ClientContext context;
    EmbMessage request;

    auto metaData = (ReqEmbsMetaData *) metaData_void;
    vector<int> &nodes = *metaData->nodes;
    int epoch = metaData->epoch;
    int layerId = metaData->layerId;
    int workerId = metaData->workerId;
    int serverId = metaData->serverId;
    int layerNum = metaData->layerNum;
    int bitNum = metaData->bitNum;
    int trend = metaData->trend;
    EmbMessage &reply = *metaData->reply;
    DGNNClient *dgnnClient = metaData->dgnnClient;
    float *ptr_result = metaData->ptr_result;
    auto oldToNewMap = *metaData->oldToNewMap;
    int localNodeSize = metaData->localNodeSize;
    int feat_num = metaData->feat_num;
    float comp_percent = -1;
//    int oneBitDim


    request.set_layerid(layerId);
    request.set_iterround(epoch);
    request.set_epoch(epoch);
    request.set_layernum(layerNum);
    request.set_workerid(workerId);
    request.set_trend(trend);
    request.set_bitnum(bitNum);

//    cout<<"aaaa"<<endl;

    // 构建request
    int node_size = nodes.size();
    for (int i = 0; i < node_size; i++) {
        request.add_nodes(nodes[i]);
    }
//    cout<<"bbbbb"<<endl;

    Status status = dgnnClient->stub_->workerPullEmbTrendSelect(&context, request, &reply);

//    cout<<"ccc"<<endl;

    int shape0 = reply.resp_node_size();
    int shape1 = reply.resp_featdim_size();
    int shape_dim = reply.shapedim();
//    cout<<"shape0,shepe1,shape_dim:"<<shape0<<","<<shape1<<","<<shape_dim<<endl;

    if (status.ok()) {
        if (epoch == 0) {
            // serverId
            map<int, map<int, vector<float>>> embs_from_server_i;
            WorkerStore::embs_from_remote.insert(
                    pair<int, map<int, map<int, vector<float>>>>(serverId, embs_from_server_i));

            // layerId
            map<int, vector<float>> embs_layer_i;
            WorkerStore::embs_from_remote[serverId].insert(
                    pair<int, map<int, vector<float>>>(layerId, embs_layer_i));

            for (int i = 0; i < shape0; i++) {
                vector<float> emb_tmp(shape1);
                int id_old = request.nodes(i);
                int id_new = oldToNewMap[id_old] - localNodeSize;
                for (int j = 0; j < shape1; j++) {
                    auto emb_j = reply.resp_none_compress_emb_concat(i * shape_dim + j);
                    ptr_result[id_new * shape_dim + j] = emb_j;
                    emb_tmp[j] = emb_j;
                }
                WorkerStore::embs_from_remote[serverId][layerId].insert(pair<int, vector<float>>(id_old, emb_tmp));
            }
        } else {
            if ((epoch + 1) % trend == 0) {
                if ((epoch + 1) / trend == 1) {

                    map<int, map<int, vector<float>>> map_map_tmp;
                    WorkerStore::embs_change_rate_worker.insert(
                            pair<int, map<int, map<int, vector<float>>>>(serverId, map_map_tmp));

                    map<int, vector<float>> map_tmp;
                    WorkerStore::embs_change_rate_worker[serverId].insert(
                            pair<int, map<int, vector<float>>>(layerId, map_tmp));
                    auto &embs = WorkerStore::embs_from_remote[serverId][layerId];
                    auto &rates = WorkerStore::embs_change_rate_worker[serverId][layerId];
                    for (int i = 0; i < shape0; i++) {
                        int id_old = request.nodes(i);
                        int id_new = oldToNewMap[id_old] - localNodeSize;
                        vector<float> rate_tmp(shape1);
                        auto embs_node = embs[id_old];
                        for (int j = 0; j < shape1; j++) {
                            auto emb_j = reply.resp_none_compress_emb_concat(i * shape_dim + j);
                            ptr_result[id_new * shape_dim + j] = emb_j;
                            embs_node[j] = emb_j;
                            rate_tmp[j] = reply.resp_none_compress_rate_concat(i * shape_dim + j);

                        }

                        rates.insert(
                                pair<int, vector<float>>(id_old, rate_tmp));
                    }

                } else {
                    auto &changeMatrix = reply.changerate().changematrix();
                    auto &changeRateLayer_ws = WorkerStore::embs_change_rate_worker[serverId][layerId];
                    auto &embs_from_remote_layer = WorkerStore::embs_from_remote[serverId][layerId];

                    for (int i = 0; i < shape0; i++) {
                        int id_old = request.nodes(i);
                        int id_new = oldToNewMap[id_old] - localNodeSize;
                        auto &embs_node = embs_from_remote_layer[id_old];
                        auto &rate_node = changeRateLayer_ws[id_old];
                        for (int j = 0; j < shape1; j++) {
                            auto emb_j = reply.resp_none_compress_emb_concat(i * shape_dim + j);
                            ptr_result[id_new * shape_dim + j] = emb_j;
                            embs_node[j] = emb_j;
                            rate_node[j] = reply.resp_none_compress_rate_concat(i * shape_dim + j);
                        }

                    }
                }

            } else {
                // decompress the value_select_flags
                comp_percent = reply.comp_data_percent();
                vector<uint> vec_flags(node_size);
                for (int i = 0; i < node_size; i++) {
                    int idInOneCompressDim = i % 16;
                    vec_flags[i] = reply.value_select_flags(i / 16) >> (32 - (idInOneCompressDim + 1) * 2) & 0x00000003;
                }

                if ((int) (epoch / trend) == 0) {
                    deCompressNew(reply, shape0, shape1, shape_dim,
                                  bitNum, nodes, localNodeSize, oldToNewMap, ptr_result, serverId, layerId, vec_flags);
                } else {
                    deCompressTrend(epoch, trend, nodes, reply, bitNum, shape0, shape1, shape_dim,
                                    ptr_result, layerId, feat_num, localNodeSize, oldToNewMap, serverId, vec_flags);
                }


            }
        }


        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
        ThreadUtil::count_respWorkerNumForEmbs++;
        if (comp_percent != -1) {
            WorkerStore::comp_percent += comp_percent * shape0;
        }
        lck.unlock();
    } else {
        cout << "worker_pull_emb_compress false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }

//    start=clock();
    // de compress
}

//void *DGNNClient::worker_pull_emb_trend_parallel(void *metaData_void) {
//    ClientContext context;
//    EmbMessage request;
//
//    auto metaData = (ReqEmbsMetaData *) metaData_void;
//    vector<int> &nodes = *metaData->nodes;
//    int epoch = metaData->epoch;
//    int layerId = metaData->layerId;
//    int workerId = metaData->workerId;
//    int serverId = metaData->serverId;
//    int layerNum = metaData->layerNum;
//    int bitNum = metaData->bitNum;
//    int trend = metaData->trend;
//    EmbMessage &reply = *metaData->reply;
//    DGNNClient *dgnnClient = metaData->dgnnClient;
//    float *ptr_result = metaData->ptr_result;
//    auto oldToNewMap = *metaData->oldToNewMap;
//    int localNodeSize = metaData->localNodeSize;
//    int feat_num = metaData->feat_num;
//
//
//    request.set_layerid(layerId);
//    request.set_iterround(epoch);
//    request.set_epoch(epoch);
//    request.set_layernum(layerNum);
//    request.set_workerid(workerId);
//    request.set_trend(trend);
//    request.set_bitnum(bitNum);
//
////    cout<<"aaaa"<<endl;
//
//    // 构建request
//    int node_size = nodes.size();
//    for (int i = 0; i < node_size; i++) {
//        request.add_nodes(nodes[i]);
//    }
////    cout<<"bbbbb"<<endl;
//
//    Status status = dgnnClient->stub_->workerPullEmbTrend(&context, request, &reply);
//
////    cout<<"ccc"<<endl;
//
//    int shape0 = reply.resp_node_size();
//    int shape1 = reply.resp_featdim_size();
//    int shape_dim = reply.shapedim();
////    cout<<"shape0,shepe1,shape_dim:"<<shape0<<","<<shape1<<","<<shape_dim<<endl;
//
//    if (status.ok()) {
//        if (epoch == 0) {
//            // serverId
//            map<int, map<int, vector<float>>> embs_from_server_i;
//            WorkerStore::embs_from_remote.insert(
//                    pair<int, map<int, map<int, vector<float>>>>(serverId, embs_from_server_i));
//
//            // layerId
//            map<int, vector<float>> embs_layer_i;
//            WorkerStore::embs_from_remote[serverId].insert(
//                    pair<int, map<int, vector<float>>>(layerId, embs_layer_i));
//
//            for (int i = 0; i < shape0; i++) {
//                vector<float> emb_tmp(shape1);
//                int id_old = request.nodes(i);
//                int id_new = oldToNewMap[id_old] - localNodeSize;
//                for (int j = 0; j < shape1; j++) {
//                    auto emb_j = reply.resp_none_compress_emb_concat(i * shape_dim + j);
//                    ptr_result[id_new * shape_dim + j] = emb_j;
//                    emb_tmp[j] = emb_j;
//                }
//                WorkerStore::embs_from_remote[serverId][layerId].insert(pair<int, vector<float>>(id_old, emb_tmp));
//            }
//        } else {
//            if ((epoch + 1) % trend == 0) {
//                if ((epoch + 1) / trend == 1) {
//
//                    map<int, map<int, vector<float>>> map_map_tmp;
//                    WorkerStore::embs_change_rate_worker.insert(
//                            pair<int, map<int, map<int, vector<float>>>>(serverId, map_map_tmp));
//
//                    map<int, vector<float>> map_tmp;
//                    WorkerStore::embs_change_rate_worker[serverId].insert(
//                            pair<int, map<int, vector<float>>>(layerId, map_tmp));
//                    auto &embs = WorkerStore::embs_from_remote[serverId][layerId];
//                    auto &rates = WorkerStore::embs_change_rate_worker[serverId][layerId];
//                    for (int i = 0; i < shape0; i++) {
//                        int id_old = request.nodes(i);
//                        int id_new = oldToNewMap[id_old] - localNodeSize;
//                        vector<float> rate_tmp(shape1);
//                        auto embs_node = embs[id_old];
//                        for (int j = 0; j < shape1; j++) {
//                            auto emb_j = reply.resp_none_compress_emb_concat(i * shape_dim + j);
//                            ptr_result[id_new * shape_dim + j] = emb_j;
//                            embs_node[j] = emb_j;
//                            rate_tmp[j] = reply.resp_none_compress_rate_concat(i * shape_dim + j);
//
//                        }
//
//                        rates.insert(
//                                pair<int, vector<float>>(id_old, rate_tmp));
//                    }
//
//                } else {
//                    auto &changeMatrix = reply.changerate().changematrix();
//                    auto &changeRateLayer_ws = WorkerStore::embs_change_rate_worker[serverId][layerId];
//                    auto &embs_from_remote_layer = WorkerStore::embs_from_remote[serverId][layerId];
//
//                    for (int i = 0; i < shape0; i++) {
//                        int id_old = request.nodes(i);
//                        int id_new = oldToNewMap[id_old] - localNodeSize;
//                        auto &embs_node = embs_from_remote_layer[id_old];
//                        auto &rate_node = changeRateLayer_ws[id_old];
//                        for (int j = 0; j < shape1; j++) {
//                            auto emb_j = reply.resp_none_compress_emb_concat(i * shape_dim + j);
//                            ptr_result[id_new * shape_dim + j] = emb_j;
//                            embs_node[j] = emb_j;
//                            rate_node[j] = reply.resp_none_compress_rate_concat(i * shape_dim + j);
//                        }
//
//                    }
//                }
//
//            } else {
//                vector<uint> vec_node(node_size);
//                for (int i = 0; i < node_size; i++) {
//                    vec_node[i] = 2;
//                }
//                if ((int) (epoch / trend) == 0) {
//                    deCompressNew(reply, shape0, shape1, shape_dim,
//                                  bitNum, nodes, localNodeSize, oldToNewMap, ptr_result, serverId, layerId, vec_node);
//                } else {
//                    deCompressTrend(epoch, trend, nodes, reply, bitNum, shape0, shape1, shape_dim,
//                                    ptr_result, layerId, feat_num, localNodeSize, oldToNewMap, serverId, vec_node);
//                }
//
//
//            }
//        }
//
//
//        unique_lock<mutex> lck(ThreadUtil::mtx_respWorkerNumForEmbs);
//        ThreadUtil::count_respWorkerNumForEmbs++;
//        lck.unlock();
//    } else {
//        cout << "worker_pull_emb_compress false" << endl;
//        cout << "error detail:" << status.error_details() << endl;
//        cout << "error message:" << status.error_message() << endl;
//        cout << "error code:" << status.error_code() << endl;
//    }
//
////    start=clock();
//    // de compress
//}


py::array_t<float> DGNNClient::getChangeRate(int serverId, int layerId) {
    int shape0 = WorkerStore::embs_change_rate_worker[serverId][layerId].size();
    int shape1 = WorkerStore::embs_change_rate_worker[serverId][layerId].begin()->second.size();
//    cout<< "[In getChangeRate]: shape0 * shape1 ="<<shape0<<"*"<<shape1<<endl;
    auto result = py::array_t<float>(shape0 * shape1);
    result.resize({shape0, shape1});
    py::buffer_info buf_result = result.request();
    float *ptr_result = (float *) buf_result.ptr;
    auto &changeRateLayer = WorkerStore::embs_change_rate_worker[serverId][layerId];
    for (int i = 0; i < shape0; i++) {
        auto &changeRateNode = changeRateLayer[i];
        for (int j = 0; j < shape1; j++) {
            ptr_result[i * shape1 + j] = changeRateNode[j];
        }
    }
    return result;

}

void DGNNClient::test_small() {
    ClientContext clientContext;
    SmallMessage request;
    SmallMessage reply;
    stub_->testSmallSize(&clientContext, request, &reply);
}

//Yu
void DGNNClient::sendTrainNode(int worker_id, py::array_t<int> list) {
    ClientContext context;
    NodeMessage request;
    BoolMessage reply;
    for(int i = 0 ;i < list.size();i++){
        request.add_nodes((int)list.at(i));
    }
    Status status = stub_->workerSendTrainNode(&context, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "update parameters false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}

py::array_t<int> DGNNClient::pullTrainNode() {
    ClientContext context;
    ContextMessage request;
    NodeMessage reply;
    Status status = stub_->serverSendTrainNode(&context, request, &reply);
    vector<int> arr;
    for(int i = 0;i < reply.nodes_size();i++){
        arr.push_back(reply.nodes(i));
    }
    py::array_t<int> result = py::array_t<double>(arr.size());
    py::buffer_info buf = result.request();
    int* ptr = (int*)buf.ptr;
    for(int i = 0 ; i < arr.size() ;i++){
        ptr[i] = arr[i];
    }
    return result;
}

void DGNNClient::sendValNode(int worker_id, py::array_t<int> list) {
    ClientContext context;
    NodeMessage request;
    BoolMessage reply;
    for(int i = 0 ;i < list.size();i++){
        request.add_nodes((int)list.at(i));
    }
    Status status = stub_->workerSendValNode(&context, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "update parameters false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}

py::array_t<int> DGNNClient::pullValNode() {
    ClientContext context;
    ContextMessage request;
    NodeMessage reply;
    Status status = stub_->serverSendValNode(&context, request, &reply);
    vector<int> arr;
    for(int i = 0;i < reply.nodes_size();i++){
        arr.push_back(reply.nodes(i));
    }
    py::array_t<int> result = py::array_t<double>(arr.size());
    py::buffer_info buf = result.request();
    int* ptr = (int*)buf.ptr;
    for(int i = 0 ; i < arr.size() ;i++){
        ptr[i] = arr[i];
    }
    return result;
}

void DGNNClient::sendTestNode(int worker_id, py::array_t<int> list) {
    ClientContext context;
    NodeMessage request;
    BoolMessage reply;
    for(int i = 0 ;i < list.size();i++){
        request.add_nodes((int)list.at(i));
    }
    Status status = stub_->workerSendTestNode(&context, request, &reply);
    if (status.ok()) {
//        cout << "okokokokok" << endl;
    } else {
        cout << "update parameters false" << endl;
        cout << "error detail:" << status.error_details() << endl;
        cout << "error message:" << status.error_message() << endl;
        cout << "error code:" << status.error_code() << endl;
    }
}

py::array_t<int> DGNNClient::pullTestNode() {
    ClientContext context;
    ContextMessage request;
    NodeMessage reply;
    Status status = stub_->serverSendTestNode(&context, request, &reply);
    vector<int> arr;
    for(int i = 0;i < reply.nodes_size();i++){
        arr.push_back(reply.nodes(i));
    }
    py::array_t<int> result = py::array_t<double>(arr.size());
    py::buffer_info buf = result.request();
    int* ptr = (int*)buf.ptr;
    for(int i = 0 ; i < arr.size() ;i++){
        ptr[i] = arr[i];
    }
    return result;
}

