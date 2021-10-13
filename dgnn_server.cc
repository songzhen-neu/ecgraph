//
// Created by songzhen on 2020/10/13.
//

#include <sys/time.h>
#include "dgnn_server.h"
// define the final ServiceImpl class (it cannot be inherited); and it extends the class Service by 'public' manner,
// it can access any public member in the parent class, expect private members.
// test git aaaabbb

struct Bucket {
    float lower_bound;
    float upper_bound;
    int bid;
    float value;
};

Status ServiceImpl::add1(ServerContext *context, const intM *request,
                         intM *reply) {
    reply->set_value(request->value() + 1);
    cout << "vvvvvvvvvvvvvv" << endl;

    return Status::OK;
}

RandomPartitioner randomPartitioner;

Status ServiceImpl::pullDataFromMaster(
        ServerContext *context, const ContextMessage *request,
        DataMessage *reply) {

    cout << "worker " << request->workerid() << " has arrived!" << endl;
    int workerid = request->workerid();
    // 除了worker 0以外，其他所有线程都进行等待，worker 0进行分区，分区完成后notify其他进程
    if (request->workerid() == 0) {
        unique_lock<mutex> lck(ThreadUtil::mtx);
        // 开始进行分区
        int workerNum = request->workernum();
        Check::check_partition_pass(
                request->workernum(),
                request->partition().datanum(),
                request->partition().datapath(),
                request->partition().featuredim(),
                request->partition().classnum());
        //int data_num, int worker_num, string filename, int feature_size, int label_size;

        randomPartitioner.init(
                request->partition().datanum(),
                request->workernum(),
                request->partition().datapath(),
                request->partition().featuredim(),
                request->partition().classnum());

        randomPartitioner.startPartition(workerNum, request->partition().partitionmethod(),
                                         request->partition().datanum(), request->partition().edgenum());
        ThreadUtil::ready = true;
        ThreadUtil::cv.notify_all();


    } else {
        unique_lock<mutex> lck(ThreadUtil::mtx);
        // 进入等待
        while (!ThreadUtil::ready) {
            ThreadUtil::cv.wait(lck);
        }
    }


    // 开始返回每个worker的数据
    // 构建nodes
    NodeMessage *nodeMessage = reply->nodelist().New();
    for (int id:randomPartitioner.nodes[workerid]) {
        nodeMessage->add_nodes(id);
    }
    reply->set_allocated_nodelist(nodeMessage);

    // 构建feature
    DataMessage_FeatureMessage *featureMessage = reply->featurelist().New();
    for (auto &id_feature : randomPartitioner.features[workerid]) {
        DataMessage_FeatureMessage_FeatureItem *item = featureMessage->add_features();
        item->set_vid(id_feature.first);
        for (float feat_dim:id_feature.second) {
            item->add_feature(feat_dim);
        }
    }
    reply->set_allocated_featurelist(featureMessage);

    // 构建label
    DataMessage_LabelMessage *labelMessage = reply->labellist().New();
    for (auto &id_label:randomPartitioner.labels[workerid]) {
        DataMessage_LabelMessage_LabelItem *item = labelMessage->add_labels();
        item->set_vid(id_label.first);
        item->set_label(id_label.second);
    }
    reply->set_allocated_labellist(labelMessage);

    // 构建adjs
    DataMessage_AdjMessage *adjMessage = reply->adjlist().New();
    for (const auto &id_neibors:randomPartitioner.adjs[workerid]) {
        DataMessage_AdjMessage_AdjItem *adjItem = adjMessage->add_adjs();
        adjItem->set_vid(id_neibors.first);
        for (auto neibor:id_neibors.second) {
            adjItem->add_neibors(neibor);
        }
    }
    reply->set_allocated_adjlist(adjMessage);


    return Status::OK;

//     vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
}

GeneralPartition generalPartition;

Status ServiceImpl::pullDataFromMasterGeneral(
        ServerContext *context, const ContextMessage *request,
        DataMessage *reply) {

    cout << "worker " << request->workerid() << " has arrived!" << endl;
    int workerid = request->workerid();
    int workerNum = request->workernum();
    // 除了worker 0以外，其他所有线程都进行等待，worker 0进行分区，分区完成后notify其他进程
    if (request->workerid() == 0) {
        unique_lock<mutex> lck(ThreadUtil::mtx);
        // 开始进行分区
        Check::check_partition_pass(
                request->workernum(),
                request->partition().datanum(),
                request->partition().datapath(),
                request->partition().featuredim(),
                request->partition().classnum());
        //int data_num, int worker_num, string filename, int feature_size, int label_size;

        generalPartition.init(
                request->partition().datanum(),
                request->workernum(),
                request->partition().datapath(),
                request->partition().featuredim(),
                request->partition().classnum());

        generalPartition.startPartition(workerNum, request->partition().partitionmethod(),
                                        request->partition().datanum(), request->partition().edgenum());
        ThreadUtil::ready = true;
        ThreadUtil::cv.notify_all();


    } else {
        unique_lock<mutex> lck(ThreadUtil::mtx);
        // 进入等待
        while (!ThreadUtil::ready) {
            ThreadUtil::cv.wait(lck);
        }
    }


    // 开始返回每个worker的数据
    // 构建nodes
    NodeMessage *nodeMessage = reply->nodelist().New();
    for (int id:GeneralPartition::nodes[workerid]) {
        nodeMessage->add_nodes(id);
    }
    reply->set_allocated_nodelist(nodeMessage);

    // 构建feature
    DataMessage_FeatureMessage *featureMessage = reply->featurelist().New();
    for (auto &id_feature : GeneralPartition::features[workerid]) {
        DataMessage_FeatureMessage_FeatureItem *item = featureMessage->add_features();
        item->set_vid(id_feature.first);
        for (float feat_dim:id_feature.second) {
            item->add_feature(feat_dim);
        }
    }
    reply->set_allocated_featurelist(featureMessage);

    // 构建label
    DataMessage_LabelMessage *labelMessage = reply->labellist().New();
    for (auto &id_label:GeneralPartition::labels[workerid]) {
        DataMessage_LabelMessage_LabelItem *item = labelMessage->add_labels();
        item->set_vid(id_label.first);
        item->set_label(id_label.second);
    }
    reply->set_allocated_labellist(labelMessage);

    // 构建adjs
    DataMessage_AdjMessage *adjMessage = reply->adjlist().New();
    for (const auto &id_neibors:GeneralPartition::adjs[workerid]) {
        DataMessage_AdjMessage_AdjItem *adjItem = adjMessage->add_adjs();
        adjItem->set_vid(id_neibors.first);
        for (auto neibor:id_neibors.second) {
            adjItem->add_neibors(neibor);
        }
    }
    reply->set_allocated_adjlist(adjMessage);

    for (int i = 0; i < workerNum; i++) {
        auto &nodes_worker = GeneralPartition::nodes[i];
        auto *nodelist = reply->add_nodesforeachworker();
        int nodeNum = nodes_worker.size();
        for (int j = 0; j < nodeNum; j++) {
            nodelist->add_nodes(nodes_worker[j]);
        }
    }


    return Status::OK;

//     vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
}

Status ServiceImpl::freeMaster(ServerContext *context, const BoolMessage *request, BoolMessage *reply) {
    vector<vector<int>>().swap(GeneralPartition::nodes);
    vector<map<int, vector<float>>>().swap(GeneralPartition::features);
    vector<map<int, int>>().swap(GeneralPartition::labels);
    vector<map<int, set<int>>>().swap(GeneralPartition::adjs);
    return Status::OK;
}

static void const addaaa();

Status ServiceImpl::workerPullEmb(
        ServerContext *context, const EmbMessage *request, EmbMessage *reply) {
    // 这里请求的nodes的顺序和返回的tensor的顺序要保持一致
//    clock_t start = clock();
    string mode = "none"; // mom mv none

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

//    ReqEmbMessage *reqEmbMessage = reply->denseembmessage().New();
    // build needed embs
    int feat_size = WorkerStore::embs.begin()->second.size();
    auto &embs_ws = WorkerStore::embs;
    map<int, vector<float>> embs;
    int layerId = request->layerid();
    int epoch = request->epoch();
    int workerId = request->workerid();


    int nodeNum = request->nodes_size();
    reply->set_shapedim(feat_size);
    reply->set_resp_node_size(nodeNum);
    reply->set_resp_featdim_size(feat_size);


    auto *mutable_emb_reply = reply->mutable_resp_none_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * feat_size);

    for (int i = 0; i < nodeNum; i++) {
        int id = request->nodes(i);
        auto &emb_nodeid = embs_ws[id];


        mutable_emb_reply->Add(emb_nodeid.begin(), emb_nodeid.end());
//        for (int j = 0; j < feat_size; j++) {
//            reply->add_resp_none_compress_emb_concat(emb_nodeid[j]);
//
//        }
    }


//    for (int i = 0; i < nodeNum; i++) {
//        TensorMessage *tensor_temp = reqEmbMessage->add_embs();
//        int id = request->nodes(i);
//        auto &emb_nodeid = embs_ws[id];
//
//        for (int j = 0; j < feat_size; j++) {
//            tensor_temp->add_tensor(emb_nodeid[j]);
//
//        }
//
//    }



//    reply->set_allocated_denseembmessage(reqEmbMessage);
    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
//    cout << "server processing emb time:" << timeuse << endl;


    return Status::OK;
}


Status ServiceImpl::workerPullG(
        ServerContext *context, const EmbMessage *request, EmbMessage *reply) {
    // 这里请求的nodes的顺序和返回的tensor的顺序要保持一致
//    cout<<"***********G_map size:"<<WorkerStore::G_map.size()<<",G2 size:"<<WorkerStore::G_map[2].size()<<endl;
    auto &G = WorkerStore::G_map[request->layerid()];
//    map<int, vector<float>> G;
    int featSize = G.begin()->second.size();
    int nodeNum = request->nodes_size();

//    for (int i = 0; i < nodeNum; i++) {
//        vector<float> vec(dimNum);
//        int id = request->nodes(i);
//        auto &G_layerId_nodeId = G_layerId[id];
//        for (int j = 0; j < dimNum; j++) {
//            vec[j] = G_layerId_nodeId[j];
//        }
//        G.insert(pair<int, vector<float>>(id, vec));
//    }

    for (int i = 0; i < nodeNum; i++) {
//        TensorMessage *tensor_temp = reply->add_embs();
        int id = request->nodes(i);
//        tensor_temp->set_vid(id);
        auto &G_layerId_nodeId = G[id];
        reply->mutable_resp_none_compress_emb_concat()->Add(G_layerId_nodeId.begin(), G_layerId_nodeId.end());
//        for (int j = 0; j < dimNum; j++) {
//            tensor_temp->add_tensor(G_layerId_nodeId[j]);
//        }
    }

    reply->set_resp_featdim_size(featSize);
    reply->set_resp_node_size(nodeNum);

    return Status::OK;
}


int getDimBucket(const vector<Bucket> &buckets, float dim, float min_value, float max_value, float interval) {

//    for (const auto &bucket:buckets) {
//        auto lower_bound=bucket.lower_bound;
//        auto upper_bound=bucket.upper_bound;
//
//        if (dim >= lower_bound && dim <= upper_bound) {
//            return bucket.bid;
//        }
//    }

    int bucketid;
    if (dim == 0) {

        return buckets.size() - 1;
    } else {
        if (min_value < 0 && max_value > 0) {
            if (dim > 0) {
                bucketid = int((dim - min_value) / interval + 1);
                if (buckets[bucketid].value > 0) {
//                    cout<<"dim1111111111:"<<dim<<",bucketid:"<<bucketid<<endl;
                    return bucketid;
                } else {
//                    cout<<"dim22222222:"<<dim<<",bucketid:"<<bucketid<<endl;
                    if(bucketid+1>=16){
                        cout<<"errrrrrrrrrrrrrrrr"<<endl;
                    }
                    return bucketid + 1;
                }
            } else {
                return int((dim - min_value) / interval);
            }
        } else {
            if (dim < 0) {
                cout << "ssssssssssssssss" << dim << endl;
            }
            return int((dim - min_value) / interval);
        }
    }

    cout << "getDimBucket error" << endl;
    return -1;
}


int getDimBucket_emb(const vector<Bucket> &buckets, float dim, float min_value, float max_value, float interval) {

//    for (const auto &bucket:buckets) {
//        auto lower_bound=bucket.lower_bound;
//        auto upper_bound=bucket.upper_bound;
//
//        if (dim >= lower_bound && dim <= upper_bound) {
//            return bucket.bid;
//        }
//    }

    int bucketid;
    if (dim == 0) {

        return buckets.size() - 1;
    } else {
        if (min_value < 0 && max_value > 0) {
            if (dim > 0) {
                bucketid = int((dim - min_value) / interval + 1);
                if (buckets[bucketid].value > 0) {
//                    cout<<"dim1111111111:"<<dim<<",bucketid:"<<bucketid<<endl;
                    return bucketid;
                } else {
//                    cout<<"dim22222222:"<<dim<<",bucketid:"<<bucketid<<endl;
                    return bucketid + 1;
                }
            } else {
                return int((dim - min_value) / interval);
            }
        } else {
            return int((dim - min_value) / interval);
        }
    }

    cout << "getDimBucket error" << endl;
    return -1;
}

//uint oneByteCompress(vector<uint> fourItemsVec) {
//
//    fourItemsVec[0] = fourItemsVec[0] << 24;
//    fourItemsVec[1] = fourItemsVec[1] << 16;
//    fourItemsVec[2] = fourItemsVec[2] << 8;
////                        fourItemsVec[3]=fourItemsVec[3];
//    uint compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
//    return compress_value;
//}


//Status ServiceImpl::workerPullEmbCompress_iter(ServerContext *context, const ReqEmbSparseMessage *request,
//                                               ReqEmbSparseMessage *reply) {
//    int bucket_num = request->bucketnum();
//    bool ifCompensate = request->ifcompensate();
//    int layerId = request->layerid();
//    int epoch = request->iterround();
//    // 先判断是否需要补偿
//    map<int, vector<float>> embs;
//    float max_value = -10000;
//    float min_value = 10000;
////    cout<<"epoch "<<epoch <<", layer id "<<layerId<< "bucket num:"<<bucket_num<< ", ifCompensate:"<<ifCompensate<<endl;
////    cout<<"WorkerStore::G_compensate size:"<< WorkerStore::G_compensate[layerId].size()<<"*"<<WorkerStore::G_compensate[layerId].begin()->second.size()<<endl;
////    cout<<"WorkerStore::G_map size:"<< WorkerStore::G_map[layerId].size()<<"*"<<WorkerStore::G_map[layerId].begin()->second.size()<<endl;
//
//    auto &emb_layerId = WorkerStore::embs;
//    auto &emb_compensate_layerId = WorkerStore::embs_compensate[layerId];
//    if (ifCompensate) {
//        if (epoch == 0) {
//            for (auto id:request->nodes()) {
//                vector<float> vec;
//                auto &emb_layerId_nodeId = emb_layerId[id];
//                for (auto feat_dim:emb_layerId_nodeId) {
//                    vec.push_back(feat_dim);
//                    // 求整个返回矩阵的元素的最大值和最小值
//                    if (feat_dim > max_value) {
//                        max_value = feat_dim;
//                    }
//                    if (feat_dim < min_value) {
//                        min_value = feat_dim;
//                    }
//                }
//                embs.insert(pair < int, vector < float >> (id, vec));
//            }
//        } else {
//            int feat_size = emb_layerId.begin()->second.size();
//            for (auto id:request->nodes()) {
//                vector<float> vec;
//                auto &emb_layerId_nodeId = emb_layerId[id];
//                auto &emb_compensate_layerId_nodeId = emb_compensate_layerId[id];
//                for (int i = 0; i < feat_size; i++) {
////                        cout<<"id:"<<layerId<<","<<id<<","<<i<<","<<"embs:"<<WorkerStore::embs[id][i]<<","<<"error compensate:"<<WorkerStore::embs_compensate[layerId][id][i]<<endl;
////                        float feat_dim=WorkerStore::embs[id][i]+WorkerStore::embs_compensate[layerId][id][i];
//                    float feat_dim = emb_layerId_nodeId[i];
////                    if(layerId==0){
//                    feat_dim = emb_layerId_nodeId[i] + emb_compensate_layerId_nodeId[i];
////                    }
//
////                        float feat_dim=WorkerStore::embs[id][i];
//                    vec.push_back(feat_dim);
//                    // 求整个返回矩阵的元素的最大值和最小值
//                    if (feat_dim > max_value) {
//                        max_value = feat_dim;
//                    }
//                    if (feat_dim < min_value) {
//                        min_value = feat_dim;
//                    }
//                }
//
//                embs.insert(pair < int, vector < float >> (id, vec));
//            }
//        }
//    } else {
//        for (auto id:request->nodes()) {
//            vector<float> vec;
//            auto &emb_layerId_nodeId = emb_layerId[id];
//            for (auto feat_dim:emb_layerId_nodeId) {
//                vec.push_back(feat_dim);
//                // 求整个返回矩阵的元素的最大值和最小值
//                if (feat_dim > max_value) {
//                    max_value = feat_dim;
//                }
//                if (feat_dim < min_value) {
//                    min_value = feat_dim;
//                }
//            }
//
//            embs.insert(pair < int, vector < float >> (id, vec));
//        }
//    }
//
//
//    // 上面是用上一轮补偿了这一轮
//    // 下面是计算压缩和发送，以及发送误差
//    vector <Bucket> buckets;
//    float interval = (max_value - min_value) / (float) (bucket_num);
//    if (min_value < 0 && max_value > 0) {
//        for (int i = 0; i < bucket_num + 1; i++) {
//            if (min_value + interval * i < 0 && min_value + interval * (i + 1) > 0) {
//                // 建两个桶,以0的分界线
//                Bucket b1;
//                b1.bid = i;
//                b1.lower_bound = min_value + interval * i;
//                b1.upper_bound = 0;
//                b1.value = (b1.lower_bound + b1.upper_bound) / 2;
//                buckets.push_back(b1);
//                reply->add_values((b1.lower_bound + b1.upper_bound) / 2);
//
//                i = i + 1;
//                Bucket b2;
//                b2.bid = i;
//                b2.lower_bound = 0;
//                b2.upper_bound = min_value + interval * (i + 1);
//                if (i == bucket_num) {
//                    b2.upper_bound = max_value;
//                }
//                b2.value = (b2.lower_bound + b2.upper_bound) / 2;
//                buckets.push_back(b2);
//                reply->add_values((b2.lower_bound + b2.upper_bound) / 2);
//            } else {
//                Bucket b;
//                b.bid = i;
//                b.lower_bound = min_value + interval * i;
//                b.upper_bound = min_value + interval * (i + 1);
//                if (i == bucket_num - 1) {
//                    b.upper_bound = max_value;
//                }
//                b.value = (b.lower_bound + b.upper_bound) / 2;
//                buckets.push_back(b);
//                reply->add_values((b.lower_bound + b.upper_bound) / 2);
//            }
//        }
//    } else {
//        for (int i = 0; i < bucket_num; i++) {
//            Bucket b;
//            b.bid = i;
////            cout<< "bid:" << b.bid<<endl;
//            b.lower_bound = min_value + interval * i;
//            b.upper_bound = min_value + interval * (i + 1);
//            if (i == bucket_num - 1) {
//                b.upper_bound = max_value;
//            }
//            b.value = (b.lower_bound + b.upper_bound) / 2;
//            buckets.push_back(b);
//            reply->add_values((b.lower_bound + b.upper_bound) / 2);
//        }
//    }
//
//    Bucket b;
//    b.bid = buckets.size();
//    b.lower_bound = 0;
//    b.upper_bound = 0;
//    b.value = 0;
//    buckets.push_back(b);
//    reply->add_values(0);
//
//    vector <uint> fourItemsVec;
//    if (ifCompensate) {
//        if (epoch == 0) {
//            // 第0轮迭代需要新建误差结构
//            for (const auto &emb:embs) {
//                IntTensorMessage *tensor = reply->add_embs();
//                vector<float> error;
//                tensor->set_vid(emb.first);
//                for (auto dim:emb.second) {
//                    int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                    tensor->add_tensor(bucket_id);
//                    fourItemsVec.push_back(bucket_id);
//                    if (fourItemsVec.size() == 4) {
//                        // compress
//                        fourItemsVec[0] = fourItemsVec[0] << 24;
//                        fourItemsVec[1] = fourItemsVec[1] << 16;
//                        fourItemsVec[2] = fourItemsVec[2] << 8;
//                        int compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
//                        tensor->add_tensor(compress_value);
//                        fourItemsVec.clear();
//                    }
//                    error.push_back(dim - buckets[bucket_id].value);
//                }
//                if (fourItemsVec.size() != 0) {
//                    int compress_value = 0;
//                    for (int i = 0; i < fourItemsVec.size(); i++) {
//                        fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
//                        compress_value = compress_value | fourItemsVec[i];
//                    }
//
//                    tensor->add_tensor(compress_value);
//                    fourItemsVec.clear();
//                }
//                WorkerStore::embs_compensate[layerId].insert(pair < int, vector < float >> (emb.first, error));
//            }
//        } else {
//            // 需要先加误差，然后算出在哪个桶中，再算误差
//
////            if(WorkerStore::G_compensate.count(layerId)==0){
////                map<int,vector<float>> map_tmp;
////                WorkerStore::G_compensate.insert(pair<int,map<int,vector<float>>>(layerId,map_tmp));
////            }
////            auto G_compensate_layerId_tmp=WorkerStore::G_compensate[layerId];
//            vector <uint> fourItemsVec;
//            for (const auto &emb:embs) {
//                IntTensorMessage *tensor = reply->add_embs();
//                tensor->set_vid(emb.first);
//                auto &emb_compensate_layerId_nodeId = emb_compensate_layerId[emb.first];
//                for (int i = 0; i < emb.second.size(); i++) {
//                    // 先加误差
//                    float dim = emb.second[i];
//                    int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                    tensor->add_tensor(bucket_id);
//                    fourItemsVec.push_back(bucket_id);
//                    if (fourItemsVec.size() == 4) {
//                        // compress
//                        fourItemsVec[0] = fourItemsVec[0] << 24;
//                        fourItemsVec[1] = fourItemsVec[1] << 16;
//                        fourItemsVec[2] = fourItemsVec[2] << 8;
//                        int compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
//                        tensor->add_tensor(compress_value);
//                        fourItemsVec.clear();
//                    }
//                    emb_compensate_layerId_nodeId[i] = (dim - buckets[bucket_id].value);
//                }
//                if (fourItemsVec.size() != 0) {
//                    int compress_value = 0;
//                    for (int i = 0; i < fourItemsVec.size(); i++) {
//                        fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
//                        compress_value = compress_value | fourItemsVec[i];
//                    }
//
//                    tensor->add_tensor(compress_value);
//                    fourItemsVec.clear();
//                }
//            }
//        }
//    } else {
//        vector <uint> fourItemsVec;
//        // 开始构建压缩后的张量
//        for (const auto &emb:embs) {
//            IntTensorMessage *tensor = reply->add_embs();
//            tensor->set_vid(emb.first);
//            for (auto dim:emb.second) {
//                int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                tensor->add_tensor(bucket_id);
//                fourItemsVec.push_back(bucket_id);
//                if (fourItemsVec.size() == 4) {
//                    // compress
//                    fourItemsVec[0] = fourItemsVec[0] << 24;
//                    fourItemsVec[1] = fourItemsVec[1] << 16;
//                    fourItemsVec[2] = fourItemsVec[2] << 8;
//                    int compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
//                    tensor->add_tensor(compress_value);
//                    fourItemsVec.clear();
//                }
//            }
//            if (fourItemsVec.size() != 0) {
//                int compress_value = 0;
//                for (int i = 0; i < fourItemsVec.size(); i++) {
//                    fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
//                    compress_value = compress_value | fourItemsVec[i];
//                }
//
//                tensor->add_tensor(compress_value);
//                fourItemsVec.clear();
//            }
//        }
//    }
//
//    reply->set_shapedim(emb_layerId.begin()->second.size());
////    cout << "G_layerId.begin()->second.size()" << reply->shapedim() << endl;
//
//    return Status::OK;
//
//
//}


struct CompensateArgs {
    const EmbMessage *request{};
    int epoch{};
    map<int, vector<float>> embs;
    EmbMessage *reply{};
    vector<Bucket> buckets;
    int layerId{};
    int changeToIter{};
    int workerId{};
    int layerNum{};
    float min_value{};
    float max_value{};
    float interval{};
};

struct LayerWorkerId {
    int layerId;
    int workerId;
};

void *layerErrorCompute(void *layerWorkerId_void) {
    // 补偿做神经网络计算
    LayerWorkerId lwid = *(LayerWorkerId *) layerWorkerId_void;
    int layerId = lwid.layerId;
    int workerId = lwid.workerId;
    map<int, vector<float>> error_tmp;
    auto &embs_compensate_layerId1 = WorkerStore::embs_compensate[layerId];
    int feat_num = embs_compensate_layerId1.begin()->second.size();
    int weight_size = WorkerStore::weights[layerId].begin()->size();
    auto &weights_layerId = WorkerStore::weights[layerId];
    for (const auto &err_pair: embs_compensate_layerId1) {
        int vid = err_pair.first;
//                        vector<float> vec_mm_weight(WorkerStore::weights[layerId+1].begin()->size(),0);
        vector<float> vec_mm_weight(weight_size, 0);
        for (int i = 0; i < feat_num; i++) {
            float dim = err_pair.second[i];
            auto &weights_layerId_i = weights_layerId[i];
            for (int j = 0; j < vec_mm_weight.size(); j++) {
//                                vec_mm_weight[j]+=dim*WorkerStore::weights[layerId+1][i][j];
                vec_mm_weight[j] += dim * weights_layerId_i[j];
            }
        }
        error_tmp.insert(pair<int, vector<float >>(vid, vec_mm_weight));
    }
    WorkerStore::embs_compensate[layerId] = error_tmp;
    WorkerStore::compFlag[workerId] = true;
}

void *mix3ErrorCompute(void *workerLayerId_void) {
    LayerWorkerId layerWorkerId = *(LayerWorkerId *) workerLayerId_void;
    int layerId = layerWorkerId.layerId;
    int workerId = layerWorkerId.workerId;
    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
    // 补偿做神经网络计算
    map<int, vector<float>> error_tmp;
    int size = embs_compensate_layerId.begin()->second.size();
    int weight_size = WorkerStore::weights[layerId].begin()->size();
    auto &weights_layerId = WorkerStore::weights[layerId];
    for (const auto &err_pair:embs_compensate_layerId) {
        int vid = err_pair.first;
        vector<float> vec_mm_weight(weight_size, 0);
        for (int i = 0; i < size; i++) {
            float dim = err_pair.second[i];
            auto &weights_layerId_nodeId = weights_layerId[i];
            for (int j = 0; j < vec_mm_weight.size(); j++) {
                vec_mm_weight[j] += dim * weights_layerId_nodeId[j];
            }
        }
        error_tmp.insert(pair<int, vector<float >>(vid, vec_mm_weight));
    }
    WorkerStore::embs_compensate[layerId] = error_tmp;
    WorkerStore::compFlag[workerId] = true;
}


void error_dir(const map<int, vector<float>> &embs, EmbMessage *reply,
               vector<Bucket> &buckets, float min_value, float max_value, float interval, int layerId) {
    vector<uint> fourItemsVec;
//    cout<<"embs compensate size:"<<WorkerStore::embs_compensate[layerId].size()<<endl;
    if (WorkerStore::embs_compensate.count(layerId) == 0) {
        map<int, vector<float>> map_tmp;
        WorkerStore::embs_compensate.insert(pair<int, map<int, vector<float>>>(layerId, map_tmp));
    }
    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
    cout << "layerid:" << layerId << endl;

    for (const auto &emb:embs) {
        IntTensorMessage *tensor = reply->add_embs();
        vector<float> error;
        tensor->set_vid(emb.first);
//        auto embs_compensate_layerId_nodeId=embs_compensate_layerId[emb.first];
        for (const auto &dim:emb.second) {
            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
            fourItemsVec.push_back(bucket_id);
            if (fourItemsVec.size() == 4) {
                // compress
                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                tensor->add_tensor(compress_value);
                fourItemsVec.clear();
            }
//            cout << dim - buckets[bucket_id].value << "," << dim << "," << buckets[bucket_id].value << " ";
            error.push_back(dim - buckets[bucket_id].value);
        }
//        cout << " " << endl;
        if (fourItemsVec.size() != 0) {
            int compress_value = 0;
            for (int i = 0; i < fourItemsVec.size(); i++) {
                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                compress_value = compress_value | fourItemsVec[i];
            }

            tensor->add_tensor(compress_value);
            fourItemsVec.clear();

        }

        if (embs_compensate_layerId.count(emb.first) == 0) {
            embs_compensate_layerId.insert(pair<int, vector<float >>(emb.first, error));
        } else {
            embs_compensate_layerId[emb.first] = error;
        }

    }


    // 输出按轮补偿误差
//    cout<<"layerid:"<< layerId<<endl;

//    for(auto p:embs_compensate_layerId){
//        for(int j=0;j<embs_compensate_layerId.begin()->second.size();j++){
//            cout<<p.second[j]<<" ";
//        }
//        cout<<" "<<endl;
//    }

//    cout<<"after embs compensate size:"<<WorkerStore::embs_compensate[layerId].size()<<endl;
}


void error_dir_hasid(const map<int, vector<float>> &embs, RespEmbSparseMessage *reply,
                     vector<Bucket> &buckets, float min_value, float max_value, float interval, int layerId) {
    vector<uint> fourItemsVec;
//    cout<<"embs compensate size:"<<WorkerStore::embs_compensate[layerId].size()<<endl;
    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
    int feat_num = embs.begin()->second.size();
    for (const auto &emb:embs) {
        IntTensorMessage *tensor = reply->add_embs();
        int nodeId = emb.first;
        auto &embs_compensate_layerId_nodeid = embs_compensate_layerId[nodeId];
        tensor->set_vid(nodeId);
        auto &emb_second = emb.second;
//        auto embs_compensate_layerId_nodeId=embs_compensate_layerId[emb.first];
        for (int i = 0; i < feat_num; i++) {
            int bucket_id = getDimBucket(buckets, emb_second[i], min_value, max_value, interval);
            fourItemsVec.push_back(bucket_id);
            if (fourItemsVec.size() == 4) {
                // compress
                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                tensor->add_tensor(compress_value);
                fourItemsVec.clear();
            }

            embs_compensate_layerId_nodeid[i] = (emb_second[i] - buckets[bucket_id].value);

//            if (layerId == 1 && nodeId == 3 && i == 5) {
//                cout << "compensate value:" << emb_second[i] << ",compress value:" << buckets[bucket_id].value
//                     << ",bucket id:" << bucket_id
//                     << ",error:" << embs_compensate_layerId_nodeid[i] << ",bucket+1+2:"<<buckets[bucket_id+1].value<<","<<buckets[bucket_id+2].value<<endl;
//            }

        }

        if (fourItemsVec.size() != 0) {
            int compress_value = 0;
            for (int i = 0; i < fourItemsVec.size(); i++) {
                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                compress_value = compress_value | fourItemsVec[i];
            }

            tensor->add_tensor(compress_value);
            fourItemsVec.clear();

        }
    }



//    cout<<"after embs compensate size:"<<WorkerStore::embs_compensate[layerId].size()<<endl;
}

void no_error(const map<int, vector<float>> &embs, EmbMessage *reply) {
    ReqEmbMessage *reqEmbSparseMessage = reply->denseembmessage().New();
    for (const auto &emb:embs) {
        TensorMessage *tensor_tmp = reqEmbSparseMessage->add_embs();
        tensor_tmp->set_vid(emb.first);
        for (auto dim:emb.second) {
            tensor_tmp->add_tensor(dim);
        }
    }
    reply->set_allocated_denseembmessage(reqEmbSparseMessage);
}


void error_layer(const map<int, vector<float>> &embs, EmbMessage *reply,
                 vector<Bucket> &buckets, float min_value, float max_value, float interval, int layerId, int workerId) {
    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
    vector<uint> fourItemsVec;
    for (const auto &emb:embs) {
        IntTensorMessage *tensor = reply->add_embs();
        vector<float> error;
        tensor->set_vid(emb.first);

        for (auto dim:emb.second) {
            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
            fourItemsVec.push_back(bucket_id);
            if (fourItemsVec.size() == 4) {
                // compress
                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                tensor->add_tensor(compress_value);
                fourItemsVec.clear();
            }
//                    tensor->add_tensor(bucket_id);
            error.push_back(dim - buckets[bucket_id].value);
        }
        if (fourItemsVec.size() != 0) {
            int compress_value = 0;
            for (int i = 0; i < fourItemsVec.size(); i++) {
                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                compress_value = compress_value | fourItemsVec[i];
            }

            tensor->add_tensor(compress_value);
            fourItemsVec.clear();
        }
        // map可以直接插入0,3这样的值，vector实现定义长度，才能用[]取值，否则需要push加入值才能[]取值
        embs_compensate_layerId[emb.first] = error;
    }
    pthread_t thread;
    auto *layerWorkerId = new LayerWorkerId;
    layerWorkerId->workerId = workerId;
    layerWorkerId->layerId = layerId;
    pthread_create(&thread, NULL, layerErrorCompute, (void *) layerWorkerId);
}


int oneByteCompress_int(vector<int> fourItemsVec) {

    fourItemsVec[0] = fourItemsVec[0] << 24;
    fourItemsVec[1] = fourItemsVec[1] << 16;
    fourItemsVec[2] = fourItemsVec[2] << 8;
//                        fourItemsVec[3]=fourItemsVec[3];
    int compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
    return compress_value;
}

//void compensateMethodForEmb(const ReqEmbSparseMessage* request,int epoch,map<int, vector<float>> embs,
//                             ReqEmbSparseMessage* reply,vector<Bucket> buckets,int layerId,int changeToIter,int workerId) {
void compensateMethodForEmb(void *compArgs_void) {
    auto compArgs = (CompensateArgs *) compArgs_void;

    const EmbMessage *request = compArgs->request;
    int epoch = compArgs->epoch;
    map<int, vector<float>> &embs = compArgs->embs;
    EmbMessage *reply = compArgs->reply;
    vector<Bucket> &buckets = compArgs->buckets;
    int layerId = compArgs->layerId;
    int changeToIter = compArgs->changeToIter;
    int workerId = compArgs->workerId;
    int layerNum = compArgs->layerNum;
    float max_value = compArgs->max_value;
    float min_value = compArgs->min_value;
    float interval = compArgs->interval;

    vector<uint> fourItemsVec;
    if (request->compensatemethod() == "accorIter") {

        error_dir(embs, reply, buckets, min_value, max_value, interval, layerId);


    } else if (request->compensatemethod() == "accorLayer") {
        // 最后一层，发送密集数据，而且不需要再计算误差
        if (layerId == (WorkerStore::layer_num - 1)) {
            no_error(embs, reply);
        } else {
            // 无论需不需要补偿，都在函数最开始加到了embs里，这里是需要算误差和结果返回的
            // 非最后一层，需要算误差（包含乘以神经网络）
            error_layer(embs, reply, buckets, min_value, max_value, interval, layerId, workerId);

        }
    } else if (request->compensatemethod() == "accorMix") {
        // 每一层都要计算误差，前k-1层的误差需要乘以神经网络，最后一层的误差不用乘以神经网络
//                cout<<"test4"<<endl;
        if (layerId == WorkerStore::layer_num - 1) {
            no_error(embs, reply);
        } else {
            error_layer(embs, reply, buckets, min_value, max_value, interval, layerId, workerId);
        }
    } else if (request->compensatemethod() == "accorMix2") {

        // 最后一层需要传送完整的，其他层压缩的
        if (layerId == (WorkerStore::layer_num - 1)) {
            // 第K-1层只需要补偿，传递完整梯度，不需要计算误差
            // 误差在函数最开始已经加进去了
            no_error(embs, reply);
        } else {
            // 不是最后一层，误差 compute as Iter Compensate method
            error_dir(embs, reply, buckets, min_value, max_value, interval, layerId);

        }
    } else if (request->compensatemethod() == "accorMix3") {
        vector<uint> fourItemsVec;
        if (layerNum == 2) {
            if (epoch < changeToIter) {
                // 每一层都要计算误差，前k-1层的误差需要乘以神经网络，最后一层的误差不用乘以神经网络
                if (layerId == WorkerStore::layer_num - 1) {

                    // 第0轮迭代需要新建误差结构
                    error_dir(embs, reply, buckets, min_value, max_value, interval, layerId);

//                    WorkerStore::compFlag[workerId] = true;
                } else {
                    // 非最后一层误差需要乘以神经网络
                    error_layer(embs, reply, buckets, min_value, max_value, interval, layerId, workerId);
                }
            } else {
                // 改成Iter补偿
                if (epoch == changeToIter && layerId == 0) {
                    WorkerStore::embs_compensate[layerId].clear();
                    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
                    // 第0轮迭代需要新建误差结构
                    for (const auto &emb:embs) {
                        IntTensorMessage *tensor = reply->add_embs();
                        vector<float> error;
                        tensor->set_vid(emb.first);
                        for (auto dim:emb.second) {
                            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
//                            tensor->add_tensor(bucket_id);
                            fourItemsVec.push_back(bucket_id);
                            if (fourItemsVec.size() == 4) {
                                // compress
                                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
                            error.push_back(dim - buckets[bucket_id].value);
                        }
                        if (fourItemsVec.size() != 0) {
                            int compress_value = 0;
                            for (int i = 0; i < fourItemsVec.size(); i++) {
                                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                compress_value = compress_value | fourItemsVec[i];
                            }

                            tensor->add_tensor(compress_value);
                            fourItemsVec.clear();
                        }
                        embs_compensate_layerId.insert(pair<int, vector<float >>(emb.first, error));
//                        WorkerStore::compFlag[workerId] = true;

                    }
                } else {
                    // 需要先加误差，然后算出在哪个桶中，再算误差
                    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
                    for (const auto &emb:embs) {
                        IntTensorMessage *tensor = reply->add_embs();
                        tensor->set_vid(emb.first);
                        auto embs_compensate_layerId_nodeId = embs_compensate_layerId[emb.first];
                        for (int i = 0; i < emb.second.size(); i++) {
                            // 先加误差
                            float dim = emb.second[i];
                            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
//                            tensor->add_tensor(bucket_id);
                            fourItemsVec.push_back(bucket_id);
                            if (fourItemsVec.size() == 4) {
                                // compress
                                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
                            embs_compensate_layerId_nodeId[i] = (dim - buckets[bucket_id].value);
//                            WorkerStore::embs_compensate[layerId][emb.first][i]=0;
                        }
                        if (fourItemsVec.size() != 0) {
                            int compress_value = 0;
                            for (int i = 0; i < fourItemsVec.size(); i++) {
                                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                compress_value = compress_value | fourItemsVec[i];
                            }

                            tensor->add_tensor(compress_value);
                            fourItemsVec.clear();
                        }
                    }
//                    WorkerStore::compFlag[workerId] = true;
                }
            }
        } else if (layerNum == 3) {
            // 这块要添加返回的emb以及计算误差
            if (epoch < changeToIter) {
                // 第2层需要计算iter误差，第0层和第1层需要计算layer误差
                if (layerId == 2) {
                    if (epoch == 0) {
                        // 第0轮迭代需要新建误差结构
                        for (const auto &emb:embs) {
                            IntTensorMessage *tensor = reply->add_embs();
                            vector<float> error;
                            tensor->set_vid(emb.first);
                            for (auto dim:emb.second) {
                                int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
//                                tensor->add_tensor(bucket_id);
                                fourItemsVec.push_back(bucket_id);
                                if (fourItemsVec.size() == 4) {
                                    uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                    tensor->add_tensor(compress_value);
                                    fourItemsVec.clear();
                                }
                                error.push_back(dim - buckets[bucket_id].value);
                            }
                            if (fourItemsVec.size() != 0) {
                                int compress_value = 0;
                                for (int i = 0; i < fourItemsVec.size(); i++) {
                                    fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                    compress_value = compress_value | fourItemsVec[i];
                                }

                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
                            WorkerStore::embs_compensate[layerId].insert(pair<int,
                                    vector<float >>(emb.first, error));

                        }
                    } else {
                        // 需要先加误差，然后算出在哪个桶中，再算误差
                        int feat_num = embs.begin()->second.size();
                        auto &compensate_layerId = WorkerStore::embs_compensate[layerId];
                        for (const auto &emb:embs) {
                            IntTensorMessage *tensor = reply->add_embs();
                            tensor->set_vid(emb.first);
                            auto &compensate_layerId_node = compensate_layerId[emb.first];
                            for (int i = 0; i < feat_num; i++) {
                                // 先加误差
                                float dim = emb.second[i];
                                int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
//                                tensor->add_tensor(bucket_id);
                                fourItemsVec.push_back(bucket_id);
                                if (fourItemsVec.size() == 4) {
                                    // compress
                                    uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                    tensor->add_tensor(compress_value);
                                    fourItemsVec.clear();
                                }
                                compensate_layerId_node[i] = (dim - buckets[bucket_id].value);
//                            WorkerStore::embs_compensate[layerId][emb.first][i]=0;
                            }
                            if (fourItemsVec.size() != 0) {
                                int compress_value = 0;
                                for (int i = 0; i < fourItemsVec.size(); i++) {
                                    fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                    compress_value = compress_value | fourItemsVec[i];
                                }

                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
                        }

                    }
                } else {
                    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
                    for (const auto &emb:embs) {
                        IntTensorMessage *tensor = reply->add_embs();
                        vector<float> error;
                        tensor->set_vid(emb.first);

                        for (auto dim:emb.second) {
                            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
//                            tensor->add_tensor(bucket_id);
                            fourItemsVec.push_back(bucket_id);
                            if (fourItemsVec.size() == 4) {
                                // compress
                                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
                            error.push_back(dim - buckets[bucket_id].value);
                        }
                        if (fourItemsVec.size() != 0) {
                            int compress_value = 0;
                            for (int i = 0; i < fourItemsVec.size(); i++) {
                                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                compress_value = compress_value | fourItemsVec[i];
                            }

                            tensor->add_tensor(compress_value);
                            fourItemsVec.clear();
                        }
                        // map可以直接插入0,3这样的值，vector实现定义长度，才能用[]取值，否则需要push加入值才能[]取值
                        if (embs_compensate_layerId.count(emb.first) > 0) {
                            embs_compensate_layerId[emb.first] = error;
                        } else {
                            embs_compensate_layerId.insert(pair<int, vector<float >>(emb.first, error));
                        }

                    }
                    pthread_t thread;
                    auto *layerWorkerId = new LayerWorkerId;
                    layerWorkerId->workerId = workerId;
                    layerWorkerId->layerId = layerId;
                    pthread_create(&thread, NULL, layerErrorCompute, (void *) layerWorkerId);

                }
            } else if (epoch >= changeToIter && epoch < changeToIter * 2) {
                // 第1,2层需要计算iter误差，第0层需要计算layer误差
                if (layerId == 0) {
                    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId];
                    for (const auto &emb:embs) {
                        IntTensorMessage *tensor = reply->add_embs();
                        vector<float> error;
                        tensor->set_vid(emb.first);

                        for (auto dim:emb.second) {
                            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
//                            tensor->add_tensor(bucket_id);
                            fourItemsVec.push_back(bucket_id);
                            if (fourItemsVec.size() == 4) {
                                // compress
                                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
                            error.push_back(dim - buckets[bucket_id].value);
                        }
                        if (fourItemsVec.size() != 0) {
                            int compress_value = 0;
                            for (int i = 0; i < fourItemsVec.size(); i++) {
                                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                compress_value = compress_value | fourItemsVec[i];
                            }

                            tensor->add_tensor(compress_value);
                            fourItemsVec.clear();
                        }
                        // map可以直接插入0,3这样的值，vector实现定义长度，才能用[]取值，否则需要push加入值才能[]取值
                        embs_compensate_layerId[emb.first] = error;
                    }
                    pthread_t thread;
                    auto *layerWorkerId = new LayerWorkerId;
                    layerWorkerId->workerId = workerId;
                    layerWorkerId->layerId = layerId;
                    pthread_create(&thread, NULL, layerErrorCompute, (void *) layerWorkerId);
                } else {
                    // 需要先加误差，然后算出在哪个桶中，再算误差
                    int feat_num = embs.begin()->second.size();
                    auto &compensate_layerId = WorkerStore::embs_compensate[layerId];
                    for (const auto &emb:embs) {
                        IntTensorMessage *tensor = reply->add_embs();
                        tensor->set_vid(emb.first);

                        vector<float> error;
                        for (int i = 0; i < feat_num; i++) {
                            // 先加误差
                            float dim = emb.second[i];
                            int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
                            fourItemsVec.push_back(bucket_id);
                            if (fourItemsVec.size() == 4) {
                                // compress
                                uint compress_value = Compress::oneByteCompress(fourItemsVec);
                                tensor->add_tensor(compress_value);
                                fourItemsVec.clear();
                            }
//                            tensor->add_tensor(bucket_id);
                            error.push_back(dim - buckets[bucket_id].value);
                        }

                        if (fourItemsVec.size() != 0) {
                            int compress_value = 0;
                            for (int i = 0; i < fourItemsVec.size(); i++) {
                                fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                                compress_value = compress_value | fourItemsVec[i];
                            }

                            tensor->add_tensor(compress_value);
                            fourItemsVec.clear();
                        }
                        compensate_layerId[emb.first] = error;
                    }
//                    WorkerStore::compFlag[workerId]= true;
                }
            } else if (epoch >= changeToIter * 2) {
                // 所有层都计算iter误差
                int feat_num = embs.begin()->second.size();
                auto &compensate_layerId = WorkerStore::embs_compensate[layerId];
                for (const auto &emb:embs) {
                    IntTensorMessage *tensor = reply->add_embs();
                    tensor->set_vid(emb.first);
                    vector<float> error;
                    auto &compensate_layerId_nodeId = compensate_layerId[emb.first];
                    auto &emb_second = emb.second;
                    for (int i = 0; i < feat_num; i++) {
                        // 先加误差
                        float dim = emb_second[i];
                        int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
                        fourItemsVec.push_back(bucket_id);
                        if (fourItemsVec.size() == 4) {
                            // compress
                            uint compress_value = Compress::oneByteCompress(fourItemsVec);
                            tensor->add_tensor(compress_value);
                            fourItemsVec.clear();
                        }
//                        tensor->add_tensor(bucket_id);
                        error.push_back(dim - buckets[bucket_id].value);
                    }
                    if (fourItemsVec.size() != 0) {
                        int compress_value = 0;
                        for (int i = 0; i < fourItemsVec.size(); i++) {
                            fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
                            compress_value = compress_value | fourItemsVec[i];
                        }

                        tensor->add_tensor(compress_value);
                        fourItemsVec.clear();
                    }
                    compensate_layerId_nodeId = error;
                }
//                cout<<"epoch:"<<epoch<<",layerId:"<<layerId<<",size:"<<compensate_layerId.begin()->second.size()<<",feat_size:"<<feat_num<<endl;

            }
        }
    }

}

struct MinMaxS {
    float min_value = 10000;
    float max_value = -10000;
};

MinMaxS cpst_non(const EmbMessage &request, map<int, vector<float>> &embs, int layerId) {
    MinMaxS s;
    float min_value = 10000;
    float max_value = -10000;
//    auto& emb_mom=WorkerStore::embs_momument[layerId][]
    map<int, vector<float>> embs_mom_layer;
    map<int, vector<float>> embs_last_layer;
//    WorkerStore::embs_momument.insert(pair<int, map<int, vector<float>>>(layerId, embs_mom_layer));
//    WorkerStore::embs_last.insert(pair<int, map<int, vector<float>>>(layerId, embs_last_layer));
    int nodeNum = request.nodes_size();
    int dimNum = WorkerStore::embs.begin()->second.size();
    for (int i = 0; i < nodeNum; i++) {
        int id = request.nodes(i);
        vector<float> vec(dimNum);
//        vector<float> vec2;
//        vector<float> vec_zero;
        auto embs_id = WorkerStore::embs[id];
        for (int j = 0; j < dimNum; j++) {
            float feat_dim = embs_id[j];
            vec[j] = feat_dim;
//            vec2.push_back(feat_dim);
            // 求整个返回矩阵的元素的最大值和最小值
//            if (feat_dim != 0) {
            if (feat_dim > max_value) {
                max_value = feat_dim;
            }
            if (feat_dim < min_value) {
                min_value = feat_dim;
            }
//            }
//            vec_zero.push_back(0);
        }
        embs.insert(pair<int, vector<float >>(id, vec));
//        WorkerStore::embs_last[layerId].insert(pair<int, vector<float>>(id, vec2));
//        WorkerStore::embs_momument[layerId].insert(pair<int, vector<float>>(id, vec_zero));
    }
    s.min_value = min_value;
    s.max_value = max_value;
    if (min_value == 10000 || max_value == -10000) {
        cout << "error!!!!" << "min_value==10000 or max value==-10000" << endl;
    }
    return s;
}

MinMaxS cpst_iter(const EmbMessage &request, map<int, vector<float>> &embs, int layerId, int epoch) {
    MinMaxS s;
    float min_value = 10000;
    float max_value = -10000;
    int feat_size = WorkerStore::embs.begin()->second.size();
//    cout<<"layer id:"<<layerId<<",embs size:"<<WorkerStore::embs.size()<<",feat size:"<<feat_size<<endl;
//    cout << "a" << endl;
//
//    cout << "epoch:" << epoch << ",,," << (1.0f / float(epoch)) << endl;
//    cout << "b" << endl;
    for (auto id:request.nodes()) {
        vector<float> vec(WorkerStore::embs.begin()->second.size());
        auto &emb = WorkerStore::embs[id];
        if (WorkerStore::embs_compensate[layerId].count(id) == 0 || WorkerStore::embs_last[layerId].count(id) == 0) {
            cout << "WorkerStore::embs_compensate[layerId].count(id)==0||WorkerStore::embs_last[layerId].count(id)==0"
                 << "errrrrrrror!!" << endl;
        }
        auto &compensate = WorkerStore::embs_compensate[layerId][id];
        auto &emb_last = WorkerStore::embs_last[layerId][id];
//        auto &emb_mom = WorkerStore::embs_momument[layerId][id];
        for (int i = 0; i < feat_size; i++) {


            vec[i] = emb[i] + compensate[i];
//            vec[i] = emb[i];
            // 求整个返回矩阵的元素的最大值和最小值
            if (vec[i] > max_value) {
                max_value = vec[i];
            }
            if (vec[i] < min_value) {
                min_value = vec[i];
            }
        }

        embs.insert(pair<int, vector<float >>(id, vec));

    }

//    for(int i=0;i<embs.size();i++){
//        auto& embs_id=embs[i];
//        for(int j=0;j<embs_id.size();j++){
//            embs_id[j]=embs_id[j]/5;
//        }
//    }
    for (auto &emb:embs) {
        for (auto &dim:emb.second) {
            dim = dim / max_value;
        }
    }

//    if (layerId == 1 && WorkerStore::embs.count(3) != 0) {
//        cout << "true value:" << WorkerStore::embs[3][5] << "," << "compensate value:" << embs[3][5] << endl;
//    }
    s.min_value = min_value;
    s.max_value = max_value;
//    if(layerId==1){
//        cout<<"min,max:"<<min_value<<","<<max_value<<endl;
//    }
    return s;
}


MinMaxS cpst_layer(const EmbMessage &request, map<int, vector<float>> &embs, int layerId) {
    MinMaxS s;
    float min_value = 10000;
    float max_value = -10000;

    int feat_num = WorkerStore::embs.begin()->second.size();
    auto &embs_compensate_layerId = WorkerStore::embs_compensate[layerId - 1];
    for (auto id:request.nodes()) {
        vector<float> vec;
        auto &embs_nodeId = WorkerStore::embs[id];
        auto &embs_compensate_layerId_nodeId = embs_compensate_layerId[id];
        for (int i = 0; i < feat_num; i++) {
            float feat_dim = embs_nodeId[i] + embs_compensate_layerId_nodeId[i];
            vec.push_back(feat_dim);
            // 求整个返回矩阵的元素的最大值和最小值
            if (feat_dim > max_value) {
                max_value = feat_dim;
            }
            if (feat_dim < min_value) {
                min_value = feat_dim;
            }
        }
        embs.insert(pair<int, vector<float >>(id, vec));
    }

    s.min_value = min_value;
    s.max_value = max_value;
    return s;
}

MinMaxS cpst_mix(const EmbMessage &request, map<int, vector<float>> &embs, int layerId) {
    MinMaxS s;
    float min_value = 10000;
    float max_value = -10000;

    for (auto id:request.nodes()) {
        vector<float> vec;
        for (int i = 0; i < WorkerStore::embs[id].size(); i++) {
            float feat_dim =
                    WorkerStore::embs[id][i] + WorkerStore::embs_compensate[layerId - 1][id][i];

            feat_dim += WorkerStore::embs_compensate[layerId][id][i];

            vec.push_back(feat_dim);
            // 求整个返回矩阵的元素的最大值和最小值
            if (feat_dim > max_value) {
                max_value = feat_dim;
            }
            if (feat_dim < min_value) {
                min_value = feat_dim;
            }
        }
        embs.insert(pair<int, vector<float >>(id, vec));
    }

    s.min_value = min_value;
    s.max_value = max_value;
    return s;
}


void compressData(int bitNum, vector<uint> &itemVector, IntTensorMessage *tensor) {
    if (bitNum == 2) {
        if (itemVector.size() == 16) {
            uint compress_value = Compress::twoBitCompress(itemVector);
            tensor->add_tensor(compress_value);
            itemVector.clear();
        }
    } else if (bitNum == 4) {
        if (itemVector.size() == 8) {
            uint compress_value = Compress::fourBitCompress(itemVector);
            tensor->add_tensor(compress_value);
            itemVector.clear();
        }
    } else if (bitNum == 8) {
        if (itemVector.size() == 4) {
            uint compress_value = Compress::eightBitCompress(itemVector);
            tensor->add_tensor(compress_value);
            itemVector.clear();
        }
    } else if (bitNum == 16) {
        if (itemVector.size() == 2) {
            uint compress_value = Compress::sixteenBitCompress(itemVector);
            tensor->add_tensor(compress_value);
            itemVector.clear();
        }
    }

}

void compressData_concat(int bitNum, vector<uint> &itemVector,
                         google::protobuf::RepeatedField<google::protobuf::uint32> *mutable_emb_reply) {
//    int vectorSize=itemVector.size();
    uint compressValue = 0;
    auto &bitMap = WorkerStore::bucketPositionBitMap;

//    cout<<"8"<<endl;
//    cout<<"itemVector.size():"<<itemVector.size()<<endl;
//    cout<<"bitMap.size():"<<bitMap.size()<<endl;
//    cout<<"bitMap.begin.size():"<<bitMap.begin()->size()<<endl;
//    for (int i = 0; i < itemVector.size(); i++) {
//        cout<<"itemVector_i:"<<itemVector[i]<<endl;
//    }
    for (int i = 0; i < itemVector.size(); i++) {
//        cout<<"9"<<endl;
//        cout<<"itemVector_i:"<<itemVector[i]<<endl;
        compressValue = compressValue | bitMap[itemVector[i]][i];
//        cout<<"10"<<endl;
//        compressValue=compressValue|0;
    }
//    cout<<"11"<<endl;
    mutable_emb_reply->Add(compressValue);
//    if (bitNum == 2) {
//
//        uint compress_value = Compress::twoBitCompress(itemVector);
//        mutable_emb_reply->Add(Compress::twoBitCompress(itemVector));
//
//    } else if (bitNum == 4) {
//
//        uint compress_value = Compress::fourBitCompress(itemVector);
//        mutable_emb_reply->Add(compress_value);
////            itemVector.clear();
//
//    } else if (bitNum == 8) {
//
//        uint compress_value = Compress::eightBitCompress(itemVector);
//        mutable_emb_reply->Add(compress_value);
////            itemVector.clear();
//
//    } else if (bitNum == 16) {
//
//        uint compress_value = Compress::sixteenBitCompress(itemVector);
//        mutable_emb_reply->Add(compress_value);
////            itemVector.clear();
//
//    }

}



void compress1BitG(const EmbMessage *request, EmbMessage *reply){
    int layerid=request->layerid();
    auto &g = WorkerStore::G_map[layerid];
    auto min_value = WorkerStore::g_min[layerid];
    auto max_value = WorkerStore::g_max[layerid];
    vector<Bucket> buckets;
    Bucket b0{};
    b0.bid = 0;
    b0.lower_bound = (float) min_value;
    b0.upper_bound = 0;
    b0.value = -1;
    buckets.push_back(b0);
    reply->add_values(b0.value);
    Bucket b1{};
    b1.bid = 1;
    b1.lower_bound = 0;
    b1.upper_bound = (float) max_value;
    b1.value = 1;
    buckets.push_back(b1);
    reply->add_values(b1.value);

//    cout << "max_min_value:" << max_value << "," << min_value << endl;
    reply->set_shapedim(g.begin()->second.size());

    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);
    int bitNum = request->bitnum();
    // 开始构建压缩后的张量
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int feat_size = g.begin()->second.size();

    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);
//        vector<uint> vec_emb(nodeNum*compress_dim_size);
//        mutable_emb_reply->Add(vec_emb.begin(),vec_emb.end());

//        auto& bitMap=WorkerStore::bucketPositionBitMap;
    uint compressValue = 0;
    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        auto &g_second = g[id];
        for (uint i = 0; i < feat_size; i++) {
            float dim = g_second[i];
            if (dim >= 0) {
                bucket_id = 1;
            } else {
                bucket_id = 0;
            }
            uint itemId = i % oneIntDimNum;
            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));

            if (itemId == (oneIntDimNum - 1)) {
                mutable_emb_reply->Add(compressValue);
                compressValue = 0;
            }

        }
        if (left_num != 0) {
            mutable_emb_reply->Add(compressValue);
            compressValue = 0;
        }
    }
    reply->set_resp_featdim_size(compress_dim_size);
}

void initGCompensate(int layerid){
    map<int,vector<float>>tmp;
    WorkerStore::G_compensate.insert(pair<int,map<int,vector<float>>>(layerid,tmp));
}

void gCompensate1Bit(const EmbMessage* request,EmbMessage* reply){
    // no errors are compensated, however, 0-th epoch need to compute the errors
    int epoch=request->epoch();
    int layerid=request->layerid();
    map<int,vector<float>> g_tmp;
    auto &g_compensate_layerid=WorkerStore::G_compensate[layerid];
    auto &g = WorkerStore::G_map[layerid];
    auto min_value = WorkerStore::g_min[layerid];
    auto max_value = WorkerStore::g_max[layerid];
    if(epoch==0){
        initGCompensate(layerid);
        g_tmp=g;
    }
    if(epoch!=0){
        // need to be compensated
        for(int i=0;i<request->nodes_size();i++){
            auto id= request->nodes(i);
            auto g_nodei=g[id];
            auto& g_comp_nodei=g_compensate_layerid[id];
            int size=g_nodei.size();
            vector<float> g_nodei_tmp(size);
            for(int j=0;j<size;j++){
                g_nodei_tmp[j]=g_nodei[j]+g_comp_nodei[j];
                if(g_nodei_tmp[j]>max_value){
                    max_value=g_nodei_tmp[j];
                }else if(g_nodei_tmp[j]<min_value){
                    min_value=g_nodei_tmp[j];
                }
            }
            if(g_tmp.count(id)==0){
                g_tmp.insert(pair<int,vector<float>>(id,g_nodei_tmp));
            }else{
                g_tmp[id]=g_nodei_tmp;
            }
        }
    }


    vector<Bucket> buckets;
    Bucket b0{};
    b0.bid = 0;
    b0.lower_bound = (float) min_value;
    b0.upper_bound = 0;
    b0.value = -1;
    buckets.push_back(b0);
    reply->add_values(b0.value);
    Bucket b1{};
    b1.bid = 1;
    b1.lower_bound = 0;
    b1.upper_bound = (float) max_value;
    b1.value = 1;
    buckets.push_back(b1);
    reply->add_values(b1.value);

    reply->set_shapedim(g_tmp.begin()->second.size());

    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);
    int bitNum = request->bitnum();
    // 开始构建压缩后的张量
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int feat_size = g_tmp.begin()->second.size();

    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);

    uint compressValue = 0;
    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        auto &g_second = g_tmp[id];
        vector<float> errors_for_noden(feat_size);
        for (uint i = 0; i < feat_size; i++) {
            float dim = g_second[i];
            if (dim >= 0) {
                bucket_id = 1;
            } else {
                bucket_id = 0;
            }

            float dim_compress=buckets[bucket_id].value;
            float error=dim - dim_compress;
            errors_for_noden[i]=error;

            uint itemId = i % oneIntDimNum;
            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));

            if (itemId == (oneIntDimNum - 1)) {
                mutable_emb_reply->Add(compressValue);
                compressValue = 0;
            }

        }
        unique_lock<mutex> lock1(ThreadUtil::mtx_gcompensate);

        if(g_compensate_layerid.count(id)==0){
            g_compensate_layerid.insert(pair<int,vector<float>>(id,errors_for_noden));
        } else{
            g_compensate_layerid[id]=errors_for_noden;
        }


        lock1.unlock();


        if (left_num != 0) {
            mutable_emb_reply->Add(compressValue);
            compressValue = 0;
        }
    }
//    cout<<"size:"<<g_compensate_layerid.size()<<endl;
    reply->set_resp_featdim_size(compress_dim_size);
}


void gCompensateBits(const EmbMessage* request,EmbMessage* reply){
    // no errors are compensated, however, 0-th epoch need to compute the errors
    int epoch=request->epoch();
    int layerid=request->layerid();
    map<int,vector<float>> g_tmp;
    auto &g_compensate_layerid=WorkerStore::G_compensate[layerid];
    auto &g = WorkerStore::G_map[layerid];
    auto min_value = WorkerStore::g_min[layerid];
    auto max_value = WorkerStore::g_max[layerid];
    if(epoch==0){
        initGCompensate(layerid);
        g_tmp=g;
    }

    if(epoch!=0){
        // need to be compensated
        for(int i=0;i<request->nodes_size();i++){
            auto id= request->nodes(i);
            auto& g_nodei=g[id];
            auto& g_comp_nodei=g_compensate_layerid[id];
            int size=g_nodei.size();
            vector<float> g_nodei_tmp(size);
            for(int j=0;j<size;j++){
                g_nodei_tmp[j]=g_nodei[j]+g_comp_nodei[j];
                if(g_nodei_tmp[j]>max_value){
                    max_value=g_nodei_tmp[j];
                }else if(g_nodei_tmp[j]<min_value){
                    min_value=g_nodei_tmp[j];
                }
            }
            if(g_tmp.count(id)==0){
                g_tmp.insert(pair<int,vector<float>>(id,g_nodei_tmp));
            }else{
                g_tmp[id]=g_nodei_tmp;
            }

        }
    }

    vector<Bucket> buckets;
    bool ifCrossZero = false;
    int bucket_num;
    int bitNum = request->bitnum();
    if (max_value > 0 && min_value < 0) {
        ifCrossZero = true;
    }
    if (ifCrossZero) {
        bucket_num = pow(2, bitNum) - 2;
    } else {
        bucket_num = pow(2, bitNum) - 1;
    }
//    cout << "max_min_value:" << max_value << "," << min_value << endl;
    reply->set_shapedim(g_tmp.begin()->second.size());

    double interval = (max_value - min_value) / (double) (bucket_num);
    int int_interval = (int) (interval * pow(10, 8));
    int int_interval_addone = int_interval + 1;
    interval = int_interval_addone / pow(10, 8);

    if (ifCrossZero) {
        // bucket_num是interval的个数，而不是真正的bucket number
        int bucket_count = 0;
        for (int i = 0; i < bucket_num; i++) {
            // 从第0个区间开始，到bucket_num-1个区间结束,这里的i都是左编号
            if (interval * i <= 0 && interval * (i + 1) >= 0) {
                Bucket b_left{};
                b_left.bid = bucket_count;
                b_left.lower_bound = min_value + interval * i;
                b_left.upper_bound = 0;
                b_left.value = (b_left.lower_bound + b_left.upper_bound) / 2;
                buckets.push_back(b_left);
                reply->add_values(b_left.value);
                bucket_count++;

                Bucket b_right{};
                b_right.bid = bucket_count;
                b_right.lower_bound = 0;
                b_right.upper_bound = min_value + interval * (i + 1);
                b_right.value = (b_right.lower_bound + b_right.upper_bound) / 2;
                buckets.push_back(b_right);
                reply->add_values(b_right.value);
                bucket_count++;
            } else {
                Bucket b{};
                b.bid = bucket_count;
                b.lower_bound = min_value + interval * i;
                b.upper_bound = min_value + interval * (i + 1);
                b.value = (b.lower_bound + b.upper_bound) / 2;
                buckets.push_back(b);
                reply->add_values(b.value);
                bucket_count++;
            }

        }
    } else {
        for (int i = 0; i < bucket_num; i++) {
            Bucket b;
            b.bid = i;
            b.lower_bound = min_value + interval * i;
            b.upper_bound = min_value + interval * (i + 1);
            b.value = (b.lower_bound + b.upper_bound) / 2;
            buckets.push_back(b);
            reply->add_values(b.value);
        }
    }

    Bucket b;
    b.bid = buckets.size();
    b.lower_bound = 0;
    b.upper_bound = 0;
    b.value = 0;
    buckets.push_back(b);
    reply->add_values(0);

    int bucketSize = buckets.size();
    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);

    // 开始构建压缩后的张量
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int feat_size = g_tmp.begin()->second.size();

    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);
    uint compressValue = 0;
    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        auto &g_second = g_tmp[id];
        vector<float> errors_for_noden(feat_size);
        for (uint i = 0; i < feat_size; i++) {
            float dim = g_second[i];
            if (dim == 0) {
                bucket_id = bucketSize - 1;
            } else if (!ifCrossZero) {
                bucket_id = (int) ((dim - min_value) / interval);
            } else {
                bucket_id = (int) ((dim - min_value) / interval);
                if (dim > 0) {
                    bucket_id += 1;
                }
            }
            float dim_compress=buckets[bucket_id].value;
            float error=dim - dim_compress;
            errors_for_noden[i]=error;

            uint itemId = i % oneIntDimNum;

            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));
//                compressValue=0;

            if (itemId == (oneIntDimNum - 1)) {
//                    compressData_concat(bitNum, itemsVec, mutable_emb_reply);
                mutable_emb_reply->Add(compressValue);
                compressValue = 0;
            }

        }

        unique_lock<mutex> lock1(ThreadUtil::mtx_gcompensate);

        if(g_compensate_layerid.count(id)==0){
            g_compensate_layerid.insert(pair<int,vector<float>>(id,errors_for_noden));
        } else{
            g_compensate_layerid[id]=errors_for_noden;
        }
        lock1.unlock();
//            cout<<endl;
        if (left_num != 0) {
            mutable_emb_reply->Add(compressValue);
            compressValue = 0;
        }
    }

    reply->set_resp_featdim_size(compress_dim_size);
}


void compensate1BitG(const EmbMessage *request, EmbMessage *reply){
    int epoch=request->epoch();
    int layerid=request->layerid();
    gCompensate1Bit(request,reply);
}

void compensateBitsG(const EmbMessage *request, EmbMessage *reply){
    int epoch=request->epoch();
    int layerid=request->layerid();
    gCompensateBits(request,reply);
    // compensate for g
}

void compress1BitEmbs(const EmbMessage *request, EmbMessage *reply) {
    // -1,1
    auto &embs = WorkerStore::embs;
    auto min_value = WorkerStore::embs_min;
    auto max_value = WorkerStore::embs_max;
    vector<Bucket> buckets;
    Bucket b0{};
    b0.bid = 0;
    b0.lower_bound = (float) min_value;
    b0.upper_bound = 0;
    b0.value = -1;
    buckets.push_back(b0);
    reply->add_values(b0.value);

    Bucket b1{};
    b1.bid = 1;
    b1.lower_bound = 0;
    b1.upper_bound = (float) max_value;
    b1.value = 1;
    buckets.push_back(b1);
    reply->add_values(b1.value);

//    cout << "max_min_value:" << max_value << "," << min_value << endl;
    reply->set_shapedim(embs.begin()->second.size());

    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);
    int bitNum = request->bitnum();
    // 开始构建压缩后的张量
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int feat_size = embs.begin()->second.size();


    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);
//        vector<uint> vec_emb(nodeNum*compress_dim_size);
//        mutable_emb_reply->Add(vec_emb.begin(),vec_emb.end());

//        auto& bitMap=WorkerStore::bucketPositionBitMap;
    uint compressValue = 0;
    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        auto &emb_second = embs[id];
        for (uint i = 0; i < feat_size; i++) {
            float dim = emb_second[i];
            if (dim >= 0) {
                bucket_id = 1;
            } else {
                bucket_id = 0;
            }
            uint itemId = i % oneIntDimNum;
            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));

            if (itemId == (oneIntDimNum - 1)) {
                mutable_emb_reply->Add(compressValue);
                compressValue = 0;
            }

        }
        if (left_num != 0) {
            mutable_emb_reply->Add(compressValue);
            compressValue = 0;
        }
    }

    reply->set_resp_featdim_size(compress_dim_size);
}

void compressBitsG(const EmbMessage *request, EmbMessage *reply) {
    int layerid=request->layerid();
    auto &g = WorkerStore::G_map[layerid];
    auto min_value = WorkerStore::g_min[layerid];
    auto max_value = WorkerStore::g_max[layerid];
    vector<Bucket> buckets;
    bool ifCrossZero = false;
    int bucket_num;
    int bitNum = request->bitnum();
    if (max_value > 0 && min_value < 0) {
        ifCrossZero = true;
    }
    if (ifCrossZero) {
        bucket_num = pow(2, bitNum) - 2;
    } else {
        bucket_num = pow(2, bitNum) - 1;
    }
//    cout << "max_min_value:" << max_value << "," << min_value << endl;
    reply->set_shapedim(g.begin()->second.size());

    double interval = (max_value - min_value) / (double) (bucket_num);
    int int_interval = (int) (interval * pow(10, 8));
    int int_interval_addone = int_interval + 1;
    interval = int_interval_addone / pow(10, 8);

    if (ifCrossZero) {
        // bucket_num是interval的个数，而不是真正的bucket number
        int bucket_count = 0;
        for (int i = 0; i < bucket_num; i++) {
            // 从第0个区间开始，到bucket_num-1个区间结束,这里的i都是左编号
            if (interval * i <= 0 && interval * (i + 1) >= 0) {
                Bucket b_left{};
                b_left.bid = bucket_count;
                b_left.lower_bound = min_value + interval * i;
                b_left.upper_bound = 0;
                b_left.value = (b_left.lower_bound + b_left.upper_bound) / 2;
                buckets.push_back(b_left);
                reply->add_values(b_left.value);
                bucket_count++;

                Bucket b_right{};
                b_right.bid = bucket_count;
                b_right.lower_bound = 0;
                b_right.upper_bound = min_value + interval * (i + 1);
                b_right.value = (b_right.lower_bound + b_right.upper_bound) / 2;
                buckets.push_back(b_right);
                reply->add_values(b_right.value);
                bucket_count++;
            } else {
                Bucket b{};
                b.bid = bucket_count;
                b.lower_bound = min_value + interval * i;
                b.upper_bound = min_value + interval * (i + 1);
                b.value = (b.lower_bound + b.upper_bound) / 2;
                buckets.push_back(b);
                reply->add_values(b.value);
                bucket_count++;
            }

        }
    } else {
        for (int i = 0; i < bucket_num; i++) {
            Bucket b;
            b.bid = i;
            b.lower_bound = min_value + interval * i;
            b.upper_bound = min_value + interval * (i + 1);
            b.value = (b.lower_bound + b.upper_bound) / 2;
            buckets.push_back(b);
            reply->add_values(b.value);
        }
    }

    Bucket b;
    b.bid = buckets.size();
    b.lower_bound = 0;
    b.upper_bound = 0;
    b.value = 0;
    buckets.push_back(b);
    reply->add_values(0);

    int bucketSize = buckets.size();
    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);

    // 开始构建压缩后的张量
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int feat_size = g.begin()->second.size();

    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);
//        vector<uint> vec_emb(nodeNum*compress_dim_size);
//        mutable_emb_reply->Add(vec_emb.begin(),vec_emb.end());

//        auto& bitMap=WorkerStore::bucketPositionBitMap;
    uint compressValue = 0;
    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        auto &g_second = g[id];
        for (uint i = 0; i < feat_size; i++) {
            float dim = g_second[i];
            if (dim == 0) {
                bucket_id = bucketSize - 1;
            } else if (!ifCrossZero) {
                bucket_id = (int) ((dim - min_value) / interval);
            } else {
                bucket_id = (int) ((dim - min_value) / interval);
                if (dim > 0) {
                    bucket_id += 1;
                }
            }
            uint itemId = i % oneIntDimNum;

            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));
//                compressValue=0;

            if (itemId == (oneIntDimNum - 1)) {
//                    compressData_concat(bitNum, itemsVec, mutable_emb_reply);
                mutable_emb_reply->Add(compressValue);
                compressValue = 0;
            }

        }
//            cout<<endl;
        if (left_num != 0) {
            mutable_emb_reply->Add(compressValue);
            compressValue = 0;
        }
    }

    reply->set_resp_featdim_size(compress_dim_size);


}


void compressBitsEmbs(const EmbMessage *request, EmbMessage *reply) {
    auto &embs = WorkerStore::embs;
    auto min_value = WorkerStore::embs_min;
    auto max_value = WorkerStore::embs_max;
    vector<Bucket> buckets;
    bool ifCrossZero = false;
    int bucket_num;
    int bitNum = request->bitnum();
    if (max_value > 0 && min_value < 0) {
        ifCrossZero = true;
    }
    if (ifCrossZero) {
        bucket_num = pow(2, bitNum) - 2;
    } else {
        bucket_num = pow(2, bitNum) - 1;
    }
//    cout << "max_min_value:" << max_value << "," << min_value << endl;
    reply->set_shapedim(embs.begin()->second.size());

    double interval = (max_value - min_value) / (double) (bucket_num);
    int int_interval = (int) (interval * pow(10, 8));
    int int_interval_addone = int_interval + 1;
    interval = int_interval_addone / pow(10, 8);

    if (ifCrossZero) {
        // bucket_num是interval的个数，而不是真正的bucket number
        int bucket_count = 0;
        for (int i = 0; i < bucket_num; i++) {
            // 从第0个区间开始，到bucket_num-1个区间结束,这里的i都是左编号
            if (interval * i <= 0 && interval * (i + 1) >= 0) {
                Bucket b_left{};
                b_left.bid = bucket_count;
                b_left.lower_bound = min_value + interval * i;
                b_left.upper_bound = 0;
                b_left.value = (b_left.lower_bound + b_left.upper_bound) / 2;
                buckets.push_back(b_left);
                reply->add_values(b_left.value);
                bucket_count++;

                Bucket b_right{};
                b_right.bid = bucket_count;
                b_right.lower_bound = 0;
                b_right.upper_bound = min_value + interval * (i + 1);
                b_right.value = (b_right.lower_bound + b_right.upper_bound) / 2;
                buckets.push_back(b_right);
                reply->add_values(b_right.value);
                bucket_count++;
            } else {
                Bucket b{};
                b.bid = bucket_count;
                b.lower_bound = min_value + interval * i;
                b.upper_bound = min_value + interval * (i + 1);
                b.value = (b.lower_bound + b.upper_bound) / 2;
                buckets.push_back(b);
                reply->add_values(b.value);
                bucket_count++;
            }

        }
    } else {
        for (int i = 0; i < bucket_num; i++) {
            Bucket b;
            b.bid = i;
            b.lower_bound = min_value + interval * i;
            b.upper_bound = min_value + interval * (i + 1);
            b.value = (b.lower_bound + b.upper_bound) / 2;
            buckets.push_back(b);
            reply->add_values(b.value);
        }
    }

    Bucket b;
    b.bid = buckets.size();
    b.lower_bound = 0;
    b.upper_bound = 0;
    b.value = 0;
    buckets.push_back(b);
    reply->add_values(0);

    int bucketSize = buckets.size();
    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);

    // 开始构建压缩后的张量
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int feat_size = embs.begin()->second.size();

    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);
//        vector<uint> vec_emb(nodeNum*compress_dim_size);
//        mutable_emb_reply->Add(vec_emb.begin(),vec_emb.end());

//        auto& bitMap=WorkerStore::bucketPositionBitMap;
    uint compressValue = 0;
    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        auto &emb_second = embs[id];
        for (uint i = 0; i < feat_size; i++) {
            float dim = emb_second[i];
            if (dim == 0) {
                bucket_id = bucketSize - 1;
            } else if (!ifCrossZero) {
                bucket_id = (int) ((dim - min_value) / interval);
            } else {
                bucket_id = (int) ((dim - min_value) / interval);
                if (dim > 0) {
                    bucket_id += 1;
                }
            }
            uint itemId = i % oneIntDimNum;

            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));
//                compressValue=0;

            if (itemId == (oneIntDimNum - 1)) {
//                    compressData_concat(bitNum, itemsVec, mutable_emb_reply);
                mutable_emb_reply->Add(compressValue);
                compressValue = 0;
            }

        }
//            cout<<endl;
        if (left_num != 0) {
            mutable_emb_reply->Add(compressValue);
            compressValue = 0;
        }
    }

    reply->set_resp_featdim_size(compress_dim_size);


}

void compressEmbs(const EmbMessage *request, EmbMessage *reply) {
    // 开始对需要的嵌入进行压缩,先进行基于range的压缩，scaling factor
    // 按照sketch压缩,但是这种实现方法需要一定时间，先试试量化这种吧
    // 按照该模式进行压缩
    // 压缩模式

    int bitNum = request->bitnum();


    // 1-bit compression is special, which cannot be divided into 3 parts when cross zero
    if (bitNum == 1) {
        compress1BitEmbs(request, reply);
    } else {
        // 如果bitnum不是1，那么至少有3个（实际4个）个桶，可以包含+-0
        compressBitsEmbs(request, reply);
    }

    // 横跨，正负优先,0随后;
    // if cross, -a,+b contains two ranges -a,0 0,+b
    // if not cross

}

void compressG(const EmbMessage *request, EmbMessage *reply){
    int bitNum = request->bitnum();
    // 1-bit compression is special, which cannot be divided into 3 parts when cross zero
    bool ifcompensate=request->ifcompensate();
//    cout<<"bitnum:"<<bitNum<<endl;

    if(!ifcompensate){
        if (bitNum == 1) {
            compress1BitG(request, reply);
        } else {
            // 如果bitnum不是1，那么至少有3个（实际4个）个桶，可以包含+-0
            compressBitsG(request, reply);
        }
    }else{
        if (bitNum == 1) {
            compensate1BitG(request, reply);
        } else {
            compensateBitsG(request,reply);

        }
    }

}

Status ServiceImpl::workerPullEmbCompress(ServerContext *context, const EmbMessage *request,
                                          EmbMessage *reply) {
//    clock_t start_time_total = clock();

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    compressEmbs(request, reply);

    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
//    cout << "server processing time:" << timeuse << "s" << endl;


    return Status::OK;

}

Status ServiceImpl::workerPullGCompress(ServerContext *context, const EmbMessage *request,
                                          EmbMessage *reply) {
//    clock_t start_time_total = clock();

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);
    compressG(request, reply);
    gettimeofday(&t2, NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
//    cout << "server processing time:" << timeuse << "s" << endl;


    return Status::OK;

}


Status ServiceImpl::sendDataToEachWorker(
        ServerContext *context, const DataMessage *request,
        BoolMessage *reply) {
    WorkerStore::set_nodes(&request->nodelist());
    WorkerStore::set_features(&request->featurelist());
    WorkerStore::set_labels(&request->labellist());
    WorkerStore::set_adjs(&request->adjlist());

    reply->set_flag(true);
    return Status::OK;
}

Status ServiceImpl::initParameter(ServerContext *context, const NetInfoMessage *request, BoolMessage *reply) {
    int serverId = ServerStore::serverId;
    cout << "Server " << serverId << ": initParameters begining!" << endl;
    // 还原request
    int worker_num = request->workernum();
    int feat_dim = request->featuredim();
    int server_num = request->servernum();
    vector<int> hid_dims;
    int wid = request->wid();

    for (auto hid_dim:request->hiddendim()) {
        hid_dims.push_back(hid_dim);
    }

    int class_dim = request->classdim();
    cout << "Server revceived:" << endl;
    if (wid == 0) {
        string hid = "hidden size:";
        for (auto dim:hid_dims) {
            hid.append(to_string(dim) + ",");
        }
        hid.pop_back();
        cout << "worker number:" << worker_num << ", server number: " << server_num << ",feature dimensions:"
             << feat_dim << ",class dimensions:"
             << class_dim << ",hidden size:" << hid << endl;
    }

    // 0号线程初始化，其他的等待
    if (wid == 0) {
        unique_lock<mutex> mutex(ThreadUtil::mtx_initParameter);
        // 初始化神经网络参数

        ServerStore::worker_num = worker_num;
        ServerStore::server_num = server_num;
        ServerStore::feat_dim = feat_dim;
        ServerStore::class_dim = class_dim;
        ServerStore::hid_dims = hid_dims;
        for(int i=0;i<request->params_size();i++){
            int size_elem=request->params(i).elems_size();
            vector<float> tmp(size_elem);
            vector<float> tmp2(size_elem);
            vector<float> tmp3(size_elem);
            vector<float> tmp4(size_elem);
            auto& param_i=request->params(i);
            for(int j=0;j<size_elem;j++){
                tmp[j]=param_i.elems(j);
            }
            ServerStore::params.insert(pair<string,vector<float>>(param_i.id(),tmp));
            ServerStore::grads_agg.insert(pair<string,vector<float>>(param_i.id(),tmp2));
            ServerStore::m_grads_t.insert(pair<string,vector<float>>(param_i.id(),tmp3));
            ServerStore::v_grads_t.insert(pair<string,vector<float>>(param_i.id(),tmp4));
        }


        Check::check_initParameter_ServerStore();


//        ServerStore::initParams(worker_num, server_num, feat_dim, class_dim, hid_dims);
        ThreadUtil::ready_initParameter = true;
        ThreadUtil::cv.notify_all();

    } else {
        unique_lock<mutex> mutex(ThreadUtil::mtx_initParameter);
        while (!ThreadUtil::ready_initParameter) {
            ThreadUtil::cv.wait(mutex);
        }
    }

    reply->set_flag(true);
    cout << "initParameters ending!" << endl;
    return Status::OK;

}

Status ServiceImpl::pullWeights(
        ServerContext *context, const IntMessage *request,
        WeightsAndBiasMessage *reply) {
    // 这里先实现同步的策略，因此在读的时候先不加锁
    int layer_id = request->id();
    for (const auto &row:ServerStore::weights[layer_id]) {
        TensorMessage *tensor = reply->add_weights();
        for (auto item :row) {
            tensor->add_tensor(item);
        }
    }

    return Status::OK;
}

Status ServiceImpl::pullBias(
        ServerContext *context, const IntMessage *request,
        WeightsAndBiasMessage *reply) {
    // 这里先实现同步的策略，因此在读的时候先不加锁
    int layer_id = request->id();
    TensorMessage *tensorMessage = reply->bias().New();
    for (auto b:ServerStore::bias[layer_id]) {
        tensorMessage->add_tensor(b);
    }
    reply->set_allocated_bias(tensorMessage);

    return Status::OK;
}

Status ServiceImpl::barrier(
        ServerContext *context, const BoolMessage *request, BoolMessage *reply) {
    unique_lock<mutex> lck(ThreadUtil::mtx_barrier);
    ThreadUtil::count_worker_for_barrier++;
    if (ThreadUtil::count_worker_for_barrier == ServerStore::worker_num) {
        ThreadUtil::cv_barrier.notify_all();
        ThreadUtil::count_worker_for_barrier = 0;
    } else {
        ThreadUtil::cv_barrier.wait(lck);
    }
    return Status::OK;
}


Status ServiceImpl::pullDataFromServer(
        ServerContext *context, const intM *request,
        DataMessage *reply) {
//        DataMessage dataMessage;
    WorkerStore::getData(reply);
    return Status::OK;

}

Status ServiceImpl::Server_SendAndUpdateModels(
        ServerContext *context, const GradientMessage *request,
        BoolMessage *reply) {
    // 所有worker等待master清空完梯度聚合后，继续执行
    if (request->worker_id() == 0) {
        unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);
        if (ServerStore::serverId == 0) {
            ServerStore::bias_grad_agg.clear();
            cout << "*********bias_grad_agg_is_cleared**********" << endl;
            cout << ServerStore::bias_grad_agg.size() << endl;
        }
        ServerStore::weights_grad_agg.clear();
        cout << "*********weights_grad_agg_is_cleared**********" << endl;
        cout << ServerStore::weights_grad_agg.size() << endl;
        ThreadUtil::ready_updateModels = true;
        ThreadUtil::cv_updateModels.notify_all();
    } else {
        unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);
        while (!ThreadUtil::ready_updateModels) {
            ThreadUtil::cv_updateModels.wait(lck);
        }
    }
    float alpha = request->lr();
    // 多个worker一起更新参数，先聚合所有worker的梯度
    // 聚合worker的梯度时，先上锁
//    pthread_mutex_lock(&ThreadUtil::mtx_updateModels_addGrad);
    unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);

    int layerNum = request->grads_size();
    cout << "**********layer number*********: " << layerNum << endl;

    // 判断是否已经被初始化了，如果未被初始化，那么直接用这台机器的梯度初始化ServerStore中的grad
    if (ServerStore::weights_grad_agg.count(0) == 0) {
        // 对于每一层的梯度聚合
        cout << request->worker_id() << "**************first add start********************" << endl;

        for (int i = 0; i < layerNum; i++) {
            const auto &grad = request->grads(i);
            int weight_size = grad.weights_size();
            int tensor_size = grad.weights().begin()->tensor_size();


            vector<vector<float>> weight_grad(weight_size);
            // 这里获取的是每一个WeightAndBiasMessage
            //先遍历Weight,这一层是遍历每一个TensorMessage

            for (int j = 0; j < weight_size; j++) {
                const auto tensor = grad.weights(j);

                vector<float> grad_row(tensor_size);
                for (int k = 0; k < tensor_size; k++) {
                    float dim = tensor.tensor(k);
                    grad_row[k] = (dim / (float) ServerStore::worker_num);
                }
                weight_grad[j] = grad_row;
            }
            ServerStore::weights_grad_agg.insert(pair<int, vector<vector<float>>>(i, weight_grad));

            // 插入第i层的bias
            if (ServerStore::serverId == 0) {
                int tensorSize = grad.bias().tensor_size();
                vector<float> bia_grad(tensorSize);
                for (int j = 0; j < tensorSize; j++) {
                    float dim = grad.bias().tensor(j);
                    bia_grad[j] = (dim / (float) ServerStore::worker_num);
                }
                ServerStore::bias_grad_agg.insert(pair<int, vector<float>>(i, bia_grad));
            }

        }
        cout << request->worker_id() << "**************first add end********************" << endl;
    } else {
        // 已经有初始化的值了，因此需要做累加
        // 对于每一层的梯度聚合
        cout << request->worker_id() << "**************second add start********************" << endl;

        for (int i = 0; i < layerNum; i++) {
            const auto &grad = request->grads(i);
            auto &weights_grad_agg_layer = ServerStore::weights_grad_agg[i];
            // 这里获取的是每一个WeightAndBiasMessage
            //先遍历Weight,这一层是遍历每一个TensorMessage
            int weight_size = grad.weights_size();
            int tensor_size = grad.weights().begin()->tensor_size();
            for (int j = 0; j < weight_size; j++) {
                const auto &tensor = grad.weights(j);
                auto &weights_grad_agg_row = weights_grad_agg_layer[j];
                for (int k = 0; k < tensor_size; k++) {
                    float dim = tensor.tensor(k);
                    weights_grad_agg_row[k] = (weights_grad_agg_row[k] +
                                               dim / (float) ServerStore::worker_num);
                }

            }
            // 插入第i层的bias
            if (ServerStore::serverId == 0) {
                int bias_size = grad.bias().tensor_size();
                auto &biass_grad_agg_layer = ServerStore::bias_grad_agg[i];
                for (int j = 0; j < bias_size; j++) {
                    float dim = grad.bias().tensor(j);
                    biass_grad_agg_layer[j] = (biass_grad_agg_layer[j] +
                                               dim / (float) ServerStore::worker_num);
                }
            }

        }
        cout << request->worker_id() << "**************second add end********************" << endl;
    }
    lck.unlock();

    // 每个worker累积完梯度就可以释放锁了
//    pthread_mutex_unlock(&ThreadUtil::mtx_updateModels_addGrad);

    // 有一个线程更新参数,更新参数的前提是所有梯度都已聚合完成
    // 确保所有机器都已到达

    lck.lock();
    ThreadUtil::count_worker_for_updateModels++;
    if (ThreadUtil::count_worker_for_updateModels == ServerStore::worker_num) {
        ThreadUtil::cv_updateModels.notify_all();
        ThreadUtil::count_worker_for_updateModels = 0;
        ThreadUtil::ready_updateModels = false;
    } else {
        ThreadUtil::cv_updateModels.wait(lck);
    }




    // 下面是做check
    if (request->worker_id() == 0) {
        cout << "Server: server store weight grad layer num: " << ServerStore::weights_grad_agg.size() << endl;
        cout << "Server: server store bias layer num: " << ServerStore::bias_grad_agg.size() << endl;

        for (auto weight = ServerStore::weights_grad_agg.begin();
             weight != ServerStore::weights_grad_agg.end(); weight++) {
            int weight_row_num = weight->second.size();
            int weight_col_num = weight->second.begin()->size();
            cout << "Server: server store weight " << weight->first << " grad size: " << weight_row_num << "*"
                 << weight_col_num << endl;
        }

        if (ServerStore::serverId == 0) {
            for (auto bia = ServerStore::bias_grad_agg.begin(); bia != ServerStore::bias_grad_agg.end(); bia++) {
                int bia_size = bia->second.size();
                cout << "Server: server store bia " << bia->first << " grad size: " << bia_size << endl;
            }
        }

    }

    // worker 0线程开始负责更新参数
    if (request->worker_id() == 0) {
        ServerStore::t++;
        float beta_1 = 0.9;
        float beta_2 = 0.999;
        float epsilon = 5e-4;
        bool isAdam = true;

        // 如果m_weight_t,v_weight_t,m_bias_t,v_bias_t为空，那么初始化
        if (ServerStore::m_weight_t.empty()) {
            for (const auto &weight_grad :ServerStore::weights_grad_agg) {
                int row_size = weight_grad.second.size();
                int col_size = weight_grad.second.begin()->size();
                vector<vector<float>> row(row_size);
                vector<vector<float>> row2(row_size);
                for (int i = 0; i < row_size; i++) {
                    vector<float> vec(col_size);
                    vector<float> vec2(col_size);
                    for (int j = 0; j < col_size; j++) {
                        vec[j] = 0;
                        vec2[j] = 0;
                    }
                    row[i] = vec;
                    row2[i] = vec2;
                }
                ServerStore::m_weight_t.insert(pair<int, vector<vector<float>>>(weight_grad.first, row));
                ServerStore::v_weight_t.insert(pair<int, vector<vector<float>>>(weight_grad.first, row2));
            }
            if (ServerStore::serverId == 0) {
                for (const auto &bia_grad: ServerStore::bias_grad_agg) {
                    int bias_size = bia_grad.second.size();
                    vector<float> vec(bias_size);
                    vector<float> vec2(bias_size);
                    for (int i = 0; i < bias_size; i++) {
                        vec[i] = 0;
                        vec2[i] = 0;
                    }
                    ServerStore::m_bias_t.insert(pair<int, vector<float>>(bia_grad.first, vec));
                    ServerStore::v_bias_t.insert(pair<int, vector<float>>(bia_grad.first, vec2));
                }
            }


        }


        for (const auto &weight_grad :ServerStore::weights_grad_agg) {
            int row_size = weight_grad.second.size();
            int col_size = weight_grad.second.begin()->size();
            auto &m_weight_t_layer = ServerStore::m_weight_t[weight_grad.first];
            auto &v_weight_t_layer = ServerStore::v_weight_t[weight_grad.first];
            auto &weight_t_layer = ServerStore::weights[weight_grad.first];
            for (int i = 0; i < row_size; i++) {
                vector<float> vec = weight_grad.second[i];
                auto &m_weight_t_row = m_weight_t_layer[i];
                auto &v_weight_t_row = v_weight_t_layer[i];
                auto &weight_t_row = weight_t_layer[i];
                for (int j = 0; j < col_size; j++) {
                    float g_t = vec[j];
                    if (isAdam) {
                        m_weight_t_row[j] =
                                beta_1 * m_weight_t_row[j] + (1 - beta_1) * g_t;
                        v_weight_t_row[j] =
                                beta_2 * v_weight_t_row[j] + (1 - beta_2) * g_t * g_t;
                        float m_cap =
                                m_weight_t_row[j] / (1 - (pow(beta_1, ServerStore::t)));
                        float v_cap =
                                v_weight_t_row[j] / (1 - (pow(beta_2, ServerStore::t)));
                        weight_t_row[j] -= (alpha * m_cap) / (sqrt(v_cap) + epsilon);
                    } else {
                        weight_t_row[j] -= alpha * g_t;
                    }
                }
            }
        }

        if (ServerStore::serverId == 0) {
            for (const auto &bia_grad: ServerStore::bias_grad_agg) {
                vector<float> vec = bia_grad.second;
                int bias_size = vec.size();
                auto &m_bias_t_layer = ServerStore::m_bias_t[bia_grad.first];
                auto &v_bias_t_layer = ServerStore::v_bias_t[bia_grad.first];
                auto &bias_layer = ServerStore::bias[bia_grad.first];
                for (int i = 0; i < bias_size; i++) {
                    float g_t = vec[i];
                    if (isAdam) {
                        m_bias_t_layer[i] =
                                beta_1 * m_bias_t_layer[i] + (1 - beta_1) * g_t;
                        v_bias_t_layer[i] =
                                beta_2 * v_bias_t_layer[i] + (1 - beta_2) * g_t * g_t;
                        float m_cap =
                                m_bias_t_layer[i] / (1 - (float) (pow(beta_1, ServerStore::t)));
                        float v_cap =
                                v_bias_t_layer[i] / (1 - (float) (pow(beta_2, ServerStore::t)));

                        bias_layer[i] -= (alpha * m_cap) / (sqrt(v_cap) + epsilon);
                    } else {
                        bias_layer[i] -= alpha * g_t;
                    }
                }
            }
        }
    }


    return Status::OK;
}

void ServiceImpl::RunServerByPy(const string &address, int serverId) {
    ServiceImpl service;

    ServerStore::serverId = serverId;
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());

    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(2147483647);
    builder.SetMaxSendMessageSize(2147483647);
    builder.SetMaxMessageSize(2147483647);


    std::unique_ptr<Server> server(builder.BuildAndStart());


    std::cout << "Server Listening on " << address << std::endl;
    std::cout << "if signal 11, please check ip :" << address << std::endl;
    server->Wait();

}

Status ServiceImpl::Test1Bit(ServerContext *context, const BitArrayMessage *request, BitArrayMessage *reply) {
    return Status::OK;
}


Status ServiceImpl::TestVariant(ServerContext *context, const TestVMessage *request, BoolMessage *reply) {


    return Status::OK;

}

Status ServiceImpl::testLargeSize(ServerContext *context, const LargeMessage *request, LargeMessage *reply) {
    clock_t start = clock();
    for (int i = 0; i < 10000000; i++) {
        reply->add_a(i);
    }
    clock_t end = clock();
    cout << "testLargeSize:" << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "largeSize:" << reply->ByteSizeLong() << endl;
    return Status::OK;
}

Status ServiceImpl::testSmallSize(ServerContext *context, const SmallMessage *request, SmallMessage *reply) {
    clock_t start = clock();
    for (int i = 0; i < 1000000; i++) {
        reply->add_a(i);
    }
    clock_t end = clock();
    cout << "testSmallSize:" << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "smallSize:" << reply->ByteSizeLong() << endl;
    return Status::OK;
}


Status ServiceImpl::sendAccuracy(ServerContext *context, const AccuracyMessage *request,
                                 AccuracyMessage *reply) {


    unique_lock<mutex> lck(ThreadUtil::mtx_accuracy);
    if (ThreadUtil::count_accuracy == 0) {
        ServerStore::val_accuracy = 0;
        ServerStore::train_accuracy = 0;
        ServerStore::test_accuracy = 0;
        ServerStore::val_f1_accuracy = 0;
        ServerStore::test_f1_accuracy = 0;

    }
    ServerStore::val_accuracy += request->val_acc();
    ServerStore::train_accuracy += request->train_acc();
    ServerStore::test_accuracy += request->test_acc();
    ServerStore::val_f1_accuracy += request->val_f1();
    ServerStore::test_f1_accuracy += request->test_f1();

    ThreadUtil::count_accuracy++;
    if (ThreadUtil::count_accuracy == ServerStore::worker_num) {
        ThreadUtil::cv_accuracy.notify_all();
        ThreadUtil::count_accuracy = 0;

    } else {
        ThreadUtil::cv_accuracy.wait(lck);
    }


    reply->set_val_acc_entire(ServerStore::val_accuracy);
    reply->set_train_acc_entire(ServerStore::train_accuracy);
    reply->set_test_acc_entire(ServerStore::test_accuracy);
    reply->set_val_f1_entire(ServerStore::val_f1_accuracy);
    reply->set_test_f1_entire(ServerStore::test_f1_accuracy);
    return Status::OK;

}

//Status ServiceImpl::workerPullEmbTrend(
//        ServerContext *context, const EmbMessage *request, EmbMessage *reply) {
//    // 这里请求的nodes的顺序和返回的tensor的顺序要保持一致
////    clock_t start = clock();
//
//    int epoch = request->epoch();
//    int layerId = request->layerid();
//    int trend = request->trend();
//    int workerId = request->workerid();
//
//
//    int feat_size = WorkerStore::embs.begin()->second.size();
//    auto &embs = WorkerStore::embs;
//    float max_value = 1.0;
//    float min_value = 0.0;
//    int bitNum = request->bitnum();
//    int bucket_num = pow(2, bitNum) - 2;
//    bool setBValueAslow = false;
//    if (bitNum == 1) {
//        bucket_num = 1;
//    }
//
//
//    reply->set_resp_node_size(request->nodes_size());
//    reply->set_shapedim(feat_size);
//
//
//    if (request->epoch() == 0) {
//        // insert worker
//        // constuct embs_last and embs_change_rate and return the non-compressed embeddings
//        map<int, map<int, vector<float>>> embs_last_worker_tmp;
//        WorkerStore::embs_last.insert(pair<int, map<int, map<int, vector<float>>>>(workerId, embs_last_worker_tmp));
//
//        // the changeRateMap for worker i
//        map<int, map<int, vector<float>>> changeRateForWorkerI;
//        map<int, vector<float>> embs_last_layer;
//        map<int, vector<float>> embs_change;
//        WorkerStore::embs_last[workerId].insert(pair<int, map<int, vector<float>>>(layerId, embs_last_layer));
//        WorkerStore::embs_change_rate.insert(
//                pair<int, map<int, map<int, vector<float >> >>(workerId, changeRateForWorkerI));
//        WorkerStore::embs_change_rate[workerId].insert(pair<int, map<int, vector<float>>>(layerId, embs_change));
//        for (auto id:request->nodes()) {
//            vector<float> vec_change(feat_size);
//            WorkerStore::embs_last[workerId][layerId].insert(pair<int, vector<float >>(id, embs[id]));
//            WorkerStore::embs_change_rate[workerId][layerId].insert(pair<int, vector<float >>(id, vec_change));
//        }
//        int nodeNum = request->nodes_size();
//        reply->set_resp_featdim_size(feat_size);
//        for (int i = 0; i < nodeNum; i++) {
//            int id = request->nodes(i);
//            reply->mutable_resp_none_compress_emb_concat()->Add(embs[id].begin(), embs[id].end());
//        }
//
//    } else {
//        auto &embs_last_layerId = WorkerStore::embs_last[workerId][layerId];
//        if ((epoch + 1) % trend == 0) {
//            // in the last of each ten epochs (e.g., ninth epochs), non-compressed embeddings and the change-rate
//
//            int nodeNum = request->nodes_size();
//            reply->set_resp_featdim_size(feat_size);
//
//            reply->mutable_resp_none_compress_rate_concat()->Reserve(nodeNum * feat_size);
//            reply->mutable_resp_none_compress_emb_concat()->Reserve(nodeNum * feat_size);
//            for (int i = 0; i < nodeNum; i++) {
//                int id = request->nodes(i);
//                reply->mutable_resp_none_compress_emb_concat()->Add(embs[id].begin(), embs[id].end());
//                auto &emb_id = embs[id];
//                auto &emb_last_node = embs_last_layerId[id];
//                for (int j = 0; j < feat_size; j++) {
//                    reply->mutable_resp_none_compress_rate_concat()->Add(emb_id[j] - emb_last_node[j]);
//                }
//            }
//
//            embs_last_layerId = embs;
//
//        } else {
//            // 发送压缩嵌入
//            vector<Bucket> buckets;
//            double interval = (max_value - min_value) / (double) (bucket_num);
//
//            int int_interval = (int) (interval * pow(10, 8));
//            int int_interval_addone = int_interval + 1;
//
//            interval = int_interval_addone / pow(10, 8);
////            cout << "bucket_num,interval_bucket:" << bucket_num << "," << interval << endl;
////            cout << "int_interval,add1:" << int_interval << "," << int_interval_addone << endl;
//
////        clock_t start_compress = clock();
//            if (min_value < 0 && max_value > 0) {
//                for (int i = 0; i < bucket_num + 1; i++) {
//                    if (min_value + interval * i < 0 && min_value + interval * (i + 1) > 0) {
//                        // 建两个桶,以0的分界线
//                        Bucket b1;
//                        b1.bid = i;
//                        b1.lower_bound = min_value + interval * i;
//                        b1.upper_bound = 0;
//
//                        b1.value = (b1.lower_bound + b1.upper_bound) / 2;
//
//                        buckets.push_back(b1);
//                        reply->add_values(b1.value);
//
//                        i = i + 1;
//                        Bucket b2;
//                        b2.bid = i;
//                        b2.lower_bound = 0;
//                        b2.upper_bound = min_value + interval * (i + 1);
//                        if (i == bucket_num) {
//                            b2.upper_bound = max_value;
//                        }
//
//                        b2.value = (b2.lower_bound + b2.upper_bound) / 2;
//
//
//                        buckets.push_back(b2);
//                        reply->add_values(b2.value);
//                    } else {
//                        Bucket b;
//                        b.bid = i;
//                        b.lower_bound = min_value + interval * i;
//                        b.upper_bound = min_value + interval * (i + 1);
//                        if (i == bucket_num - 1) {
//                            b.upper_bound = max_value;
//                        }
//                        if (b.lower_bound < 0 && setBValueAslow) {
//                            b.value = b.upper_bound;
//                        } else if (b.lower_bound > 0 && setBValueAslow) {
//                            b.value = b.lower_bound;
//                        } else {
//                            b.value = (b.lower_bound + b.upper_bound) / 2;
//                        }
//
//                        buckets.push_back(b);
//                        reply->add_values(b.value);
//                    }
//                }
//            } else {
//                for (int i = 0; i < bucket_num; i++) {
//                    Bucket b;
//                    b.bid = i;
//                    b.lower_bound = min_value + interval * i;
//                    b.upper_bound = min_value + interval * (i + 1);
//                    if (i == bucket_num - 1) {
//                        b.upper_bound = max_value;
//                    }
//
//                    if (b.lower_bound < 0 && setBValueAslow) {
//                        b.value = b.upper_bound;
//                    } else if (b.lower_bound > 0 && setBValueAslow) {
//                        b.value = b.lower_bound;
//                    } else {
//                        b.value = (b.lower_bound + b.upper_bound) / 2;
//                    }
//                    buckets.push_back(b);
//                    reply->add_values(b.value);
//                }
//            }
//
//            Bucket b;
//            b.bid = buckets.size();
//            b.lower_bound = 0;
//            b.upper_bound = 0;
//            b.value = 0;
//            buckets.push_back(b);
//            reply->add_values(0);
//            int bucketSize = buckets.size();
//            int nodeNum = request->nodes_size();
//            reply->set_resp_node_size(nodeNum);
//            int oneIntDimNum = 32 / bitNum;
//            vector<uint> itemsVec(oneIntDimNum);
//            int left_num = feat_size % oneIntDimNum;
//            int compress_dim_size;
//            if (left_num == 0) {
//                compress_dim_size = feat_size / oneIntDimNum;
//            } else {
//                compress_dim_size = feat_size / oneIntDimNum + 1;
//            }
//            auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
//            mutable_emb_reply->Reserve(nodeNum * compress_dim_size);
////        vector<uint> vec_emb(nodeNum*compress_dim_size);
////        mutable_emb_reply->Add(vec_emb.begin(),vec_emb.end());
//
////        auto& bitMap=WorkerStore::bucketPositionBitMap;
////            cout<<"11111111"<<endl;
//
//            uint compressValue = 0;
//            for (int n = 0; n < nodeNum; n++) {
//                auto id = request->nodes(n);
//                uint bucket_id;
////                if(embs.count(id)==0){
////                    cout<<"2222222222222222"<<endl;
////                }
//                auto &emb_second = embs[id];
//
//                for (uint i = 0; i < feat_size; i++) {
//                    float dim = emb_second[i];
//                    if (dim == 0) {
//                        bucket_id = bucketSize - 1;
//                    } else {
//                        bucket_id = (int) (dim / interval);
//                    }
//                    uint itemId = i % oneIntDimNum;
//
////                compressValue=compressValue|bitMap[bucket_id][itemId];
//                    compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));
////                compressValue=0;
//
//                    if (itemId == (oneIntDimNum - 1)) {
////                    compressData_concat(bitNum, itemsVec, mutable_emb_reply);
//                        mutable_emb_reply->Add(compressValue);
//                        compressValue = 0;
//                    }
//
//                }
//                if (left_num != 0) {
//                    mutable_emb_reply->Add(compressValue);
//                    compressValue = 0;
//                }
//            }
//
//            reply->set_resp_featdim_size(compress_dim_size);
//
//        }
//        reply->set_shapedim(embs.begin()->second.size());
//    }
//
//
//
////    clock_t end = clock();
////    cout << "222222222222222222:" << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;
////    cout << "reply->ByteSizeLong():" << reply->ByteSizeLong() << endl;
////    cout<<"embs size::"<<reply->embs_size()<<",dim size:"<<reply->embs(0).tensor_size()<<endl;
//
//    return Status::OK;
//}


int getIndexWithMinError(float v0, float v1, float v2) {

    if (v1 <= v0 && v1 <= v2) {
        return 1;
    } else if (v2 <= v0 && v2 <= v1) {
        return 2;
    } else {
        return 0;
    }
    // @test 1
//    return 2;
}

void compress1BitEmbsTrend(const EmbMessage *request, EmbMessage *reply, map<int, vector<float>> &embs) {
    vector<Bucket> buckets;
    auto min_value = WorkerStore::embs_min;
    auto max_value = WorkerStore::embs_max;
    int bitNum = request->bitnum();
    int workerId = request->workerid();
    int layerId = request->layerid();
    int epoch = request->epoch();
    int trend = request->trend();


    Bucket b0{};
    b0.bid = 0;
    b0.lower_bound = (float) min_value;
    b0.upper_bound = 0;
    b0.value = -1;
    buckets.push_back(b0);
    reply->add_values(b0.value);

    Bucket b1{};
    b1.bid = 1;
    b1.lower_bound = 0;
    b1.upper_bound = (float) max_value;
    b1.value = 1;
    buckets.push_back(b1);
    reply->add_values(b1.value);

//    cout << "max_min_value:" << max_value << "," << min_value << endl;
    reply->set_shapedim(embs.begin()->second.size());
    int feat_size = embs.begin()->second.size();
    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);

    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);

    int node_left_num = nodeNum % 16;
    int compress_node_size;
    if (node_left_num == 0) {
        compress_node_size = nodeNum / 16;
    } else {
        compress_node_size = nodeNum / 16 + 1;
    }
    auto *mutable_value_flag_reply = reply->mutable_value_select_flags();
    mutable_value_flag_reply->Reserve(compress_node_size);

    uint compressValue = 0;
    uint compressNode = 0;

    vector<float> count_flag_num(3);

    auto last_embs = WorkerStore::embs_last[workerId][layerId];
    auto rates = WorkerStore::embs_change_rate[workerId][layerId];
    reply->mutable_value_select_flags()->Reserve(nodeNum);

    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        // the non-error embs
        auto &emb_second = embs[id];
        auto last_embs_node = last_embs[id];
        auto rates_node = rates[id];
        float comp_error = 0;
        float pred_error = 0;
        float mix_error = 0;

        vector<uint> vec_feat_comp_tmp(compress_dim_size);

        for (uint i = 0; i < feat_size; i++) {
            float dim = emb_second[i];
            if (dim >= 0) {
                bucket_id = 1;
            } else {
                bucket_id = 0;
            }
            uint itemId = i % oneIntDimNum;
            // compare the values of compress, predict and (c+p)/2
            float compDim = buckets[bucket_id].value;
            float predDim = 0;
            if (epoch / trend == 0) {
                // predictive value is the last embs
                predDim = last_embs_node[i];
            } else {
                int round = (epoch + 1) % trend;
                predDim = round * rates_node[i] + last_embs_node[i];
            }
            float mixDim = (predDim + compDim) / 2;


            comp_error += abs(compDim - dim);
            pred_error += abs(predDim - dim);
            mix_error += abs(mixDim - dim);

            // @test 3
//                    comp[i] = compDim;
//                    pred[i] = predDim;
//                    mix[i] = mixDim;

            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));

            if (itemId == (oneIntDimNum - 1)) {
                vec_feat_comp_tmp[i / oneIntDimNum] = compressValue;
                compressValue = 0;
            }

        }


        if (left_num != 0) {
            vec_feat_comp_tmp[feat_size / oneIntDimNum] = compressValue;
            compressValue = 0;
        }

        uint index_WithMinError = getIndexWithMinError(comp_error, pred_error, mix_error);


        count_flag_num[index_WithMinError]++;


        if (index_WithMinError != 1) {
            // predictive value does not need to be sent,others need to be sent the value of compression
            mutable_emb_reply->Add(vec_feat_comp_tmp.begin(), vec_feat_comp_tmp.end());

        }

        uint itemId_node = n % 16;
        compressNode = compressNode | (index_WithMinError << (32 - (itemId_node + 1) * 2));
        if (itemId_node == (16 - 1)) {
            mutable_value_flag_reply->Add(compressNode);
            compressNode = 0;
        }

    }

    if (node_left_num != 0) {
        mutable_value_flag_reply->Add(compressNode);
        compressNode = 0;
    }


    reply->set_resp_featdim_size(compress_dim_size);
    if (layerId == 0) {
        reply->set_comp_data_percent(0);
    } else {
        reply->set_comp_data_percent((float) (count_flag_num[0] + count_flag_num[2]) / (float) nodeNum);
    }


//    reply->set_shapedim(embs.begin()->second.size());

}

void compressBitsEmbsTrend(const EmbMessage *request, EmbMessage *reply, map<int, vector<float>> &embs) {
    // 发送压缩嵌入
    vector<Bucket> buckets;
    auto min_value = WorkerStore::embs_min;
    auto max_value = WorkerStore::embs_max;
    int bitNum = request->bitnum();
    int workerId = request->workerid();
    int layerId = request->layerid();
    int epoch = request->epoch();
    int trend = request->trend();

    bool ifCrossZero = false;
    int bucket_num;
    int feat_size=embs.begin()->second.size();
    if (max_value > 0 && min_value < 0) {
        ifCrossZero = true;
    }
    if (ifCrossZero) {
        bucket_num = pow(2, bitNum) - 2;
    } else {
        bucket_num = pow(2, bitNum) - 1;
    }
    reply->set_shapedim(embs.begin()->second.size());

    double interval = (max_value - min_value) / (double) (bucket_num);
    uint int_interval = (int) (interval * pow(10, 8));
    uint int_interval_addone = int_interval + 1;
    interval = int_interval_addone / pow(10, 8);

    if (ifCrossZero) {
        // bucket_num是interval的个数，而不是真正的bucket number
        int bucket_count = 0;
        for (int i = 0; i < bucket_num; i++) {
            // 从第0个区间开始，到bucket_num-1个区间结束,这里的i都是左编号
            if (interval * i <= 0 && interval * (i + 1) >= 0) {
                Bucket b_left{};
                b_left.bid = bucket_count;
                b_left.lower_bound = min_value + interval * i;
                b_left.upper_bound = 0;
                b_left.value = (b_left.lower_bound + b_left.upper_bound) / 2;
                buckets.push_back(b_left);
                reply->add_values(b_left.value);
                bucket_count++;

                Bucket b_right{};
                b_right.bid = bucket_count;
                b_right.lower_bound = 0;
                b_right.upper_bound = min_value + interval * (i + 1);
                b_right.value = (b_right.lower_bound + b_right.upper_bound) / 2;
                buckets.push_back(b_right);
                reply->add_values(b_right.value);
                bucket_count++;
            } else {
                Bucket b{};
                b.bid = bucket_count;
                b.lower_bound = min_value + interval * i;
                b.upper_bound = min_value + interval * (i + 1);
                b.value = (b.lower_bound + b.upper_bound) / 2;
                buckets.push_back(b);
                reply->add_values(b.value);
                bucket_count++;
            }

        }
    } else {
        for (int i = 0; i < bucket_num; i++) {
            Bucket b;
            b.bid = i;
            b.lower_bound = min_value + interval * i;
            b.upper_bound = min_value + interval * (i + 1);
            b.value = (b.lower_bound + b.upper_bound) / 2;
            buckets.push_back(b);
            reply->add_values(b.value);
        }
    }

    Bucket b;
    b.bid = buckets.size();
    b.lower_bound = 0;
    b.upper_bound = 0;
    b.value = 0;
    buckets.push_back(b);
    reply->add_values(0);


    int bucketSize = buckets.size();
    int nodeNum = request->nodes_size();
    reply->set_resp_node_size(nodeNum);
    int oneIntDimNum = 32 / bitNum;
    vector<uint> itemsVec(oneIntDimNum);
    int left_num = feat_size % oneIntDimNum;
    int compress_dim_size;
    if (left_num == 0) {
        compress_dim_size = feat_size / oneIntDimNum;
    } else {
        compress_dim_size = feat_size / oneIntDimNum + 1;
    }
    auto *mutable_emb_reply = reply->mutable_resp_compress_emb_concat();
    mutable_emb_reply->Reserve(nodeNum * compress_dim_size);

    int node_left_num = nodeNum % 16;
    int compress_node_size;
    if (node_left_num == 0) {
        compress_node_size = nodeNum / 16;
    } else {
        compress_node_size = nodeNum / 16 + 1;
    }
    auto *mutable_value_flag_reply = reply->mutable_value_select_flags();
    mutable_value_flag_reply->Reserve(compress_node_size);

    uint compressValue = 0;
    uint compressNode = 0;

    vector<float> count_flag_num(3);

    auto last_embs = WorkerStore::embs_last[workerId][layerId];
    auto rates = WorkerStore::embs_change_rate[workerId][layerId];
    reply->mutable_value_select_flags()->Reserve(nodeNum);

    for (int n = 0; n < nodeNum; n++) {
        auto id = request->nodes(n);
        uint bucket_id;
        // the non-error embs
        auto &emb_second = embs[id];
        auto last_embs_node = last_embs[id];
        auto rates_node = rates[id];
        float comp_error = 0;
        float pred_error = 0;
        float mix_error = 0;


        vector<uint> vec_feat_comp_tmp(compress_dim_size);


        for (uint i = 0; i < feat_size; i++) {
            float dim = emb_second[i];
            if (dim == 0) {
                bucket_id = bucketSize - 1;
            } else if (!ifCrossZero) {
                bucket_id = (int) ((dim - min_value) / interval);
            } else {
                bucket_id = (int) ((dim - min_value) / interval);
                if (dim > 0) {
                    bucket_id += 1;
                }
            }

            uint itemId = i % oneIntDimNum;
            // compare the values of compress, predict and (c+p)/2
            float compDim = buckets[bucket_id].value;
            float predDim = 0;
            if (epoch / trend == 0) {
                // predictive value is the last embs
                predDim = last_embs_node[i];
            } else {
                int round = (epoch + 1) % trend;
                predDim = round * rates_node[i] + last_embs_node[i];
            }
            float mixDim = (predDim + compDim) / 2;


            comp_error += abs(compDim - dim);
            pred_error += abs(predDim - dim);
            mix_error += abs(mixDim - dim);


            compressValue = compressValue | (bucket_id << (32 - (itemId + 1) * bitNum));

            if (itemId == (oneIntDimNum - 1)) {
                vec_feat_comp_tmp[i / oneIntDimNum] = compressValue;
                compressValue = 0;
            }

        }


        if (left_num != 0) {
            vec_feat_comp_tmp[feat_size / oneIntDimNum] = compressValue;
            compressValue = 0;
        }

        uint index_WithMinError = getIndexWithMinError(comp_error, pred_error, mix_error);


        count_flag_num[index_WithMinError]++;


        if (index_WithMinError != 1) {
            // predictive value does not need to be sent,others need to be sent the value of compression
            mutable_emb_reply->Add(vec_feat_comp_tmp.begin(), vec_feat_comp_tmp.end());

        }

        uint itemId_node = n % 16;
        compressNode = compressNode | (index_WithMinError << (32 - (itemId_node + 1) * 2));
        if (itemId_node == (16 - 1)) {
//                    compressData_concat(bitNum, itemsVec, mutable_emb_reply);
            mutable_value_flag_reply->Add(compressNode);
            compressNode = 0;
        }

    }

    if (node_left_num != 0) {
        mutable_value_flag_reply->Add(compressNode);
        compressNode = 0;
    }


    reply->set_resp_featdim_size(compress_dim_size);
    if (layerId == 0) {
        reply->set_comp_data_percent(0);
    } else {
        reply->set_comp_data_percent((float) (count_flag_num[0] + count_flag_num[2]) / (float) nodeNum);
    }
}

Status ServiceImpl::workerPullEmbTrendSelect(
        ServerContext *context, const EmbMessage *request, EmbMessage *reply) {
    // 这里请求的nodes的顺序和返回的tensor的顺序要保持一致
//    clock_t start = clock();

    int epoch = request->epoch();
    int layerId = request->layerid();
    int trend = request->trend();
    int workerId = request->workerid();


    int feat_size = WorkerStore::embs.begin()->second.size();
    auto &embs = WorkerStore::embs;

    int bitNum = request->bitnum();


    reply->set_resp_node_size(request->nodes_size());
    reply->set_shapedim(feat_size);


    if (request->epoch() == 0) {
        // insert worker
        // constuct embs_last and embs_change_rate and return the non-compressed embeddings
        map<int, map<int, vector<float>>> embs_last_worker_tmp;
        WorkerStore::embs_last.insert(pair<int, map<int, map<int, vector<float>>>>(workerId, embs_last_worker_tmp));

        // the changeRateMap for worker i
        map<int, map<int, vector<float>>> changeRateForWorkerI;
        map<int, vector<float>> embs_last_layer;
        map<int, vector<float>> embs_change;
        WorkerStore::embs_last[workerId].insert(pair<int, map<int, vector<float>>>(layerId, embs_last_layer));
        WorkerStore::embs_change_rate.insert(
                pair<int, map<int, map<int, vector<float >> >>(workerId, changeRateForWorkerI));
        WorkerStore::embs_change_rate[workerId].insert(pair<int, map<int, vector<float>>>(layerId, embs_change));
        for (auto id:request->nodes()) {
            vector<float> vec_change(feat_size);
            WorkerStore::embs_last[workerId][layerId].insert(pair<int, vector<float >>(id, embs[id]));
            WorkerStore::embs_change_rate[workerId][layerId].insert(pair<int, vector<float >>(id, vec_change));
        }
        int nodeNum = request->nodes_size();
        reply->set_resp_featdim_size(feat_size);
        for (int i = 0; i < nodeNum; i++) {
            int id = request->nodes(i);
            reply->mutable_resp_none_compress_emb_concat()->Add(embs[id].begin(), embs[id].end());
        }

    } else {
        auto &embs_last_layerId = WorkerStore::embs_last[workerId][layerId];
        auto &embs_change_rate_layerId = WorkerStore::embs_change_rate[workerId][layerId];
        if ((epoch + 1) % trend == 0) {
            // in the last of each ten epochs (e.g., ninth epochs), non-compressed embeddings and the change-rate

            int nodeNum = request->nodes_size();
            reply->set_resp_featdim_size(feat_size);

            reply->mutable_resp_none_compress_rate_concat()->Reserve(nodeNum * feat_size);
            reply->mutable_resp_none_compress_emb_concat()->Reserve(nodeNum * feat_size);
            for (int i = 0; i < nodeNum; i++) {
                int id = request->nodes(i);
                reply->mutable_resp_none_compress_emb_concat()->Add(embs[id].begin(), embs[id].end());
                auto &emb_id = embs[id];
                auto &emb_last_node = embs_last_layerId[id];
                auto &embs_change_rate_node = embs_change_rate_layerId[id];
                for (int j = 0; j < feat_size; j++) {
                    float rate_dim = (emb_id[j] - emb_last_node[j]) / (float) (trend - 1);
                    reply->mutable_resp_none_compress_rate_concat()->Add(rate_dim);
                    embs_change_rate_node[j] = rate_dim;

                }

            }

            embs_last_layerId = embs;

        } else {
            if (bitNum == 1) {
                compress1BitEmbsTrend(request, reply, embs);
            } else {
                compressBitsEmbsTrend(request, reply, embs);
            }
        }
//        reply->set_shapedim(embs.begin()->second.size());
    }


    return Status::OK;
}


//Status ServiceImpl::workerPullGCompress(ServerContext *context, const EmbMessage *request,
//                                        EmbMessage *reply) {
//    int bitNum = request->bitnum();
//    int bucket_num = pow(2, bitNum) - 2;
////    cout<<"bucket_num:"<<bucket_num<<endl;
//    bool ifCompensate = request->ifcompensate();
//    int layerId = request->layerid();
//    int epoch = request->iterround();
//
//
//    // 先判断是否需要补偿
//    map<int, vector<float>> G;
//    float max_value = -10000;
//    float min_value = 10000;
//
//    cout<<"epoch "<<epoch <<", layer id "<<layerId<< "bucket num:"<<bucket_num<< ", ifCompensate:"<<ifCompensate<<endl;
//    cout<<"WorkerStore::G_compensate size:"<< WorkerStore::G_compensate[layerId].size()<<"*"<<WorkerStore::G_compensate[layerId].begin()->second.size()<<endl;
//    cout<<"WorkerStore::G_map size:"<< WorkerStore::G_map[layerId].size()<<"*"<<WorkerStore::G_map[layerId].begin()->second.size()<<endl;
//
//    auto &G_layerId = WorkerStore::G_map[layerId];
//    auto &G_compensate_layerId = WorkerStore::G_compensate[layerId];
//
//    int featNum = G_layerId.begin()->second.size();
//    int nodeNum = request->nodes_size();
//    if (ifCompensate) {
//        if (epoch == 0) {
//            for (int i = 0; i < nodeNum; i++) {
//                auto id = request->nodes(i);
//                vector<float> vec(featNum);
//                auto &G_layerId_nodeId = G_layerId[id];
//                for (int j = 0; j < featNum; j++) {
//                    auto feat_dim = G_layerId_nodeId[j];
//                    vec[j] = feat_dim;
//                    // 求整个返回矩阵的元素的最大值和最小值
//                    if (feat_dim > max_value) {
//                        max_value = feat_dim;
//                    }
//                    if (feat_dim < min_value) {
//                        min_value = feat_dim;
//                    }
//                }
//                G.insert(pair<int, vector<float >>(id, vec));
//            }
//        } else {
//
//            for (int i = 0; i < nodeNum; i++) {
//                vector<float> vec(featNum);
//                auto id = request->nodes(i);
//                auto &G_layerId_nodeId = G_layerId[id];
//                auto &G_compensate_layerId_nodeId = G_compensate_layerId[id];
//                for (int j = 0; j < featNum; j++) {
////                        cout<<"id:"<<layerId<<","<<id<<","<<i<<","<<"embs:"<<WorkerStore::embs[id][i]<<","<<"error compensate:"<<WorkerStore::embs_compensate[layerId][id][i]<<endl;
////                        float feat_dim=WorkerStore::embs[id][i]+WorkerStore::embs_compensate[layerId][id][i];
//                    float feat_dim = G_layerId_nodeId[j] + G_compensate_layerId_nodeId[j];
////                    }
//
//                    vec[j] = feat_dim;
//                    // 求整个返回矩阵的元素的最大值和最小值
//                    if (feat_dim > max_value) {
//                        max_value = feat_dim;
//                    }
//                    if (feat_dim < min_value) {
//                        min_value = feat_dim;
//                    }
//                }
//
//                G.insert(pair<int, vector<float >>(id, vec));
//            }
//        }
//    } else {
//        for (int i = 0; i < nodeNum; i++) {
//            auto id = request->nodes(i);
//            vector<float> vec(featNum);
//            auto &G_layerId_nodeId = G_layerId[id];
//            for (int j = 0; j < featNum; j++) {
//                auto feat_dim = G_layerId_nodeId[j];
//                vec[j] = feat_dim;
//                // 求整个返回矩阵的元素的最大值和最小值
//                if (feat_dim > max_value) {
//                    max_value = feat_dim;
//                }
//                if (feat_dim < min_value) {
//                    min_value = feat_dim;
//                }
//            }
//
//            G.insert(pair<int, vector<float >>(id, vec));
//        }
//    }
//
//    cout<<"max,min:"<<max_value<<","<<min_value<<endl;
//    // 上面是用上一轮补偿了这一轮
//    // 下面是计算压缩和发送，以及发送误差
//    vector<Bucket> buckets;
//    float interval = (max_value - min_value) / (float) (bucket_num);
//    if (min_value < 0 && max_value > 0) {
//        for (int i = 0; i < bucket_num + 1; i++) {
//            if (min_value + interval * i < 0 && min_value + interval * (i + 1) > 0) {
//                // 建两个桶,以0的分界线
//                Bucket b1;
//                b1.bid = i;
//                b1.lower_bound = min_value + interval * i;
//                b1.upper_bound = 0;
//                b1.value = (b1.lower_bound + b1.upper_bound) / 2;
//                buckets.push_back(b1);
//                reply->add_values((b1.lower_bound + b1.upper_bound) / 2);
//
//                i = i + 1;
//                Bucket b2;
//                b2.bid = i;
//                b2.lower_bound = 0;
//                b2.upper_bound = min_value + interval * (i + 1);
//                if (i == bucket_num) {
//                    b2.upper_bound = max_value;
//                }
//                b2.value = (b2.lower_bound + b2.upper_bound) / 2;
//                buckets.push_back(b2);
//                reply->add_values((b2.lower_bound + b2.upper_bound) / 2);
//            } else {
//                Bucket b;
//                b.bid = i;
//                b.lower_bound = min_value + interval * i;
//                b.upper_bound = min_value + interval * (i + 1);
//                if (i == bucket_num - 1) {
//                    b.upper_bound = max_value;
//                }
//                b.value = (b.lower_bound + b.upper_bound) / 2;
//                buckets.push_back(b);
//                reply->add_values((b.lower_bound + b.upper_bound) / 2);
//            }
//        }
//    } else {
//        for (int i = 0; i < bucket_num; i++) {
//            Bucket b;
//            b.bid = i;
////            cout<< "bid:" << b.bid<<endl;
//            b.lower_bound = min_value + interval * i;
//            b.upper_bound = min_value + interval * (i + 1);
//            if (i == bucket_num - 1) {
//                b.upper_bound = max_value;
//            }
//            b.value = (b.lower_bound + b.upper_bound) / 2;
//            buckets.push_back(b);
//            reply->add_values((b.lower_bound + b.upper_bound) / 2);
//        }
//    }
//
//    Bucket b;
//    b.bid = buckets.size();
//    b.lower_bound = 0;
//    b.upper_bound = 0;
//    b.value = 0;
//    buckets.push_back(b);
//    reply->add_values(0);
//
//    int feat_size = G_layerId.begin()->second.size();
//    int oneIntDimNum = 32 / bitNum;
//    vector<uint> itemsVec(oneIntDimNum);
//
//    int left_num = feat_size % oneIntDimNum;
//    int compress_dim_size;
//    if (left_num == 0) {
//        compress_dim_size = feat_size / oneIntDimNum;
//    } else {
//        compress_dim_size = feat_size / oneIntDimNum + 1;
//    }
//
//    auto *mutable_reply = reply->mutable_resp_compress_emb_concat();
//    mutable_reply->Reserve(nodeNum * compress_dim_size);
//
//    cout<<"bucket size:"<<buckets.size()<<endl;
//
//    if (ifCompensate) {
//        if (epoch == 0) {
//            // 第0轮迭代需要新建误差结构
//
//            for (int m = 0; m < nodeNum; m++) {
//                auto id = request->nodes(m);
//                auto &g_node = G[id];
////                const auto &g=G[id];
//                vector<float> error(featNum);
//                for (int j = 0; j < feat_size; j++) {
//                    float dim = g_node[j];
//                    int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                    tensor->add_tensor(bucket_id);
//                    int itemId = j % oneIntDimNum;
//                    itemsVec[itemId] = bucket_id;
//                    if (itemId == (oneIntDimNum - 1)) {
//                        compressData_concat(bitNum, itemsVec, mutable_reply);
//                    }
//
//                    error[j] = (dim - buckets[bucket_id].value);
//                }
//
//                if (left_num != 0) {
//                    uint compress_value = 0;
//                    if (bitNum == 2) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (30 - 2 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    } else if (bitNum == 4) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (28 - 4 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    } else if (bitNum == 8) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (24 - 8 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    } else if (bitNum == 16) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (16 - 16 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    }
//                    mutable_reply->Add(compress_value);
////                    tensor->add_tensor(compress_value);
////                    itemsVec.clear();
//                }
//                WorkerStore::G_compensate[layerId].insert(pair<int, vector<float >>(id, error));
//            }
//        } else {
//            // 需要先加误差，然后算出在哪个桶中，再算误差
//
////            if(WorkerStore::G_compensate.count(layerId)==0){
////                map<int,vector<float>> map_tmp;
////                WorkerStore::G_compensate.insert(pair<int,map<int,vector<float>>>(layerId,map_tmp));
////            }
////            auto G_compensate_layerId_tmp=WorkerStore::G_compensate[layerId];
//
//
//            for (int m = 0; m < nodeNum; m++) {
//                auto id = request->nodes(m);
//                const auto &g = G[id];
////                IntTensorMessage *tensor = reply->add_embs();
////                tensor->set_vid(id);
//                auto &G_compensate_layerId_nodeId = G_compensate_layerId[id];
//                for (int i = 0; i < featNum; i++) {
//                    float dim = g[i];
//                    int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                    tensor->add_tensor(bucket_id);
////                    itemsVec.push_back(bucket_id);
//                    int itemId = i % oneIntDimNum;
//                    itemsVec[itemId] = bucket_id;
//                    if (itemId == (oneIntDimNum - 1)) {
//                        compressData_concat(bitNum, itemsVec, mutable_reply);
//                    }
//                    G_compensate_layerId_nodeId[i] = (dim - buckets[bucket_id].value);
//                }
//
//                if (left_num != 0) {
//                    uint compress_value = 0;
//                    if (bitNum == 2) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (30 - 2 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    } else if (bitNum == 4) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (28 - 4 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    } else if (bitNum == 8) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (24 - 8 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    } else if (bitNum == 16) {
//                        for (int i = 0; i < left_num; i++) {
//                            itemsVec[i] = itemsVec[i] << (16 - 16 * i);
//                            compress_value = compress_value | itemsVec[i];
//                        }
//                    }
//                    mutable_reply->Add(compress_value);
////                    tensor->add_tensor(compress_value);
////                    itemsVec.clear();
//                }
//
//            }
//        }
//    } else {
//
//        // 开始构建压缩后的张量
//        cout<<"compress begin"<<endl;
//        for (int m = 0; m < nodeNum; m++) {
//            auto id = request->nodes(m);
////            cout<<"1"<<endl;
//            const auto &g = G[id];
////            cout<<"2"<<endl;
////            IntTensorMessage *tensor = reply->add_embs();
////            tensor->set_vid(id);
//            for (int n = 0; n < featNum; n++) {
////                cout<<"3"<<endl;
//                auto dim = g[n];
////                cout<<"4"<<endl;
//                int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                cout<<"5"<<endl;
////                tensor->add_tensor(bucket_id);
//                int itemId = n % oneIntDimNum;
//                itemsVec[itemId] = bucket_id;
//                if (itemId == (oneIntDimNum - 1)) {
////                    cout<<"6"<<endl;
//                    compressData_concat(bitNum, itemsVec, mutable_reply);
////                    cout<<"7"<<endl;
//                }
//
//
//            }
//            if (left_num != 0) {
////                cout<<"8"<<endl;
//                uint compress_value = 0;
//                if (bitNum == 2) {
//                    for (int i = 0; i < left_num; i++) {
//                        itemsVec[i] = itemsVec[i] << (30 - 2 * i);
//                        compress_value = compress_value | itemsVec[i];
//                    }
//                } else if (bitNum == 4) {
//                    for (int i = 0; i < left_num; i++) {
//                        itemsVec[i] = itemsVec[i] << (28 - 4 * i);
//                        compress_value = compress_value | itemsVec[i];
//                    }
//                } else if (bitNum == 8) {
//                    for (int i = 0; i < left_num; i++) {
//                        itemsVec[i] = itemsVec[i] << (24 - 8 * i);
//                        compress_value = compress_value | itemsVec[i];
//                    }
//                } else if (bitNum == 16) {
//                    for (int i = 0; i < left_num; i++) {
//                        itemsVec[i] = itemsVec[i] << (16 - 16 * i);
//                        compress_value = compress_value | itemsVec[i];
//                    }
//                }
////                cout<<"9 "<<endl;
//                mutable_reply->Add(compress_value);
////                itemsVec.clear();
//            }
//        }
////        cout<<"compress end"<<endl;
//    }
//    reply->set_resp_node_size(nodeNum);
//    reply->set_resp_featdim_size(compress_dim_size);
//    reply->set_shapedim(G_layerId.begin()->second.size());
////    cout << "G_layerId.begin()->second.size()" << reply->shapedim() << endl;
//
//    return Status::OK;
//
//
//}

Status ServiceImpl::workerSendTrainNode(ServerContext *context, const NodeMessage *request, BoolMessage *reply) {
    unique_lock<mutex> lck(ThreadUtil::mtx_sendNode);
    for (int i = 0; i < request->nodes_size(); i++) {
        int n = request->nodes(i);
        ServerStore::train_nodes.push_back(n);
    }
    return Status::OK;
}

Status ServiceImpl::serverSendTrainNode(ServerContext *context, const ContextMessage *request, NodeMessage *reply) {
    vector<int> train_nodes = ServerStore::train_nodes;
    for (int n : train_nodes) {
        reply->add_nodes(n);
    }
    return Status::OK;
}

Status ServiceImpl::workerSendValNode(ServerContext *context, const NodeMessage *request, BoolMessage *reply) {
    unique_lock<mutex> lck(ThreadUtil::mtx_sendNode);
    for (int i = 0; i < request->nodes_size(); i++) {
        int n = request->nodes(i);
        ServerStore::val_nodes.push_back(n);
    }
    return Status::OK;
}

Status ServiceImpl::serverSendValNode(ServerContext *context, const ContextMessage *request, NodeMessage *reply) {
    vector<int> train_nodes = ServerStore::val_nodes;
    for (int n : train_nodes) {
        reply->add_nodes(n);
    }
    return Status::OK;
}

Status ServiceImpl::workerSendTestNode(ServerContext *context, const NodeMessage *request, BoolMessage *reply) {
    unique_lock<mutex> lck(ThreadUtil::mtx_sendNode);
    for (int i = 0; i < request->nodes_size(); i++) {
        int n = request->nodes(i);
        ServerStore::test_nodes.push_back(n);
    }
    return Status::OK;
}

Status ServiceImpl::serverSendTestNode(ServerContext *context, const ContextMessage *request, NodeMessage *reply) {
    vector<int> train_nodes = ServerStore::test_nodes;
    for (int n : train_nodes) {
        reply->add_nodes(n);
    }
    return Status::OK;
}

Status ServiceImpl::server_PullParams(ServerContext *context,const StringM *request, Param* reply){
    const string& lay_id=request->value();
    reply->mutable_elems()->Add(ServerStore::params[lay_id].begin(),ServerStore::params[lay_id].end());
    return Status::OK;
}

Status ServiceImpl::server_updateModels(ServerContext *context, const GradMessage* request, BoolMessage *reply){
    if (request->wid() == 0) {
        unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);
        ServerStore::grads_agg[request->grad().id()].clear();
        // vector is initialized as 0 by default
        vector<float> tmp(request->grad().elems_size());
        ServerStore::grads_agg[request->grad().id()]=tmp;
        cout<<"********server_updateModels-clear gradient aggregations******"<<endl;
        ThreadUtil::ready_updateModels = true;
        ThreadUtil::cv_updateModels.notify_all();
    } else {
        unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);
        while (!ThreadUtil::ready_updateModels) {
            ThreadUtil::cv_updateModels.wait(lck);
        }
    }
    int grad_size=request->grad().elems_size();
    string grad_id=request->grad().id();
    float alpha = request->lr();
    int wid=request->wid();

    // 多个worker一起更新参数，先聚合所有worker的梯度
    // 聚合worker的梯度时，先上锁
//    pthread_mutex_lock(&ThreadUtil::mtx_updateModels_addGrad);
    unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);

    auto& grad_agg=ServerStore::grads_agg[grad_id];
    // add gradients to grads_agg
    for(int i=0;i<grad_size;i++){
        grad_agg[i]+=grad_agg[i]+request->grad().elems(i);
    }
    cout<<"********server_updateModels----gradient aggregating end******"<<endl;
    lck.unlock();

    // 每个worker累积完梯度就可以释放锁了
//    pthread_mutex_unlock(&ThreadUtil::mtx_updateModels_addGrad);

    // 有一个线程更新参数,更新参数的前提是所有梯度都已聚合完成
    // 确保所有机器都已到达

    lck.lock();
    ThreadUtil::count_worker_for_updateModels++;
    if (ThreadUtil::count_worker_for_updateModels == ServerStore::worker_num) {
        ThreadUtil::cv_updateModels.notify_all();
        ThreadUtil::count_worker_for_updateModels = 0;
        ThreadUtil::ready_updateModels = false;
    } else {
        ThreadUtil::cv_updateModels.wait(lck);
    }

    // 下面是做check
    if (wid == 0) {
        cout<<ThreadUtil::count_worker_for_updateModels<<" workers have been added into the gradient aggregations!"<<endl;
        cout<<"param id:"<<grad_id<<","<<"grad size:"<<grad_agg.size()<<endl;
    }

    // worker 0线程开始负责更新参数
    if (wid == 0) {
        ServerStore::t++;
        float beta_1 = 0.9;
        float beta_2 = 0.999;
        float epsilon = 5e-4;
        bool isAdam = true;
        auto& m_grads_t=ServerStore::m_grads_t[grad_id];
        auto& v_grads_t=ServerStore::v_grads_t[grad_id];
        auto& param=ServerStore::params[grad_id];
        // 如果m_weight_t,v_weight_t,m_bias_t,v_bias_t为空，那么初始化
        for(int i=0;i<grad_size;i++){
            float g_t=grad_agg[i];
            if(isAdam){
                m_grads_t[i]=beta_1* m_grads_t[i]+(1-beta_1)*g_t;
                v_grads_t[i]=beta_2*v_grads_t[i]+(1-beta_2)*g_t*g_t;
                float m_cap=m_grads_t[i]/(1-(pow(beta_1,ServerStore::t)));
                float v_cap=v_grads_t[i]/(1-(pow(beta_2,ServerStore::t)));
                param[i]-=(alpha*m_cap)/(sqrt(v_cap)+epsilon);
            }else{
                param[i]-=alpha*g_t;
            }
        }


    }


    return Status::OK;
}


Status ServiceImpl::server_aggGrad(ServerContext *context, const GradMessage *request, GradMessage *reply) {
    // fix the adam
    if (request->wid() == 0) {
        unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);
        ServerStore::grads_agg[request->grad().id()].clear();
        // vector is initialized as 0 by default
        vector<float> tmp(request->grad().elems_size());
        ServerStore::grads_agg[request->grad().id()] = tmp;
        cout << "********server_updateModels-clear gradient aggregations******" << endl;
        ThreadUtil::ready_updateModels = true;
        ThreadUtil::cv_updateModels.notify_all();
    } else {
        unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);
        while (!ThreadUtil::ready_updateModels) {
            ThreadUtil::cv_updateModels.wait(lck);
        }
    }
    int grad_size = request->grad().elems_size();
    string grad_id = request->grad().id();
    float alpha = request->lr();
    int wid = request->wid();

    // 多个worker一起更新参数，先聚合所有worker的梯度
    // 聚合worker的梯度时，先上锁
//    pthread_mutex_lock(&ThreadUtil::mtx_updateModels_addGrad);
    unique_lock<mutex> lck(ThreadUtil::mtx_updateModels);

    auto &grad_agg = ServerStore::grads_agg[grad_id];
    // add gradients to grads_agg
    for (int i = 0; i < grad_size; i++) {
        grad_agg[i] += grad_agg[i] + request->grad().elems(i);
    }
    cout << "********server_updateModels----gradient aggregating end******" << endl;
    lck.unlock();

    // 每个worker累积完梯度就可以释放锁了
//    pthread_mutex_unlock(&ThreadUtil::mtx_updateModels_addGrad);

    // 有一个线程更新参数,更新参数的前提是所有梯度都已聚合完成
    // 确保所有机器都已到达

    lck.lock();
    ThreadUtil::count_worker_for_updateModels++;
    if (ThreadUtil::count_worker_for_updateModels == ServerStore::worker_num) {
        ThreadUtil::cv_updateModels.notify_all();
        ThreadUtil::count_worker_for_updateModels = 0;
        ThreadUtil::ready_updateModels = false;
    } else {
        ThreadUtil::cv_updateModels.wait(lck);
    }

    // 下面是做check
    if (wid == 0) {
        cout << ThreadUtil::count_worker_for_updateModels << " workers have been added into the gradient aggregations!"
             << endl;
        cout << "param id:" << grad_id << "," << "grad size:" << grad_agg.size() << endl;
    }


    auto* grad_message_tmp=reply->grad().New();
    grad_message_tmp->mutable_elems()->Add(grad_agg.begin(),grad_agg.end());
    grad_message_tmp->set_id(grad_id);
    reply->set_allocated_grad(grad_message_tmp);

    return Status::OK;
}