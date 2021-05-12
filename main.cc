//
// Created by songzhen on 2020/9/26.
//

#include "core/service/dgnn_client.h"
#include "cpptest/bittest.h"
#include <math.h>
#include <map>

using namespace std;

void testSelect( EmbMessage *request);
void testCopyMap() {
    map<int, vector<float>> a;
    for (int i = 0; i < 10; i++) {
        vector<float> vec_tmp(5);
        for (int j = 0; j < 5; j++) {
            vec_tmp[j] = j;
        }
        a.insert(pair<int, vector<float>>(i, vec_tmp));
    }
    map<int, vector<float>> b;
    b = a;
    a[5][0] = 100;
    cout << "a" << endl;
}

//GeneralPartition generalPartition;
void testPartition() {
//    generalPartition.init(0, 2,"a",0,0);
    int workerNum = 5;
    char *buffer;
    buffer = getcwd(NULL, 0);
    string pwd = buffer;
    if (pwd[pwd.length() - 1] != '/') {
        pwd += '/';
    }
    // 开始处理数据集
//    string adjFile = pwd+"data_raw/cora/edges.txt";
//    string featFile = pwd+"data_raw/cora/featsClass.txt";

    string adjFile = pwd + "../data/test" + "/edges.txt";
    string featFile = pwd + "../data/test" + "/featsClass.txt";
    string partitionFile = pwd + "../data/test" + "/nodesPartition.txt";
    cout << "adjfile:" << adjFile << endl;
    cout << "featFle:" << featFile << endl;
    cout << "nodesPartition:" << partitionFile << endl;


    ifstream partitionInFile(partitionFile);
    string temp;
    if (!partitionInFile.is_open()) {
        cout << "partitionInFile 未成功打开文件" << endl;
    }

    int count_worker = 0;
    while (getline(partitionInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vSize = v.size();
        vector<int> v_tmp(vSize);
        GeneralPartition::nodes.push_back(v_tmp);
        auto &nodesWorker = GeneralPartition::nodes[count_worker];
//        vector<int> a(vSize);


        for (int i = 0; i < vSize; i++) {
            nodesWorker[i] = (atoi(v[i].c_str()));
        }

        count_worker++;
    }
    cout << GeneralPartition::nodes[0].size() << endl;
    cout << GeneralPartition::nodes[1].size() << endl;


    ifstream adjInFile(adjFile);

    if (!adjInFile.is_open()) {
        cout << "adjInFile 未成功打开文件" << endl;
    }


    map<int, set<int>> adj_map;
    int count = 0;
    int count_flag = 0;
    cout << "正在处理邻接表数据" << endl;
    while (getline(adjInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        int neibor_id = atoi(v[1].c_str());

        // 开始构造邻接表

        if (adj_map.count(vertex_id) == 0) {
            set<int> set_temp;
            set_temp.insert(neibor_id);
            adj_map[vertex_id] = set_temp;
        } else {
            set<int> set_temp = adj_map[vertex_id];
            set_temp.insert(neibor_id);
            adj_map[vertex_id] = set_temp;
        }

        if (adj_map.count(neibor_id) == 0) {
            set<int> set_temp;
            set_temp.insert(vertex_id);
            adj_map[neibor_id] = set_temp;
        } else {
            set<int> set_temp = adj_map[neibor_id];
            set_temp.insert(vertex_id);
            adj_map[neibor_id] = set_temp;
        }
        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }

    }


    int edge_num = count_flag;
    int data_num = 20;
    int feature_size = 5;
    adjInFile.close();

    map<int, vector<float>> feature;
    vector<int> label_array(data_num); // 如果需要获取length，那么这块只能赋值常量
//    map<string, int> label_map;
    int count_label = 0;

    ifstream featInFile(featFile);
    if (!featInFile.is_open()) {
        cout << "未成功打开文件" << endl;
    }

    count = 0;
    count_flag = 0;
    cout << "正在处理特征数据 " << endl;
    while (getline(featInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        vector<float> vec_feat;
        for (int i = 1; i < feature_size + 1; i++) {
            vec_feat.push_back(atof(v[i].c_str()));
        }
        feature.insert(pair<int, vector<float>>(vertex_id, vec_feat));

//        cout<<"label:"<<label_new<<endl;
        label_array[vertex_id] = atoi(v[feature_size + 1].c_str());
        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }
    }

    featInFile.close();


    for (int i = 0; i < workerNum; i++) {
        auto &node_worker_i = GeneralPartition::nodes[i];
        int nodeSize = node_worker_i.size();
        map<int, set<int> > adjForWorkerI_tmp;
        GeneralPartition::adjs.push_back(adjForWorkerI_tmp);
        auto &adjForWorkerI = GeneralPartition::adjs[i];

        map<int, int> label_tmp;
        GeneralPartition::labels.push_back(label_tmp);
        auto &labelWorkerI = GeneralPartition::labels[i];

        map<int, vector<float>> feat_tmp;
        GeneralPartition::features.push_back(feat_tmp);
        auto &featWorkerI = GeneralPartition::features[i];

        for (int j = 0; j < nodeSize; j++) {
            int nodeId = node_worker_i[j];
            auto &neiborVecForNode = adj_map[nodeId];
            auto &featVecForNode = feature[nodeId];
            adjForWorkerI.insert(pair<int, set<int>>(nodeId, neiborVecForNode));
            labelWorkerI.insert(pair<int, int>(nodeId, label_array[nodeId]));
            featWorkerI.insert(pair<int, vector<float>>(nodeId, featVecForNode));
        }
    }
    cout << "aaa" << endl;

}

void testVector() {
    int len = 1000000;
    vector<float> vec_1;
    vector<float> vec_2(len);

    clock_t start = clock();
    for (int i = 0; i < len; i++) {
        vec_1.push_back(0);
    }
    clock_t end = clock();
    cout << " time1:" << (double) (end - start) / CLOCKS_PER_SEC << endl;


    start = clock();
    for (int i = 0; i < len; i++) {
        vec_2[i] = 0;
    }
    end = clock();
    cout << " time2:" << (double) (end - start) / CLOCKS_PER_SEC << endl;
}


void testVectorAndArray() {
    vector<float> vec(1000000);
    float a[1000000];

    clock_t start = clock();
    for (int i = 0; i < 1000000; i++) {
        vec[i] += 1;
    }
    clock_t end = clock();
    cout << " time1:" << (double) (end - start) / CLOCKS_PER_SEC << endl;

    start = clock();
    for (float &i : a) {
        i += 1;
    }
    end = clock();
    cout << " time2:" << (double) (end - start) / CLOCKS_PER_SEC << endl;

    start = clock();
    for (int i = 0; i < 1000000; i++) {
        a[i] += 1;
    }
    end = clock();
    cout << " time3:" << (double) (end - start) / CLOCKS_PER_SEC << endl;


    start = clock();
    for (auto &i:vec) {
        i += 1;
    }
    end = clock();
    cout << " time4:" << (double) (end - start) / CLOCKS_PER_SEC << endl;
}

void testbit() {
    BitClass bitClass;
    bitClass.print_a_bit();

}

void test_vec_clear() {
    vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    cout << "size:" << vec.size() << endl;
    vec.clear();
    cout << "clear size:" << vec.size() << endl;
}

void testCompress() {
    int aa = 250;
    uint a = aa;
    uint b = 2;
    uint c = 3;
    uint d = 250;
    a = a << 24;
    b = b << 16;
    c = c << 8;
    uint e = a | b | c | d;
//    cout<<e<<endl;

    uint a_1 = e >> 24;
    uint b_1 = e >> 16 & 0x000000ff;
    uint c_1 = e >> 8 & 0x000000ff;
    uint d_1 = e & 0x000000ff;


    cout << a_1 << "," << b_1 << "," << c_1 << "," << d_1 << endl;

}

void testCompressServerCode() {


    vector<int> fourItemsVec;
    vector<int> emb;
    vector<int> compressVs;
    for (int i = 0; i < 9; i++) {
        emb.push_back(i);
    }
    emb.push_back(100);

    for (auto dim:emb) {
        fourItemsVec.push_back(dim);
        if (fourItemsVec.size() == 4) {
            // compress
            fourItemsVec[0] = fourItemsVec[0] << 24;
            fourItemsVec[1] = fourItemsVec[1] << 16;
            fourItemsVec[2] = fourItemsVec[2] << 8;
//                        fourItemsVec[3]=fourItemsVec[3];
            int compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
            compressVs.push_back(compress_value);
            fourItemsVec.clear();
        }
//                    tensor->add_tensor(bucket_id);
    }
    // 某个嵌入向量全部压缩wan, haiyou sheng yu de meiyasuo
    if (fourItemsVec.size() != 0) {
        int compress_value = 0;
        for (int i = 0; i < fourItemsVec.size(); i++) {
            fourItemsVec[i] = fourItemsVec[i] << (8 * (3 - i));
            compress_value = compress_value | fourItemsVec[i];
        }
        compressVs.push_back(compress_value);

        fourItemsVec.clear();
    }



    // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
    int shape_dim = 10;
    for (int i = 0; i < compressVs.size(); i++) {
        if (i * 4 + 3 < shape_dim) {
            int dim = compressVs[i];
            int a_1 = dim >> 24;
            int a_2 = dim >> 16 & 0x000f;
            int a_3 = dim >> 8 & 0x000f;
            int a_4 = dim & 0x000f;

        } else {
            int num = shape_dim - i * 4;
            int dim = compressVs[i];
            vector<int> a;
            a.push_back(dim >> 24);
            a.push_back(dim >> 16 & 0x000f);
            a.push_back(dim >> 8 & 0x000f);
            a.push_back(dim & 0x000f);

        }
    }


}

void testCode() {
    map<int, vector<float>> embs_compensate_layerId;

    for (int i = 0; i < 5; i++) {
        vector<float> tmp;
        for (int j = 0; j < 10; j++) {
            tmp.push_back(1);
        }
        embs_compensate_layerId.insert(pair<int, vector<float>>(i, tmp));
    }

    cout << "layerid:" << 0 << endl;
    for (int i = 0; i < embs_compensate_layerId.size(); i++) {
        for (int j = 0; j < embs_compensate_layerId.begin()->second.size(); j++) {
            cout << embs_compensate_layerId[i][j] << " ";
        }
        cout << " " << endl;
    }
}

struct Bucket {
    float lower_bound;
    float upper_bound;
    int bid;
    float value;
};


int getDimBucket_main(const vector<Bucket> &buckets, float dim, float min_value, float max_value, float interval) {

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

uint oneByteCompress(vector<uint> fourItemsVec) {

    fourItemsVec[0] = fourItemsVec[0] << 24;
    fourItemsVec[1] = fourItemsVec[1] << 16;
    fourItemsVec[2] = fourItemsVec[2] << 8;
//                        fourItemsVec[3]=fourItemsVec[3];
    uint compress_value = fourItemsVec[0] | fourItemsVec[1] | fourItemsVec[2] | fourItemsVec[3];
    return compress_value;
}

//void testChangeRate(int epoch) {
//
//    int layerId = 0;
//    int trend = 3;
//    int bucket_num = 3;
//    vector<int> nodes;
//    nodes.push_back(0);
//    nodes.push_back(1);
//    nodes.push_back(2);
//    int workerId=0;
//
//
//    int feat_size = WorkerStore::embs.begin()->second.size();
//    auto &embs_ws = WorkerStore::embs;
//    map<int, vector<float>> embs;
//    float max_value = -10000;
//    float min_value = 10000;
//
//    bool setBValueAslow = false;
//
//    if (epoch == 0) {
//        map<int, vector<float>> embs_last_layer;
//        map<int, vector<float>> embs_change;
//        WorkerStore::embs_last.insert(pair<int, map<int, vector<float>>>(layerId, embs_last_layer));
//        WorkerStore::embs_change_rate[workerId].insert(pair<int, map<int, vector<float>>>(layerId, embs_change));
//        for (auto id:nodes) {
//            vector<float> vec;
//            vector<float> vec_change;
//            auto &embs_node_ws = embs_ws[id];
//            for (int i = 0; i < feat_size; i++) {
//                vec.push_back(embs_node_ws[i]);
//                vec_change.push_back(0);
//                if (embs_node_ws[i] > max_value) {
//                    max_value = embs_node_ws[i];
//                }
//                if (embs_node_ws[i] < min_value) {
//                    min_value = embs_node_ws[i];
//                }
//            }
//            embs.insert(pair<int, vector<float>>(id, vec));
//            WorkerStore::embs_last[layerId].insert(pair<int, vector<float>>(id, vec));
//            WorkerStore::embs_change_rate[workerId][layerId].insert(pair<int, vector<float>>(id, vec_change));
//        }
//    } else {
//        for (auto id:nodes) {
//            auto &emb_last = WorkerStore::embs_last[layerId][id];
//            auto &emb_change = WorkerStore::embs_change_rate[workerId][layerId][id];
//            vector<float> vec;
//            auto &embs_node_ws = embs_ws[id];
//
//            for (int i = 0; i < feat_size; i++) {
//                float change = (embs_node_ws[i] - emb_last[i]) / trend;
//                emb_last[i] = embs_node_ws[i];
//                emb_change[i] += change;
//                vec.push_back(embs_node_ws[i]);
//
//                if (embs_node_ws[i] > max_value) {
//                    max_value = embs_node_ws[i];
//                }
//                if (embs_node_ws[i] < min_value) {
//                    min_value = embs_node_ws[i];
//                }
//
//            }
//
//            embs.insert(pair<int, vector<float>>(id, vec));
//        }
//    }
//
//    if ((epoch + 1) % trend == 0) {
//        // 发送不压缩嵌入，并发送变化率
//
//        for (auto id:nodes) {
//
//
//            auto &emb_id = embs[id];
//            auto &change_id = WorkerStore::embs_change_rate[workerId][layerId][id];
//            for (int i = 0; i < feat_size; i++) {
//
//                change_id[i] = 0;
//            }
//        }
//
//
//    } else {
//        // 发送压缩嵌入
//        vector<Bucket> buckets;
//        float interval = (max_value - min_value) / (float) (bucket_num);
////        clock_t start_compress = clock();
//        if (min_value < 0 && max_value > 0) {
//            for (int i = 0; i < bucket_num + 1; i++) {
//                if (min_value + interval * i < 0 && min_value + interval * (i + 1) > 0) {
//                    // 建两个桶,以0的分界线
//                    Bucket b1;
//                    b1.bid = i;
//                    b1.lower_bound = min_value + interval * i;
//                    b1.upper_bound = 0;
//
//                    b1.value = (b1.lower_bound + b1.upper_bound) / 2;
//
//                    buckets.push_back(b1);
//
//
//                    i = i + 1;
//                    Bucket b2;
//                    b2.bid = i;
//                    b2.lower_bound = 0;
//                    b2.upper_bound = min_value + interval * (i + 1);
//                    if (i == bucket_num) {
//                        b2.upper_bound = max_value;
//                    }
//
//                    b2.value = (b2.lower_bound + b2.upper_bound) / 2;
//
//
//                    buckets.push_back(b2);
//
//                } else {
//                    Bucket b;
//                    b.bid = i;
//                    b.lower_bound = min_value + interval * i;
//                    b.upper_bound = min_value + interval * (i + 1);
//                    if (i == bucket_num - 1) {
//                        b.upper_bound = max_value;
//                    }
//                    if (b.lower_bound < 0 && setBValueAslow) {
//                        b.value = b.upper_bound;
//                    } else if (b.lower_bound > 0 && setBValueAslow) {
//                        b.value = b.lower_bound;
//                    } else {
//                        b.value = (b.lower_bound + b.upper_bound) / 2;
//                    }
//
//                    buckets.push_back(b);
//
//                }
//            }
//        } else {
//            for (int i = 0; i < bucket_num; i++) {
//                Bucket b;
//                b.bid = i;
//                b.lower_bound = min_value + interval * i;
//                b.upper_bound = min_value + interval * (i + 1);
//                if (i == bucket_num - 1) {
//                    b.upper_bound = max_value;
//                }
//
//                if (b.lower_bound < 0 && setBValueAslow) {
//                    b.value = b.upper_bound;
//                } else if (b.lower_bound > 0 && setBValueAslow) {
//                    b.value = b.lower_bound;
//                } else {
//                    b.value = (b.lower_bound + b.upper_bound) / 2;
//                }
//                buckets.push_back(b);
//
//            }
//        }
//
//        Bucket b;
//        b.bid = buckets.size();
//        b.lower_bound = 0;
//        b.upper_bound = 0;
//        b.value = 0;
//        buckets.push_back(b);
//
//
//        vector<uint> fourItemsVec;
//
//        for (const auto &emb:embs) {
//
//            int bucket_id;
//            auto &emb_second = emb.second;
//            for (int i = 0; i < feat_size; i++) {
//                bucket_id = getDimBucket(buckets, emb_second[i], min_value, max_value, interval);
//                fourItemsVec.push_back(bucket_id);
//                if (fourItemsVec.size() == 4) {
//                    // compress
//                    uint compress_value = oneByteCompress(fourItemsVec);
//
//                    fourItemsVec.clear();
//                }
////                    if (layerId == 1 && emb.first == 3 && i == 5) {
////                        cout << "compensate value:" << emb.second[5] << ",compress value:" << buckets[bucket_id].value
////                             << ",bucket id:" << bucket_id
////                             << ",error:" << emb.second[5] - buckets[bucket_id].value << endl;
////                    }
//
//            }
//
//            // 某个嵌入向量全部压缩wan, haiyou sheng yu de meiyasuo
//            if (fourItemsVec.size() != 0) {
//                uint compress_value = 0;
//                for (int i = 0; i < fourItemsVec.size(); i++) {
//                    fourItemsVec[i] = fourItemsVec[i] << 8 * (3 - i);
//                    compress_value = compress_value | fourItemsVec[i];
//                }
//
//
//                fourItemsVec.clear();
//            }
//
//        }
//
//
//    }
//}

//void test_ChangeRate() {
//
//    for (int i = 0; i < 3; i++) {
//        vector<float> vector_tmp;
//        for (int j = 0; j < 10; j++) {
//            vector_tmp.push_back((float) j / 10);
//        }
//        WorkerStore::embs.insert(pair<int, vector<float>>(i, vector_tmp));
//    }
//
//    for (int i = 0; i < 100; i++) {
//        testChangeRate(i);
//        for (int m = 0; m < 3; m++) {
//            for (int j = 0; j < 10; j++) {
//                WorkerStore::embs[m][j] += 1;
//            }
//
//        }
//    }
//}

void changeRateWorker() {
//    int shape0 = 0;
//    int shape1 = 0;
//    int shape_dim = reply.shapedim();
//
//
//    if ((epoch+1)%trend==0) {
//        shape0 = reply.denseembmessage().embs_size();
//        shape1 = reply.denseembmessage().embs().begin()->tensor_size();
//    } else  {
//        shape0 = reply.embs_size();
//        shape1 = reply.embs().begin()->tensor_size();
//    }
//    auto result = py::array_t<float>(shape0 * shape_dim);
//    result.resize({shape0, shape_dim});
//
//    py::buffer_info buf_result = result.request();
//    float *ptr_result = (float *) buf_result.ptr;
//
//    cout<<"shape0,shape1,shapedim:"<<shape0<<","<<shape1<<","<<shape_dim<<endl;
//
//    // 两种情况，1是请求压缩的，2是请求完整的加变化率
//    if((epoch+1)%trend==0){
//        if((epoch+1)/trend==1){
//            auto & changeMatrix=reply.changerate().changematrix();
//            map<int,vector<float>> map_tmp;
//            WorkerStore::embs_change_rate_worker.insert(pair<int,map<int,vector<float>>>(layerId,map_tmp));
//            for (int i = 0; i < shape0; i++) {
//                vector<float> vec_tmp;
//                const auto &tm = reply.denseembmessage().embs(i);
//                auto &changeVector=changeMatrix.Get(i);
//                for (int j = 0; j < shape1; j++) {
//                    vec_tmp.push_back(changeVector.tensor(j));
//                    ptr_result[i * shape1 + j] = tm.tensor(j);
//                }
//                WorkerStore::embs_change_rate_worker[layerId].insert(pair<int,vector<float>>(i,vec_tmp));
//            }
//        }else{
//            auto & changeMatrix=reply.changerate().changematrix();
//            map<int,vector<float>> map_tmp;
//            auto &changeRateLayer_ws=WorkerStore::embs_change_rate_worker[layerId];
//            for (int i = 0; i < shape0; i++) {
//                auto &changeRateNode_ws=changeRateLayer_ws[i];
//                const auto &tm = reply.denseembmessage().embs(i);
//                auto &changeVector=changeMatrix.Get(i);
//                for (int j = 0; j < shape1; j++) {
//                    changeRateNode_ws[j]=changeVector.tensor(j);
//                    ptr_result[i * shape1 + j] = tm.tensor(j);
//                }
//
//            }
//        }
//
//        cout<<"WorkerStore::embs_change_rate_worker shape"<<WorkerStore::embs_change_rate_worker.size()<<
//            "*"<<WorkerStore::embs_change_rate_worker.begin()->second.size()<<endl;
//
//    }else{
//
//        // 返回的是压缩的后的值
//        vector<float> bucket;
//        for (auto value:reply.values()) {
//            bucket.push_back(value);
//        }
//
//        for (int i = 0; i < shape0; i++) {
//            const IntTensorMessage &tm = reply.embs(i);
//            for (int j = 0; j < shape1; j++) {
//                // transform to 4 int data_raw
////                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
//                if (j * 4 + 3 < shape_dim) {
//                    uint dim = tm.tensor(j);
//                    int a_1 = dim >> 24;
//                    int a_2 = dim >> 16 & 0x000000ff;
//                    int a_3 = dim >> 8 & 0x000000ff;
//                    int a_4 = dim & 0x000000ff;
//                    ptr_result[i * shape_dim + j * 4] = bucket[a_1];
//                    ptr_result[i * shape_dim + j * 4 + 1] = bucket[a_2];
//                    ptr_result[i * shape_dim + j * 4 + 2] = bucket[a_3];
//                    ptr_result[i * shape_dim + j * 4 + 3] = bucket[a_4];
//                } else {
//                    int num = shape_dim - j * 4;
//                    uint dim = tm.tensor(j);
//                    vector<int> a;
//                    a.push_back(dim >> 24);
//                    a.push_back(dim >> 16 & 0x000000ff);
//                    a.push_back(dim >> 8 & 0x000000ff);
//                    a.push_back(dim & 0x000000ff);
//                    for (int k = 0; k < num; k++) {
//                        ptr_result[i * shape_dim + j * 4 + k] = bucket[a[k]];
//                    }
//
//                }
//            }
//        }
//    }
}

void testTwoBitCompress() {
    vector<uint> fourItemsVec(18);
    for (int i = 0; i < 4; i++) {
        fourItemsVec[i] = 0;
        fourItemsVec[i + 4] = 1;
        fourItemsVec[i + 8] = 2;
        fourItemsVec[i + 12] = 3;
    }
    fourItemsVec[16] = 3;
    fourItemsVec[17] = 3;

    uint compress_value = 0;
    uint compress_value2 = 0;
    for (int i = 0; i < 16; i++) {
        fourItemsVec[i] = fourItemsVec[i] << (32 - (i + 1) * 2);
        compress_value = compress_value | fourItemsVec[i];
    }

    compress_value2 = compress_value2 | (fourItemsVec[16] << 30) | fourItemsVec[17] << 28;

    vector<uint> compressV;
    compressV.push_back(compress_value);
    compressV.push_back(compress_value2);


    for (int i = 0; i < 1; i++) {

        for (int j = 0; j < 2; j++) {
            // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
            if (j * 16 + 15 < 18) {
                uint dim = compressV[j];
                for (int l = 0; l < 16; l++) {
                    int a = dim >> (30 - l * 2) & 0x00000003;
                    cout << a << ",";
                }
                cout << endl;
            } else {
                int num = 18 - j * 16;
                uint dim = compressV[j];
                for (int k = 0; k < num; k++) {
                    int a = dim >> (30 - k * 2) & 0x00000003;
                    cout << a << ",";
                }
                cout << endl;

            }
        }
    }
}


void testFourBitCompress() {
    vector<uint> fourItemsVec(10);
    for (int i = 0; i < 8; i++) {
        fourItemsVec[i] = 5;

    }
    fourItemsVec[8] = 15;
    fourItemsVec[9] = 13;

    uint compress_value = 0;
    uint compress_value2 = 0;
    for (int i = 0; i < 8; i++) {
        fourItemsVec[i] = fourItemsVec[i] << (28 - 4 * i);
        compress_value = compress_value | fourItemsVec[i];
    }

    compress_value2 = compress_value2 | (fourItemsVec[8] << 28) | fourItemsVec[9] << 24;

    vector<uint> compressV;
    compressV.push_back(compress_value);
    compressV.push_back(compress_value2);


    for (int i = 0; i < 1; i++) {

        for (int j = 0; j < 2; j++) {
            // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];
            if (j * 8 + 7 < 10) {
                uint dim = compressV[j];
                for (int l = 0; l < 8; l++) {
                    int a = dim >> (28 - l * 4) & 0x0000000f;
                    cout << a << ",";
                }
                cout << endl;
            } else {
                int num = 10 - j * 8;
                uint dim = compressV[j];
                for (int k = 0; k < num; k++) {
                    int a = dim >> (28 - k * 4) & 0x0000000f;
                    cout << a << ",";
                }
                cout << endl;

            }
        }
    }
}


void testEightBitCompress() {
    vector<uint> fourItemsVec(6);
    for (int i = 0; i < 4; i++) {
        fourItemsVec[i] = 5;

    }
    fourItemsVec[4] = 255;
    fourItemsVec[5] = 240;

    uint compress_value = 0;
    uint compress_value2 = 0;
    for (int i = 0; i < 4; i++) {
        fourItemsVec[i] = fourItemsVec[i] << (24 - 8 * i);
        compress_value = compress_value | fourItemsVec[i];
    }

    compress_value2 = compress_value2 | (fourItemsVec[4] << 24) | fourItemsVec[5] << 16;

    vector<uint> compressV;
    compressV.push_back(compress_value);
    compressV.push_back(compress_value2);


    for (int i = 0; i < 1; i++) {

        for (int j = 0; j < 2; j++) {
            // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];

            if (j * 4 + 3 < 6) {
                uint dim = compressV[j];
                for (int l = 0; l < 4; l++) {
                    int a = dim >> (24 - l * 8) & 0x000000ff;
                    cout << a << ",";
                }
                cout << endl;
            } else {
                int num = 6 - j * 4;
                uint dim = compressV[j];
                for (int k = 0; k < num; k++) {
                    int a = dim >> (24 - k * 8) & 0x000000ff;
                    cout << a << ",";
                }
                cout << endl;

            }
        }
    }
}


void testSixteenBitCompress() {
    vector<uint> fourItemsVec(3);
    for (int i = 0; i < 2; i++) {
        fourItemsVec[i] = 5;

    }
    fourItemsVec[2] = 65535;

    uint compress_value = 0;
    uint compress_value2 = 0;
    for (int i = 0; i < 2; i++) {
        fourItemsVec[i] = fourItemsVec[i] << (16 - 16 * i);
        compress_value = compress_value | fourItemsVec[i];
    }

    compress_value2 = compress_value2 | (fourItemsVec[2] << 16);

    vector<uint> compressV;
    compressV.push_back(compress_value);
    compressV.push_back(compress_value2);


    for (int i = 0; i < 1; i++) {

        for (int j = 0; j < 2; j++) {
            // transform to 4 int data_raw
//                ptr_result[i*shape1+j]=bucket[tm.tensor(j)];

            if (j * 2 + 1 < 3) {
                uint dim = compressV[j];
                for (int l = 0; l < 2; l++) {
                    int a = dim >> (16 - l * 16) & 0x0000ffff;
                    cout << a << ",";
                }
                cout << endl;
            } else {
                int num = 3 - j * 2;
                uint dim = compressV[j];
                for (int k = 0; k < num; k++) {
                    int a = dim >> (16 - k * 16) & 0x0000ffff;
                    cout << a << ",";
                }
                cout << endl;
            }
        }
    }
}

void readEdges() {
    char *buffer;
    buffer = getcwd(NULL, 0);
    string pwd = buffer;
    if (pwd[pwd.length() - 1] != '/') {
        pwd += '/';
    }
    string adjFile = pwd + "../data/reddit-small" + "/edges.txt";
    cout << "adjFile:" << adjFile << endl;

    ifstream adjInFile(adjFile);

    if (!adjInFile.is_open()) {
        cout << "adjInFile 未成功打开文件" << endl;
    }

    int nodeNum = 232965;
    vector<set<int>> adj_map(nodeNum);
    for (int i = 0; i < nodeNum; i++) {
        set<int> set_tmp;
        adj_map[i] = set_tmp;
    }
    int count = 0;
    int count_flag = 0;

    string temp;
    cout << "正在处理邻接表数据" << endl;
    while (getline(adjInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        int neibor_id = atoi(v[1].c_str());

        // 开始构造邻接表

        adj_map[vertex_id].insert(neibor_id);
        adj_map[neibor_id].insert(vertex_id);

        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }

    }
    int edge_num = count_flag;
    adjInFile.close();

}

void deleteTest() {
    vector<vector<int>> vec(100000);


    for (int i = 0; i < 100000; i++) {
        vector<int> vec_tmp(10000);
        vec[i] = vec_tmp;
    }

    vector<vector<int>>().swap(vec);

    cout << "a" << endl;
}

struct GrainRho {
    int key;
    double rho;
};

void deleteTestCap() {
    static vector<struct GrainRho> rhovec;

    static vector<struct GrainRho>::iterator itrho;

    GrainRho grainRho;

    for (int i = 0; i < 100; i++) {
        grainRho = {i, 0.5 + double(i)};
        rhovec.push_back(grainRho);
    }

    rhovec.clear();
    cout << "rhovec.size(): " << rhovec.size() << endl;
    cout << "rhovec.capacity(): " << rhovec.capacity() << endl;

    vector<struct GrainRho>().swap(rhovec);

    grainRho = {1995, 6.28};
    rhovec.push_back(grainRho);
    cout << "rhovec.size(): " << rhovec.size() << endl;
    cout << "rhovec.capacity(): " << rhovec.capacity() << endl;

    system("pause");
}

//void compressData_concat(int bitNum, vector<uint> &itemVector,
//                         google::protobuf::RepeatedField<google::protobuf::uint32> *mutable_emb_reply) {
////    int vectorSize=itemVector.size();
//    uint compressValue = 0;
//    auto &bitMap = WorkerStore::bucketPositionBitMap;
//
//
//    for (int i = 0; i < 16; i++) {
//        compressValue = compressValue | bitMap[itemVector[i]][i];
//    }
//    mutable_emb_reply->Add(compressValue);
//
//
//}

//void pullNeighborG() {
//    auto *request = new EmbMessage();
//    auto *reply = new EmbMessage();
//    int layerId = 2;
//    request->set_layerid(layerId);
//    request->set_ifcompensate(true);
//    request->set_iterround(1);
//    request->set_bucketnum(0);
//    request->set_bitnum(2);
//    int feat_dim = 5;
//
//    int nodeNum = 5;
//    map<int, vector<float>> map_tmp;
//    map<int, vector<float>> map_tmp2;
//    WorkerStore::G_map.insert(pair<int, map<int, vector<float>>>(layerId, map_tmp));
//    WorkerStore::G_compensate.insert(pair<int, map<int, vector<float>>>(layerId, map_tmp2));
//
//    vector<int> nodeVec(nodeNum);
//    for (int i = 0; i < nodeNum; i++) {
//        nodeVec[i] = i;
//        vector<float> vec_dim(feat_dim);
//        vector<float> vec_dim2(feat_dim);
//        for (int j = 0; j < feat_dim; j++) {
//            vec_dim[j] = j;
//            vec_dim2[j]=float(j)*0.1;
//        }
//        WorkerStore::G_map[layerId].insert(pair<int, vector<float>>(i, vec_dim));
//        WorkerStore::G_compensate[layerId].insert(pair<int, vector<float>>(i, vec_dim2));
//    }
//
//    request->mutable_nodes()->Reserve(nodeNum);
//    for (int i = 0; i < nodeNum; i++) {
//        request->mutable_nodes()->Add(nodeVec[i]);
//    }
//
//
//    int bitNum = request->bitnum();
//    int bucket_num = pow(2, bitNum) - 2;
////    cout<<"bucket_num:"<<bucket_num<<endl;
//    bool ifCompensate = request->ifcompensate();
//    int epoch = request->iterround();
//
//    // 先判断是否需要补偿
//    map<int, vector<float>> G;
//    float max_value = -10000;
//    float min_value = 10000;
////    cout<<"epoch "<<epoch <<", layer id "<<layerId<< "bucket num:"<<bucket_num<< ", ifCompensate:"<<ifCompensate<<endl;
////    cout<<"WorkerStore::G_compensate size:"<< WorkerStore::G_compensate[layerId].size()<<"*"<<WorkerStore::G_compensate[layerId].begin()->second.size()<<endl;
////    cout<<"WorkerStore::G_map size:"<< WorkerStore::G_map[layerId].size()<<"*"<<WorkerStore::G_map[layerId].begin()->second.size()<<endl;
//
//    auto &G_layerId = WorkerStore::G_map[layerId];
//    auto &G_compensate_layerId = WorkerStore::G_compensate[layerId];
//
//    int featNum = G_layerId.begin()->second.size();
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
//
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
//        for (int m = 0; m < nodeNum; m++) {
//            auto id = request->nodes(m);
//            const auto &g = G[id];
////            IntTensorMessage *tensor = reply->add_embs();
////            tensor->set_vid(id);
//            for (int n = 0; n < featNum; n++) {
//                auto dim = g[n];
//                int bucket_id = getDimBucket(buckets, dim, min_value, max_value, interval);
////                tensor->add_tensor(bucket_id);
//                int itemId = n % oneIntDimNum;
//                itemsVec[itemId] = bucket_id;
//                if (itemId == (oneIntDimNum - 1)) {
//                    compressData_concat(bitNum, itemsVec, mutable_reply);
//                }
//
//
//            }
//            if (left_num != 0) {
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
//
//                mutable_reply->Add(compress_value);
////                itemsVec.clear();
//            }
//        }
//    }
//    reply->set_resp_node_size(nodeNum);
//    reply->set_resp_featdim_size(compress_dim_size);
//    reply->set_shapedim(G_layerId.begin()->second.size());
////    cout << "G_layerId.begin()->second.size()" << reply->shapedim() << endl;
//
//
//}

void test_trend() {
    DGNNClient dgnnClient;
    DGNNClient *dgnn2;
    auto *metaData = new ReqEmbsMetaData;
    float *ptr_result;
    map<int, int> oldToNewMap;

    EmbMessage replyVec;
    vector<int> nodes;
    metaData->reply = &replyVec;
    metaData->serverId = 1;
    metaData->workerId = 0;
    metaData->epoch = 0;
    metaData->nodes = &nodes;
    metaData->layerId = 1;
    metaData->dgnnClient = dgnn2;
    metaData->ifCompress = true;
    metaData->layerNum = 1;
    metaData->bitNum = 2;
    metaData->trend = 10;
    metaData->ptr_result = ptr_result;
    metaData->oldToNewMap = &oldToNewMap;
    metaData->localNodeSize = 10;
    metaData->feat_num = 100;


    DGNNClient *dgnnClient2 = metaData->dgnnClient;

//    dgnnClient.worker_pull_emb_trend_parallel((void *) metaData);
}

//void test_abs(){
//    cout<<abs(-555.15)<<endl;
//}

void decompress() {
    vector<float> bucket;
    for (int i = 0; i < 3; i++) {
        bucket.push_back((float) i / 10);
    }


    map<int, vector<float>> embs_from_remote;
    for (int i = 0; i < 3; i++) {
        vector<float> tmp(3);
        for (int j = 0; j < 3; j++) {
            tmp[j] = j;
        }
        embs_from_remote.insert(pair<int, vector<float>>(i, tmp));
    }

    // comp_error,pred_error,mix_error

//    auto &embReply = reply.resp_compress_emb_concat();
    vector<int> flag_vec;
    for (int i = 0; i < 10; i++) {
        flag_vec.push_back(i % 3);
    }
    int count_non_remove_comp = 0;
    if (2 == 2) {
        for (int i = 0; i < 10; i++) {
            int flag = flag_vec[i];
            if (flag == 0) {
                for (int j = 0; j < 4; j++) {

                    if (j * 16 + 15 < 49) {
                        for (int l = 0; l < 16; l++) {
                            cout << "" << endl;
                        }
                    } else {
                        int num = 49 - j * 16;
                        for (int l = 0; l < num; l++) {
                            cout << "" << endl;
                        }
                    }
                    count_non_remove_comp++;
                }

            } else if (flag == 1) {
                for (int j = 0; j < 4; j++) {
                    if (j * 16 + 15 < 49) {
                        for (int l = 0; l < 16; l++) {

                        }
                    } else {
                        int num = 49 - j * 16;
                        for (int l = 0; l < num; l++) {

                        }
                    }
                }
            } else {
                for (int j = 0; j < 4; j++) {

                    if (j * 16 + 15 < 49) {
                        for (int l = 0; l < 16; l++) {

                        }
                    } else {
                        int num = 49 - j * 16;
                        for (int l = 0; l < num; l++) {

                        }
                    }
                    count_non_remove_comp++;
                }
            }

        }
    }
}

void testPartition2() {
    char *buffer;
    buffer = getcwd(NULL, 0);
    string pwd = buffer;
    cout << "pwd:" << pwd << ",buffer:" << buffer << endl;
//    string pwd =buffer;
    if (pwd[pwd.length() - 1] != '/') {
        pwd += '/';
    }
    string filename="../data/test";
    int data_num=4;
    int feature_size=34;
    string partitionMethod="metis";
    int worker_num=2;
    int nodeNum=4;
    // 开始处理数据集
//    string adjFile = pwd+"data_raw/cora/edges.txt";
//    string featFile = pwd+"data_raw/cora/featsClass.txt";

    string adjFile = pwd +  filename+ "/edges.txt";
    string featFile = pwd + filename + "/featsClass.txt";
    string partitionFile =
            pwd + filename + "/nodesPartition" + "." + partitionMethod + to_string(worker_num) + ".txt";
    cout << "adjfile:" << adjFile << endl;
    cout << "featFle:" << featFile << endl;
    cout << "partition file path:" << partitionFile << endl;


    ifstream partitionInFile(partitionFile);
    string temp;
    if (!partitionInFile.is_open()) {
        cout << "partitionInFile 未成功打开文件" << endl;
    }

    int count_worker = 0;
    while (getline(partitionInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vSize = v.size();
        vector<int> v_tmp(vSize);
        GeneralPartition::nodes.push_back(v_tmp);
        auto &nodesWorker = GeneralPartition::nodes[count_worker];
        for (int i = 0; i < vSize; i++) {
            nodesWorker[i] = atoi(v[i].c_str());
//            cout<<nodesWorker[i]<<",";
        }
//        cout<<endl;

        cout << "nodes num for worker " << count_worker << " :" << GeneralPartition::nodes[count_worker].size() << endl;
        count_worker++;
    }


    ifstream adjInFile(adjFile);

    if (!adjInFile.is_open()) {
        cout << "adjInFile 未成功打开文件" << endl;
    }


    vector<set<int>> adj_map(nodeNum);
    for (int i = 0; i < nodeNum; i++) {
        set<int> set_tmp;
        adj_map[i] = set_tmp;
    }
//    vector<vector<int>> adj_vec(2);
//    for(int i=0;i<adj_vec.size();i++){
//        vector<int> vec_tmp(edgeNum);
//        adj_vec[i]=vec_tmp;
//    }

    int count = 0;
    int count_flag = 0;
    cout << "正在处理邻接表数据" << endl;
    while (getline(adjInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        int neibor_id = atoi(v[1].c_str());

        // 开始构造邻接表
        adj_map[vertex_id].insert(neibor_id);
        adj_map[neibor_id].insert(vertex_id);


        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }

    }
    int edge_num = count_flag;
    adjInFile.close();


    // 开始处理feature和label,同样使用in file stream



    map<int, vector<float>> feature;
    vector<int> label_array(data_num); // 如果需要获取length，那么这块只能赋值常量
//    map<string, int> label_map;
    int count_label = 0;

    ifstream featInFile(featFile);
    if (!featInFile.is_open()) {
        cout << "未成功打开文件" << endl;
    }

    count = 0;
    count_flag = 0;
    cout << "正在处理特征数据 " << endl;

    while (true) {
        getline(featInFile, temp);
        if(temp.empty()){
            break;
        }
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        vector<float> vec_feat;
        for (int i = 1; i < feature_size + 1; i++) {
            vec_feat.push_back(atof(v[i].c_str()));
        }
        feature.insert(pair<int, vector<float>>(vertex_id, vec_feat));

//        cout<<"label:"<<label_new<<endl;
        label_array[vertex_id] = atoi(v[feature_size + 1].c_str());
        count_flag++;
        if (count_flag % (10000) == 0) {
            cout << "正在处理第" << count_flag << "个数据" << endl;
        }
    }


    featInFile.close();

    // 开始划分，邻接表、顶点map、属性、标签
    // 这里顶点按照哈希（取余数）的方式进行划分，因此不需要建立map
    // 邻接表：map<int, map<int,set>>

    cout << "adj_map size:" << adj_map.size() << endl;
    cout << "边数:" << edge_num << endl;

    for (int i = 0; i < worker_num; i++) {
        auto &node_worker_i = GeneralPartition::nodes[i];
        int nodeSize = node_worker_i.size();
        map<int, set<int> > adjForWorkerI_tmp;
        GeneralPartition::adjs.push_back(adjForWorkerI_tmp);
        auto &adjForWorkerI = GeneralPartition::adjs[i];

        map<int, int> label_tmp;
        GeneralPartition::labels.push_back(label_tmp);
        auto &labelWorkerI = GeneralPartition::labels[i];

        map<int, vector<float>> feat_tmp;
        GeneralPartition::features.push_back(feat_tmp);
        auto &featWorkerI = GeneralPartition::features[i];

        for (int j = 0; j < nodeSize; j++) {
            int nodeId = node_worker_i[j];
            auto &neiborVecForNode = adj_map[nodeId];
            auto &featVecForNode = feature[nodeId];
            adjForWorkerI.insert(pair<int, set<int>>(nodeId, neiborVecForNode));
            labelWorkerI.insert(pair<int, int>(nodeId, label_array[nodeId]));
            featWorkerI.insert(pair<int, vector<float>>(nodeId, featVecForNode));
        }
    }



}


int getIndexWithMinError_main(float v0, float v1, float v2) {

    if (v1 <= v0  && v1 <= v2) {
        return 1;
    } else if (v2 <= v0  && v2 <= v1) {
        return 2;
    } else {
        return 0;
    }
    // @test 1
//    return 2;
}

void testSelectMain(){


    for(int l=0;l<10;l++){
        vector<float> vec_tmp(20);
        for(int j=0;j<20;j++){
            vec_tmp[j]=0;
        }
        WorkerStore::embs.insert(pair<int,vector<float>>(l,vec_tmp));
    }
    for(int i=0;i<50;i++){
        auto *request=new EmbMessage();

        for(int l=0;l<10;l++){
            vector<float> vec_tmp(20);
            for(int j=0;j<20;j++){
                WorkerStore::embs[l][j]=float(i);
            }

        }

        for(int j=0;j<3;j++){
            request->set_epoch(i);
            request->set_layerid(j);
            request->set_trend(10);
            request->set_workerid(0);
            request->set_bitnum(1);
            for(int n=0;n<10;n++){
                request->add_nodes(n);
            }
            testSelect(request);
        }
    }
}

void testSelect( EmbMessage *request){

    auto *reply=new EmbMessage();

    int epoch = request->epoch();
    int layerId = request->layerid();
    int trend = request->trend();
    int workerId = request->workerid();


    int feat_size = WorkerStore::embs.begin()->second.size();
    auto &embs = WorkerStore::embs;
    float max_value = 1.0;
    float min_value = 0.0;
    int bitNum = request->bitnum();
    int bucket_num = pow(2, bitNum) - 2;
    if (bitNum == 1) {
        bucket_num = 1;
    }
    bool setBValueAslow = false;

    // @test1
//    cout << "epoch " << epoch << ", layer " << layerId << endl;

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
            // @test4
//            for (float emb_di m:embs[id]) {
//                cout << emb_dim << " ";
//            }
//            cout << endl;
        }

    } else {
        auto &embs_last_layerId = WorkerStore::embs_last[workerId][layerId];
        auto &embs_change_rate_layerId=WorkerStore::embs_change_rate[workerId][layerId];
        if ((epoch + 1) % trend == 0) {
            // in the last of each ten epochs (e.g., ninth epochs), non-compressed embeddings and the change-rate

            int nodeNum = request->nodes_size();
            reply->set_resp_featdim_size(feat_size);

            reply->mutable_resp_none_compress_rate_concat()->Reserve(nodeNum * feat_size);
            reply->mutable_resp_none_compress_emb_concat()->Reserve(nodeNum * feat_size);
            for (int i = 0; i < nodeNum; i++) {
                int id = request->nodes(i);
                reply->mutable_resp_none_compress_emb_concat()->Add(embs[id].begin(), embs[id].end());
                // @test 4
//                for (float emb_dim:embs[id]) {
//                    cout << emb_dim << " ";
//                }
//                cout << endl;
                auto &emb_id = embs[id];
                auto &emb_last_node = embs_last_layerId[id];
                auto &embs_change_rate_node=embs_change_rate_layerId[id];
                for (int j = 0; j < feat_size; j++) {
                    float rate_dim=(emb_id[j] - emb_last_node[j])/(float)trend;
                    reply->mutable_resp_none_compress_rate_concat()->Add(rate_dim);
                    embs_change_rate_node[j]=rate_dim;
                    // @test 1
//                    cout << emb_id[j] - emb_last_node[j] << " ";
                }
                // @test 1
//                cout << endl;
            }

            embs_last_layerId = embs;

        } else {
            // 发送压缩嵌入
            vector<Bucket> buckets;
            double interval = (max_value - min_value) / (double) (bucket_num);

            uint int_interval = (int) (interval * pow(10, 8));
            uint int_interval_addone = int_interval + 1;

            interval = int_interval_addone / pow(10, 8);
//            cout << "bucket_num,interval_bucket:" << bucket_num << "," << interval << endl;
//            cout << "int_interval,add1:" << int_interval << "," << int_interval_addone << endl;

//        clock_t start_compress = clock();
            if (min_value < 0 && max_value > 0) {
                for (int i = 0; i < bucket_num + 1; i++) {
                    if (min_value + interval * i < 0 && min_value + interval * (i + 1) > 0) {
                        // 建两个桶,以0的分界线
                        Bucket b1;
                        b1.bid = i;
                        b1.lower_bound = min_value + interval * i;
                        b1.upper_bound = 0;

                        b1.value = (b1.lower_bound + b1.upper_bound) / 2;

                        buckets.push_back(b1);
                        reply->add_values(b1.value);

                        i = i + 1;
                        Bucket b2;
                        b2.bid = i;
                        b2.lower_bound = 0;
                        b2.upper_bound = min_value + interval * (i + 1);
                        if (i == bucket_num) {
                            b2.upper_bound = max_value;
                        }

                        b2.value = (b2.lower_bound + b2.upper_bound) / 2;


                        buckets.push_back(b2);
                        reply->add_values(b2.value);
                    } else {
                        Bucket b;
                        b.bid = i;
                        b.lower_bound = min_value + interval * i;
                        b.upper_bound = min_value + interval * (i + 1);
                        if (i == bucket_num - 1) {
                            b.upper_bound = max_value;
                        }
                        if (b.lower_bound < 0 && setBValueAslow) {
                            b.value = b.upper_bound;
                        } else if (b.lower_bound > 0 && setBValueAslow) {
                            b.value = b.lower_bound;
                        } else {
                            b.value = (b.lower_bound + b.upper_bound) / 2;
                        }

                        buckets.push_back(b);
                        reply->add_values(b.value);
                    }
                }
            } else {
                for (int i = 0; i < bucket_num; i++) {
                    Bucket b;
                    b.bid = i;
                    b.lower_bound = min_value + interval * i;
                    b.upper_bound = min_value + interval * (i + 1);
                    if (i == bucket_num - 1) {
                        b.upper_bound = max_value;
                    }

                    if (b.lower_bound < 0 && setBValueAslow) {
                        b.value = b.upper_bound;
                    } else if (b.lower_bound > 0 && setBValueAslow) {
                        b.value = b.lower_bound;
                    } else {
                        b.value = (b.lower_bound + b.upper_bound) / 2;
                    }
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
//            mutable_emb_reply->Reserve(nodeNum * compress_dim_size);

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

//                float max_error=0;
//                int flag=0;


                vector<uint> vec_feat_comp_tmp(compress_dim_size);

                // @test 3
//                vector<float> comp(feat_size);
//                vector<float> pred(feat_size);
//                vector<float> mix(feat_size);

                for (uint i = 0; i < feat_size; i++) {
                    float dim = emb_second[i];
                    if (dim == 0) {
                        bucket_id = bucketSize - 1;
                    } else {
                        bucket_id = (int) (dim / interval);
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
                // @test for for
//                if(id==0){
//                    for (int i = 0; i < 4; i++) {
//                        if (i == 0) {
//                            cout << "compress:";
//                        } else if (i == 1) {
//                            cout << "predict:";
//                        } else if (i == 2) {
//                            cout << "mix:";
//                        } else {
//                            cout << "real:";
//                        }
//
//                        for (int j = 0; j < feat_size; j++) {
//                            if (i == 0) {
//                                cout << comp[j] << " ";
//                            } else if (i == 1) {
//                                cout << pred[j] << " ";
//                            } else if (i == 2) {
//                                cout << mix[j] << " ";
//                            } else {
//                                cout << emb_second[j] << " ";
//                            }
//                        }
//                        cout << endl;
//                    }
//                }



                if (left_num != 0) {
                    vec_feat_comp_tmp[feat_size / oneIntDimNum] = compressValue;
                    compressValue = 0;
                }

                uint index_WithMinError = getIndexWithMinError_main(comp_error, pred_error, mix_error);

                // @test if
//                if(id==1){
//                    index_WithMinError=0;
//                }else if(id==2){
//                    index_WithMinError=1;
//                }else if(id==3){
//                    index_WithMinError=2;
//                }

                count_flag_num[index_WithMinError]++;


//                index_WithMinError=0;
//                index_WithMinError=2;

                // @test 2
//                cout<<"comp,pred,mix:"<<comp_error<<","<<pred_error<<","<<mix_error<<endl;
//                cout<<"index_WithMinError:"<<index_WithMinError<<endl;

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

//            cout<<"epoch "<<epoch<<",layer:"<<layerId<<","<<count_flag_num[0]<<","<<count_flag_num[1]<<","<<count_flag_num[2]<<endl;


            reply->set_resp_featdim_size(compress_dim_size);
            reply->set_comp_data_percent((float) count_flag_num[0] / (float) nodeNum);

        }
        reply->set_shapedim(embs.begin()->second.size());
    }


}

// 测试可变长度的int
// make重新编译，  出现问题，是因为core和client_lib之间互相依赖的原因，把client_lib对core的依赖删掉即可
int main() {
    testSelectMain();
//    testPartition2();
//    decompress();
//    test_trend();
//    test_abs();
//    char *buffer;
//    buffer=getcwd(NULL,0);
//    cout<<buffer<<endl;

//    dgnnClient.init_by_address("127.0.0.1:4001");
//
//
//    clock_t start=clock();
//    dgnnClient.test_large();
//    clock_t end=clock();
//    cout<<"time1:"<<(double)(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
//
//    clock_t start1=clock();
//    dgnnClient.test_small();
//    clock_t end1=clock();
//    cout<<"time2:"<<(double)(end1-start1)/CLOCKS_PER_SEC<<"s"<<endl;


//    DGNNClient master;
//    master.set_serverAddress("192.168.184.157:4001");
//    master.startClientServer();
//    dgnnClient.init_by_address("192.168.184.157:4001");
//    sleep(2);
//    dgnnClient.pullDataFromMaster(0,2,5,"aa",1433,4);

    // 开启client自带server
//    dgnnClient.set_serverAddress("192.168.184.142:4001");
//    dgnnClient.startClientServer();
//    sleep(2);
//    dgnnClient.init_by_address("192.168.184.142:4001");
//    dgnnClient.testVariant();
//    dgnnClient.test1Bit();
//    dgnnClient.test_workerPullEmbCompress();
//    vector<int> hid;
//    hid.push_back(16);
//    dgnnClient.initParameter(2,1433,hid,7,0);


//    map<int, vector<float>> vec_map;
//    for(int i=0;i<3000;i++){
//        vector<float> vec_temp;
//        for(int j=0;j<14333;j++){
//            vec_temp.push_back(j);
//        }
//        vec_map.insert(pair<int,vector<float>>(i,vec_temp));
//    }
//    dgnnClient.worker_setEmbs(vec_map);
//    testbit();
//    clock_t start=clock();
//    for(long i=0;i<5000000;i++){
//        testCompress();
//    }
//    clock_t end=clock();
//    cout<<"time:"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
//    return 0;

//    test_vec_clear();
//    testCompressServerCode();

//    testCode();
//        test_ChangeRate();
//    cout<<pow(2,5)<<endl;
//    testTwoBitCompress();
//    testFourBitCompress();
//    testEightBitCompress();
//    testSixteenBitCompress();
//    testVectorAndArray();
//    testVector();
//    testPartition();
//    testCopyMap();
//    readEdges();
//    deleteTest();
//    pullNeighborG();
}