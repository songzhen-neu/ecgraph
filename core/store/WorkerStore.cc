//
// Created by songzhen on 2020/10/8.
//

#include "WorkerStore.h"
// vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
// node;feature;label;adj
//vector<int> WorkerStore::nodes;
//map<int,vector<int>> WorkerStore::features;
//map<int,int> WorkerStore::labels;
//map<int, set<int>> WorkerStore::adjs;

vector<int> WorkerStore::nodes;
map<int, vector<float>> WorkerStore::features;
map<int, int> WorkerStore::labels;
map<int, set<int>> WorkerStore::adjs;
string WorkerStore::testString;
map<int,vector<float>> WorkerStore::embs;
map<int,map<int,vector<float>>> WorkerStore::embs_compensate;
map<int,vector<vector<float>>> WorkerStore::weights;
int WorkerStore::layer_num;
map<int,map<int,vector<float>>> WorkerStore::G_map;
map<int,map<int,vector<float>>> WorkerStore::G_compensate;
vector<bool> WorkerStore::compFlag;
map<int,map<int,vector<float>>>  WorkerStore::embs_momument;
map<int,map<int,map<int,vector<float>>> > WorkerStore::embs_last;
map<int,map<int,map<int,vector<float>>>>  WorkerStore::embs_change_rate;
map<int,map<int,map<int,vector<float>>>>  WorkerStore::embs_change_rate_worker;
vector<vector<int>> WorkerStore::nodesForEachWorker;
double WorkerStore::embs_max;
double WorkerStore::embs_min;
vector<vector<uint>> WorkerStore::bucketPositionBitMap;

float WorkerStore::comp_percent;

// worker layer node feature
map<int,map<int,map<int,vector<float>>>> WorkerStore::embs_from_remote;
void WorkerStore::set_nodes(const NodeMessage *nodeMessage) {
    // 将nodeMessage解析成nodes
    for (auto i:nodeMessage->nodes()) {
        WorkerStore::nodes.push_back(i);
    }
    cout << "server:node number:" << nodes.size() << endl;
}

void WorkerStore::set_nodes_for_each_worker(const DataMessage *reply){
    auto& nodes_tmp=reply->nodesforeachworker();
    int workerNum=nodes_tmp.size();
    for(int i=0;i<workerNum;i++){
        auto& nodes_worker_i=nodes_tmp.Get(i);
        int nodesNum=nodes_worker_i.nodes_size();
        vector<int> vec_tmp;
        WorkerStore::nodesForEachWorker.push_back(vec_tmp);
        auto& nodesWorkerI_ws=WorkerStore::nodesForEachWorker[i];
        for(int j=0;j<nodesNum;j++){
            nodesWorkerI_ws.push_back(nodes_worker_i.nodes(j));
        }
    }
};

void WorkerStore::set_features(const DataMessage_FeatureMessage *featureMessage) {
    for (const auto &featureItem:featureMessage->features()) {
        int vid = featureItem.vid();
//        cout<<"vid:"<<vid<<endl;
        vector<float> feature;
        for (auto featureDim:featureItem.feature()) {
            feature.push_back(featureDim);
        }
        WorkerStore::features.insert(pair<int, vector<float>>(vid, feature));
    }
    Check::check_features(WorkerStore::features);

}

void WorkerStore::set_labels(const DataMessage_LabelMessage *labelMessage) {
    for (const auto &labelItem:labelMessage->labels()) {
        int vid = labelItem.vid();
        int label = labelItem.label();
//        cout<<"label::setlabel::"<<label<<endl;
        WorkerStore::labels.insert(pair<int, int>(vid, label));
    }
    Check::check_labels(WorkerStore::labels);
}

void WorkerStore::set_adjs(const DataMessage_AdjMessage *adjMessage) {
    for (const auto &adjItem:adjMessage->adjs()) {
        int vid = adjItem.vid();
        set<int> neibors;
        for (auto neiborId:adjItem.neibors()) {
            neibors.insert(neiborId);
        }
        WorkerStore::adjs.insert(pair<int, set<int>>(vid, neibors));
    }
    Check::check_adjs(WorkerStore::adjs);
}


void WorkerStore::getData(DataMessage *reply) {
    // nodes
    NodeMessage *nm = reply->nodelist().New();
    for (auto node:WorkerStore::nodes) {
        nm->add_nodes(node);
    }
    reply->set_allocated_nodelist(nm);

    // features
    DataMessage_FeatureMessage *fm = reply->featurelist().New();
    for (auto feature:WorkerStore::features) {
        DataMessage_FeatureMessage_FeatureItem *fm_fi = fm->add_features();
        fm_fi->set_vid(feature.first);
        for (auto dim:feature.second) {
            fm_fi->add_feature(dim);
        }
    }
    reply->set_allocated_featurelist(fm);

//
//    // labels
    DataMessage_LabelMessage *lm = reply->labellist().New();
    for (auto label:WorkerStore::labels) {
        DataMessage_LabelMessage_LabelItem *lm_li = lm->add_labels();
        lm_li->set_vid(label.first);
        lm_li->set_label(label.second);
//        cout<<label.second<<endl;
    }
    reply->set_allocated_labellist(lm);

    // adjs
    DataMessage_AdjMessage *am = reply->adjlist().New();
    for (auto adj:WorkerStore::adjs) {
        DataMessage_AdjMessage_AdjItem *am_ai = am->add_adjs();
        am_ai->set_vid(adj.first);
        for (auto nid:adj.second) {
            am_ai->add_neibors(nid);
        }
    }

    reply->set_allocated_adjlist(am);
}