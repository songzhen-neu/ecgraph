//
// Created by songzhen on 2020/10/15.
//

#include "ServerStore.h"


int ServerStore::feat_dim;
int ServerStore::class_dim;
vector<int> ServerStore::hid_dims;
map<int, vector<vector<float>>> ServerStore::weights;
map<int, vector<float>> ServerStore::bias;
int ServerStore::worker_num;
int ServerStore::server_num;
map<int,vector<vector<float>>> ServerStore::weights_grad_agg;
map<int,vector<float>> ServerStore::bias_grad_agg;
map<int,vector<vector<float>>> ServerStore::m_weight_t;
map<int,vector<vector<float>>> ServerStore::v_weight_t;
int ServerStore::t;
map<int,vector<float>> ServerStore::m_bias_t;
map<int,vector<float>> ServerStore::v_bias_t;
int ServerStore::serverId;
float ServerStore::val_accuracy=0;
float ServerStore::train_accuracy=0;
float ServerStore::test_accuracy=0;
float ServerStore::val_f1_accuracy=0;
float ServerStore::test_f1_accuracy=0;
vector<int> ServerStore::train_nodes;
vector<int> ServerStore::val_nodes;
vector<int> ServerStore::test_nodes;

void ServerStore::initParams(const int &workerN,const int &serverN,const int & fd,const int & cd,const vector<int> &hd) {
    ServerStore::worker_num=workerN;
    ServerStore::server_num=serverN;



    ServerStore::feat_dim=fd;
    ServerStore::class_dim=cd;
    ServerStore::hid_dims=hd;
    bool initAsOne= false;

//    cout<<"feat_dim:"<<fd<<",class_dim:"<<cd<<",hid_dim:"<<hd.size()<<endl;
    srand(2);
    // 先构建fd和h[0]的
    vector<vector<float>> feat_emb_layer;

    int featPerWorkerNum;
    if(serverId==server_num-1){
        featPerWorkerNum=fd-int(fd/server_num)*(server_num-1);
    } else{
        featPerWorkerNum=int(fd/server_num);
    }

    for(int i=0;i<featPerWorkerNum;i++){
        vector<float> row(0);
        for(int j=0;j<hid_dims[0];j++){
            if(initAsOne){
                row.push_back(1);
            } else{
                row.push_back(rand()%1000/(float)1000);
            }

        }
        feat_emb_layer.push_back(row);
    }
    ServerStore::weights.insert(pair<int,vector<vector<float>>>(0,feat_emb_layer));

    if(serverId==0){
        vector<float> feat_emb_bias(0);
        for(int i=0;i<hid_dims[0];i++){
            if(initAsOne){
                feat_emb_bias.push_back(1);
            } else{
                feat_emb_bias.push_back(rand()%1000/(float)1000);
            }
        }
        ServerStore::bias.insert(pair<int,vector<float>>(0,feat_emb_bias));
    }


    // 构建隐藏层
    for(int i=0;i<hid_dims.size()-1;i++){
        vector<vector<float>> hid_emb_layer;
        int hidPerWorkerNum;
        if(serverId==server_num-1){
            hidPerWorkerNum=hd[i]-int(hd[i]/server_num)*(server_num-1);
        } else{
            hidPerWorkerNum=int(hd[i]/server_num);
        }

        for(int m=0;m<hidPerWorkerNum;m++){
            vector<float> row;
            for(int n=0;n<hd[i+1];n++){
                if(initAsOne){
                    row.push_back(1);
                } else{
                    row.push_back(rand()%1000/(float)1000);
                }
            }
            hid_emb_layer.push_back(row);
        }
        ServerStore::weights.insert(pair<int,vector<vector<float>>>(i+1,hid_emb_layer));


        if(serverId==0){
            vector<float> hid_emb_bias(0);
            for(int l=0;l<hid_dims[i+1];l++){
                if(initAsOne){
                    hid_emb_bias.push_back(1);
                } else{
                    hid_emb_bias.push_back(rand()%1000/(float)1000);
                }
            }
            ServerStore::bias.insert(pair<int,vector<float>>(i+1,hid_emb_bias));
        }

    }

    // 构建最后一层
    int last_hid_dim=hd[hd.size()-1];
    int lastHidPerWorkerNum;
    if(serverId==server_num-1){
        lastHidPerWorkerNum=last_hid_dim-int(last_hid_dim/server_num)*(server_num-1);
    } else{
        lastHidPerWorkerNum=int(last_hid_dim/server_num);
    }

    vector<vector<float>> last_weights_emb_layer(0);
    for(int i=0;i<lastHidPerWorkerNum;i++){
        vector<float> row(0);
        for(int j=0;j<cd;j++){
            if(initAsOne){
                row.push_back(1);
            } else{
                row.push_back(rand()%1000/(float)1000);
            }
        }
        last_weights_emb_layer.push_back(row);
    }
    ServerStore::weights.insert(pair<int,vector<vector<float>>>(hd.size(),last_weights_emb_layer));

    if(serverId==0){
        vector<float> last_bias_layer(0);
        for(int i=0;i<cd;i++){
            if(initAsOne){
                last_bias_layer.push_back(1);
            }else{
                last_bias_layer.push_back(rand()%1000/(float)1000);
            }
        }
        ServerStore::bias.insert(pair<int,vector<float>>(hd.size(),last_bias_layer));
    }


    // 检查weights和bias个数；维度；

    Check::check_initParameter_ServerStore();



}
