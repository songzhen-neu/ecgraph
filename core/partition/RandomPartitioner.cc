//
// Created by songzhen on 2020/10/3.
//
#include <zconf.h>
#include "RandomPartitioner.h"



using namespace std;

RandomPartitioner::RandomPartitioner(): Partitioner() {}

void RandomPartitioner::init(
        const int &data_num, const int & worker_num,
        const string &filename, const int & feature_size, const int & label_size) {
    this->data_num=data_num;
    this->worker_num=worker_num;
    this->filename=filename;
    this->feature_size=feature_size;
    this->label_size=label_size;
}



void check_pam(const map<int, set<int>> map_temp[],int worker_num){
    int count=0;
    for(int i=0;i<worker_num;i++){
        count+=map_temp[i].size();
    }
    cout<<"adj_worker size:"<<count<<endl;
}

void check_lwtn(const map<int,int> map_temp[],int worker_num){
    int count=0;
//    map_temp[0];
//    map_temp[1];
//    cout<<"map_temp size:"<<map_temp->size()<<endl;
    for(int i=0;i<worker_num;i++){
        count+=map_temp[i].size();
    }

//    int length=sizeof(map_temp)/sizeof(map_temp[0]);
//    for(int i=0;i<length;i++){
//        count+=map_temp[i].size();
//    }
    cout<<"label_worker size:"<<count<<endl;
}

void check_feature_worker(const map<int,vector<float>> vec[],int worker_num){
    int count=0;
    for(int i=0;i<worker_num;i++){
        count+=vec[i].size();
    }
    cout<<"feature_worker count:"<<count<<endl;
}

int RandomPartitioner::startPartition(int worker_num,string partitionMethod,int nodeNum,int edgeNum){
    char *buffer;
    buffer=getcwd(NULL,0);
    string pwd=buffer;
    if(pwd[pwd.length()-1]!='/'){
        pwd+='/';
    }
    // 开始处理数据集
//    string adjFile = pwd+"data_raw/cora/edges.txt";
//    string featFile = pwd+"data_raw/cora/featsClass.txt";

    string adjFile = pwd+this->filename+"/edges.txt";
    string featFile = pwd+this->filename+"/featsClass.txt";
    cout<<"adjfile:"<<adjFile<<endl;
    cout<<"featFle:"<<featFile<<endl;
    ifstream adjInFile(adjFile);

    string temp;
    if (!adjInFile.is_open()) {
        cout << "未成功打开文件" << endl;
    }

    // 从0开始重新编码
    map<int, int> m;
    map<int, set<int>> adj_map;
    int count = 0;
    int count_flag=0;
    cout<<"正在处理邻接表数据"<<endl;
    while (getline(adjInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = atoi(v[0].c_str());
        int neibor_id = atoi(v[1].c_str());
        if (m.count(vertex_id) == 0) {
            m[vertex_id] = count;
            count++;
        }
        if (m.count(neibor_id) == 0) {
            m[neibor_id] = count;
            count++;
        }

        // 开始构造邻接表
        int new_vertex_id = m[vertex_id];
        int new_neibor_id = m[neibor_id];

        if (adj_map.count(new_vertex_id) == 0) {
            set<int> set_temp;
            set_temp.insert(new_neibor_id);
            adj_map[new_vertex_id] = set_temp;
        } else {
            set<int> set_temp = adj_map[new_vertex_id];
            set_temp.insert(new_neibor_id);
            adj_map[new_vertex_id] = set_temp;
        }

        if (adj_map.count(new_neibor_id) == 0) {
            set<int> set_temp;
            set_temp.insert(new_vertex_id);
            adj_map[new_neibor_id] = set_temp;
        } else {
            set<int> set_temp = adj_map[new_neibor_id];
            set_temp.insert(new_vertex_id);
            adj_map[new_neibor_id] = set_temp;
        }
        count_flag++;
        if(count_flag%(10000)==0){
            cout<<"正在处理第"<<count_flag<<"个数据"<<endl;
        }

    }
    int edge_num=count_flag;
    adjInFile.close();
    cout<<"m size:"<<m.size()<<endl;

    // 开始处理feature和label,同样使用in file stream

//    float2DArray feature(data_num, feature_size);

    map<int,vector<float>> feature;
    vector<int> label_array(data_num); // 如果需要获取length，那么这块只能赋值常量
    map<string, int> label_map;
    int count_label = 0;

    ifstream featInFile(featFile);
    if (!featInFile.is_open()) {
        cout << "未成功打开文件" << endl;
    }

    count = 0;
    count_flag=0;
    cout<<"正在处理特征数据 "<<endl;
    while (getline(featInFile, temp)) {
        vector<string> v;
        split(temp, v, "\t");
        int vertex_id = m[atoi(v[0].c_str())];
        vector<float> vec_feat;
        for (int i = 1; i < feature_size + 1; i++) {
            vec_feat.push_back(atof(v[i].c_str()));
        }
        feature.insert(pair<int,vector<float>>(vertex_id,vec_feat));
        string label_str = v[1 + feature_size];
        int label_new;
        if (label_map.count(label_str) == 0) {
            label_map[label_str] = count;
            label_new = count;
            count++;
        } else {
            label_new = label_map[label_str];
        }
//        cout<<"label:"<<label_new<<endl;
        label_array[vertex_id] = label_new;
        count_flag++;
        if(count_flag%(10000)==0){
            cout<<"正在处理第"<<count_flag<<"个数据"<<endl;
        }
    }

    featInFile.close();

    // 开始划分，邻接表、顶点map、属性、标签
    // 这里顶点按照哈希（取余数）的方式进行划分，因此不需要建立map
    // 邻接表：map<int, map<int,set>>


    cout<<"adj_map size:"<<adj_map.size()<<endl;
    cout<<"边数:"<<edge_num<<endl;

    for (auto it = adj_map.begin(); it != adj_map.end(); it++) {
        int vertex_id = it->first;
        set<int> neibors = it->second;
        int worker_id = vertex_id % worker_num;
        this->adjs[worker_id].insert(pair<int,set<int>>(vertex_id,neibors));
    }
    check_pam(this->adjs,worker_num);

    // 划分label
//    cout<<"label_array size:"<<length(label_array)<<endl;
    for(int i=0;i<data_num;i++){
        int worker_id=i%worker_num;
        this->labels[worker_id].insert(pair<int,int>(i,label_array[i]));
//        if(label_array[i]!=0){
//            cout<<"划分label:"<<label_array[i]<<endl;
//        }
    }

    check_lwtn(this->labels,worker_num);

    // 划分feature

    for(int i=0;i<data_num;i++){
        int worker_id=i%worker_num;
        vector<float> vec(feature_size);
        for(int f=0;f<feature_size;f++){
            vec[f]=feature[i][f];
        }
        // 顶点i的一阶邻居的属性也要传过去(先不传了，只计算本地的，然后在图层传播的时候再传)
//        for(int nei_id:adj_map[i]){
//            vector<float> vec_temp(feature_size);
//            for(int f_n=0;f_n<feature_size;f_n++){
//                vec_temp[f_n]=feature[nei_id][f_n];
//            }
//            // insert插入方式，当插入存在的key时，无法替换
//            this->features[worker_id].insert(pair<int,vector<float>>(nei_id,vec_temp));
//        }
        if(i==0){
            cout<<"i:"<<i<<",worker_id:"<<worker_id<<",worker_num:"<<worker_num<<",feature size:"<<vec.size()<<endl;
        }
        this->features[worker_id].insert(pair<int,vector<float>>(i,vec));
    }

//    // 检查 worker 1的feature 0是不是有问题
//    cout<< "features 1 0 :" << this->features[1][0].size()<<endl;
//    cout<< "features 1 0 :" << this->features[1][1].size()<<endl;

    check_feature_worker(this->features,worker_num);

    // 划分顶点
    for(int i=0;i<data_num;i++){
        int worker_id=i%worker_num;
        this->nodes[worker_id].push_back(i);
    }


    // print partition info
    vector<int> adj_num_each_worker(worker_num);
    // init data_raw structure
    for(int i=0;i<worker_num;i++){
        for(auto tmp:adjs[i]){
            adj_num_each_worker[i]+=tmp.second.size();
        }
        cout<<"worker "<<i<<" edge num:"<<adj_num_each_worker[i]<<endl;
    }



    // vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>


    return 0;
}


