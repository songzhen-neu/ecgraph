//
// Created by songzhen on 2021/3/23.
//


#include "GeneralPartition.h"

GeneralPartition::GeneralPartition() = default;;
vector<vector<int>> GeneralPartition::nodes;
vector<map<int, vector<float>>>  GeneralPartition::features;
vector<map<int, int>> GeneralPartition::labels;
vector<map<int, set<int>>> GeneralPartition::adjs;

void GeneralPartition::init(const int &data_num, const int &worker_num, const string &filename, const int &feature_size,
                            const int &label_size) {
    this->data_num = data_num;
    this->worker_num = worker_num;
    this->filename = filename;
    this->feature_size = feature_size;
    this->label_size = label_size;
//    for(int i=0;i<worker_num;i++){
//        vector<int> nodes_tmp;
//        map<int,vector<float>> feature_tmp;
//        map<int,int> label_tmp;
//        map<int,set<int>> adj_tmp;
//
//        GeneralPartition::nodes.push_back(nodes_tmp);
//        GeneralPartition::features.push_back(feature_tmp);
//        GeneralPartition::labels.push_back(label_tmp);
//        GeneralPartition::adjs.push_back(adj_tmp);
//
//    }
};

int GeneralPartition::startPartition(int worker_num, string partitionMethod, int nodeNum, int edgeNum) {
    char *buffer;
    buffer = getcwd(NULL, 0);
    string pwd = buffer;
    cout << "pwd:" << pwd << ",buffer:" << buffer << endl;
//    string pwd =buffer;
    if (pwd[pwd.length() - 1] != '/') {
        pwd += '/';
    }
    // 开始处理数据集
//    string adjFile = pwd+"data_raw/cora/edges.txt";
//    string featFile = pwd+"data_raw/cora/featsClass.txt";

//    string adjFile = pwd + this->filename + "/edges.txt";
//    string featFile = pwd + this->filename + "/featsClass.txt";
//    string partitionFile =
//            pwd + this->filename + "/nodesPartition" + "." + partitionMethod + to_string(worker_num) + ".txt";
    string adjFile =  this->filename + "/edges.txt";
    string featFile =  this->filename + "/featsClass.txt";
    string partitionFile =
            this->filename + "/nodesPartition" + "." + partitionMethod + to_string(worker_num) + ".txt";

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


    return 0;
};
// vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
