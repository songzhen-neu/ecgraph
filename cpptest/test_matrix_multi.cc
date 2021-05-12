//
// Created by songzhen on 2020/10/28.
//
#include <vector>
#include <map>
#include <iostream>
using namespace std;

int main() {
    vector<float> embs_compensage;
    embs_compensage.push_back(1);
    embs_compensage.push_back(2);
    embs_compensage.push_back(3);
    vector<vector<float>> weights;
    for (int i = 0; i < 3; i++) {
        vector<float> vector1;
        for (int j = 0; j < 2; j++) {
            vector1.push_back(2);
        }
        weights.push_back(vector1);
    }

    vector<float> vec_mm_weight(weights.begin()->size(), 0);
    for (int i = 0; i < embs_compensage.size(); i++) {
        float dim = embs_compensage[i];
        for (int j = 0; j < vec_mm_weight.size(); j++) {
            vec_mm_weight[j] += dim * weights[i][j];
//                                cout<<"vec_mm_weight"<<vec_mm_weight[j]<<endl;
        }
    }

    for(auto dim:vec_mm_weight){
        cout<<dim<<endl;
    }

}