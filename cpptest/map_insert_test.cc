//
// Created by songzhen on 2020/10/12.
//

#include <map>
#include <iostream>
#include <vector>
using namespace std;
int main(){
    map<int,int> map1;
    map<int,vector<float>> map_float;

    cout<<map_float.size()<<endl;
    cout<<map_float[0].empty()<<endl;

    cout<<map_float.size()<<endl;

    cout<<map1.empty()<<endl;
    if(map1.empty()){
        cout<<"empty"<<endl;
    }
    // 当有对应key的时候，无法插入
    map1.insert(pair<int,int>(0,1));
    map1.insert(make_pair(1,2));
    cout<<map1.empty()<<endl;

    cout<<map1[0]<<endl;
    cout<<map1[1]<<endl;

    // 数组插入方式,可以改值
    map1[0]=2;
    cout<<map1[0]<<endl;
    map1[2]=3;
    cout<<map1[2]<<endl;
};