//
// Created by songzhen on 2020/10/5.
//
#include <iostream>
#include <set>
#include <map>
#include <vector>
using namespace std;
int main(){

    srand(1);
    for(int i=0;i<10;i++){
        cout<<rand()%1000/(float)1000<<endl;
    }

    // 除了指针操作，其他的赋值都是对值进行操作
    map<int,set<int>> map_temp;

    // (7,{3})
    set<int> set_tem2;
    set_tem2.insert(3);
    map_temp[7]=set_tem2;

    // （6，{2}）
    set<int> *set_temp_point=&map_temp[6];
    set_temp_point->insert(2);

    // （5，{}）
    set<int> set_temp=map_temp[5];
    set_temp.insert(1);




//    set<int> aa=set_temp;
//    aa.insert(4);
    int b=3;
    int a=b;
    a=5;

    return 0;
}