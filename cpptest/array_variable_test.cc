//
// Created by songzhen on 2020/10/5.
//
#include <iostream>
using namespace std;

int main(){
    int a=4;
    int b[4];
    int c[5];
    c[0]=1;
    b[0]=2;
    b[3]=3;
    b[4]=3;
    b[5]=3;
    b[6]=3;
    const int size=100;
    int d[size];
    d[0]=100;
    for(int i=0;i<10;i++){
        cout<<"b:"<<i<<":"<<b[i]<<endl;
        cout<<"b address:"<<i<<":"<<&b[i]<<endl;
    }

    for(int i=0;i<15;i++){
        cout<<"c"<<i<<":"<<c[i]<<endl;
        cout<<"c address:"<<i<<":"<<&c[i]<<endl;
    }




    cout<<d[0]<<","<<&d[0]<<endl;
    cout<<d[1]<<","<<&d[1]<<endl;
    cout<<d[2]<<","<<&d[2]<<endl;
    cout<<d[3]<<","<<&d[3]<<endl;
    cout<<d[4]<<","<<&d[4]<<endl;

}