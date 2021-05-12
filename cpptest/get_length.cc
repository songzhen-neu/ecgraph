//
// Created by songzhen on 2020/10/5.
//

#include<iostream>
#include "../core/util/Length.cc"
using namespace std;

//template<class T>
//int length(T& arr)
//{
//    //cout << sizeof(arr[0]) << endl;
//    //cout << sizeof(arr) << endl;
//    return sizeof(arr) / sizeof(arr[0]);
//}

int main()
{
    //int b=10;int arr[b];这种写法不对
    int a=1;
    const int c=a;
    const int b=10;
    int arr[b] ;
    // 方法一
    cout << "数组的长度为：" << length(arr) << endl;
    // 方法二
    //cout << end(arr) << endl;
    //cout << begin(arr) << endl;
    cout << "数组的长度为：" << end(arr)-begin(arr) << endl;
    system("pause");
    return 0;
}