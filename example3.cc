//
// Created by songzhen on 2020/12/21.
//
#include <pybind11/pybind11.h>
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <map>
#include <vector>
#include <time.h>


namespace py=pybind11;
using namespace std;

py::array_t<double> add_arrays_2d(py::array_t<double>& input1, py::array_t<double>& input2) {

    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
    {
        throw std::runtime_error("numpy.ndarray dims must be 2!");
    }
    if ((buf1.shape[0] != buf2.shape[0])|| (buf1.shape[1] != buf2.shape[1]))
    {
        throw std::runtime_error("two array shape must be match!");
    }

    //申请内存
    auto result = py::array_t<double>(buf1.size);
    //转换为2d矩阵
    result.resize({buf1.shape[0],buf1.shape[1]});


    py::buffer_info buf_result = result.request();

    //指针访问读写 numpy.ndarray
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr_result = (double*)buf_result.ptr;

    for (int i = 0; i < buf1.shape[0]; i++)
    {
        for (int j = 0; j < buf1.shape[1]; j++)
        {
            auto value1 = ptr1[i*buf1.shape[1] + j];
            auto value2 = ptr2[i*buf2.shape[1] + j];

            ptr_result[i*buf_result.shape[1] + j] = value1 + value2;
        }
    }

    return result;

}

void sendMatrix( map<int,vector<float>> map_int_float, map<int,vector<float>> map_int_float2){
//    cout<<"aaaaaaa"<<endl;
//    return map_int_float;
    clock_t start_totle=clock();
    map<int,vector<float>> map_1;
//    for(int i=0;i<2700;i++){
//        vector<float> vec_1;
//        for(int j=0;j<1433;j++){
//            vec_1.push_back(j);
//        }
//        map_1.insert(pair<int,vector<float>>(i,vec_1));
//    }
    clock_t end_totle=clock();
    cout<<"for in c++ sendmatrix:"<<(double)(end_totle-start_totle)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
//    cout<<map_int_float[0][0]<<map_int_float[0][1]<<map_int_float[0][2]<<map_int_float[1][0]<<endl;
//    return map_int_float;
}

map<int,vector<float>> receiveMatrix(){
    clock_t start_totle=clock();
    map<int,vector<float>> map_1;
    for(int i=0;i<2700;i++){
        vector<float> vec_1;
        for(int j=0;j<1433;j++){
            vec_1.push_back(j);
        }
        map_1.insert(pair<int,vector<float>>(i,vec_1));
    }
    clock_t end_totle=clock();
    cout<<"for in c++:"<<(double)(end_totle-start_totle)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    return map_1;
}

PYBIND11_MODULE(example3,m){
    m.def("add_arrays_2d",&add_arrays_2d);
    m.def("sendMatrix",&sendMatrix);
    m.def("receiveMatrix",&receiveMatrix);

}