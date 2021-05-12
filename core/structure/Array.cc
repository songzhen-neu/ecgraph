//
// Created by songzhen on 2020/10/3.
//

#include <iostream>
using namespace std;
class int2DArray{
public:
    int2DArray(int row,int column):row(row),column(column){
        arr=new int*[row];
        for(int i=0;i<row;i++){
            arr[i]=new int[column];
        }
    }
    ~int2DArray(){
       for(int i=0;i<column;i++){
           delete [] arr[i];
       }
        delete[] arr;
    }

    // 还不太明白这个函数的具体语法和含义
    int* const operator[](const int row) const{
        return arr[row];
    }


private:
    int **arr;
    const int row;
    const int column;
};

class float2DArray{
public:
    float2DArray(int row,int column):row(row),column(column){
        arr=new float*[row];
//        float *a[row];
//        arr= a;
        for(int i=0;i<row;i++){
            arr[i]=new float[column];
        }
    }
    ~float2DArray(){
        for(int i=0;i<column;i++){
            delete [] arr[i];
        }
        delete[] arr;
    }

    // 还不太明白这个函数的具体语法和含义
    float* const operator[](const int row) const{
        return arr[row];
    }


private:
    float **arr;
    const int row;
    const int column;
};