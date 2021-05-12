//
// Created by songzhen on 2020/10/5.
//
#include <iostream>
int main() {
    int **arr;
    arr = new int *[5];
    for (int i = 0; i < 5; i++) {
        arr[i] = new int[5];
        for (int j = 0; j < 5; j++) {
            arr[i][j] = 1;
        }
    }
    std::cout<<arr[0][0]<<","<<arr[0][1]<<std::endl;
    return 0;
}
