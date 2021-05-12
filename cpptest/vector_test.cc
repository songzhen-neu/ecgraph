//
// Created by songzhen on 2020/10/6.
//
#include <vector>
using namespace std;

class A{
public:
    int a=1;
};
int main(){
    int b=5;
    vector<A> vec(5);
    // vector<A> vec; 这会报错
    A a;
    // 段错误
    for(int i=0;i<5;i++){
        vec[i]=a;
    }

    return 0;
}