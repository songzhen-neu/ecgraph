//
// Created by songzhen on 2020/11/5.
//

#include <mutex>
#include <pthread.h>
#include <iostream>
using namespace std;

pthread_mutex_t mutex_p;
mutex mutex_p2;
int a=0;
int b=1;

void* addone(void *){
    std::unique_lock<std::mutex> lck(mutex_p2);

//    lck.unlock();
//    lck.lock();
//    pthread_mutex_lock(&mutex_p);
    a++;
    cout<<"a"<< a <<endl;
    lck.unlock();

//    std::unique_lock<std::mutex> lck_1(mutex_p2);
    lck.lock();
//    lck.unlock();
//    lck.lock();
//    pthread_mutex_lock(&mutex_p);
    b++;
    cout<<"b"<<b<<endl;
//    lck_1.unlock();
    lck.unlock();

//    pthread_mutex_unlock(&mutex_p);
}
int main(){
    pthread_t pthreads[10];
    pthread_t p;
    // 创建多线程，对全局变量做加法
    for(int i=0;i<10;i++){
        pthread_create(&pthreads[i],NULL,addone,NULL);

    }



    cout<<"aaaaaa:"<<a<<endl;
    pthread_exit(NULL);



}

