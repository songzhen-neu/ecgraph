//
// Created by songzhen on 2020/10/14.
//

#ifndef DGNN_TEST_THREADUTIL_H
#define DGNN_TEST_THREADUTIL_H
#include<pthread.h>
#include <condition_variable>
#include <unistd.h>
#include <mutex>
using namespace std;

class ThreadUtil {
public:
    static mutex mtx; // 全局互斥锁.
    static bool ready;
    static mutex mtx_initParameter;
    static bool ready_initParameter;
    static condition_variable cv;
    static void addone(int &count);
    static int arrived_worker_num;

    static mutex mtx_sendNode;

    static mutex mtx_barrier;
    static condition_variable cv_barrier;
    static int count_worker_for_barrier;

    static mutex mtx_updateModels;
    static pthread_mutex_t mtx_updateModels_addGrad;
    static mutex mtx_updateModels_barrier;
    static condition_variable cv_updateModels;
    static int count_worker_for_updateModels;
    static bool ready_updateModels;
    static bool ready_updateModels_2;

    static int count_respWorkerNumForEmbs;
    static mutex mtx_respWorkerNumForEmbs;

    static int count_accuracy;
    static mutex mtx_accuracy;
    static condition_variable cv_accuracy; // 全局条件变量.

    static mutex mtx_gcompensate;



};


#endif //DGNN_TEST_THREADUTIL_H
