//
// Created by songzhen on 2020/10/14.
//

#include "threadUtil.h"

// 只在head中声明，不要在head中定义，否则会报multi definition错误
mutex ThreadUtil::mtx; // 全局互斥锁.
bool ThreadUtil::ready= false;
condition_variable ThreadUtil::cv;
int ThreadUtil::arrived_worker_num=0;

mutex ThreadUtil::mtx_initParameter;
bool ThreadUtil::ready_initParameter= false;

void ThreadUtil::addone(int &count) {
    unique_lock<mutex> lock(mtx);
    count++;
}

 int ThreadUtil::count_accuracy=0;
 mutex ThreadUtil::mtx_accuracy;
condition_variable ThreadUtil::cv_accuracy;

mutex ThreadUtil::mtx_barrier;
condition_variable ThreadUtil::cv_barrier;

int ThreadUtil::count_worker_for_barrier;

mutex ThreadUtil::mtx_updateModels;
condition_variable ThreadUtil::cv_updateModels;
int ThreadUtil::count_worker_for_updateModels;
bool ThreadUtil::ready_updateModels= false;
bool ThreadUtil::ready_updateModels_2=false;

pthread_mutex_t ThreadUtil::mtx_updateModels_addGrad;
mutex ThreadUtil::mtx_updateModels_barrier;

int ThreadUtil::count_respWorkerNumForEmbs=0;
mutex ThreadUtil::mtx_respWorkerNumForEmbs;


