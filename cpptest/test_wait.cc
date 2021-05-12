//
// Created by songzhen on 2020/10/14.
//

#include <iostream>                // std::cout

#include <thread>                // std::thread

#include <mutex>                // std::mutex, std::unique_lock

#include <condition_variable>    // std::condition_variable

#include <pthread.h>
#include <unistd.h>

std::mutex mtx_test; // 全局互斥锁.

std::condition_variable cv_test; // 全局条件变量.

bool ready = false; // 全局标志位.



// pthread version
void *do_print_id_pthread(void *id_tmp) {

    std::unique_lock<std::mutex> lck(mtx_test);
    // 如果标志位不为 true, 则等待...
    // 当前线程被阻塞, 当全局标志位变为 true 之后,线程被唤醒, 继续往下执行打印线程编号id.
    cv_test.wait(lck);

    std::cout<< "aaa"<<std::endl;
}


void go() {
//    std::cout << "go" << std::endl;
    std::unique_lock<std::mutex> lck(mtx_test);

    cv_test.notify_all(); // 唤醒所有线程.

}

void pthread_test() {
    pthread_t threads[10];
    // spawn 10 threads:

    for (int i = 0; i < 10; ++i) {
        int *tmp = new int(i);
        pthread_create(&threads[i], NULL, do_print_id_pthread, (void *) tmp);
    }

//    std::cout << "10 threads ready to race...\n";

    sleep(5);
    go(); // go!
    for (auto th:threads)
        pthread_join(th, NULL);


}


int main() {
//    thread_test();
    pthread_test();

    return 0;

}

