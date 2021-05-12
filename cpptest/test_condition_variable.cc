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
struct cat {
    int *tmp;
    int *tmp2;
};


// pthread version
void *do_print_id_pthread(void *id_tmp) {
    sleep(2);
//    int id = *((int *) id_tmp);
    cat a= *(cat*) id_tmp;
    std::cout<<std::to_string(*a.tmp)+" lock b "+"\n";
    std::unique_lock<std::mutex> lck(mtx_test);
    std::cout<<std::to_string(*a.tmp2)+" lock a"+"\n";
    // 如果标志位不为 true, 则等待...
    // 当前线程被阻塞, 当全局标志位变为 true 之后,线程被唤醒, 继续往下执行打印线程编号id.
    while (!ready){
//        std::cout<<id<<std::endl;
        cv_test.wait(lck);
//        sleep(3);
    }
    std::cout<<std::to_string(*a.tmp)+" lock free"+"\n";


//    std::cout << "thread " << id << '\n';
}


void go() {
    std::cout<<"go"<<std::endl;
    std::unique_lock<std::mutex> lck(mtx_test);
    std::cout<<"go2"<<std::endl;
    // 当设置成true，表明go函数要执行的执行完了，子线程不用在等待了
    // 所以，还没有wait的线程，不需要执行wait，进而可以畅通无阻
    // 而已经wait的线程，通过notify_all也继续执行
    ready = true; // 设置全局标志位为 true.
    std::cout<<"go3"<<std::endl;
    cv_test.notify_all(); // 唤醒所有线程.
    std::cout<<"go4"<<std::endl;

}

void pthread_test() {
    pthread_t threads[10];
    // spawn 10 threads:

    for (int i = 0; i < 10; ++i) {
        cat* a=new cat;
        a->tmp=new int(i);
        a->tmp2=new int(i+1);
        pthread_create(&threads[i], NULL, do_print_id_pthread, (void *) a);

    }

    std::cout << "10 threads ready to race...\n";

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

