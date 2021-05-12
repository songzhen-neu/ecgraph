//
// Created by songzhen on 2020/10/13.
//

#include <iostream>
#include <unistd.h>
using namespace std;



class Context1{
public:
     static int aa;
//     static void init(int aa){
//         Context1::aa=aa;
//     }
};

int Context1::aa =5;

void* hello(void *args){
    cout<<"hello"<<endl;
    cout<<"from thread:"<<Context1::aa<<endl;
    return 0;}
int main(){
    pthread_t p;
    pthread_create(&p,NULL,hello,NULL);


    cout<<Context1::aa<<endl;
    sleep(2);
    Context1::aa=10;
    cout<<Context1::aa<<endl;
    pthread_exit(NULL);
//    cout<<Context1::aa<<endl;


}