//
// Created by songzhen on 2020/10/6.
//

#ifndef DGNN_TEST_CONTEXT_H
#define DGNN_TEST_CONTEXT_H
#include <map>
#include <vector>
#include "../service/dgnn_client.h"

//#include "grpcpp/grpcpp.h"

using namespace std;

class Context {
public:
    static Context* getInstance();
    static vector<string> address;
    static int worker_num;
    static vector<DGNNClient> dgnnWorkerRoute;


private:
    Context();
    Context(const Context&);
    Context& operator=(const Context&);

    static Context* context;


//    struct CG{
//        ~CG(){
//            if(Context::context){
//                delete context;
//            }
//        }
//    };
};


#endif //DGNN_TEST_CONTEXT_H
