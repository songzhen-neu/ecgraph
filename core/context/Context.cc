//
// Created by songzhen on 2020/10/6.
//

#include "Context.h"

Context::Context() = default;
Context::Context(const Context &) = default;
Context& Context::operator=(const Context &) {}


Context* Context::context=new Context();
vector<DGNNClient> Context::dgnnWorkerRoute;
Context* Context::getInstance() {
    return context;
}

int Context::worker_num=2;


vector<string> Context::address=vector<string>({"192.168.184.138:2001","192.168.184.138:2002"});



