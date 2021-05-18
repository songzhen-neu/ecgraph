//
// Created by songzhen on 2020/10/8.
//

#include <pybind11/pybind11.h>
#include "cpptest/Animal.cc"
//#include "core/partition/RandomPartitioner.h"
#include <pybind11/stl.h>
#include "dgnn_server.h"
#include "cmake/build/dgnn_test.grpc.pb.h"
#include "core/service/dgnn_client.h"
#include "core/store/WorkerStore.h"
#include "pybind11/numpy.h"
#include "core/service/router.h"
//
//#include <pybind11/numpy.h>
//namespace py=pybind11;
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


// set_embs
void set_embs(py::array_t<int>& ids,py::array_t<float>& embs){
    WorkerStore::embs.clear();
    py::buffer_info buf1=ids.request();
    py::buffer_info buf2=embs.request();
//    cout<<buf2.shape[0]<<endl;
//    cout<<buf2.shape[1]<<endl;

    if(buf1.ndim!=1 || buf2.ndim!=2){
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    if(buf1.shape[0]!=buf2.shape[0]){
        throw std::runtime_error("ids dim cannot match node number of features");
    }

    int* ptr1=(int*) buf1.ptr;
    float* ptr2=(float*) buf2.ptr;

    for(int i=0;i<buf1.shape[0];i++){
        auto id=ptr1[i];
        vector<float> vec(buf2.shape[1]);
        for(int j=0;j<buf2.shape[1];j++){
            auto dim=ptr2[i*buf2.shape[1]+j];
            vec[j]=dim;
        }
        WorkerStore::embs.insert(pair<int,vector<float>>(id,vec));
    }


}



PYBIND11_MODULE(example2, m) {


//    PYBIND11_NUMPY_DTYPE(AA, a,b);


    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function which adds two numbers");
    m.def("set_embs",&set_embs,"set local embs");


    // 创建server服务器
//    m.def("run_server",&RunServer);
//    py::class_<ServiceImpl>(m,"Server")
//            .def(py::init<>());

    py::class_<ServiceImpl>(m,"ServiceImpl")
            .def(py::init<>())
            .def("RunServerByPy",&ServiceImpl::RunServerByPy);


    py::class_<Router>(m,"Router")
            .def(py::init<>())
            .def("getNeededEmb",&Router::getNeededEmb)
            .def("get_comp_percent",&Router::get_comp_percent)
            .def("initWorkerRouter",&Router::initWorkerRouter);

    // 创建client
    py::class_<DGNNClient>(m,"DGNNClient")
            .def(py::init<>())
            .def("init_by_address",&DGNNClient::init_by_address)
            .def("sendDataToEachWorker",&DGNNClient::sendDataToEachWorker)
            .def("add1",&DGNNClient::add1)
            .def("pullDataFromServer",&DGNNClient::pullDataFromServer)
            .def("add",&DGNNClient::add)
            .def("startClientServer",&DGNNClient::startClientServer)
            .def("pullDataFromMaster",&DGNNClient::pullDataFromMaster)
            .def("pullDataFromMasterGeneral",&DGNNClient::pullDataFromMasterGeneral)
            .def("initParameter",&DGNNClient::initParameter)
            .def("server_PullWeights",&DGNNClient::server_PullWeights)
            .def("server_PullBias",&DGNNClient::server_PullBias)
            .def("server_Barrier",&DGNNClient::server_Barrier)
            .def("worker_pull_needed_emb",&DGNNClient::worker_pull_needed_emb)
            .def("sendAndUpdateModels",&DGNNClient::sendAndUpdateModels)
            .def("worker_setEmbs",&DGNNClient::worker_setEmbs)
            .def("worker_pull_emb_compress",&DGNNClient::worker_pull_emb_compress)
            .def("setG",&DGNNClient::setG)
            .def("worker_pull_needed_G",&DGNNClient::worker_pull_needed_G)
            .def("worker_pull_needed_G_compress",&DGNNClient::worker_pull_needed_G_compress)
            .def("sendAccuracy",&DGNNClient::sendAccuracy)
            .def("freeMaster",&DGNNClient::freeMaster)
//            .def("worker_pull_needed_emb_compress_iter",&DGNNClient::worker_pull_needed_emb_compress_iter)
//            .def("worker_pull_emb_trend",&DGNNClient::worker_pull_emb_trend)
            .def("getChangeRate",&DGNNClient::getChangeRate)
            .def("initCompressBitMap",&DGNNClient::initCompressBitMap)
            .def("freeSpace",&DGNNClient::freeSpace)
            .def("sendTrainNode",&DGNNClient::sendTrainNode) //Yu
            .def("pullTrainNode",&DGNNClient::pullTrainNode) //Yu
            .def("sendValNode",&DGNNClient::sendValNode) //Yu
            .def("pullValNode",&DGNNClient::pullValNode) //Yu
            .def("sendTestNode",&DGNNClient::sendTestNode) //Yu
            .def("pullTestNode",&DGNNClient::pullTestNode) //Yu               
            .def_property("nodesForEachWorker",&DGNNClient::get_nodesForEachWorker,&DGNNClient::set_nodesForEachWorker)
            .def_property("nodes",&DGNNClient::get_nodes,&DGNNClient::set_nodes)
            .def_property("features",&DGNNClient::get_features,&DGNNClient::set_features)
            .def_property("labels",&DGNNClient::get_labels,&DGNNClient::set_labels)
            .def_property("adjs",&DGNNClient::get_adjs,&DGNNClient::set_adjs)
            .def_property("serverAddress",&DGNNClient::get_serverAddress,&DGNNClient::set_serverAddress)
            .def_property("testString",&DGNNClient::get_testString,&DGNNClient::set_testString)
            .def_property("embs",&DGNNClient::get_embs,&DGNNClient::set_embs)
            .def_property("layerNum",&DGNNClient::get_layerNum,&DGNNClient::set_layerNum);


    // 创建随机划分的类
//    py::class_<RandomPartitioner>(m,"RandomPartitioner")
//            .def(py::init<int,int,string,int,int>())
//            .def("startPartition",&RandomPartitioner::startPartition);





}
