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


using namespace std;
namespace py = pybind11;


int add(int i, int j) {
    return i + j;
}

//void test(py::array_t<int> &id){
//
//}


void test_list(const py::array_t<int> &list){
//    int size=list.size();
//    int* ptr=(int *)list.request().ptr;
//    for( int i=0;i<size;i++){
//        ptr[i]=ptr[i];
//    }
//    cout<<list.data(0)<<endl;
//    cout<<list[1]<<endl;
//    list.request().ptr;
}


void test_list_list(const vector<vector<int>> &list){
//    int size=list.size();
//    int* ptr=(int *)list.request().ptr;
//    for( int i=0;i<size;i++){
//        ptr[i]=ptr[i];
//    }
//    cout<<list.data(0)<<endl;
//    cout<<list[1]<<endl;
//    list.request().ptr;
}




void test_list2(const vector<int> &list){
//    cout<<list.data(0)<<endl;
//    for( auto it:list){
//        it=it;
//    }
//    cout<<list[1]<<endl;
}

void test_list3(py::list &list){
//    int* ptr=(int *)list.;
//    int size=list.size();
//    for( int i=0;i<size;i++){
//        ptr[i]=ptr[i];
//    }


//    for( auto it:list){
//        cout<<it<<endl;
//        int a = it.cast<int>();
//        cout<<a<<endl;
//    }
//    cout<<list.data(0)<<endl;
//    cout<<list[1]<<endl;
//    list.request().ptr;
}


void test_dict(const map<int,int> &a_map){
//    cout<<a_map[0]<<endl;
//    cout<<id.size()<<endl;
//    for(auto i : id){
//        i.first=i.first+i.second;
//    }
}

void set_embs_ptr(py::array_t<float>& embs,double embs_max,double embs_min,int epoch){
//    WorkerStore::embs.clear();

    py::buffer_info buf2=embs.request();


    if( buf2.ndim!=2){
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }

//    cout<<"embs shape[1]:"<<buf2.shape[1]<<endl;

    float* ptr2=(float*) buf2.ptr;
//    WorkerStore::embs_ptr=new float[buf2.shape[0]*buf2.shape[1]];


    WorkerStore::embs_ptr=ptr2;

    WorkerStore::embs_max=embs_max;
    WorkerStore::embs_min=embs_min;
}


//void set_embs_ptr(py::array_t<float>& embs,double embs_max,double embs_min,int epoch){
////    WorkerStore::embs.clear();
//
//    py::buffer_info buf2=embs.request();
//
//
//    if( buf2.ndim!=2){
//        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
//    }
//
////    cout<<"embs shape[1]:"<<buf2.shape[1]<<endl;
//
//    float* ptr2=(float*) buf2.ptr;
////    WorkerStore::embs_ptr=new float[buf2.shape[0]*buf2.shape[1]];
//
//
//    WorkerStore::embs_ptr=ptr2;
//
//    WorkerStore::embs_max=embs_max;
//    WorkerStore::embs_min=embs_min;
//}

void set_g_ptr(py::array_t<int>& ids,py::array_t<float>& g,double g_max,double g_min,int epoch){
//    WorkerStore::embs.clear();
    py::buffer_info buf1=ids.request();
    py::buffer_info buf2=g.request();


    if(buf1.ndim!=1 || buf2.ndim!=2){
        throw std::runtime_error("ids dim size!=1 or embs dim size!=2");
    }
    if(buf1.shape[0]!=buf2.shape[0]){
        throw std::runtime_error("ids dim cannot match node number of features");
    }
//    cout<<"embs shape[1]:"<<buf2.shape[1]<<endl;
    int* ptr1=(int*) buf1.ptr;
    float* ptr2=(float*) buf2.ptr;
//    WorkerStore::embs_ptr=new float[buf2.shape[0]*buf2.shape[1]];

    if(epoch==0){
        for(int i=0;i<buf1.shape[0];i++){
            auto id=ptr1[i];
            WorkerStore::oid2nid_g.insert(pair<int,int>(id,i));

//            for(int j=0;j<buf2.shape[1];j++){
//                WorkerStore::embs_ptr[i*buf2.shape[1]+j]=ptr2[i*buf2.shape[1]+j];
//            }
        }
        WorkerStore::g_ptr=ptr2;
    }else{
        WorkerStore::g_ptr=ptr2;
//        for(int i=0;i<buf1.shape[0];i++){
//            for(int j=0;j<buf2.shape[1];j++){
//                WorkerStore::embs_ptr[i*buf2.shape[1]+j]=ptr2[i*buf2.shape[1]+j];
//            }
//        }
    }




    WorkerStore::g_max_ptr=g_max;
    WorkerStore::g_min_ptr=g_min;
}

// set_embs
void set_embs(py::array_t<int>& ids,py::array_t<float>& embs,double embs_max,double embs_min,int epoch){
//    WorkerStore::embs.clear();
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

    if(epoch==0){
        for(int i=0;i<buf1.shape[0];i++){
            auto id=ptr1[i];
//        vector<float> vec(buf2.shape[1]);
//        for(int j=0;j<buf2.shape[1];j++){
//            auto dim=ptr2[i*buf2.shape[1]+j];
//            vec[j]=dim;
//        }
            vector<float> vec(ptr2+i*(buf2.shape[1]),ptr2+(i+1)*(buf2.shape[1]));
//        vector<float> vec(buf2.shape[1]);
            WorkerStore::embs.insert(pair<int,vector<float>>(id,vec));
        }
    }else{
        for(int i=0;i<buf1.shape[0];i++){
            auto id=ptr1[i];
//        vector<float> vec(buf2.shape[1]);
//        for(int j=0;j<buf2.shape[1];j++){
//            auto dim=ptr2[i*buf2.shape[1]+j];
//            vec[j]=dim;
//        }
            vector<float> vec(ptr2+i*(buf2.shape[1]),ptr2+(i+1)*(buf2.shape[1]));
//        vector<float> vec(buf2.shape[1]);
            WorkerStore::embs[id]=vec;
        }
    }



    WorkerStore::embs_max=embs_max;
    WorkerStore::embs_min=embs_min;
}




PYBIND11_MODULE(example2, m) {


//    PYBIND11_NUMPY_DTYPE(AA, a,b);


    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function which adds two numbers");
    m.def("set_embs",&set_embs,"set local embs");
    m.def("set_embs_ptr",&set_embs_ptr,"set local embs");
    m.def("set_g_ptr",&set_g_ptr,"set local g");

    m.def("test_list",&test_list,"test list");
    m.def("test_dict",&test_dict,"test dict");
    m.def("test_list2",&test_list2,"test list");
    m.def("test_list3",&test_list3,"test list");
    m.def("test_list_list",&test_list_list,"test list");


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
            .def("getNeededEmb_train",&Router::getNeededEmb_train)
            .def("get_comp_percent",&Router::get_comp_percent)
            .def("getG",&Router::getG)
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
            .def("server_updateModels",&DGNNClient::server_updateModels)
//            .def("worker_pull_needed_emb_compress_iter",&DGNNClient::worker_pull_needed_emb_compress_iter)
//            .def("worker_pull_emb_trend",&DGNNClient::worker_pull_emb_trend)
            .def("getChangeRate",&DGNNClient::getChangeRate)
            .def("initCompressBitMap",&DGNNClient::initCompressBitMap)
            .def("freeSpace",&DGNNClient::freeSpace)
            .def("server_PullParams",&DGNNClient::server_PullParams)
            .def("server_aggGrad",&DGNNClient::server_aggGrad)
            .def("sendTrainNode",&DGNNClient::sendTrainNode) //Yu
            .def("pullTrainNode",&DGNNClient::pullTrainNode) //Yu
            .def("sendValNode",&DGNNClient::sendValNode) //Yu
            .def("pullValNode",&DGNNClient::pullValNode) //Yu
            .def("sendTestNode",&DGNNClient::sendTestNode) //Yu
            .def("pullTestNode",&DGNNClient::pullTestNode) //Yu
            .def("sendNode",&DGNNClient::sendNode)
            .def("pullNode",&DGNNClient::pullNode)
            .def("setCtxForCpp",&DGNNClient::setCtxForCpp)
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
