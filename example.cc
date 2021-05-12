////
//// Created by songzhen on 2020/9/22.
////
//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
//#include "core/context/Context.h"
//#include<vector>
//#include <pybind11/stl.h>
//
//namespace py=pybind11;
//
//int add(int i, int j ){
//    return i+j;
//}
//
//
//py::array_t<double> add_arrays_1d(py::array_t<double>& input1, py::array_t<double>& input2) {
//    // 获取input1, input2的信息
//    py::buffer_info buf1 = input1.request();
//    py::buffer_info buf2 = input2.request();
//
//    if (buf1.ndim !=1 || buf2.ndim !=1)
//    {
//        throw std::runtime_error("Number of dimensions must be one");
//    }
//
//    if (buf1.size !=buf2.size)
//    {
//        throw std::runtime_error("Input shape must match");
//    }
//
//    //申请空间
//    auto result = py::array_t<double>(buf1.size);
//    py::buffer_info buf3 = result.request();
//
//    //获取numpy.ndarray 数据指针
//    double* ptr1 = (double*)buf1.ptr;
//    double* ptr2 = (double*)buf2.ptr;
//    double* ptr3 = (double*)buf3.ptr;
//
//    //指针访问numpy.ndarray
//    for (int i = 0; i < buf1.shape[0]; i++)
//    {
//        ptr3[i] = ptr1[i] + ptr2[i];
//    }
//
//    cout<<"worker_num:"<<Context::worker_num<<endl;
//
//    return result;
//
//}
//
//
//py::array_t<double> add_arrays_2d(py::array_t<double>& input1, py::array_t<double>& input2) {
//
//    py::buffer_info buf1 = input1.request();
//    py::buffer_info buf2 = input2.request();
//
//    if (buf1.ndim != 2 || buf2.ndim != 2)
//    {
//        throw std::runtime_error("numpy.ndarray dims must be 2!");
//    }
//    if ((buf1.shape[0] != buf2.shape[0])|| (buf1.shape[1] != buf2.shape[1]))
//    {
//        throw std::runtime_error("two array shape must be match!");
//    }
//
//    //申请内存
//    auto result = py::array_t<double>(buf1.size);
//    //转换为2d矩阵
//    result.resize({buf1.shape[0],buf1.shape[1]});
//
//
//    py::buffer_info buf_result = result.request();
//
//    //指针访问读写 numpy.ndarray
//    double* ptr1 = (double*)buf1.ptr;
//    double* ptr2 = (double*)buf2.ptr;
//    double* ptr_result = (double*)buf_result.ptr;
//
//    for (int i = 0; i < buf1.shape[0]; i++)
//    {
//        for (int j = 0; j < buf1.shape[1]; j++)
//        {
//            auto value1 = ptr1[i*buf1.shape[1] + j];
//            auto value2 = ptr2[i*buf2.shape[1] + j];
//
//            ptr_result[i*buf_result.shape[1] + j] = value1 + value2;
//        }
//    }
//
//    return result;
//
//}
//
//std::vector<std::vector<double>> modify(const std::vector<std::vector<double>>& input)
//{
//    std::vector<std::vector<double>> output;
//
//    std::transform(
//            input.begin(),
//            input.end(),
//            std::back_inserter(output),
//            [](const std::vector<double> &iv) {
//                std::vector<double> dv;
//                std::transform(iv.begin(), iv.end(), std::back_inserter(dv), [](double x) -> double { return 2.*x; });
//                return dv;
//            }
//    );
//
//    return output;
//}
//
////struct TestS{
////    vector<int> a;
////    int b;
////};
////
////TestS test3(){
////    TestS test;
////    test.a.push_back(1);
////    test.a.push_back(2);
////    test.b=6;
////    return test;
////}
//
//map<int,vector<int>> modify2()
//{
////    map<int,vector<int>> output;
//    map<int,vector<int>> input;
//    vector<int> v1;
//    v1.push_back(1);
//    vector<int> v2;
//    v2.push_back(1);
//    v2.push_back(2);
//
//    input.insert(pair<int,vector<int>>(1,v1));
//    input.insert(pair<int,vector<int>>(2,v2));
//
////    std::transform(
////            input.begin(),
////            input.end(),
////            std::back_inserter(output),
////            [](const std::pair<int,vector<int>> &iv) {
////                std::vector<int> dv;
////                std::transform(iv.second.begin(), iv.second.end(), std::back_inserter(dv), [](int x) -> int { return 2*x; });
////                return pair<int,vector<int>>(iv.first,dv);
////            }
////    );
//
//    return input;
//}
//
//// vector<int>;map<int,vector<int>>; map<int,int>;map<int, set<int>>
//
//class Dog{
//public:
//    string name;
//    vector<string> food;
//    map<int,vector<int>> noname;
//    void set_name(string name){
//        this->name=name;
//    }
//    string get_name(){
//        return this->name;
//    }
//    void set_food(vector<string> &vec){
//        for(auto str:vec){
//            this->food.push_back(str);
//        }
//    }
//
//    vector<string> get_food(){
//        return this->food;
//    }
//    void set_noname(map<int,vector<int>> &map){
//        for(auto pair_tmp:map){
//            this->noname.insert(pair_tmp);
//        }
//    }
//
//    map<int,vector<int>> get_noname(){
//        return this->noname;
//    }
//};
//
//
//PYBIND11_MODULE(example,m){
//
////    struct FeatureItem{
////        int vid;
////        py::array_t<int> feature;
////    };
////    struct Features{
////        py::array_t<FeatureItem> features;
////    };
////    struct LabelItem{
////        int vid;
////        int label;
////    };
////    struct Labels{
////        py::array_t<LabelItem> labels;
////    };
////
////    struct AdjItem{
////        int vid;
////        py::array_t<int> adj;
////    };
////
////    struct Adjs{
////        py::array_t<AdjItem> adjs;
////    };
//
//
////    m.def("test3", &test3, "Multiply all entries of a nested list by 2.0");
////    py::class_<Dog>(m,"Dog")
////            .def(py::init<>())
////            .def_property("name",&Dog::get_name,&Dog::set_name)
////            .def_property("food",&Dog::get_food,&Dog::set_food)
////            .def_property("noname",&Dog::get_noname,&Dog::set_noname);
////
////    m.def("modify", &modify, "Multiply all entries of a nested list by 2.0");
////    m.def("modify2", &modify2, "Multiply all entries of a nested list by 2.0");
////
////
////    m.doc()="pybind11 example plugin";
////    m.def("add",&add,"A function which adds two numbers");
////    m.def("add_arrays_1d",&add_arrays_1d);
////    m.def("add_arrays_2d",&add_arrays_2d);
//}