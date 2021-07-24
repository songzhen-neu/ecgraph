
cmake_minimum_required(VERSION 3.17)
project(dgnn_test C CXX)

# linux cxx environment setting
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)
#find_package(Threads REQUIRED)


# find protobuf installation
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)

message(STATUS "Using protobuf ${Protobuf_VERSION}")


set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP grpc++)

set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# find grpc

#add_subdirectory(grpc)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)

set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(pybind11)
add_compile_options(-fPIC)

#pybind11_add_module(example example.cc)
#target_link_libraries(example PRIVATE core)
#add_compile_options(-fPIC)

pybind11_add_module(example3 example3.cc)
#target_link_libraries(example3 PRIVATE dgnn_server_lib)
#target_link_libraries(example3 PRIVATE core)

pybind11_add_module(example2 example2.cc)
target_link_libraries(example2 PRIVATE bittest)
#target_link_libraries(example2 PRIVATE dgnn_server_lib)
target_link_libraries(example2 PRIVATE core)
target_link_libraries(example2 PRIVATE router)
#add_compile_options(-fPIC)



# Proto file
get_filename_component(hw_proto "protos/dgnn_test.proto" ABSOLUTE)
# hw_proto_path doesn't contain the file, only shows the path;
get_filename_component(hw_proto_path "${hw_proto}" PATH)
message(STATUS "hw_proto: ${hw_proto}")
message(STATUS "hw_proto_path: ${hw_proto_path}")

# Generated sources
message(STATUS "CURRENT BINARY PATH: ${CMAKE_CURRENT_BINARY_DIR}")

set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/dgnn_test.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/dgnn_test.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/dgnn_test.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/dgnn_test.grpc.pb.h")

message(STATUS "proto_src ${hw_proto_srcs}")


add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}"
)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "${CMAKE_CURRENT_BINARY_DIR}")

#aux_source_directory(. DIR_SRCS)
add_subdirectory(cpptest)

add_subdirectory(core)

# Targets greeter_[async_](client|server)

find_package(Threads)



#foreach(_target
#        dgnn_server)
#    message(STATUS "_target: ${_target}")
#
#
#endforeach()
add_library(client_lib core/service/dgnn_client.cc core/service/dgnn_client.h ${hw_proto_srcs} ${hw_grpc_srcs} )
target_link_libraries(client_lib ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF}  dgnn_server_lib)



add_library(dgnn_server_lib dgnn_server.cc dgnn_server.h
        ${hw_proto_srcs}
        ${hw_grpc_srcs})

target_link_libraries(dgnn_server_lib
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

target_link_libraries(client_lib core)
target_link_libraries(client_lib workerStore)

target_link_libraries(client_lib pybind11::embed)
target_link_libraries(dgnn_server_lib core)
add_executable(mainTest main.cc)
target_link_libraries(mainTest client_lib workerStore bittest)

