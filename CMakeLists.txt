cmake_minimum_required(VERSION 3.17)
project(dgnn_test VERSION 1.0)

# linux cxx environment setting
set(CMAKE_CXX_STANDARD 11)


# find protobuf installation, need protobufConfig.cmake
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")


set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP grpc++)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# find grpc, need grpcConfig.cmake
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

#find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11
set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(third_party/pybind11)
add_compile_options(-fPIC)
#include_directories(eigen3)



# Proto file
get_filename_component(hw_proto "core/protos/dgnn_test.proto" ABSOLUTE)
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

add_subdirectory(cpptest)
add_subdirectory(core)
find_package(Threads)



add_library(hw_grpc_proto
        ${hw_grpc_srcs}
        ${hw_grpc_hdrs}
        ${hw_proto_srcs}
        ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})


add_executable(mainTest main.cc)
target_link_libraries(mainTest store bittest service compress partition)

