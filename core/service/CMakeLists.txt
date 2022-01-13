aux_source_directory(. DIR_LIB_SRCS)
add_library(service ${DIR_LIB_SRCS})
target_link_libraries(service hw_grpc_proto store pybind11::embed compress util partition)