

aux_source_directory(. DIR_LIB_SRCS)

add_library(core ${DIR_LIB_SRCS} util/string_split.cc structure/Array.cc util/Length.cc
        partition/Partitioner.cc partition/Partitioner.h partition/RandomPartitioner.cc partition/RandomPartitioner.h
        store/WorkerStore.cc store/WorkerStore.h util/check.cc util/check.h util/string_split.h util/threadUtil.cc util/threadUtil.h
        store/ServerStore.cc store/ServerStore.h)

#pybind11_add_module(dgnnclient service/dgnn_client.cc)

target_link_libraries(core client_lib)


