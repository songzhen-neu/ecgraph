

#aux_source_directory(. DIR_LIB_SRCS)

add_library(core ${DIR_LIB_SRCS} util/string_split.cc structure/Array.cc util/Length.cc
        partition/Partitioner.cc partition/Partitioner.h partition/RandomPartitioner.cc partition/RandomPartitioner.h
        util/check.cc util/check.h util/string_split.h util/threadUtil.cc util/threadUtil.h
        store/ServerStore.cc store/ServerStore.h compress/compress.cc compress/compress.h partition/GeneralPartition.cc partition/GeneralPartition.h)

add_library(router service/router.cc service/router.h)
target_link_libraries(router client_lib)




add_library(workerStore store/WorkerStore.cc store/WorkerStore.h )
target_link_libraries(workerStore client_lib)


#pybind11_add_module(dgnnclient service/dgnn_client.cc)

target_link_libraries(core client_lib)



