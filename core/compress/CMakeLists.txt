aux_source_directory(. DIR_LIB_SRCS)
add_library(compress ${DIR_LIB_SRCS})
target_link_libraries(compress service)