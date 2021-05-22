FROM python:3.6
ADD . /code
# RUN python -m pip install --upgrade pip

RUN pip3 install -r /code/python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn
ENV role_id="",worker_server_num="",ifctx_mode="",data_path="",hidden="",vtx_edge_feat_class_train_val_test=""
ENV if_cprs_trend_backcprs_backcpst_changeBit="",bit_backbit_trend="",iter_lr_pttMethod="",servers="",workers="",master=""


RUN apt update && apt install -y wget vim git libz-dev net-tools inetutils-ping lsof
# RUN apt-get -y install gawk && apt-get -y install bison
# RUN apt install -y libstdc++6

# RUN ln -sf /lib/x86_64-linux-gnu/libc.so.6 /lib64/libc.so.6

# RUN wget https://mirror.tuna.tsinghua.edu.cn/gnu/glibc/glibc-2.29.tar.gz \
#     && tar -zxvf glibc-2.29.tar.gz \
#     && cd /glibc-2.29 \
#     && mkdir build && cd build \
#     && ../configure --prefix=/opt/glibc-2.29 \
#     && make -j4 \
#     && make install




RUN apt install -y nfs-kernel-server
RUN apt install -y nfs-common
# root in docker, don't need chmod
RUN mkdir -p /mnt/data
RUN mkdir -p /mnt/data/nfs/graph-learn/distributed/

CMD mount -o nolock -t nfs 219.216.64.103:/var/data /mnt/data && python3 $code_path $role_id $worker_server_num $ifctx_mode $data_path \
$data_path $hidden $vtx_edge_feat_class_train_val_test $if_cprs_trend_backcprs_backcpst_changeBit $bit_backbit_trend \
$iter_lr_pttMethod $servers $workers $master
