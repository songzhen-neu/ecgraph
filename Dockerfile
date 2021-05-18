FROM python:3.6
ADD . /code
# RUN python -m pip install --upgrade pip

RUN pip3 install -r /code/python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn/simple
ENV role="",id="",mode="",worker_num="",server_num="" data_num="",feature_dim="",class_num="",hidden=""
ENV ifcompress="",ifcompensate="",isneededexactbackprop="",bucketnum="",iternum="",ifbackpropcompress=""
ENV ifbackpropcompensate="",bucketnum_backprop="",changetoiter="",compensatemethod="",data_path="",isChangeRate=""
ENV bitNum="",trend="",bitNum_backProp="",partitionMethod="",raw_data_path="",edge_num="",localCodeMode="",lr=""
ENV train_num="",val_num="",test_num="",code_path=""
ENV workers="",servers="",master=""


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

CMD mount -o nolock -t nfs 219.216.64.103:/var/data /mnt/data && python3 $code_path $role $id $mode $worker_num $server_num $data_num $feature_dim $class_num $hidden $ifcompress \
$ifcompress $ifcompensate $isneededexactbackprop $bucketnum $iternum $ifbackpropcompress \
$ifbackpropcompensate $bucketnum_backprop $changetoiter $compensatemethod $data_path $isChangeRate \
$bitNum $trend $bitNum_backProp $partitionMethod $raw_data_path $edge_num $localCodeMode $lr \
$train_num $val_num $test_num \
$workers $master $servers