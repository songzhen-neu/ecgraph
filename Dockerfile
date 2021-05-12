FROM python:3.6
ADD . /code
RUN pip3 install -r /code/python/requirements.txt -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple
ENV role="",id="",mode="",worker_num="",server_num="" data_num="",feature_dim="",class_num="",hidden=""
ENV ifcompress="",ifcompensate="",isneededexactbackprop="",bucketnum="",iternum="",ifbackpropcompress=""
ENV ifbackpropcompensate="",bucketnum_backprop="",changetoiter="",compensatemethod="",data_path="",isChangeRate=""
ENV bitNum="",trend="",bitNum_backProp="",partitionMethod="",raw_data_path="",edge_num="",localCodeMode="",lr=""
ENV train_num="",val_num="",test_num="",code_path=""

CMD python3 $code_path $role $id $mode $worker_num $server_num $data_num $feature_dim $class_num $hidden $ifcompress \
$ifcompress $ifcompensate $isneededexactbackprop $bucketnum $iternum $ifbackpropcompress \
$ifbackpropcompensate $bucketnum_backprop $changetoiter $compensatemethod $data_path $isChangeRate \
$bitNum $trend $bitNum_backProp $partitionMethod $raw_data_path $edge_num $localCodeMode $lr \
$train_num $val_num $test_num