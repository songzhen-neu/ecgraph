# EC-Graph: A Distributed GNN System with Error-Compensted Compression.

This is a system for distributed GNN training, which adopts a Graph-Centered based framework. There are three logical roles in this system, i.e., masters, workers and servers. Master reads the partition profiles and a input graph to assign sub-graphs to different training nodes (workers). The workers are responsible for the majority of computations (including forward propagation and backward propagation), while the servers manage the parameters (aggregating gradients and then updating the parameters). EC-Graph allows the communication among workers for exchanging the embeddings (FP) and the gradients of embeddings (BP). 

![image](https://user-images.githubusercontent.com/20767715/139521729-c2b5a7ca-8a3a-47ab-9668-2ebb452837ca.png)


The argument list:
```c++
--role_id=master,0
--worker_server_num=2,2
--ifctx_mode=true,test
--data_path=/mnt/data/cora
--hidden=16
--vtx_edge_feat_class_train_val_test=2708,5278,1433,7,1408,300,1000
--if_cprs_trend_backcprs_backcpst_changeBit=false,false,false,false,false
--bit_backbit_trend_printepoch=2,2,10,5
--iter_lr_pttMethod=1000,0.2,hash
--servers=127.0.0.1:2001,127.0.0.1:2002,127.0.0.1:2003
--workers=127.0.0.1:3001,127.0.0.1:3002,127.0.0.1:3003
--master=127.0.0.1:4001
--distgnnr=5
```
Note that, /mnt/data/cora is the common-shared nfs directory. 
Also, you can create the directory in the local to simulate the 
distributed environment. 

##**_How to Install EC-Graph_**

If you just want to using EC-Graph to build your own GNN models, you need to install the python dependencies:
`python3.6` and `requirements` in "python/requirements.txt" on Ubuntu16.04 (other versions of Ubuntu can also work).
Then use "ldd cmake/build/example2.cpython-36m-x86_64-linux-gnu.so" to detect if all dependencies are satisfied. 

**_otherwise_**:

If you intend to modify the core codes of EC-Graph in c++, beyond the python dependencies, you need to install `cmake, grpc, protobuf, pybind11`

```
mkdir cmake/build && cd cmake/build
cmake ../..
make
```

If build successfully, it will generate new 4 grpc and protobuf files (".grpc.pb.cc and .h","pb.cc and .h" )
 and the dynamic link library example2.cpython-36m-x86_64-linux-gnu.so. Then, you can run EC-Graph
 following the instructions in "How to Run an Example"


You can use docker to run EC-Graph on the servers, please see details in Dockerfile.

##**_How to Run an Example_**

1: Install a distributed file system, e.g., NFS, HDFS. Set the shared-directory as "/mnt/data". 
If you just want to run EC-Graph on a single-machine, you can just "mkdir /mnt/data" without installing NFS.

2: Processing the data format to the EC-Graph format by using programs in "python/data_processing".
```
Two files will be created (all separators are "\t"). 

featsClass.txt (id   feat (dim = 5)   class):
0 1 0 1 1 1 0
1 0 1 1 0 1 1
2 0 0 0 1 1 0

edges.txt (src   dst)
0   1
1   2
0   2
``` 

3: Move these two files to their directory "/mnt/data/cora"

4: Set the number of workers and servers in "python/context/context.py"

5: Run 1 master, 1 worker and 1 server as an example
```
Run "python/dist_gcn_agg_grad/dist_start.py" with "--role_id=master,0"
Run "python/dist_gcn_agg_grad/dist_start.py" with "--role_id=server,0"
Run "python/dist_gcn_agg_grad/dist_start.py" with "--role_id=worker,0"
```
The other settings are shown at the beginning "argument list".



