This is a distributed graph neural network system.

The argument list:
```c++
--role_id=master,0
--worker_server_num=2,2
--ifctx_mode=false,test
--data_path=/mnt/data/cora
--hidden=16
--vtx_edge_feat_class_train_val_test=2708,5278,1433,7,140,500,1000
--if_cprs_trend_backcprs_backcpst_changeBit=false,false,false,false,false
--bit_backbit_trend=2,2,10
--iter_lr_pttMethod=1000,0.2,metis
--servers=127.0.0.1:2001,127.0.0.1:2002
--workers=127.0.0.1:3001,127.0.0.1:3002
--master=127.0.0.1:4001
```
Note that, /mnt/data/cora is the common-shared nfs directory. 
Also, you can create the directory in the local to simulate the 
distributed environment.