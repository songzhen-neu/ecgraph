class Graph():
    def __init__(self,agg_node_old,feat,label,adj,id_old2new_dict,id_new2old_dict,fsthop_for_worker):
        # 这里除了agg_node_old和fsthop_for_worker是old id，其他的全部是编码好的
        # agg_node_old是需要进行聚合全部顶点
        self.agg_node_old=agg_node_old
        self.feat_data=feat
        self.label=label
        self.adj=adj
        self.id_old2new_dict=id_old2new_dict
        self.id_new2old_dict=id_new2old_dict
        self.fsthop_for_worker=fsthop_for_worker