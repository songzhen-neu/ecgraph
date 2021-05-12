from dist_sage.dist_start import load_cora
import numpy as np
from operator import itemgetter
import random

if __name__ == '__main__':
    # dict is disorder
    information = {}
    feat_data, labels, adj_lists = load_cora()
    config = {
        'worker_num': 5,
        'vertex_num': feat_data.shape[0]
    }
    # data_raw num per work
    d_num_pw = int(config['vertex_num'] / config['worker_num']) + 1
    feat_data_worker = {}
    labels_worker = {}
    adj_lists_worker = {}

    # define 'worker_num' arrays to record the map (node -> worker_id)
    arrays = []

    for id in range(config['worker_num']):
        if id != config['worker_num'] - 1:
            feat_data_worker[id] = feat_data[id * d_num_pw:(id + 1) * d_num_pw, :]
            labels_worker[id] = labels[id * d_num_pw:(id + 1) * d_num_pw]
            arrays.append([i for i in range(id * d_num_pw, (id + 1) * d_num_pw)])
            adj_lists_worker[id] = itemgetter(*arrays[id])(adj_lists)

        else:
            feat_data_worker[id] = feat_data[id * d_num_pw:, :]
            labels_worker[id] = labels[id * d_num_pw:]
            arrays.append([i for i in range(id * d_num_pw, (config['vertex_num'] - 1))])
            adj_lists_worker[id] = itemgetter(*arrays[id])(adj_lists)

    # calculate communications on each worker
    counts_remote_edge_w = np.zeros(config['worker_num'], dtype=int)
    counts_all_edge_w = np.zeros(config['worker_num'], dtype=int)
    for wid in range(config['worker_num']):
        for v_adj in adj_lists_worker[wid]:
            for v_neigh in v_adj:
                counts_all_edge_w[wid] += 1
                if v_neigh not in arrays[wid]:
                    counts_remote_edge_w[wid] += 1
    count_all_edge = np.sum(counts_all_edge_w)
    count_all_remote_edge = np.sum(counts_remote_edge_w)

    information['remote_edge_w'] = counts_remote_edge_w
    information['all_edge_w'] = counts_all_edge_w
    information['all_edge'] = count_all_edge
    information['all_remote_edge'] = count_all_remote_edge
    print(information)
