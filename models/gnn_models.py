import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from models.basic_model import BasicModel
from models.layers import RandomInputLayer, NameInputLayer, GCNLayer, GATLayer


def get_adj_matrix(triples, ent_num):
    edges = dict()
    for (h, r, t) in triples:
        if h == t:
            continue
        if h not in edges.keys():
            edges[h] = set()
        if t not in edges.keys():
            edges[t] = set()
        edges[h].add(t)
        edges[t].add(h)

    row, col = list(), list()
    for i in range(ent_num):
        if i not in edges.keys():
            continue
        key = i
        value = edges[key]
        multi_key = (key * np.ones(len(value))).tolist()
        row.extend(multi_key)
        col.extend(list(value))
    data = np.ones(len(row))
    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(ent_num, ent_num))
    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
    adj_matrix = adj_matrix.tocoo()
    return adj_matrix


def normalize_adj(adj_matrix):
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj_normalized


class GCN(BasicModel):
    def __init__(self):
        super(GCN, self).__init__()

        self.ent_dim = None
        self.layer_num = None
        self.highway = None
        self.all_layers = None
        self.inputs = None
        self.model = None

    def init(self):
        if self.args.ent_name_init:
            self.ent_dim = len(self.pretrained_embeds[0])
        else:
            self.ent_dim = self.args.ent_dim
        self.layer_num = self.args.layer_num
        self.highway = False
        self.all_layers = False
        if self.args.skip_conn == 'highway':
            self.highway = True
        elif self.args.skip_conn == 'concat':
            self.all_layers = True

        adj_matrix = get_adj_matrix(self.triples, self.ent_num)
        adj = normalize_adj(adj_matrix)
        adj_indices = np.stack((adj.row, adj.col), axis=1)
        adj_data = np.array(adj.data)
        inputs = [adj_indices, adj_data]
        self.inputs = [np.expand_dims(item, axis=0) for item in inputs]
        self.model = self.build()

    def update_adj(self):
        new_triples = self.generate_new_triples()
        adj_matrix = get_adj_matrix(self.triples + list(new_triples), self.ent_num)
        adj = normalize_adj(adj_matrix)
        adj_indices = np.stack((adj.row, adj.col), axis=1)
        adj_data = np.array(adj.data)
        inputs = [adj_indices, adj_data]
        self.inputs = [np.expand_dims(item, axis=0) for item in inputs]

    def build(self):
        adj_indices_input = tf.keras.Input(shape=(None, 2))
        adj_data_input = tf.keras.Input(shape=(None, ))
        if self.args.ent_name_init:
            ent_embeds = NameInputLayer(tf.convert_to_tensor(self.pretrained_embeds, dtype=tf.float32))\
                (adj_indices_input)
        else:
            ent_embeds = RandomInputLayer(self.ent_num, self.ent_dim)(adj_indices_input)
        ent_features = ent_embeds
        all_layer_outputs = [ent_features]
        for i in range(self.layer_num):
            inputs = [adj_indices_input, adj_data_input, ent_features]
            gnn_layer = GCNLayer(self.ent_num, self.ent_dim, self.ent_dim,
                                 self.args.dropout, self.args.activation, self.highway)
            ent_features = gnn_layer(inputs)
            all_layer_outputs.append(ent_features)
        if self.all_layers:
            model = tf.keras.Model(inputs=[adj_indices_input, adj_data_input], outputs=all_layer_outputs)
        else:
            model = tf.keras.Model(inputs=[adj_indices_input, adj_data_input], outputs=ent_features)
        return model


class GAT(BasicModel):
    def __init__(self):
        super(GAT, self).__init__()

        self.ent_dim = None
        self.layer_num = None
        self.head_num = None
        self.highway = None
        self.all_layers = None
        self.inputs = None
        self.model = None

    def init(self):
        if self.args.ent_name_init:
            self.ent_dim = len(self.pretrained_embeds[0])
        else:
            self.ent_dim = self.args.ent_dim
        self.layer_num = self.args.layer_num
        self.head_num = self.args.head_num
        self.highway = False
        self.all_layers = False
        if self.args.skip_conn == 'highway':
            self.highway = True
        elif self.args.skip_conn == 'concat':
            self.all_layers = True

        adj_matrix = get_adj_matrix(self.triples, self.ent_num)
        adj_indices = np.stack((adj_matrix.row, adj_matrix.col), axis=1)
        inputs = [adj_indices]
        self.inputs = [np.expand_dims(item, axis=0) for item in inputs]
        self.model = self.build()

    def update_adj(self):
        new_triples = self.generate_new_triples()
        adj_matrix = get_adj_matrix(self.triples + list(new_triples), self.ent_num)
        adj_indices = np.stack((adj_matrix.row, adj_matrix.col), axis=1)
        inputs = [adj_indices]
        self.inputs = [np.expand_dims(item, axis=0) for item in inputs]

    def build(self):
        adj_indices_input = tf.keras.Input(shape=(None, 2))
        if self.args.ent_name_init:
            ent_embeds = NameInputLayer(tf.convert_to_tensor(self.pretrained_embeds, dtype=tf.float32))\
                (adj_indices_input)
        else:
            ent_embeds = RandomInputLayer(self.ent_num, self.ent_dim)(adj_indices_input)
        ent_features = ent_embeds
        all_layer_outputs = [ent_features]
        for i in range(self.layer_num):
            inputs = [adj_indices_input, ent_features]
            ent_features = GATLayer(self.ent_num, self.ent_dim, self.ent_dim, self.head_num,
                                    self.args.dropout, self.args.activation, self.highway)(inputs)
            all_layer_outputs.append(ent_features)
        if self.all_layers:
            model = tf.keras.Model(inputs=[adj_indices_input], outputs=all_layer_outputs)
        else:
            model = tf.keras.Model(inputs=[adj_indices_input], outputs=ent_features)
        return model

