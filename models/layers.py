import tensorflow as tf
from tensorflow.python.keras import initializers, activations


class RandomInputLayer(tf.keras.layers.Layer):
    def __init__(self, ent_num, ent_dim):
        super(RandomInputLayer, self).__init__()
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.embeds = self.add_weight('embeds',
                                      shape=(ent_num, ent_dim),
                                      dtype='float32',
                                      initializer=self.kernel_initializer,
                                      trainable=True)

    def call(self, inputs):
        return self.embeds


class NameInputLayer(tf.keras.layers.Layer):
    def __init__(self, pretrained_embeds):
        super(NameInputLayer, self).__init__()
        self.ent_embeds = tf.Variable(pretrained_embeds, trainable=True)
        # self.ent_embeds = tf.nn.l2_normalize(self.ent_embeds, 1)

    def call(self, inputs):
        return self.ent_embeds


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self,
                 ent_num,
                 in_dim,
                 out_dim,
                 dropout,
                 activation,
                 highway):
        super(GCNLayer, self).__init__()
        self.ent_num = ent_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.highway = highway

        self.weight_initializer = initializers.get('ones')
        self.weight = None
        self.gate_weight, self.gate_bias = None, None

    def build(self, input_shape):
        self.weight = self.add_weight('kernel',
                                      shape=(1, self.in_dim),
                                      initializer=self.weight_initializer,
                                      dtype='float32',
                                      trainable=True)

        if self.highway:
            self.gate_weight = self.add_weight('gate weight',
                                               shape=(self.in_dim, self.in_dim),
                                               initializer=initializers.get('glorot_uniform'),
                                               dtype='float32',
                                               trainable=True)
            self.gate_bias = self.add_weight('bias',
                                             shape=(self.in_dim, ),
                                             initializer=initializers.get('glorot_uniform'),
                                             dtype='float32',
                                             trainable=True)

    def call(self, inputs, training=True):
        adj = tf.SparseTensor(indices=tf.cast(tf.squeeze(inputs[0], axis=0), dtype=tf.int64),
                              values=tf.squeeze(inputs[1], axis=0), dense_shape=(self.ent_num, self.ent_num))
        features = inputs[2]
        # features = tf.keras.layers.BatchNormalization()(features)
        if training and self.dropout > 0.0:
            features = tf.nn.dropout(features, self.dropout)

        h = tf.multiply(features, self.weight)
        outputs = tf.sparse.sparse_dense_matmul(tf.cast(adj, tf.float32), h)
        outputs = self.activation(outputs)
        if self.highway:
            gate = tf.matmul(features, self.gate_weight)
            gate = tf.add(gate, self.gate_bias)
            gate = tf.keras.activations.sigmoid(gate)
            outputs = tf.add(tf.multiply(outputs, gate), tf.multiply(features, 1.0 - gate))
        return outputs


class GATLayer(tf.keras.layers.Layer):
    def __init__(self,
                 ent_num,
                 in_dim,
                 out_dim,
                 head_num,
                 dropout,
                 activation,
                 highway):
        super(GATLayer, self).__init__()
        self.ent_num = ent_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_heads = head_num
        self.attn_kernels = []
        self.self_kernels = []

        self.dropout = dropout
        self.highway = highway
        self.activation = activations.get(activation)

        self.weight_initializer = initializers.get('ones')
        self.attn_initializer = initializers.get('glorot_uniform')

        self.gate_weight, self.gate_bias = None, None

    def build(self, input_shape):
        for head in range(self.attn_heads):
            self_kernel = self.add_weight('kernel',
                                          shape=[1, self.out_dim],
                                          initializer=self.weight_initializer,
                                          dtype='float32',
                                          trainable=True)

            attn_kernel = self.add_weight('attention vector',
                                          shape=[1, 2 * self.out_dim],
                                          initializer=self.attn_initializer,
                                          dtype='float32',
                                          trainable=True)
            self.self_kernels.append(self_kernel)
            self.attn_kernels.append(attn_kernel)

        if self.highway:
            self.gate_weight = self.add_weight('gate weight',
                                               shape=[self.in_dim, self.out_dim],
                                               initializer=initializers.get('glorot_uniform'),
                                               dtype='float32',
                                               trainable=True)
            self.gate_bias = self.add_weight('bias',
                                             shape=[self.out_dim, ],
                                             initializer=initializers.get('glorot_uniform'),
                                             dtype='float32',
                                             trainable=True)

    def call(self, inputs, training=True):
        adj = tf.SparseTensor(indices=tf.cast(tf.squeeze(inputs[0], axis=0), dtype=tf.int64),
                              values=tf.ones_like(inputs[0][0, :, 0]), dense_shape=(self.ent_num, self.ent_num))
        features = inputs[1]
        # features = tf.keras.layers.BatchNormalization()(features)
        if training and self.dropout > 0.0:
            features = tf.nn.dropout(features, self.dropout)

        outputs_list = []
        for head in range(self.attn_heads):
            h = tf.multiply(features, self.self_kernels[head])
            nbr_h = tf.gather(h, adj.indices[:, 1])
            self_h = tf.gather(h, adj.indices[:, 0])
            edge_h = tf.concat([self_h, nbr_h], axis=1)

            att_h = tf.matmul(self.attn_kernels[head], tf.transpose(edge_h))
            att_h = tf.squeeze(att_h)
            att_h = tf.SparseTensor(indices=adj.indices,
                                    values=tf.nn.leaky_relu(att_h),
                                    dense_shape=adj.dense_shape)
            att_h = tf.sparse.softmax(att_h)

            outputs = tf.sparse.sparse_dense_matmul(att_h, h)
            outputs_list.append(outputs)

        outputs = tf.concat(outputs_list, axis=1)
        outputs = tf.reshape(outputs, [outputs.shape[0], self.attn_heads, -1])
        outputs = tf.reduce_mean(outputs, axis=1)
        outputs = self.activation(outputs)

        if self.highway:
            gate = tf.matmul(features, self.gate_weight)
            gate = tf.add(gate, self.gate_bias)
            gate = tf.keras.activations.sigmoid(gate)
            outputs = tf.add(tf.multiply(outputs, gate), tf.multiply(features, 1.0 - gate))
        return outputs

