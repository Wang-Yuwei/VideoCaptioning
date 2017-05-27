from keras import backend as K
from keras.layers import Layer
from utilities import stanh

class AttentionLayer(Layer):
    def __init__(self, internal_dim, **kwargs):
        self.internal_dim = internal_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        feature_dim = input_shape[0][2]
        hidden_dim = input_shape[1][2]
        self.W = self.add_weight(shape = (feature_dim, self.internal_dim),
                                 initializer = 'uniform', trainable = True,
                                 name = '{}_W'.format(self.name))
        self.U = self.add_weight(shape = (hidden_dim, self.internal_dim),
                                 initializer = 'uniform', trainable = True,
                                 name = '{}_U'.format(self.name))
        self.b = self.add_weight(shape = (self.internal_dim,),
                                 initializer = 'uniform', trainable = True,
                                 name = '{}_b'.format(self.name))
        self.w = self.add_weight(shape = (self.internal_dim, 1),
                                 initializer = 'uniform', trainable = True,
                                 name = '{}_w'.format(self.name))
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask = None):
        v = inputs[0]
        h = inputs[1]
        Uh = K.dot(h, self.U)
        Wv = K.dot(v, self.W)
        Uh = K.reshape(Uh, (-1, K.int_shape(Uh)[1], 1, K.int_shape(Uh)[2]))
        Wv = K.reshape(Wv, (-1, 1, K.int_shape(Wv)[1], K.int_shape(Wv)[2]))

        f = stanh(Wv + Uh + self.b)
        print(f.shape)
        q = K.dot(f, self.w)
        print(q.shape)
        beta = K.exp(q) / K.sum(K.exp(q), axis = -2, keepdims = True)

        v = K.reshape(v, (-1, 1, K.int_shape(v)[1], K.int_shape(v)[2]))
        u = beta * v
        u = K.permute_dimensions(u, [2, 3, 1, 0])
        u = K.sum(u, axis = 0)
        self.u = K.permute_dimensions(u, [2, 1, 0])
        return self.u

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        output_shape = K.int_shape(self.u)
        return output_shape
