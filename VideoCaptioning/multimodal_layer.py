from keras import backend as K
from keras.layers import Layer
from utilities import stanh

class MultimodalLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MultimodalLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        assert type(input_shapes) == list
        self.W = []
        for i, input_shape in enumerate(input_shapes):
            self.W.append(self.add_weight(shape = (input_shape[2], self.output_dim),
                                          initializer = 'uniform', trainable = True,
                                          name = '{}_W{}'.format(self.name, i)))
        self.b = self.add_weight(shape = (self.output_dim,),
                                 initializer = 'uniform', trainable = True,
                                 name = '{}_b'.format(self.name))
        super(MultimodalLayer, self).build(input_shape)

    def call(self, inputs, mask = None):
        self.out = self.b
        for i, input in enumerate(inputs):
            self.out = self.out + K.dot(input, self.W[i])
        self.out = stanh(self.out)
        return self.out

    def compute_output_shape(self, input_shape):
        output_shape = K.int_shape(self.out)
        return (None, output_shape[1], output_shape[2])