import keras.backend as K
from keras.layers import Layer, Dense, InputSpec

class DenseTranspose(Layer):
    def __init__(self, other_layer, **kwargs):
        super().__init__(**kwargs)
        self.other_layer = other_layer

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.output_dim = self.other_layer.input_shape[-1]
        self.W = K.transpose(self.other_layer.weights[0])

        self.b = self.add_weight(
            (self.output_dim,), initializer = 'zero',
            name = '{}_b'.format(self.name), trainable = True)
        self.built = True

    def call(self, x):
        return K.dot(x, self.W) + self.b

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.output_dim)