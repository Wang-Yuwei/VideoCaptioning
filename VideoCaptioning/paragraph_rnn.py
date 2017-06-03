from keras import backend as K
from keras.layers import Layer, Input
from keras.models import Model
from recurrentshop import GRUCell

class ParagraphRNN(Layer):
    def __init__(self, units, sentence_model, **kwargs):
        self.units = units
        self.sentence_model = sentence_model
        self.input_dim = K.int_shape(sentence_model.output[1])[-1] + \
            K.int_shape(sentence_model.output[2])[-1]
        print(self.input_dim)
        self.cell = GRUCell(units = units, input_dim = self.input_dim)
        super(ParagraphRNN, self).__init__(**kwargs)

    def rnn_step(self, input, states, video_features):
        first_sentence_state = states[0]
        paragraph_input_state = states[1]
        sentence_input = input[:, :, 1:]
        mask = input[:, :, 0]
        sentence_output, sentence_output_state, words_embeded = self.sentence_model(
            [sentence_input, first_sentence_state, video_features])
        sum_word_embeded = K.sum(words_embeded, axis = 1)
        length = K.sum(mask, axis = 1)
        average_word_embeded = sum_word_embeded / length
        input_state = K.concatenate([average_word_embeded, sentence_output_state])
        next_sentence_input_state, paragraph_output_state = self.cell(
            [input_state, paragraph_input_state])
        return sentence_output, [next_sentence_input_state, paragraph_output_state]
    
    def call(self, inputs, mask = None):
        assert len(inputs) == 3
        sentence_input = inputs[0]
        mask = inputs[1]
        video_features = inputs[2]
        initial_state = K.zeros_like(sentence_input)
        initial_state = K.sum(initial_state, axis = (1, 2, 3))
        initial_state = K.expand_dims(initial_state)
        initial_state = K.tile(initial_state, [1, self.units])
        initial_states = [initial_state, initial_state]
        combined_input = K.concatenate([K.expand_dims(mask), sentence_input])
        output = K.rnn(lambda input, states: self.rnn_step(input, states, video_features),
                       combined_input, initial_states)
        return output[1]

    def compute_output_shape(self, input_shape):
        print(input_shape[0])
        return input_shape[0]

    def get_weights(self):
        return self.sentence_model.get_weights() + self.cell.get_weights()
        
    @property
    def trainable_weights(self):
        return self.sentence_model.trainable_weights + self.cell.trainable_weights
