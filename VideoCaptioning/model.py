import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, GRU, Dropout, TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback
from attention_layer import AttentionLayer
from multimodal_layer import MultimodalLayer
import keras.backend as K
from dense_transpose import DenseTranspose
import numpy as np

HIDDEN_NUMBER = 512
MULTIMODEL_NUMBER = 1024
MAX_TIME = 20
ATTENTION_SIZE = 32

class HistoryLogger(Callback):
    def __init__(self, filename):
        self.logfile = open(filename, 'w+')

    def on_epoch_begin(self, epoch, logs = {}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs = {}):
        print('epoch %d, batch %d, loss %f' % (self.epoch, batch, logs.get('loss')), file = self.logfile)
        self.logfile.flush()

class CaptioningModel:
    def __init__(self, **kwargs):
        self.feature_shape = kwargs.pop('feature_shape', [50, 4096])
        self.feature_number = self.feature_shape[0]
        self.words_number = kwargs.pop('words_number', 12)
        self.batch_size = kwargs.pop('batch_size', 2)
        self.max_time = kwargs.pop('max_time', MAX_TIME)

    def embedding_layer(self, input_data):
        model = Sequential()
        model.add(Dense(HIDDEN_NUMBER, input_shape = (self.max_time, self.words_number)))
        model.add(Activation('relu'))
        model.compile('rmsprop', 'mse')
        embedding_weights = model.get_weights()
        output_array = model.predict(input_data)
        self.embedding_weights = model.get_weights()
        output_weights = np.asarray(self.embedding_weights[0]).T
        self.embedding_weights[0] = output_weights
        self.embedding_weights[1] = np.ones((self.words_number,))
        return output_array

    def get_loss_function(self, mask):
        length = K.sum(mask, axis = -1)
        def compute_loss(y_true, y_pred):
            prob = y_true * y_pred
            print(prob.shape)
            log_prob = -K.log(K.sum(prob, axis = -1) * mask + 1e-7)
            return K.sum(log_prob, axis = -1) / length

        return compute_loss

    def create_model(self):
        words_input = Input(shape = (self.max_time, self.words_number),
                            dtype = 'float32')
        video_feature_pool = Input(shape = self.feature_shape, dtype = 'float32')
        mask = Input(shape = (self.max_time,), dtype = 'float32')
        embeded_layer = Dense(HIDDEN_NUMBER, input_shape = (self.max_time, self.words_number))
        words_embeded = embeded_layer(words_input)
        rnn_output = GRU(HIDDEN_NUMBER, return_sequences = True,
                  input_shape = (self.max_time, HIDDEN_NUMBER),
                  activation = 'relu')(words_embeded)
        attention_output = AttentionLayer(internal_dim = 256, name = 'attention')([video_feature_pool, rnn_output])
        multimodal_output = MultimodalLayer(output_dim = 1024)([rnn_output, attention_output])
        dropout = Dropout(0.5)(multimodal_output)
        decode_output = TimeDistributed(Dense(activation = 'tanh', units = HIDDEN_NUMBER))(dropout)
        softmax_input = DenseTranspose(embeded_layer)(decode_output)
        softmax_output = Activation(activation = 'softmax')(softmax_input)
        model = Model(inputs = [words_input, mask, video_feature_pool], outputs = [softmax_output])
        loss_function = self.get_loss_function(mask)
        model.summary()
        model.compile(optimizer = RMSprop(lr = 0.0001), loss = loss_function)
        self.model = model

    def train(self, generator, epochs, savepath):
        filepath = savepath + '/word-weights-improvement-{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(filepath, save_weights_only = True, monitor = 'loss')
        logger = HistoryLogger(savepath + '/log.txt')
        self.model.fit_generator(generator.read_generator(),
                       steps_per_epoch = int(generator.sample_number / generator.batch_size), epochs = epochs, callbacks = [logger])
        self.model.save_weights(savepath + '/final-weights.hdf5')
        input, output = next(generator.read_generator())
        output = self.model.predict(input)
        args = np.argmax(output, axis = -1)
        print(args)

    def generate(self, generator, savepath):
        self.model.load_weights(savepath, by_name = True)
        input, output = next(generator.read_generator())
        output = self.model.predict(input)
        args = np.argmax(output, axis = -1)
        print(args)