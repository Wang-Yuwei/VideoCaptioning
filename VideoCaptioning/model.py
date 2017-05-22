import tensorflow as tf

HIDDEN_NUMBER = 512
MULTIMODEL_NUMBER = 1024
MAX_TIME = 20
ATTENTION_SIZE = 32

def stanh(input):
    return tf.tanh(input * 2 / 3) * 1.7159

class Model:
    def __init__(self, **kwargs):
        self.feature_shape = kwargs.pop('feature_shape', [50, 4096])
        self.feature_number = self.feature_shape[0]
        self.words_number = kwargs.pop('words_number', 12)
        self.gru_cell = tf.contrib.rnn.GRUCell(HIDDEN_NUMBER, activation = tf.nn.relu)
        self.batch_size = kwargs.pop('batch_size', 2)
        self.max_time = kwargs.pop('max_time', MAX_TIME)

    def create_model(self):
        self.input_shape = [self.batch_size, self.max_time, self.words_number]
        self.input = tf.placeholder(tf.float32,
                                    shape = self.input_shape,
                                    name = 'onehot-word')
        self.length = tf.placeholder(tf.float32, shape = [self.batch_size], name = 'word-length')
        # Shape of feature pool (batch_size, depth, number of features)
        self.video_feature_pool = tf.placeholder(tf.float32,
                                                 [self.batch_size] + self.feature_shape)
        embed_table = tf.get_variable('embed_table', [HIDDEN_NUMBER, self.words_number])
        embeded_word = self.time_matmul(self.input, embed_table)
        self.share = False
        rnn_output, rnn_state = tf.nn.dynamic_rnn(self.gru_cell, embeded_word, dtype = tf.float32, initial_state = None)
        attention_output = self.attention(self.video_feature_pool, rnn_output)
        multimodal_output = self.multimodal([attention_output, rnn_output], HIDDEN_NUMBER)
        multimodal_output = tf.nn.dropout(multimodal_output, 0.5)
        hidden_input = stanh(multimodal_output)
        hidden_output = self.time_matmul(hidden_input, tf.transpose(embed_table))
        self.next_words = tf.nn.softmax(hidden_output)
        self.next_words_sliced = tf.slice(self.next_words, [0, 0, 0], [-1, self.max_time - 1, -1])
        self.input_sliced = tf.slice(self.input, [0, 1, 0], [-1, -1, -1])
        # TODO add mask here
        # shape of probability (batch_size, max_sentence_length - 1, dict_size)
        self.probability = self.input_sliced * self.next_words_sliced
        # shape of probability (batch_size, max_sentence_length - 1)
        self.probability = tf.reduce_sum(self.probability, [-1]) + 1e-6
        self.mask = tf.sequence_mask(self.length, self.max_time)
        self.mask = tf.slice(self.mask, [0, 1], [-1, -1])
        self.log_prob = -tf.log(self.probability) * tf.cast(self.mask, tf.float32)
        self.sum_log = tf.reduce_sum(self.log_prob, [1])
        self.ppl = tf.reduce_sum(self.sum_log / self.length)
        self.optimizer = tf.train.RMSPropOptimizer(0.0001, momentum = 0.5).minimize(self.ppl)
        ppl_summary = tf.summary.scalar('ppl', self.ppl)
        self.summary = tf.summary.merge_all()

    def time_matmul(self, input, weight):
        input_dim = input.get_shape().as_list()[2]
        input_reshaped = tf.reshape(input, [-1, input_dim])
        output_reshaped = tf.matmul(input_reshaped, tf.transpose(weight))
        output_dim = weight.get_shape().as_list()[0]
        return tf.reshape(output_reshaped, [self.batch_size, self.max_time, output_dim])

    def multimodal(self, input_list, output_dim, name = 'multimodal'):
        with tf.variable_scope(name, reuse = False):
            bias = tf.get_variable('bias', [output_dim],
                                   initializer = tf.constant_initializer(0))
            result = bias
            for i in range(len(input_list)):
                input_dim = int(input_list[i].get_shape()[2])
                weight = tf.get_variable('weight%d' % i,
                                         [input_dim, output_dim])
                input_reshaped = tf.reshape(input_list[i],
                                            [self.batch_size * self.max_time, input_dim])
                output_reshaped = tf.matmul(input_reshaped, weight)
                output = tf.reshape(output_reshaped,
                                    [self.batch_size, self.max_time, output_dim])
                result = result + output
            return result

    # Shape of input (batch_size, max_time, depth)
    def dense(self, input, output_dim, name = 'dense'):
        with tf.variable_scope(name, reuse = self.share):
            input_dim = input.get_shape()[2]
            weight = tf.get_variable('weight',
                                     [input_dim, output_dim])
            bias = tf.get_variable('bias',
                                   [output_dim],
                                   initializer = tf.constant_initializer(0))
            input_reshaped = tf.reshape(input,
                                        [self.batch_size * self.max_time, input_dim])
            output_reshaped = tf.matmul(weight, input_reshaped) + bias
            output = tf.reshape(output_reshaped,
                                [self.batch_size, self.max_time, input_dim])
            return output

    def attention(self, batch_feature, batch_input, name = 'attention'):
        input_list = tf.unstack(batch_input, axis = 0)
        feature_list = tf.unstack(batch_feature, axis = 0)
        assert(len(input_list) == len(feature_list))
        shared = False
        result = []
        for i in range(len(input_list)):
            # Shape of input (max_time, depth)
            input = input_list[i]
            # Shape of feature (feature_number, feature_dimension)
            feature = feature_list[i]
            feature_dimension = int(feature.shape[1])
            result_list = []
            for state in tf.unstack(input):
                with tf.variable_scope('attention', reuse = shared):
                    wq = tf.get_variable('wq', [ATTENTION_SIZE, feature_dimension])
                    u = tf.get_variable('u', [ATTENTION_SIZE, HIDDEN_NUMBER])
                    b = tf.get_variable('b', [ATTENTION_SIZE])
                    output = tf.matmul(wq, tf.transpose(feature)) + \
                        tf.matmul(u, tf.expand_dims(state, axis = -1)) + \
                        tf.stack([b] * self.feature_number, axis = -1)
                    output = stanh(output)
                    w = tf.get_variable('w', [1, ATTENTION_SIZE])
                    q = tf.matmul(w, output)
                    beta = tf.nn.softmax(q)
                    average = tf.matmul(beta, feature)
                    average = tf.reshape(average, shape = [feature_dimension])
                    shared = True
                result_list.append(average)
            result.append(tf.stack(result_list))
        return tf.stack(result, axis = 0, name = name)

    def train(self, train_generator):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(10):
                for i in range(0, train_generator.sample_number, self.batch_size):
                    data, length, feature = train_generator.next()
                    _, loss_value, summary = session.run(
                        [self.optimizer, self.ppl, self.summary],
                        feed_dict = {self.input : data,
                                     self.length: length,
                                     self.video_feature_pool: feature})
                    print(loss_value)