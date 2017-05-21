from model import Model
import numpy


feature_shape = [50, 4096]
words_number = 12
max_time = 20

model = Model(feature_shape = feature_shape,
              words_number = words_number,
              batch_size = 1,
              max_time = max_time)
model.create_model()

train_data = numpy.zeros(shape = [1, max_time, words_number])
train_feature = numpy.zeros([1, feature_shape[0], feature_shape[1]])
train_length = numpy.array([max_time])

for i in range(max_time):
    train_data[0, i, i % words_number] = 1

model.train(train_data, train_feature, train_length)
  