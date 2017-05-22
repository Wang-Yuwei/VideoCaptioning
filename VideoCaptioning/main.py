from model import Model
import numpy
from youtube import preprocess_data, process_dict
from youtube_generator import YouTubeGenerator


batch_size = 1
generator = YouTubeGenerator('index.txt', 'features', batch_size)

feature_shape = [50, 4096]
words_number = generator.words_number
max_time = generator.max_sentence_length
model, length, feature = generator.next()

model = Model(feature_shape = feature_shape,
              words_number = words_number,
              batch_size = batch_size,
              max_time = max_time)
model.create_model()

model.train(generator)


print(feature.shape)