from model import Model
import numpy
from youtube import preprocess_data, process_dict
from youtube_generator import YouTubeGenerator


preprocess_data('youtubeclips/video-descriptions.csv',
                'youtubeclips/youtube_mapping.txt',
                'youtubeclips/result.txt')

process_dict('youtubeclips/result.txt', 'youtubeclips/dict.txt', 'youtubeclips/index.txt')

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

model.train(generator, 100, 'models/youtube', 'models/log.txt')

dictionary = []
with open('youtubeclips/dict.txt', encoding = 'utf8') as f:
    dictionary = f.readlines()

model.load_generating_model('models/youtube-1')

word_index = numpy.array([0])

state = None
for i in range(20):
    word_index, state = model.generate_next(word_index, feature, state)
    print(dictionary[word_index[0]])