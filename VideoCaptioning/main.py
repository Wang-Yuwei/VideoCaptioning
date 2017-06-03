from model import CaptioningModel
import numpy
from youtube import preprocess_data, process_dict
from youtube_generator import YouTubeGenerator

#preprocess_data('youtubeclips')
#word2idx, idx_to_word = process_dict('youtubeclips')

generator = YouTubeGenerator('youtubeclips/result/train-index.txt', 'features', 1)
model = CaptioningModel(feature_shape=generator.feature_shape,
                        words_number=generator.words_number,
                        batch_size=1,
                        max_time=generator.max_sentence_length,
                        max_sentence_number = 2)
model.create_paragraph_model(train=True)
model.train(generator,500, 'models')
# model.generate(generator, idx_to_word=idx_to_word, video='vid1216')