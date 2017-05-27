from model import CaptioningModel
import numpy
from youtube import preprocess_data, process_dict
from youtube_generator import YouTubeGenerator

#preprocess_data('youtubeclips')
#word2idx, idx_to_word = process_dict('youtubeclips')
batch_size = 1
generator = YouTubeGenerator('youtubeclips/result/train-index.txt', 'features', batch_size)
#need to make a new folder, called save
#train(generator, save_path='models/youtube')
model = CaptioningModel(feature_shape=generator.feature_shape,
                        words_number=generator.words_number,
                        batch_size=batch_size,
                        max_time=generator.max_sentence_length)
model.create_model()
#model.train(generator, 200, 'models')
model.generate(generator, 'models/word-weights-improvement-60.hdf5')
#tf.reset_default_graph()
#generation(generator, save_path='models', videos=['vid15'], idx_to_word=idx_to_word)
