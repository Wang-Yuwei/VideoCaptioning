from model import Model
import numpy
from youtube import preprocess_data, process_dict
from youtube_generator import YouTubeGenerator
from train import *

preprocess_data('youtubeclips')

word2idx, idx_to_word = process_dict('youtubeclips')

#batch_size = 2
#generator = YouTubeGenerator('index.txt', 'features', batch_size)
##need to make a new folder, called save
#train(generator, save_path='save')
#generation(generator, save_path='save', videos=['vid1'], idx_to_word=idx_to_word)
