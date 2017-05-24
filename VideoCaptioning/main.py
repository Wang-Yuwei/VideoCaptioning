from model import Model
import numpy
from youtube import preprocess_data, process_dict
from youtube_generator import YouTubeGenerator
from train import *
import tensorflow as tf

preprocess_data('youtubeclips')
word2idx, idx_to_word = process_dict('youtubeclips')
batch_size = 200
generator = YouTubeGenerator('youtubeclips/result/train-index.txt', 'features', batch_size)
#need to make a new folder, called save
train(generator, save_path='models')
#tf.reset_default_graph()
#generation(generator, save_path='models', videos=['vid1'], idx_to_word=idx_to_word)
