import os
import keras.backend as K

def stanh(x):
    return 1.732 * K.tanh(x * 2 / 3)

def checked_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

