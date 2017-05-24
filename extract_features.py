import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import os

model = VGG16()
f = K.function([model.layers[0].input],
               [model.get_layer('fc2').output])

img = image.load_img('result/vid1/result001.jpg', target_size=(224, 224))
if not os.path.exists('features'):
    os.makedirs('features')
for basename in os.listdir('result'):
    features = []
    print(basename)
    if not os.path.isdir('result/' + basename):
        continue
    for file in os.listdir('result/' + basename):
        fullname = 'result/' + basename + '/' + file
        img = image.load_img(fullname,
                             target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, 0);
        x = preprocess_input(x)
        feature = f([x])[0]
        features.append(feature)
    features = np.concatenate(features)
    np.save('features/' + basename + '.npy', features)