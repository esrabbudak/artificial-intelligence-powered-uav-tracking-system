from PIL import Image
import numpy as np
from tensorflow import keras
from skimage import transform

#{'heavyTraffic': 0, 'openTraffic': 1, 'pits': 2, 'rubbles': 3, 'trees': 4}

#model.save = ('model_10.h5') S

model = keras.models.load_model('./models/h5format/model_10.h5')

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (128, 128, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

# image = load('./converted/test/pits/pits_350.jpg')
#image = load('./converted/test/rubbles/rubbles_68.jpg')
#image = load('./converted/test/trees/trees_224.jpg')
#image = load('./converted/test/openTraffic_yatay/open_538_(8).jpg')
image = load('./converted/test/rubbles_yatay/rubbles_144.jpg')
predict = model.predict(image)
y_classes = predict.argmax(axis=-1)
print(y_classes)
