from keras.preprocessing import image
from PIL import ImageFile
from matplotlib import pyplot as plt
import numpy as np
from keras.models import model_from_json                      # To save the model on json
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import plot_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(123456789)

test_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

print(test_generator.class_indices)

filenames = test_generator.filenames
nb_samples = len(filenames)

#### LOAD JSON MODEL and WEIGHTS ####
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")

plot_model(loaded_model, to_file = 'model.png')

predict = loaded_model.predict_generator(test_generator, steps = 1)

for i, p in enumerate(predict):
    print (filenames[i] , ' --------------> ' , p)