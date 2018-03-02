#### IMPORT ####
from keras.preprocessing import image
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  # Core layers: These are the layers that are used in almost any NN
from keras.layers import Convolution2D, MaxPooling2D          # CNN layers: These will help us efficiently train on image data
from keras.utils import np_utils
from PIL import ImageFile
from matplotlib import pyplot as plt
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
history = History()
np.random.seed(123456789)

batch_size = 16

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size = batch_size,
        class_mode='categorical')

print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size = batch_size,
        class_mode='categorical')

model = Sequential()                                          # Declare Model
# The first 3 parameters represent:
#  1. The number of convolution filters to use
#  2. The number of rows in each convolution kernel
#  3. The number of columns in each convolution kernel
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(150,150,3))) # CNN input layers
model.add(Convolution2D(32, 3, 3, activation='relu'))         
model.add(MaxPooling2D(pool_size=(2,2)))                      # Equivalent to the reducer step
model.add(Dropout(0.25))                                      # Regularization step
model.add(Flatten())                                          # Necessary before the Dense (makes it 1 dimensional)
model.add(Dense(128, activation='relu'))                      # Output size of the the Layer
model.add(Dropout(0.5))                     
model.add(Dense(2, activation='softmax'))                     # The final layer has 2 output node, as the cathegories

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

output = model.fit_generator(
        train_generator,
        steps_per_epoch = batch_size,
        epochs = 25,
        validation_data = validation_generator,
        validation_steps = 160 // batch_size)
print(output.history)

# plot
plt.figure(1)

plt.subplot(121)  
plt.plot(output.history['acc'], 'r*')  
plt.plot(output.history['val_acc'], 'go')  
plt.title('Model Accuracy')  
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')
plt.ylim(0,1)
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)  
plt.plot(output.history['loss'], 'r*')  
plt.plot(output.history['val_loss'], 'go')  
plt.title('Model Loss')  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend(['Train', 'Validation'], loc='upper left')  

plt.show()
        
#### SAVE JSON MODEL and WEIGHTS ####
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_weights.h5")
print("Saved model to disk")
        
keras.callbacks.EarlyStopping(monitor='val_acc', min_delta= 0.01, patience=0, verbose=1, mode='auto')