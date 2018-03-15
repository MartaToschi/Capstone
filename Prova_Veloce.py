#### IMPORT ####
from keras.preprocessing import image
from keras.callbacks import History, EarlyStopping
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras import optimizers

from PIL import ImageFile
from matplotlib import pyplot as plt
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
early_stop = EarlyStopping(monitor='val_loss', min_delta= 0.01, patience=5, verbose=1, mode='auto')

history = History()
np.random.seed(12345678)

### TRAINING PARAMETERS ###
image_size = 150
batch_size = 64
activation = 'relu'
activation_last = 'sigmoid'
class_mode = 'binary'

### IMPORT DATA ###
train_datagen = image.ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(
        rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size = (image_size, image_size),
        batch_size = batch_size,
        class_mode = class_mode)
print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size = (image_size, image_size),
        batch_size = batch_size,
        class_mode = class_mode)

model = Sequential()                                          # Declare Model
# The first 3 parameters represent:
#  1. The number of convolution filters to use
#  2. The number of rows in each convolution kernel
#  3. The number of columns in each convolution kernel
model.add(Convolution2D(32, 3, 3, input_shape=(image_size, image_size, 3))) # CNN input layers
#model.add(BatchNormalization())
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2)))                      # Equivalent to the reducer step
model.add(Dropout(0.25))                                      # Regularization step

model.add(Convolution2D(32, 3, 3))
#model.add(BatchNormalization())
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2)))                      
model.add(Dropout(0.5))

model.add(Convolution2D(32, 3, 3))
#model.add(BatchNormalization())     
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2)))                      
model.add(Dropout(0.3))

model.add(Flatten())                                          # Necessary before the Dense (makes it 1 dimensional)
model.add(Dense(64))                                          # Output size of the the Layer
model.add(Activation(activation))
model.add(Dropout(0.5))                     
model.add(Dense(1))                                           # The final layer has 2 output node, as the cathegories
model.add(Activation(activation_last))

#sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

output = model.fit_generator(
        train_generator,
        steps_per_epoch = 5052 // batch_size,                 # Number of training images // batch_size (ideally: images % batch = 0)
        epochs = 50,
        callbacks = [early_stop],
        validation_data = validation_generator,
        validation_steps = 2526 // batch_size)
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

model.save("model.h5")
print("Saved model to disk")
        
