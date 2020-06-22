import os
import numpy as np
from config import get_config
from utils import setup_gpu
from dataloader import load_data
from module import Pupil, Illumination, IFFT, get_intensity, \
clip_data, PupilConstraintsEndCallback
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Lambda, Permute, GaussianNoise
from tensorflow.keras.models import Model
from keras.optimizers import Adam


config, unparsed = get_config()
setup_gpu(config.gpu_number)

x_train, y_train, x_test, y_test = load_data(config.data, config.random_seed)
_, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH = x_train.shape

my_pupil = Pupil([28, 28], 6, config.is_pupil_train)
output_start = my_pupil.pupil
my_illumination = Illumination(config.is_illu)
output_start2 = my_illumination.illumination

inputs = Input((IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), dtype = 'complex128')

# physical layer - pupil
x = my_pupil(inputs)
my_pupil.pupil_constraints()
x = Lambda(lambda x: IFFT(x, (-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH))) (x)
x = Lambda(lambda x: get_intensity(x)) (x)
x = Permute((2,3,1))(x)

# physical layer - illumination
x = my_illumination(x)
x = Lambda(lambda x: clip_data(x)) (x)
inputs_print = GaussianNoise(config.noise_std)(x)

# digital layers
x = Conv2D(6, (3, 3), activation='relu', padding='same') (inputs_print)
x = Conv2D(6, (3, 3), activation='relu', padding='same') (x)
x = MaxPooling2D((2, 2)) (x)
x = Conv2D(6, (3, 3), activation='relu', padding='same') (x)
x = Conv2D(6, (3, 3), activation='relu', padding='same') (x)
x = MaxPooling2D((2, 2)) (x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=[inputs], outputs=[outputs])
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[PupilConstraintsEndCallback(my_pupil)])
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# print pupil
# output = model.layers[1].pupil
# pupil_weight = output.get_weights()

# print illumination
# output = model.layers[5].illumination
# LED_weight = output.get_weights()[0][0,0,:,0]
# LED_weight = np.multiply(LED_weight, np.concatenate((np.repeat(1.0, 13), np.repeat(1/3, 12))))
