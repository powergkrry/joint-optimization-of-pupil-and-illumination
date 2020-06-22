import numpy as np
import tensorflow as tf
import tf_fftshift as sh


class Pupil(tf.keras.layers.Layer):
    """
    Here, the pupil has certain constraints.
    1. The range of value: (0, 1].
    2. The pupil only modulates amplitude.
    """
    def __init__(self, shape, radius, is_pupil_train):
        super(Pupil, self).__init__()
        self.shape = shape
        self.radius = radius
        self.is_pupil_train = is_pupil_train
        self.init_pupil()
    
    def trim_pupil(self, input_pupil):
        # this makes the outside of pupil to be 0
        a, b = self.shape[0] / 2, self.shape[1] / 2
        n = self.shape[0]
        r = self.radius
        y,x = np.ogrid[-a:n-a, -b:n-b]
        pupil = x*x + y*y <= r*r
        pupil = pupil.reshape(self.shape)
        pupil = tf.convert_to_tensor(pupil, np.float32)
        
        return tf.math.multiply(input_pupil, pupil)
    
    def pupil_constraints(self, is_end = False):
        # the pupil has only positive value
        self.pupil = tf.maximum(self.pupil, 0.001)
        self.pupil = self.trim_pupil(self.pupil)
        # the maximum value of the pupil is 1
        self.pupil = tf.math.divide(self.pupil, tf.reduce_max(self.pupil))
        if self.is_pupil_train and not is_end:
            self.pupil = tf.Variable(self.pupil)
    
    def init_pupil(self):
        if self.is_pupil_train:
            # random seed is already setup in dataloader.py
            self.pupil = np.random.normal(0.5, 0.05, self.shape)
        else:
            self.pupil = np.ones(self.shape)
        self.pupil = tf.convert_to_tensor(self.pupil, np.float32)
        self.pupil_constraints()
        
        if self.is_pupil_train:
            self.pupil = tf.Variable(self.pupil)
        
    def call(self, input_frequency):
        # input frequency's amplitude is trimmed by trained pupil
        return tf.math.multiply(input_frequency, tf.cast(self.pupil, tf.complex128))

class PupilConstraintsEndCallback(tf.keras.callbacks.Callback):
    """
    Backpropagation is performed after pupil constraints are applied.
    To make the pupil satisfy the constraints,
    backpropagation will not be performed at the last epoch.
    """
    def __init__(self, pupil):
        super(PupilConstraintsEndCallback, self).__init__()
        self.pupil = pupil
        
    def on_train_end(self, pupil):
        self.pupil.pupil_constraints(is_end = True)
        output = self.pupil.pupil

class Illumination(tf.keras.layers.Layer):
    def __init__(self, is_illu):
        super(Illumination, self).__init__()
        self.illumination = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1,
                                                   use_bias=False, kernel_initializer='TruncatedNormal')
        self.is_illu = is_illu
    
    def call(self, inputs):
        if self.is_illu:
            return self.illumination(inputs)
        else:
            return inputs

def IFFT(input_frequency, shape):
    output_images = sh.tf_ifftshift2(tf.signal.ifft2d(sh.tf_fftshift2(input_frequency)))
    return tf.reshape(output_images, shape)

def get_intensity(input_image):
    return tf.math.pow(tf.dtypes.cast(tf.math.abs(input_image), tf.float32), tf.constant([2.0]))

def clip_data(input_image):
    return tf.clip_by_value(input_image, clip_value_min=-1., clip_value_max=1.)