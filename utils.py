import os
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session


def setup_gpu(gpu_number=0):
    # set system setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    
    config = tf.compat.v1.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)