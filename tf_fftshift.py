import tensorflow as tf


def tf_fftshift3(A):
    # 3D fftshift
    # apply fftshift to the last 3 dims
    # for some reason if I use A.shape tf says <NOT CONVERTIBLE TO TENSOR>
    # s = A.shape
    s = tf.shape(A)
    s1 = s[-3]+1
    s2 = s[-2]+1
    s3 = s[-1]+1
    A = tf.concat([A[..., s1//2:, :, :], A[..., :s1//2, :, :]], axis=-3)
    A = tf.concat([A[..., :, s2//2:, :], A[..., :, :s2//2, :]], axis=-2)
    A = tf.concat([A[..., :, :, s3//2:], A[..., :, :, :s3//2]], axis=-1)
    return A

def tf_ifftshift3(A):
    # 3D ifftshift
    # apply ifftshift to the last 3 dims
    # s = A.shape
    s = tf.shape(A)
    s1 = s[-3]
    s2 = s[-2]
    s3 = s[-1]
    A = tf.concat([A[..., s1//2:, :, :], A[..., :s1//2, :, :]], axis=-3)
    A = tf.concat([A[..., :, s2//2:, :], A[..., :, :s2//2, :]], axis=-2)
    A = tf.concat([A[..., :, :, s3//2:], A[..., :, :, :s3//2]], axis=-1)
    return A

def tf_fftshift2(A):
    # 2D fftshift
    # apply fftshift to the last two dims
    # s = A.shape
    s = tf.shape(A)
    s1 = s[-2]+1
    s2 = s[-1]+1
    A = tf.concat([A[..., s1//2:, :], A[..., :s1//2, :]], axis=-2)
    A = tf.concat([A[..., :, s2//2:], A[..., :, :s2//2]], axis=-1)
    return A

def tf_ifftshift2(A):
    # 2D ifftshift
    # apply ifftshift to the last two dims
    # s = A.shape
    s = tf.shape(A)
    s1 = s[-2]
    s2 = s[-1]
    A = tf.concat([A[..., s1//2:, :], A[..., :s1//2, :]], axis=-2)
    A = tf.concat([A[..., :, s2//2:], A[..., :, :s2//2]], axis=-1)
    return A
