import tensorflow as tf

# Gradient function
def my_gradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]
