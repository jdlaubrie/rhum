import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

phi_0 = 0.2 * np.pi
phi_f = tf.constant(phi_0, dtype='float32')
two = tf.constant(2.0, dtype='float32')
four = tf.constant(2.0, dtype='float32')


# Tension stress
def stress_circ_th(inputs, stretch2):
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, stretch1, I1) = inputs

    stretch3 = stretch1 * stretch2

    minus = two * (dPsidI1 + dPsidI2 * (I1 - K.pow(stretch3, 2))) * K.pow(stretch3, 2)

    stress = two * (
            dPsidI1 + dPsidI2 * (I1 - K.pow(stretch1, 2)) + two * dPsidI4 * K.cos(phi_f) * K.cos(
        phi_f) + four * dPsidI5 * K.pow(stretch1, 2) * K.cos(phi_f) * K.cos(phi_f)) * K.pow(
        stretch1, 2) - minus

    return stress

# Tension stress
def stress_long_th(inputs, stretch2):
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, stretch1, I1) = inputs

    stretch3 = stretch1 * stretch2

    minus = two * (dPsidI1 + dPsidI2 * (I1 - K.pow(stretch3, 2))) * K.pow(stretch3, 2)

    stress = two * (
            dPsidI1 + dPsidI2 * (I1 - K.pow(stretch2, 2)) + two * dPsidI4 * K.sin(phi_f) * K.sin(
        phi_f) + four * dPsidI5 * K.pow(stretch2, 2) * K.sin(phi_f) * K.sin(phi_f)) * K.pow(
        stretch2, 2) - minus

    return stress
