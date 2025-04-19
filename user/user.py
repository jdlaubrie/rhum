import numpy as np
from keras.src.initializers.initializer import Initializer
from keras.src.regularizers import L1, L2, L1L2


class Variables:
    def __init__(self, epochs: int, batch_size: int, regulazier: L1 | L2 | L1L2, initializer_def: str,
                 initializer_exp: Initializer):
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularizer = regulazier
        self.initializer_def = initializer_def
        self.initializer_exp = initializer_exp


class TrainData:
    def __init__(self):
        np.random.seed(0)
        self.input_train = np.linspace(1.0, 2.0, 25).astype(np.float32)
        self.output_train = (
                2.5 * (self.input_train ** 2 - 1.0 / self.input_train) + 0.1 * np.random.randn(
            *self.input_train.shape)).astype(
            np.float32)
        self.sample_weights = np.array([1.0] * self.input_train.shape[0], dtype=np.float32)
