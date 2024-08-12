import tensorflow as tf
from typing import List, Dict


class Processor(tf.keras.layers.Layer):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        pass

    def define_field(self):
        pass

    def build_layers(self):
        pass

    @tf.function
    def call(self, x):
        pass