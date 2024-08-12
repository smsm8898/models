import tensorflow as tf
from model.fm import FMComponent
from model.deep import DeepComponent
from model.processor import Processor


class DeepFM(Model):
    def __init__(
        self,
        config: dict
    ):
        super(DeepFM, self).__init__()
        self.config = config

        # models
        self.processor = Processor(config)

        self.fm = FMComponent(
            field_dict=config.field.field_dict,
            field_index=config.field.field_index,
            embedding_size=config.embedding_size,
        )
        self.deep = DeepComponent(config.num_layers, dropout)
        self.final = tf.keras.layers.Dense(units=1, activation="sigmoid")

    @tf.function
    def call(self, x):
        x = self.processor(x)
        y_fm, new_x = self.fm(x)
        new_x = tf.reshape(new_x, [-1, len(self.config.field.field_index) * self.config.embedding_size])

        # 2) Deep Component
        y_deep = self.deep(new_x)

        # Concatenation
        y_pred = tf.concat([y_fm, y_deep], 1)
        y_pred = self.final(y_pred)
        y_pred = tf.reshape(y_pred, [-1])

        return y_pred

    # To save & load model
    def get_config(self):
        config = super(DeepFM, self).get_config()
        config.update({
            "config": self.config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)