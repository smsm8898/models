import tensorflow as tf


class WideAndDeep(tf.keras.Model):
    def __init__(self, config):
        super().__init__(**kwargs)
        self.config = config
        self.user_embedding_model = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=config.unique_users, mask_token=None),
            tf.keras.layers.Embedding(len(config.unique_users) + 1, config.embedding_dim),
        ])
        self.seller_embedding_model = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=config.unique_sellers, mask_token=None),
            tf.keras.layers.Embedding(len(config.unique_sellers) + 1, config.embedding_dim),
        ])
        
        self.feature_lookup = tf.keras.layers.StringLookup(vocabulary=self.config.feature, mask_token=None)
        self.wide = ...
        self.deep = tf.keras.Sequential([...])
        self.final = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        user_embedding = self.user_embedding_model(inputs["user"])
        seller_embedding = self.gripper_embedding_model(inputs["seller"])
        deep_input = tf.keras.layers.concatenate([
            user_embedding,
            seller_embedding,
            ...
        ])

        wide_input = tf.keras.layers.concatenate([
            tf.multiply(user_embedding, gripper_embedding),
            tf.add(user_embedding, gripper_embedding),
        ])

        output = self.final(
            tf.keras.layers.concatenate([
                self.deep(deep_input),
                self.wide(wide_input),
            ])
        )

        return output
