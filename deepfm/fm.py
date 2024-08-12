import tensorflow as tf


class FMComponent(tf.keras.layers.Layer):
    def __init__(self, field_dict, field_index, embedding_size):
        super().__init__()
        self.num_features = len(field_index)  
        self.num_fields = len(field_dict) 
        self.field_index = field_index  
        self.embedding_size = embedding_size

    def build(self, batch_input_size):
        w_init = tf.random_normal_initializer()

        self.w = tf.Variable(initial_value=w_init(shape=[self.num_features]), dtype="float32", name="W")
        self.V = tf.Variable(
            initial_value=w_init(shape=[self.num_fields, self.embedding_size]),
            dtype="float32",
            name="V",
        )

    @tf.function
    def call(self, x):
        # x: [batch, num_features], lookup을 통해 같은 field는 row를 공유
        x_batch = tf.reshape(x, [-1, self.num_features, 1])
        embedding = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)

        # deep input
        new_inputs = tf.math.multiply(x_batch, embedding)

        # linear
        linear_terms = tf.reduce_sum(tf.math.multiply(self.w, x), axis=1, keepdims=False)
        linear_terms = tf.reshape(linear_terms, [-1, 1])

        # interaction
        interactions = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(new_inputs, [1, 2])),
            tf.reduce_sum(tf.square(new_inputs), [1, 2]),
        )
        interactions = tf.reshape(interactions, [-1, 1])

        # concat
        y_fm = tf.concat([linear_terms, interactions], 1)

        return y_fm, new_inputs