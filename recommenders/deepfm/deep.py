import tensorflow as tf

class DeepComponent(tf.keras.layers.Layer):
    def __init__(self, num_layers, dropout):
        super().__init__()
        if num_layers >= 5:
            print(f"Number of layers:{num_layers} is invalid(under 5)")
            raise ValueError
        
        self.layers = tf.keras.models.Sequential()
        for i in range(num_layers):
            self.layers.add(tf.keras.layers.Dense(2 ** (6 - i), kernel_initializer=tf.keras.initializers.HeNormal()))
            self.layers.add(tf.keras.layers.BatchNormalization())
            self.layers.add(tf.keras.layers.ReLU())
            self.layers.add(tf.keras.layers.Dropout(dropout))
        self.out = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, x):
        x = self.layers(x)
        x = self.out(x)
        return x