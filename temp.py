import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
import numpy as np

class PolicyNetwork(tf.keras.Model):
    def __init__(self, n_actions, hidden_layers_dims):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_layers_dims = hidden_layers_dims

    def call(self, X):
        hidden = Dense(units=self.hidden_layers_dims[0], activation='relu')(X)
        for i in range(1, len(self.hidden_layers_dims)):
            hidden = Dense(units=self.hidden_layers_dims[i], activation='relu')(hidden)
        pi = Dense(units=self.n_actions, activation='softmax')(hidden)
        return pi

model_a = PolicyNetwork(3, [5, 5])
# model_a.compile(optimizer='adam')
prob = model_a(tf.convert_to_tensor([[1, 2, 3, 4, 5]], dtype=tf.float32))
a = prob.get_weights()
# model_a.summary()
# model_a.save('test.h5')
print(a)