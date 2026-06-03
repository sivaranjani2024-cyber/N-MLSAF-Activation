
import tensorflow as tf
from tensorflow.keras import layers

alpha = 0.5
beta = 0.5

def exp_approximation_using_gamma(x, num_terms=10):
    x_abs = tf.abs(x)
    x_abs = tf.clip_by_value(x_abs, 0.0, 10.0)
    x_abs = tf.cast(x_abs, tf.float64)
    result = tf.ones_like(x_abs, dtype=tf.float64)

    for i in range(1, num_terms):
        gamma_i_plus_1 = tf.exp(tf.math.lgamma(tf.cast(alpha * i + beta, tf.float64)))
        term = tf.pow(x_abs, i) / gamma_i_plus_1
        result += term

    result = tf.where(x < 0, 1 / result, result)
    return result

class MLSAF(layers.Layer):
    def __init__(self, num_terms=10, **kwargs):
        super().__init__(**kwargs)
        self.num_terms = num_terms

    def call(self, inputs):
        neg_inputs = exp_approximation_using_gamma(-inputs, self.num_terms)
        neg_inputs = tf.cast(neg_inputs, tf.float32)
        return tf.pow(inputs, 2) / (1 + neg_inputs)
