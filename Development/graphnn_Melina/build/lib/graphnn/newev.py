# =============================================================================
# import tensorflow as tf
# 
# # Tensorflow evidential layer
# class DenseNormalGamma(Layer):
#     def __init__(self, units):
#         super(DenseNormalGamma, self).__init__()
#         self.units = int(units)
#         self.dense = Dense(4 * self.units, activation=None) # Dense layer with 4*units outputs
# 
#     def evidence(self, x):
#         # return tf.exp(x)
#         return tf.nn.softplus(x)
# 
#     def call(self, x):
#         output = self.dense(x)
#         mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
#         v = self.evidence(logv)
#         alpha = self.evidence(logalpha) + 1
#         beta = self.evidence(logbeta)
#         return tf.concat([mu, v, alpha, beta], axis=-1)
# 
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], 4 * self.units)
# 
#     def get_config(self):
#         base_config = super(DenseNormalGamma, self).get_config()
#         base_config['units'] = self.units
#         return base_config
# =============================================================================
    
from torch.nn import Module, Linear, Softplus
from torch import split, cat

# Torch evidential layer
class DenseNormalGamma_torch(Module):
    def __init__(self, units):
        super(DenseNormalGamma_torch, self).__init__()
        self.units = int(units)
        self.dense = Linear(in_features=self.units, out_features=4 * self.units)

    def evidence(self, x):
        return Softplus()(x) # Formula slightly different from the tensorflow one
    
    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = split(output, 1, dim=1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return cat((mu, v, alpha, beta), axis=-1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)
    
    def get_config(self):
        base_config = super(DenseNormalGamma_torch, self).get_config()
        base_config['units'] = self.units
        return base_config