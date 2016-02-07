# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations, initializations
from ..layers.core import Layer, MaskedLayer

class PairMerge(Layer):
    """
    merge a pair of inputs
    """
    def __init__(self, input1, input2, mode='sum', concat_axis=-1, dot_axes=-1):
        if mode not in {'sum', 'mul', 'concat', 'ave', 'join', 'cos', 'dot',
                        'l2', 'l1', 'hamming'}:
            raise Exception('Invalid merge mode: ' + str(mode))
        
        self.mode = mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        self.layers = [input1, input2]
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.trainable_weights:
                    self.trainable_weights.append(p)
                    self.constraints.append(c)
        super(PairMerge, self).__init__()
        
    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        input_shape = input_shapes[0]
        if self.mode in ['sum', 'mul', 'ave']:
            return input_shape
        elif self.mode in ['l2', 'l1']:
            return (input_shape[0],1)
        
    def get_params(self):
        return self.trainable_weights, self.regularizers, self.constraints, self.updates
    
    def get_output(self, train=False):
        s1 = self.layers[0].get_output(train)
        s2 = self.layers[1].get_output(train)
        if self.mode == 'sum':
            return s1 + s2
        elif self.mode == 'ave':
            return (s1+s2)/2
        elif self.mode == 'mul':
            return s1 * s2
        elif self.mode == 'l2':
            return K.sum(K.square(s1-s2), axis=-1, keepdims=True)
        elif self.mode == 'l1':
            return K.sum(K.abs(s1-s2))
        else:
            raise Exception('Unknown merge mode.')

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()
    
    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].trainable_weights)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layers': [l.get_config() for l in self.layers],
                  'mode': self.mode,
                  'concat_axis': self.concat_axis,
                  'dot_axes': self.dot_axes}
        base_config = super(PairMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
