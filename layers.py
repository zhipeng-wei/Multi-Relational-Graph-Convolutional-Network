from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        # res = tf.matmul(x, y)
        res = tf.keras.backend.dot(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolution_relational_gcn(Layer):
    '''relational-gcn, https://github.com/tkipf/relational-gcn/blob/master/rgcn/layers/graph.py'''
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, num_bases = 0,**kwargs):
        super(GraphConvolution_relational_gcn, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_bases = num_bases
        self.input_dim = input_dim
        self.output_dim = output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        with tf.variable_scope(self.name + '_vars'):
            if num_bases>0:

                self.vars['weights'] = tf.concat([glorot([input_dim, output_dim], name='weights_concat_'+str(i)) \
                                                   for i in range(self.num_bases)], axis=0)
                self.vars['weights_comp'] = glorot([len(self.support), self.num_bases])
                
            else:
                self.vars['weights'] = tf.concat([glorot([input_dim, output_dim], name='weights_concat_'+str(i)) \
                                                   for i in range(len(self.support))], axis=0)
                
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # if featureless = True: sum(adj_i * weights_i)
        # if featureless = False: sum(adj_i * x * weights_i)
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                supports.append(dot(self.support[i], x, sparse=True))
            else:
                supports.append(self.support[i])
        supports = tf.concat(supports, axis=1)
        if self.num_bases>0:
            self.vars['weights'] = tf.reshape(self.vars['weights'], (self.num_bases, self.input_dim, self.output_dim))
            self.vars['weights'] = tf.transpose(self.vars['weights'], perm=[1,0,2])
            V = dot(self.vars['weights_comp'], self.vars['weights'], sparse=False)
            V = tf.reshape(V, (len(self.support)*self.input_dim, self.output_dim))
            output = dot(supports, V, sparse=False)
        else:
            output = dot(supports, self.vars['weights'])
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    
    
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # if featureless = True: sum(adj_i * weights_i)
        # if featureless = False: sum(adj_i * x * weights_i)
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        
        # change this command, Computes Python style division of output by the length of adjs.
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_Norm(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_Norm, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = uniform([input_dim, output_dim], scale=0.001,
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_weights_adjs(Layer):
    """Graph convolution layer with weighted adjs."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_weights_adjs, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
            
            self.vars['weights_adjs'] = glorot([1, len(self.support)], name='weights_adjs')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # if featureless = True: sum(adj_i * weights_i)
        # if featureless = False: sum(adj_i * x * weights_i)
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        
        # different adjs are assigned different weights.
        conbines_output = tf.transpose(tf.stack(supports, axis=0), perm=[1,0,2])
        
        V = dot(tf.nn.softmax(self.vars['weights_adjs']), conbines_output, sparse=False)
        output = tf.squeeze(V)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_concat(Layer):
    """Graph convolution layer. Concat different graph features, and mapped to output dimension."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_concat, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                
            self.vars['weights'] = tf.concat([glorot([output_dim, output_dim], name='weights_concat_'+str(i)) \
                                                   for i in range(len(self.support))], axis=0)
    

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # if featureless = True: sum(adj_i * weights_i)
        # if featureless = False: sum(adj_i * x * weights_i)
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        
        supports = tf.concat(supports, axis=1)
        # concat features.        
        output = dot(supports, self.vars['weights'])
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

    
class GraphConvolution_multi_dimension(Layer):
    """
    Graph convolution layer.
    Conference: Multi-dimensional Graph Convolutional Networks
    """
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_multi_dimension, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        
        # self.input_n = 519
        self.alpha = 0.5
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
            
            #self.vars['weights_concat'] = tf.concat([glorot([input_n, input_n *3], name='weights_concat_'+str(i)) \
            #                                       for i in range(len(self.support))], axis=0)
            
            if len(self.support) == 1:
                pass
            else:
                self.vars['weights_concat'] = tf.concat([glorot([output_dim, output_dim], name='weights_concat_'+str(i)) \
                                                   for i in range(len(self.support))], axis=0)
                self.vars['weights_bilinear'] = glorot([input_dim,  input_dim], name='weights_bilinear')
            
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # if featureless = True: sum(adj_i * weights_i)
        # if featureless = False: sum(adj_i * x * weights_i)
        supports = list()
        per_sups = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
            per_sups.append(pre_sup)
        
        if len(self.support) == 1:
            output = support
            if self.bias:
                output += self.vars['bias']
            return self.act(output)
        
        # bilinear
        weights = list()
        transpose_weights = list()
        for i in range(len(self.support)):
            weights.append(self.vars['weights_'+str(i)])
            transpose_weights.append(tf.matmul(tf.transpose(self.vars['weights_'+str(i)]), self.vars['weights_bilinear']))
        
        coefs_matrixs = tf.matmul(tf.concat(transpose_weights,axis=0), tf.concat(weights,axis=1))
        
        # generate coefs
        coefs = []
        for i in range(len(self.support)):
            coef = []
            for j in range(len(self.support)):
                this_slice = tf.slice(coefs_matrixs, [i*self.output_dim,j*self.output_dim], [self.output_dim, self.output_dim])
                this_coef = tf.linalg.trace(this_slice)
                coef.append(this_coef)
            coefs.append(coef)
        
        coefs_tensor = tf.convert_to_tensor(coefs, dtype=tf.float32)
        
        coefs_softmax = tf.nn.softmax(coefs_tensor, axis=1)
        
        # generate coef_features (3output_dims*3output_dims)
        coef_features = []
        for i in range(len(self.support)):
            this_coef_features = []
            for j in range(len(self.support)):
                this_slice = tf.slice(coefs_softmax, [i,j], [1, 1])
                this_coef_features.append(tf.multiply(this_slice, tf.eye(self.output_dim)))
            coef_features.append(tf.concat(this_coef_features, axis=0))
        coef_features = tf.concat(coef_features, axis=1)
        
        before_output = tf.math.add(tf.multiply(1 - self.alpha, tf.concat(supports, axis=1))             ,tf.multiply(self.alpha,tf.matmul(tf.concat(per_sups, axis=1), coef_features))) # (1-alpha)*nodes*3output_dims+alpha*nodes*3output_dims
        
        
        # different adjs are assigned different weights.
        output = tf.matmul(before_output, self.vars['weights_concat'], ) # nodes*3output_dims matmul 3output_dims*output_dims
        # output = tf.math.divide(tf.add_n(supports), len(self.support))
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

                                    
class GraphConvolution_nesting(Layer):
    """
    Graph convolution layer. 
    Rethinking Knowledge Graph Propagation for Zero-Shot Learning.
    """
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_nesting, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        gra = (output_dim-input_dim)/len(self.support)
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim + int(gra*i), input_dim + int(gra*(i+1))],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # if featureless = True: sum(adj_i * weights_i)
        # if featureless = False: sum(adj_i * x * weights_i)
        
        for i in range(len(self.support)):
            pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            
            su = dot(self.support[i], pre_sup, sparse=True)
            x = self.act(su)
            
        
        
        # concat features.
        output = x

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)