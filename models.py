from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class Model_dense(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'layer_structure', 'layer_types', 'layer'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        
        layer_name = kwargs.get('layer')
        self.layer_name = eval(layer_name)
        
        layer_structure = kwargs.get('layer_structure')
        self.layer_structure = layer_structure
        
        layer_types = kwargs.get('layer_types')
        self.layer_types = layer_types
        
        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        self.summary_op = tf.summary.merge_all()
        # calculate gradient with respect to input, only for dense model
        self.grads = tf.gradients(self.loss, self.inputs)[0]  # does not work on sparse vector

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, unique_name, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "../new_data/tmp/{0}_{1}.ckpt".format(self.name, unique_name))
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN_dense_mse(Model_dense):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_dense_mse, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                   self.placeholders['labels_mask'])
        tf.summary.scalar('train_loss', self.loss)

    def _accuracy(self):
        self.accuracy = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                      self.placeholders['labels_mask'])
        
        tf.summary.scalar('train_accuracy', self.accuracy)

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
        flag = 0
        for layer_type, layer_output in zip(self.layer_types[:-1], self.layer_structure[:-1]):
            
            # certerfy the input_dim for each layer.
            if flag == 0:
                this_input_dim = self.input_dim
            else:
                this_input_dim = self.layer_structure[flag-1]
            flag += 1
            
            if layer_type == 'no-dropout':
                self.layers.append(self.layer_name(input_dim=this_input_dim,
                                            output_dim = layer_output,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))
            elif layer_type == 'dropout':
                self.layers.append(self.layer_name(input_dim=this_input_dim,
                                            output_dim=layer_output,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))
                
        self.layers.append(self.layer_name(input_dim = self.layer_structure[-2],
                                            output_dim = self.layer_structure[-1],
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging))

class GCN_layer_six(Model_dense):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_layer_six, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                   self.placeholders['labels_mask'])
        tf.summary.scalar('train_loss', self.loss)

    def _accuracy(self):
        self.accuracy = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                      self.placeholders['labels_mask'])
        
        tf.summary.scalar('train_accuracy', self.accuracy)

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):        
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=2048,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=2048,
                                            output_dim=2048,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=2048,
                                            output_dim=1024,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=1024,
                                            output_dim=1024,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=1024,
                                            output_dim=512,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=512,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging))
        

    def predict(self):
        return self.outputs

class GCN_weights_adj(Model_dense):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_dense_mse, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                   self.placeholders['labels_mask'])
        tf.summary.scalar('train_loss', self.loss)

    def _accuracy(self):
        self.accuracy = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                      self.placeholders['labels_mask'])
        
        tf.summary.scalar('train_accuracy', self.accuracy)

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
        flag = 0
        for layer_type, layer_output in zip(self.layer_types[:-1], self.layer_structure[:-1]):
            
            # certerfy the input_dim for each layer.
            if flag == 0:
                this_input_dim = self.input_dim
            else:
                this_input_dim = self.layer_structure[flag-1]
            flag += 1
            
            if layer_type == 'no-dropout':
                self.layers.append(GraphConvolution_weights_adjs(input_dim=this_input_dim,
                                            output_dim = layer_output,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))
            elif layer_type == 'dropout':
                self.layers.append(GraphConvolution_weights_adjs(input_dim=this_input_dim,
                                            output_dim=layer_output,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))
                
        self.layers.append(GraphConvolution_weights_adjs(input_dim = self.layer_structure[-2],
                                            output_dim = self.layer_structure[-1],
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging))
        
    
    
    
class relational_gcn(Model_dense):
    '''realational-gcn'''
    def __init__(self, placeholders, input_dim, **kwargs):
        super(relational_gcn, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                   self.placeholders['labels_mask'])
        tf.summary.scalar('train_loss', self.loss)

    def _accuracy(self):
        self.accuracy = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                      self.placeholders['labels_mask'])
        
        tf.summary.scalar('train_accuracy', self.accuracy)

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
        self.layers.append(GraphConvolution_relational_gcn(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_relational_gcn(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_relational_gcn(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_relational_gcn(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_relational_gcn(input_dim=FLAGS.hidden4,
                                            output_dim=FLAGS.hidden5,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_relational_gcn(input_dim=FLAGS.hidden5,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            dropout=True,
                                            logging=self.logging))
        

    def predict(self):
        return self.outputs    