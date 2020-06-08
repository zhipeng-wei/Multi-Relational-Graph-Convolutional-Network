import tensorflow as tf
from models import GCN_layer_six

flags.DEFINE_list('layer_structure',[2048, 2048, 1024, 1024, 512, 2048],'output dimensions for each layer.')
flags.DEFINE_list('layer_types',['dropout','dropout'],'Whether each layer uses dropout.')
flags.DEFINE_string('layer','GraphConvolution_multi_dimension','the type of layer.')

model_func = GCN_layer_six

# You should use your own dataset to construct the placeholder.
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(features.shape[0], features.shape[1])),  # sparse_placeholder
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0.5, shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=())
}

# Create model, features is the input w2v.
model = model_func(placeholders, input_dim=features.shape[1], logging=True, layer_structure=FLAGS.layer_structure, layer_types=FLAGS.layer_types, layer=FLAGS.layer)

