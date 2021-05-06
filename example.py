import tensorflow as tf
from models import GCN_layer_six

flags.DEFINE_list('layer_structure',[2048, 2048, 1024, 1024, 512, 2048],'output dimensions for each layer.')
flags.DEFINE_list('layer_types',['dropout','dropout'],'Whether each layer uses dropout.')
flags.DEFINE_string('layer','GraphConvolution_multi_dimension','the type of layer.')

model_func = GCN_layer_six

# load dataset
with open('UEC_data.pkl','rb') as ipt:
    dataset_dict = pkl.load(ipt)

save_information = ['hierarchy', 'ingredient', 'occurrence', 'occurrence1m', 'name_df',
                    'feature_matrix',
                    'feature_images','label_images',
                    'gt_classifier', 'gt_bias',
                    'gt_classifier_train', 'gt_classifier_test','gcn_train_mask', 'gcn_test_mask',
                    'evaluation_val_mask','evaluation_test_mask','evaluation_train_mask',
                    'path_list']

adj1, adj2, adj3_0, adj3_1 = dataset_dict['hierarchy'], dataset_dict['ingredient'], dataset_dict['occurrence'], dataset_dict['occurrence1m']
features = dataset_dict['feature_matrix']
shape = dataset_dict['gt_classifier'].shape[0]
y_train, y_test, train_mask, test_mask = dataset_dict['gt_classifier_train'], dataset_dict['gt_classifier_test'], dataset_dict['gcn_train_mask'], dataset_dict['gcn_test_mask']
evaluation_trian_mask,evaluation_test_mask,evaluation_val_mask = dataset_dict['evaluation_train_mask'], dataset_dict['evaluation_test_mask'], dataset_dict['evaluation_val_mask'],
feature_images = dataset_dict['feature_images']
label_images = dataset_dict['label_images']
path_list = dataset_dict['path_list'] 
  
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

