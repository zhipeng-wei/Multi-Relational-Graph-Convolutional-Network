import tensorflow as tf
import numpy as np

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_sigmoid_cross_entropy(preds, labels, mask):

    """Sigmoid cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def sample_mask_sigmoid(h, w):
    """Create mask."""
    idx = [i for i in range(353)]
    mask = np.zeros((h, w))
    matrix_one = np.ones((h, w))
    mask[idx, :] = matrix_one[idx,:]
    return np.array(mask, dtype=np.bool)

def mask_mse_loss(preds, labels, mask):
    """Sigmoid cross-entropy loss with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    labels *= mask 
    preds  *= mask

    # loss = tf.losses.mean_squared_error(labels=labels, predictions=preds, weights=mask)
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=preds, weights=1.0)
    loss = tf.nn.l2_loss(tf.subtract(labels, preds))

    # return tf.reduce_mean(loss)
    return loss

def mask_mae_loss(preds, labels, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    labels *= mask 
    preds  *= mask
    return tf.reduce_mean(tf.abs(tf.subtract(labels, preds)))