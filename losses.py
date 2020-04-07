import tensorflow as tf

def LossFunc(loss_type , logit_scale=None):
    def softmax_loss(y_true, y_pred, embeded):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        indx = tf.range(tf.size(y_true))
        indx = tf.stack([indx, y_true], axis=1)

        log_sof = tf.nn.log_softmax(y_pred, axis=1)
        picked_log_sof = tf.gather_nd(log_sof, indx)
        return tf.math.negative(tf.reduce_mean(picked_log_sof))
    
    def compact_softmax(y_true, y_pred, embeded):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        indx = tf.range(tf.size(y_true))
        indx = tf.stack([indx, y_true], axis=1)

        log_sof = tf.nn.log_softmax(y_pred, axis=1)
        picked_log_sof = tf.gather_nd(log_sof, indx)
        soft_loss = tf.math.negative(tf.reduce_mean(picked_log_sof))

        picked = tf.gather_nd(y_pred, indx)
        rescaled = tf.divide(picked, logit_scale)
        angles = tf.math.acos(tf.clip_by_value(rescaled, -1., 1.))
        additional_loss = tf.reduce_mean(angles)
        return soft_loss, additional_loss

    if loss_type == 'softmax': return softmax_loss
    elif loss_type == 'compact_softmax': return compact_softmax
    else: raise ValueError('No such loss_type')
