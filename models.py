import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.regularizers import l2

def EmbedLayer(embd_shape, w_l2factor=5e-4, training=True):
    def linear_embed(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Flatten()(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=training)(x, training=training)
        x = Dropout(rate=0.5)(x, training=training)
        x = Dense(embd_shape, kernel_regularizer=l2(w_l2factor), use_bias=True, bias_regularizer=l2(w_l2factor))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-4, trainable=training)(x, training=training)
        return Model(inputs, x, name='linear_output')(x_in)
    return linear_embed

def LogitLayer(logit_type, num_classes, w_l2factor=5e-4, margin=0.5, logist_scale=64):
    def linear_logit(x_in, logist_scale, margin):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=l2(w_l2factor), use_bias=True, bias_regularizer=l2(w_l2factor))(x)
        return Model(inputs, x, name='linear_logit')(x_in)
    
    def angle_logit(x_in, logist_scale, margin):
        x = inputs = Input(x_in.shape[1:])
        x = tf.math.l2_normalize(x, axis=1)
        x = Dense(num_classes, use_bias=False, kernel_constraint=UnitNorm(axis=0))(x)
        x = tf.math.scalar_mul(logist_scale, x)
        return Model(inputs, x, name='angle_logit')(x_in)
    
    if logit_type == 'linear': return linear_logit
    elif logit_type == 'angle': return angle_logit
    else: raise ValueError('No such logit_type')

                   
def ArcFaceModel(logit_type, size=112, num_classes=None, embd_shape=512, training=False, margin=0.5, w_l2factor=5e-4, logist_scale=64):
    x = inputs = Input((size, size, 3), name='input_images')
    x = ResNet50(input_shape=inputs.shape[1:], include_top=False, weights='imagenet')(x)
    embds = EmbedLayer(embd_shape, w_l2factor=w_l2factor, training=training)(x)
    
    logits = None
    
    if training:
        assert num_classes is not None
        logits = LogitLayer(logit_type, num_classes, w_l2factor)(embds, logist_scale, margin)
        return Model(inputs, (embds, logits), name='arc_face_model')
    else: return Model(inputs, embds, name='arc_face_model')
