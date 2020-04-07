import tensorflow as tf

def _parse_tfrecord(training):
    def parse_train(tfrecord):
        features = {'label': tf.io.FixedLenFeature([], tf.int64),
                    'image': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        x_train = tf.reshape(tf.io.decode_raw(x['image'], tf.uint8), [112,112,3])
        y_train = x['label']
        
        x_train = _transform_images()(x_train)
        return x_train, y_train
    
    def parse_test(tfrecord):
        features = {'img1': tf.io.FixedLenFeature([], tf.string),
                    'img2': tf.io.FixedLenFeature([], tf.string),
                    'same': tf.io.FixedLenFeature([], tf.int64)}
        x = tf.io.parse_single_example(tfrecord, features)
        img1 = tf.reshape(tf.io.decode_raw(x['img1'], tf.uint8), [112,112,3]) / 255
        img2 = tf.reshape(tf.io.decode_raw(x['img2'], tf.uint8), [112,112,3]) / 255
        return img1, img2, x['same']
    
    if training: return parse_train
    else: return parse_test

def _transform_images():
    def transform_images(x_train):
        #x_train = tf.image.random_crop(x_train, (112,112,3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images

def load_tfrecord_dataset(tfrecord_name, batch_size, training, shuffle=True, buffer_size=1024):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    if training: raw_dataset = raw_dataset.repeat()
    if shuffle: raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(_parse_tfrecord(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset