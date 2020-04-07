from absl import app, flags, logging
from absl.flags import FLAGS
import os, sys, yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_virtual_device_configuration(gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5300)])
tf.autograph.set_verbosity(0)

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from models import ArcFaceModel
from utils import get_ckpt_inf
from losses import LossFunc
from dataset import load_tfrecord_dataset

flags.DEFINE_string('cfg_path', './configs/.yaml', 'config file path')

def main(_):
    cfg = None
    with open(FLAGS.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    model = ArcFaceModel(logit_type=cfg['logit_type'], size=cfg['input_size'],
                        num_classes=cfg['num_classes'], embd_shape=cfg['embd_shape'],
                        training=True, margin=cfg['margin'],
                        w_l2factor=cfg['w_l2factor'], logist_scale=cfg['logist_scale'])

    dataset_len = cfg['num_samples']
    steps_per_epoch = dataset_len // cfg['batch_size']
    train_dataset = load_tfrecord_dataset(cfg['train_dataset'], cfg['batch_size'], training=True)
    train_dataset = iter(train_dataset)
    
    learning_rate = tf.constant(cfg['base_lr'])
    print('learning rate = {:.9f}'.format(learning_rate))
    #optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    loss_func = LossFunc(cfg['loss_func'], cfg['logist_scale'])
    
    ckpt_dir = cfg['ckpt_dir']
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is not None:
        print('[*] load ckpt from {}'.format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print('[*] train from scratch')
        epochs, steps = 1, 1
    summary_writer = tf.summary.create_file_writer(cfg['log_dir'])
    additional_loss_weight = cfg['additional_loss_weight']
    
    
    for layer in model.layers:
        layer.trainable = True
    model.summary(line_length=80)

    total_loss_sum = tf.Variable(0.)
    reg_loss_sum = tf.Variable(0.)
    pred_loss_sum = tf.Variable(0.)
    additional_loss_sum = tf.Variable(0.)
    acc_sum = tf.Variable(0.)
    
    while epochs <= cfg['epochs']:
        total_loss_sum = 0.
        reg_loss_sum = 0.
        pred_loss_sum = 0.
        additional_loss_sum = 0.
        acc_sum = 0.
        
        for batch_num in range(steps_per_epoch):
            imgs, labels = next(train_dataset)
            with tf.GradientTape() as tape:
                embededs, logits = model(imgs)
                
                acc = tf.reduce_sum(tf.cast(tf.math.equal(labels, tf.math.argmax(logits, axis=1)), tf.float32))
                acc_sum = tf.math.add(acc_sum, acc)
                
                reg_loss = tf.reduce_sum(model.losses)
                pred_loss, additional_loss = loss_func(labels, logits, embededs)
                total_loss = tf.math.add(tf.math.add(pred_loss, reg_loss), tf.multiply(additional_loss, additional_loss_weight))
                
                reg_loss_sum = tf.math.add(reg_loss_sum, reg_loss)
                pred_loss_sum = tf.math.add(pred_loss_sum, pred_loss)
                additional_loss_sum = tf.math.add(additional_loss_sum, additional_loss)
                total_loss_sum = tf.math.add(total_loss_sum, total_loss)
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            steps += 1
            
            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                pre_ckpt_path = ckpt_path
                ckpt_path = os.path.join(ckpt_dir, 'e_{}_b_{}.ckpt'.format(epochs, steps%steps_per_epoch))
                model.save_weights(ckpt_path)
                if pre_ckpt_path != None:
                    os.system('rm -f {}*'.format(pre_ckpt_path))
        
        reg_loss_sum = tf.math.divide(reg_loss_sum, steps_per_epoch)
        pred_loss_sum = tf.math.divide(pred_loss_sum, steps_per_epoch)
        additional_loss_sum = tf.math.divide(additional_loss_sum, steps_per_epoch)
        total_loss_sum = tf.math.divide(total_loss_sum, steps_per_epoch)
        acc_sum = tf.math.multiply(tf.math.divide(acc_sum, dataset_len), 100)
        print('Epoch {}/{}, loss={:.4f}, acc={:.2f}'.format(epochs, cfg['epochs'], total_loss_sum.numpy(), acc_sum.numpy()), flush=True, end='')
        print(', reg={:.4f}, pred={:.4f}, add={:.4f}'.format(reg_loss_sum.numpy(), pred_loss_sum.numpy(), additional_loss_sum.numpy()), flush=True)
        
        with summary_writer.as_default():
            tf.summary.scalar('loss/reg loss', reg_loss_sum, step=steps)
            tf.summary.scalar('loss/pred loss', pred_loss_sum, step=steps)
            tf.summary.scalar('loss/additional_loss', additional_loss_sum, step=steps)
            tf.summary.scalar('loss/total loss', total_loss_sum, step=steps)
            tf.summary.scalar('acc', acc_sum, step=steps)
        
        epochs += 1
    
if __name__ == '__main__':
    app.run(main)
