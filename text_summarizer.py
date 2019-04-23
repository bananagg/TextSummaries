import data_util

import tensorflow as tf

import os
import sys

class Largeconfig(object):
    learning_rate = 1.0
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 4096
    batch_size = 64
    size = 256
    num_layers = 4
    vocab_size = 50000

class Largeconfig(object):
    learning_rate = 0.5
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 2048
    batch_size = 64
    size = 64
    num_layers = 2
    vocab_size = 10000

config = Largeconfig()
train_dir = os.path.join(data_util.root_path, 'train')
data_path = data_util.root_path

tf.app.flags.DEFINE_float('learning_rate', config.learning_rate, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', config.learning_rate_decay_factor, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', config.max_gradient_norm, 'clip max gradient to this norm')
tf.app.flags.DEFINE_integer('num_samples', config.num_samples, 'Number of Samples for Sampled softmax.')
tf.app.flags.DEFINE_integer('batch_size', config.batch_size, 'Batch size to user during training.')
tf.app.flags.DEFINE_integer('size', config.size, 'size of each model layer')
tf.app.flags.DEFINE_integer('num_layers', config.num_layers, 'Number of layers in the model')
tf.app.flags.DEFINE_integer('vocab_size', config.vocab_size, 'vocabulary size.')

tf.app.flags.DEFINE_string('data_dir', data_path, 'Data directory')
tf.app.flags.DEFINE_string('train_dir', train_dir, 'Training directory')

tf.app.flags.DEFINE_string('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_string('steps_per_checkpoint', 1000, 'How many training steps to do per checkpoint.')
tf.app.flags.DEFINE_boolean('decode', False, 'Set to Tre for interactive decoding')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Train using flp16 instead of fp32')

tf.app.flags.DEFINE_string('headline_scope_name', 'headline_var_scope', 'Variable scope of Headline textsum model')

FLAGS = tf.app.flags.FLAGS

buckets = [(120, 30), (200,35), (300, 40), (400, 40), (500, 40)]

def create_model(session, forward_only):
    pass

def read_data(source_path, target_path, max_size=None):
    """
    读取原文件和目标文件 加入buckets
    :param source_path:
    :param target_path:
    :param max_size: 读取的最大行数，其他的将会被忽略，如果为0或None，将会全部读取，没有限制
    :return:
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        with tf.gfile.GFile(target_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 10000 == 0:
                    print('  reading data line %d' % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_util.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
    print(data_set)
    return data_set
    # pass

def train():
    print("train function")
    # 准备数据
    print(FLAGS.data_dir)
    print(FLAGS.vocab_size)
    # src_train, dest_train, src_dev, dest_dev = data_util.prepare_headline_data(FLAGS.data_dir, FLAGS.vocab_size)
    src_train = os.path.join(FLAGS.data_dir,'train\\content_train_id')
    dest_train = os.path.join(FLAGS.data_dir,'train\\title_train_id')
    src_dev = os.path.join(FLAGS.data_dir,'dev\\content_dev_id')
    dest_dev = os.path.join(FLAGS.data_dir,'dev\\title_dev_id')
    print(src_train)
    print(dest_train)
    print(src_dev)
    print(dest_dev)

    # dev_config = tf.ConfigProto(devic)
    with tf.Session() as sess:
        print('start......')
        #创建模型
        model = create_model(sess, False)
        dev_set = read_data(src_dev, dest_dev)
        print('finish')

    pass

def main():
    train()

if __name__ == '__main__':
    tf.app.run(main())


