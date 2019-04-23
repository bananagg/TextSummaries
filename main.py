import numpy as np
import time
import os

from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# batch_size=32

vacab_path = ''

def get_vocab(vacab_path):
    vocab_list = []
    with open(vacab_path, 'r', encoding='utf-8') as f:
        for item in f.readlines():
            vocab_list.append(item.strip())
    int_to_vocab = {idx: word for idx, word in enumerate(vocab_list)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

source_int_to_letter, source_letter_to_int = get_vocab(vacab_path)
target_int_to_letter, target_letter_to_int = source_int_to_letter, source_letter_to_int

print(source_int_to_letter)

# 模型构建
def get_inputs():
    """
    输入层
    模型输入tensor
    :return:
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.int32, name='learning_rate')

    target_sequence_length = tf.placeholder(tf.int32, [None,], name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    souce_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, souce_sequence_length

def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_sie, encodeing_embedding_size):
    '''
    构建Encoder层
    :param input_data: 输入tensor
    :param rnn_size: rnn隐层节点数量
    :param num_layers: 堆叠的rnn cell数量
    :param source_sequence_length: 源数据的序列长度
    :param source_vocab_sie: 源数据的词典大小
    :param encodeing_embedding_size:  embedding的大小
    :return:
    '''

    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_sie, encodeing_embedding_size)

    # Rnn cell
    def get_lstm_cell(rnn_size):
        return tf.contrib.layers.embed_sequence(rnn_size, initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length, dtype=tf.float32)
    return encoder_output, encoder_state
    # pass

def process_decoder_input(data, vocab_to_int, batch_size):
    """
    补充<GO>,并移除最后一个字符
    这里是为了构建Decoder训练时的输入数据， 使用target而不是预测出的数据，提高经度
    :param data:
    :param vocab_to_int:
    :param batch_size:
    :return:
    """
    # cut 掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [-1,1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input

def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size, target_sequence_length,
                   max_target_sequence_length,encoder_state, decoder_input, encoder_outputs, source_sequence_length):
    """
    构建Decoder层
    :param target_letter_to_int: target数据的映射表
    :param decoding_embedding_size: embed向量大小
    :param num_layers: 堆叠的RNN单元数量
    :param rnn_size: RNN单元的隐层结点数量
    :param target_sequence_length: target数据序列长度
    :param max_target_sequence_length: target数据序列最大长度
    :param encoder_state: encoder端编码的状态向量
    :param decoder_input: decoder端输入
    :param encoder_outputs: 添加一个注意力机制
    :param source_sequence_length: 源数据长度
    :return:
    """

    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embded_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 2. 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer = tf.random_uniform_initializer(-0.1,0.1, seed=2))
        return decoder_cell

    # 2.1 添加注意力机制的RNN单元
    def get_decoder_cell_attention(rnn_size):
        attention_state = encoder_state
        # create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuogAttention(rnn_size, attention_state, memory_sequence_length=source_sequence_length)
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_lyaer_size = rnn_size)
        return decoder_cell
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell_attention(rnn_size) for _ in range(num_layers)])

    # output full layer
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4 Training decoder
    with tf.variable_scope('decode'):
        # 得到help对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embded_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper,
                                                           initial_state=cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                           output_layer=output_layer)
        training_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                         maximum_iterations=max_target_sequence_length)
    # 5 predicting decoder
    # 与training共享参数
    with tf.variable_scope('decode', reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.float32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     target_letter_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper,
                                                             inital_state=cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                             output_layer=output_layer)
        predicting_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)
    return training_decoder_output, predicting_decoder_output

def seq2seq_model(input_data, targets,lr, target_sequence_length, max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size, encoder_embnedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
    """

    :param input_data:
    :param target:
    :param target_sequence_length:
    :param max_target_sequence_length:
    :param source_sequence_length:
    :param source_vocab_size:
    :param target_vocab_size:
    :param encoder_embnedding_size:
    :param decoder_embedding_size:
    :param rnn_size:
    :param num_layers:
    :return:
    """
    # 获取encoder的状态输出
    encoder_outputs, encoder_state = get_encoder_layer(input_data,rnn_size,num_layers,source_sequence_length,
                                                       source_vocab_size, encoder_embnedding_size)

    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoder_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input,
                                                                        encoder_outputs,
                                                                        source_sequence_length)
    return training_decoder_output, predicting_decoder_output

epochs = 50
batch_size = 32
rnn_size = 50
num_layers = 2
encoding_embedding_size = 128
decoding_embedding_size = 128

learning_rate = 0.0005

# Batcher
train_graph = tf.Graph()
with train_graph.as_default():
    # 获得模型输入
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       learning_rate,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_letter_to_int),
                                                                       len(target_letter_to_int),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)
    training_logits = tf.identity(training_decoder_output.rnn_out, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope('optimization'):
        # loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        #optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
def pad_sentence_batch(sentence_batch, pad_int):
    """
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    :param sentence_batch:
    :param pad_int: <PAD>对应的索引号
    :return:
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_barches(file_list, tokenize_path, barch_size, pad_int):
    """
    定义生成器，用来获取tokenize下的所有content
    :param file_list:
    :param tokenizer_path:
    :param barch_size:
    :param pad_int:
    :return:
    """

    for item in file_list:
        source_path = os.path.join(tokenize_path, '')
        target_path = os.path.join(tokenize_path, '')
        with open(source_path, 'r', encoding='utf-8') as sf:
            sources = [[int(word) for word in sentence.strip().split(' ')] for sentence in sf.readlines()]
        with open(target_path, 'r', encoding='utf-8') as tf:
            targets = [[int(word) for word in sentence.strip().split(' ')] for sentence in tf.readlines()]

        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * barch_size
            sources_batch = sources[start_i:start_i+barch_size]
            targets_batch = targets[start_i:start_i+barch_size]

            # 补全序列
            pad_source_batch = np.array(pad_sentence_batch(sources_batch, pad_int))
            pad_target_batch = np.array(pad_sentence_batch(targets_batch, pad_int))

            targets_length = []
            for target in targets_batch:
                targets_length.append(len(target))
            source_length = []
            for source in sources_batch:
                source_length.append(len(source))
            yield pad_target_batch, pad_source_batch, targets_length, source_length
    pass

data_path = ''
file_list = os.listdir(data_path)
tokenize_path = ''
batch_size = 32
pad_int = source_letter_to_int['<PAD>']

(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
    next(get_barches(file_list, tokenize_path, batch_size, pad_int))

display_step = 50
checkpoint_path = ''
#查看是有有checkpoint这个文件夹，无则新建一个
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

checkpoint = checkpoint_path+'/trained_model.ckpt'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs+1):
        for batch_i, (targets_batch, source_batch, targets_lengths, source_lengths) in enumerate(get_barches(file_list, tokenize_path, batch_size, pad_int)):
            _, loss = sess.run([train_op, cost], {input_data:source_batch, targets: targets_batch,
                                                  lr:learning_rate,
                                                  target_sequence_length: targets_lengths,
                                                  source_sequence_length: source_lengths})
            if batch_i % display_step == 0:
                # 计算validation loss
                validation_loss = sess.run([cost], {input_data:source_batch, targets: targets_batch,
                                                  lr:learning_rate,
                                                  target_sequence_length: targets_lengths,
                                                  source_sequence_length: source_lengths})
                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i, epochs, batch_i, '未知', loss, validation_loss[0]))
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print("Model Trained and Saved")

# def source_to_seq(text):
#     """
#     对源数据进行转换
#     :param text:
#     :return:
#     """
#     sequence_length = 120
#     return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in
#             text.split(' ')] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))
#
# #输入一个单词
# input_word = '''
#  NUMBER   反对 宪法 基本 原则 危害 国家 安全 政权 稳定 统一 的 煽动 民族 仇恨 民族 歧视 的   NUMBER   宣扬 邪教 和 封建迷信 的 散布 谣言 破坏 社会 稳定 的 侮辱 诽谤 他人 侵害 他人 合法权益 的   NUMBER   散布 淫秽 色情 赌博 暴力 凶杀 恐怖 或者 教唆 犯罪 的
#  '''
#
#  text = source_to_seq(input_word)
#  checkpoint = checkpoint_path + 'trainded_model.ckpt'
#