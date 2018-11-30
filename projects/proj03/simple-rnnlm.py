import tensorflow as tf;
print(tf.__version__)
import numpy as np
import time
from tqdm import tqdm, trange

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())


word2idx = {'<stop>': 0}
idx2word = ['<stop>']
index = 1
max_seq_len = 0

def load_voc(filename):
    print("loading %s"%filename)
    global index, max_seq_len, word2idx, idx2word
    sentences = []
    num_tokens = 0
    with open(filename, 'r') as f:
        for line in f:
            stn = []
            for w in line.rstrip().split(' '):
                if w not in word2idx: 
                    word2idx[w] = index
                    index += 1
                    idx2word.append(w)
                stn.append(word2idx[w])
            num_tokens += len(stn)
            max_seq_len = max(max_seq_len, len(stn))
            sentences.append( np.array(stn, dtype=int) )
    print("#sentences {}, #tokens {}".format(len(sentences), num_tokens))
    return sentences      


trn_sentences = load_voc('trn-wiki.txt')
dev_sentences = load_voc('dev-wiki.txt')
tst_sentences = load_voc('tst-wiki.txt')
print('vocb size %d'%index)
print('max stn len %d'%max_seq_len)


vocabulary_size = index
input_size = 32
hidden_size = 32
batch_size = 1
seq_len = max_seq_len


tf.reset_default_graph()

wordids_placeholder = tf.placeholder(tf.int64, [batch_size, seq_len])
word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, input_size])
embedded_words = tf.nn.embedding_lookup(word_embeddings, wordids_placeholder)

lstm = tf.contrib.rnn.LSTMCell(hidden_size)
output2wordid = tf.layers.Dense(vocabulary_size, activation=tf.nn.softmax)

initial_c_state = tf.get_variable("initial_c_hidden_state", [batch_size, hidden_size])
initial_m_state = tf.get_variable("initial_m_hidden_state", [batch_size, hidden_size])

embedded_word_series = tf.unstack(embedded_words, axis=1)

state = (initial_c_state, initial_m_state)
probabilities = []
losses = []
acc = []
for i in range(len(embedded_word_series)-1):
    embedded_word = embedded_word_series[i]
    output, state = lstm(embedded_word, state)
    
    correct_pred = wordids_placeholder[:, i+1]
    
    prob = output2wordid(output)
    acc.append( tf.cast(tf.equal(tf.argmax(prob, axis=1), correct_pred), tf.float32) )
    losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob, labels = correct_pred))
    
    
total_loss = tf.reduce_mean(losses)
tf.summary.scalar('loss', total_loss)

accuracy = tf.reduce_mean(acc)
tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('train'):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = opt.compute_gradients(total_loss)
    capped_grads_and_vars = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in grads_and_vars]
    train_step = opt.apply_gradients(capped_grads_and_vars)

merged_summary = tf.summary.merge_all()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

epoches = 10

with tf.Session(config=config) as sess:
    now = time.strftime("%c")
    train_writer = tf.summary.FileWriter('./logs/'+now, sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch_idx in range(epoches):
        num_batch = len(trn_sentences) / batch_size
        for batch_id, stn in enumerate(tqdm(trn_sentences, desc='training epoch %d'%epoch_idx)):
            stn_len = stn.shape[0]
            i = 0
            input_stn = np.expand_dims(np.pad( stn[i:i+seq_len], (0,  i+seq_len - stn_len), 'edge'), axis=0)
            summary_, _ = sess.run(
                [merged_summary, train_step], 
                feed_dict = {
                    learning_rate : 0.1,
                    wordids_placeholder: input_stn
                })              
            train_writer.add_summary(summary_, num_batch*epoch_idx+batch_id)
        
    train_writer.close()
