import tensorflow as tf;
print(tf.__version__)
import numpy as np
import time
from tqdm import tqdm, tnrange, tqdm_notebook, trange

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
            sentences.append( np.array(stn, dtype=int) )
    print("#sentences {}, #tokens {}".format(len(sentences), num_tokens))
    return sentences 


trn_sentences = load_voc('trn-wiki.txt')
dev_sentences = load_voc('dev-wiki.txt')
tst_sentences = load_voc('tst-wiki.txt')
print('vocb size %d'%index)

vocabulary_size = index
input_size = 32
hidden_size = 32
batch_size = 24

tf.reset_default_graph()

wordids_placeholder = tf.placeholder(tf.int64, [batch_size, None])
word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, input_size], trainable=True)
embedded_words = tf.nn.embedding_lookup(word_embeddings, wordids_placeholder)

lstm = tf.contrib.cudnn_rnn.CudnnLSTM(2, hidden_size)
output2wordid = tf.layers.Dense(vocabulary_size)

seq_weight = tf.cast(tf.sign(wordids_placeholder[:,1:]), tf.float32)
seq_length = tf.cast(tf.reduce_sum(seq_weight, axis=1), tf.int32)
total_seq_length = tf.cast(tf.reduce_sum(seq_length, axis=0), tf.float32)

inputs = embedded_words[:, :-1, ]
outputs, state = lstm(inputs)

labels = wordids_placeholder[:,1:]
logits = tf.map_fn(lambda x: output2wordid(x), outputs)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits) * seq_weight
    
total_loss = tf.reduce_sum(losses) / total_seq_length
tf.summary.scalar('loss', total_loss)

perplexity = tf.exp(total_loss)
tf.summary.scalar('perplexity', perplexity)

with tf.name_scope('train'):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate)
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
        num_batch = len(trn_sentences) // batch_size - 1
        loss_arr = []
        weight_arr = []
        for batch_id in trange(num_batch, desc='training epoch %d'%epoch_idx):
            len_arr = [sent.shape[0] for sent in trn_sentences[batch_id*batch_size:(batch_id+1)*batch_size]]
            max_len = max(len_arr)
            total_len = sum(len_arr)
            
            padded = [np.pad( sent, (0,  max_len - sent.shape[0]), 'edge') for sent in trn_sentences[batch_id*batch_size:(batch_id+1)*batch_size] ]
            batch_stn = np.stack(padded, axis=0)
            summary_, _, total_loss_ = sess.run(
                [merged_summary, train_step, total_loss], 
                feed_dict = {
                    learning_rate : 0.01,
                    wordids_placeholder: batch_stn
                })                    
            train_writer.add_summary(summary_, num_batch*epoch_idx+batch_id)
            loss_arr.append(total_loss_)
            weight_arr.append(total_len)
        print( np.exp(np.average(loss_arr, weights=weight_arr)) )
        
        loss_arr = []
        weight_arr = []
        # eval
        num_batch = len(dev_sentences) // batch_size - 1
        for batch_id in trange(num_batch, desc='validate epoch %d'%epoch_idx):
            len_arr = [sent.shape[0] for sent in trn_sentences[batch_id*batch_size:(batch_id+1)*batch_size]]
            max_len = max(len_arr)
            total_len = sum(len_arr)
            
            padded = [np.pad( sent, (0,  max_len - sent.shape[0]), 'edge') for sent in trn_sentences[batch_id*batch_size:(batch_id+1)*batch_size] ]
            batch_stn = np.stack(padded, axis=0)
            total_loss_ = sess.run(
                total_loss, 
                feed_dict = {
                    wordids_placeholder: batch_stn
                })  
            loss_arr.append(total_loss_)
            weight_arr.append(total_len)
        print( np.exp(np.average(loss_arr, weights=weight_arr)) )
