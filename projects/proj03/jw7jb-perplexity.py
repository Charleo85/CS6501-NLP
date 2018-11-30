import tensorflow as tf;
print(tf.__version__)
import numpy as np
import time
from tqdm import tqdm, tnrange, tqdm_notebook


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

word2idx = {'<stop>': 0}
idx2word = ['<stop>']
index = 1

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
batch_size = 1

tf.reset_default_graph()

wordids_placeholder = tf.placeholder(tf.int64, [1, None])
word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, input_size], trainable=True)
embedded_words = tf.nn.embedding_lookup(word_embeddings, wordids_placeholder)

lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, hidden_size)
output2wordid = tf.layers.Dense(vocabulary_size)

inputs = embedded_words[:, :-1, ]
outputs, state = lstm(inputs)

labels = wordids_placeholder[:,1:]
logits = tf.map_fn(lambda x: output2wordid(x), outputs)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)

preds = tf.argmax(logits, axis=-1)
acces = tf.cast(tf.equal(preds, labels), tf.float32)
    
total_loss = tf.reduce_mean(losses)
tf.summary.scalar('loss', total_loss)

perplexity = tf.exp(total_loss)
tf.summary.scalar('perplexity', perplexity)

accuracy = tf.reduce_mean(acces)
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
        loss_arr = []
        weight_arr = []
        for batch_id, stn in enumerate(tqdm(trn_sentences, desc='training epoch %d'%epoch_idx)):
            summary_, _, total_loss_ = sess.run(
                [merged_summary, train_step, total_loss], 
                feed_dict = {
                    learning_rate : 0.1,
                    wordids_placeholder: np.expand_dims(stn, 0)
                })              
            train_writer.add_summary(summary_, num_batch*epoch_idx+batch_id)
            loss_arr.append(total_loss_)
            weight_arr.append(stn.shape[0]-1)
        print( np.exp(np.average(loss_arr, weights=weight_arr)) )
        
        loss_arr = []
        weight_arr = []
        # eval
        num_batch = len(dev_sentences) / batch_size
        for batch_id, stn in enumerate(tqdm(dev_sentences, desc='validate epoch %d'%epoch_idx)):
            total_loss_ = sess.run(
                total_loss, 
                feed_dict = {
                    wordids_placeholder: np.expand_dims(stn, 0)
                })  
            loss_arr.append(total_loss_)
            weight_arr.append(stn.shape[0]-1)
        print( np.exp(np.average(loss_arr, weights=weight_arr)) )
        
    # test
    f = open('jw7jb-tst-logprob.txt', 'w')
    for batch_id, stn in enumerate(tqdm(tst_sentences, desc='testing')):
        losses_ = sess.run(
            losses, 
            feed_dict = {
                wordids_placeholder: np.expand_dims(stn, 0)
            })  
        for wid, prob in zip(stn[1:], losses_[0]) :
            f.write( '{}\t{}\n'.format(idx2word[wid], -prob) )
    f.close()