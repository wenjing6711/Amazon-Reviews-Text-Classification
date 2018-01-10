from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import os
import gensim
from model import word2vec_conv
NUM_CLASSES = 5 #5

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
batch_size = 100
DOC_LENGTH = 150

prod ='video_surveillance'
# Overall Review
X_train = np.load("X_2_train_"+prod+".npy")
y_train = np.load("y_5_train_"+prod+".npy")
X_test =  np.load("X_2_test_"+prod+".npy")
y_test= np.load("y_5_test_"+prod+".npy")
# 5 class
#y_train = np.load("y_5_train.npy")
#y_test = np.load("y_5_test.npy")

def train_inputs(batch_size,i):
    index = np.random.choice(X_train.shape[0], batch_size,replace=False)
    #X_train_batch = X_train[i*batch_size:(i+1)*batch_size]
    X_train_batch = X_train[index]
    #X_train_batch = tf.reshape(X_train_batch, (batch_size, DOC_LENGTH))
    #y_train_batch = y_train[i*batch_size:(i+1)*batch_size]
    y_train_batch = y_train[index]
    #y_train_batch = tf.reshape(y_train_batch, (batch_size, 2))
    return X_train_batch, y_train_batch

def test_inputs(batch_size,i):
    index = np.random.choice(X_test.shape[0], batch_size,replace=False)
    #index = np.random.choice(X_test.shape[0],   batch_size)
    #X_test_batch = X_test[i*batch_size:(i+1)*batch_size,:]
    X_test_batch = X_test[index]
    #X_test_batch = tf.reshape(X_test_batch, (batch_size, DOC_LENGTH))
    #y_test_batch = y_test[i*batch_size:(i+1)*batch_size]
    y_test_batch = y_test[index]
    #y_test_batch = tf.reshape(y_test_batch, (batch_size, 2))
    return X_test_batch, y_test_batch

#### TRAIN
# with tf.device('/cpu:0'):
#     num_batch_train = int(X_train.shape[0]/batch_size)
#     num_batch_test = int(X_test.shape[0]/batch_size)
#     i_train = np.random.randint(0,num_batch_train,1)[0]
#     i_test =np.random.randint(0,num_batch_test,1)[0]
#     X_train_batch, y_train_batch = train_inputs(batch_size, i_train)
#     X_test_batch, y_test_batch= test_inputs(batch_size,i_test)

with tf.variable_scope('placeholder'):
    X = tf.placeholder(tf.int32, [batch_size, DOC_LENGTH])
    y = tf.placeholder(name='label', dtype=tf.int32, shape=[batch_size, 5])
    embedding_matrix = tf.placeholder(tf.float32, shape=(3000000, 300))  # embedding matrix
    keep_prob = tf.placeholder(tf.float32, shape=())

with tf.variable_scope('model'):
    output = word2vec_conv(X, embedding_matrix, keep_prob=keep_prob)

with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#with tf.device('/cpu:0'):
#var = tf.get_variable(name, shape, initializer=initializer)
tvar = tf.trainable_variables()
word2vec_var = [var for var in tvar if 'word2vec_conv' in var.name]

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, var_list=word2vec_var)

saver = tf.train.Saver(tvar)
sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

if not os.path.exists('current_model/'):
    os.makedirs('current_model/')

# saver.restore(sess,tf.train.latest_checkpoint('current_model/'))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

#prob = np.array([0.75, 0.8, 0.9, 1])
print("pre-trained word embedding load...")
pre_trained = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vocabulary = list(pre_trained.wv.vocab.keys())
embeddings = np.array([pre_trained[word] for word in vocabulary])
print("Done!")
#train_trend_loss = []
#train_trend_acc = []
train_loss = np.zeros(15)
train_acc = np.zeros(15)
test_acc = np.zeros(15)
for k in range(5):
    keep_probability = 0.5
    print("dropout rate is ", 1-keep_probability)
    tf.train.start_queue_runners()
    #loss_print = 0
    #accuracy_print = 0
    t = time.time()
    train_epoch_loss = np.zeros(350)
    train_epoch_acc = np.zeros(350)
    for i in range(0, 80):

        #X_batch, y_batch = sess.run([X_train_batch, y_train_batch])train_inputs(batch_size,i)
        X_batch, y_batch= train_inputs(batch_size,i)
        #print(y_batch)
        #y_batch = np.zeros((batch_size, NUM_CLASSES))
        #y_batch[np.arange(batch_size), labels_batch] = 1

        _, loss_print, accuracy_print = sess.run([train_step, loss, accuracy],
                                                 feed_dict={X: X_batch, y: y_batch, embedding_matrix: embeddings, keep_prob: keep_probability})

        train_epoch_loss[i] = loss_print
        train_epoch_acc[i] = accuracy_print
        #loss_out = np.sum(train_epoch_loss)/(i+1)
        #acc_out = np.sum(train_epoch_acc)/(i+1)
        #train_trend_loss.append(loss_print)
        #train_trend_acc.append(accuracy_print)
        if i % 20 == 0:
            print('time: %f epoch: %d iteration:%d loss:%f accuracy:%f' % (float(time.time() - t), k, i, loss_print, accuracy_print))
            t = time.time()

        if i % 100 == 0:

            test_accuracy = 0.0
            #accuracy_count = 0

            for j in range(10):
                X_tst_batch, y_tst_batch = test_inputs(batch_size,j)
                #y_batch = np.zeros((batch_size, NUM_CLASSES))
                #y_batch[np.arange(batch_size), labels_batch] = 1

                accuracy_print = sess.run([accuracy], feed_dict={X: X_tst_batch, y: y_tst_batch,embedding_matrix:embeddings, keep_prob: 1.0})

                test_accuracy += accuracy_print[0]
                #accuracy_count += 1
            test_accuracy = test_accuracy / 10
            print('TEST:%f' % test_accuracy)
    train_loss[k] = np.average(train_epoch_loss)
    train_acc[k] = np.average(train_epoch_acc)
    #saver.save(sess, 'current_model/model', global_step=30000)

    ### JUST TEST
    tf.train.start_queue_runners()
    t = time.time()
    #test_accuracy = 0.0
    #accuracy_count = 0
    test_epoch_acc = np.zeros(80)

    for i in range(20):
        #X_tst, y_tst = sess.run([X_test_batch, y_test_batch])
        X_tst, y_tst = test_inputs(batch_size, i)
        #y_batch = np.zeros((batch_size, NUM_CLASSES))
        #y_batch[np.arange(batch_size), labels_batch] = 1

        accuracy_print = sess.run([accuracy], feed_dict={X: X_tst, y: y_tst, embedding_matrix: embeddings, keep_prob: 1.0})

        #test_accuracy += accuracy_print[0]
        #accuracy_count += 1
        test_epoch_acc[i]= accuracy_print[0]
        if i % 10 == 0:
            test_accuracy = np.sum(test_epoch_acc) / (i+1)
            print('time: %f accuracy:%f' % (float(time.time() - t), test_accuracy))
            t = time.time()
    test_acc[k] = np.average(test_epoch_acc)
np.save("train_loss__word2vec_cnn5_"+prod,np.array(train_loss))
np.save("train_acc__word2vec_cnn5_"+prod, np.array(train_acc))
np.save("test_acc_word2vec_cnn5_"+prod, np.array(test_acc))