import numpy as np
import time
import tensorflow as tf
import gensim

def train_inputs(x,y,vocabulary,batch_size,epoch):
    n, doc_len = x.shape
    emb_len = 300
    choose_index = epoch
    temp = x[choose_index*batch_size:(choose_index+1)*batch_size, ]
    use_batch = np.zeros((batch_size, doc_len, emb_len))
    for i in range(batch_size):
        for j in range(doc_len):
            use_batch[i, j, :] = vocabulary[temp[i, j],]
    use_batch_x = use_batch
    use_batch_y = y[choose_index*batch_size:(choose_index+1)*batch_size,:]
    use_batch_x0= np.reshape(use_batch_x, (batch_size, 315,300,1))
    return use_batch_x0, use_batch_y

def test_inputs(x,y,vocabulary,batch_size,epoch):
    n, doc_len = x.shape
    emb_len = 300
    choose_index = epoch
    temp = x[choose_index*batch_size:(choose_index+1)*batch_size, ]
    use_batch = np.zeros((batch_size, doc_len, emb_len))
    for i in range(batch_size):
        for j in range(doc_len):
            use_batch[i, j, :] = vocabulary[temp[i, j]-1,]

    use_batch_x = use_batch
    use_batch_y = y[choose_index*batch_size:(choose_index+1)*batch_size,:]
    use_batch_x0= np.reshape(use_batch_x, (batch_size, 315,300,1))
    return use_batch_x0, use_batch_y


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
vocabulary = list(model.wv.vocab.keys())
vocabulary = np.array([model[v] for v in vocabulary])
X_in_train = np.load("X_train.npy")
y_in_train = np.load("y_train.npy")
L_Y = len(y_in_train)
y2_train = np.int32(np.zeros((L_Y, 2)))
for i in range(L_Y):
    if int(y_in_train[i]) > 3:
        y2_train[i,1] = 1
    else:
        y2_train[i,0] = 1
#for i in range(L_Y):
#    if int(float(y_in[i]))> 3:
#        y2[i, 2] = 1
#    elif int(float(y_in[i])) ==3:
#        y2[i, 1] = 1
#    else:
#        y2[i, 0] = 1
X_in_test = np.load("X_test.npy")
y_in_test = np.load("y_test.npy")
L_Y_test = len(y_in_test)
y2_test = np.int32(np.zeros((L_Y_test, 2)))
for i in range(L_Y_test):
    if int(y_in_test[i]) > 3:
        y2_test[i,1] = 1
    else:
        y2_test[i,0] = 1
batch_size = 100
num_batch = int(X_in_train.shape[0]/batch_size)
num_epochs = 50
# learning rate
LR = .01
#training_epochs = 1000
keep_prob0 = .8
num_H1 = 10

print("Learning rate: %f" % LR)
print("Batch size: %d" % batch_size)
print("Dropout probability: %f" % keep_prob0)
print("Number of units in H1: %d" % num_H1)

time1 = time.time()
#with tf.device('/gpu:0'):
#with tf.device('/cpu:0'):
x = tf.placeholder(tf.float32, shape=[None,315,300,1])
y_ = tf.placeholder(tf.float32, shape=[None,2])
keep_prob = tf.placeholder(tf.float32)

#x_image = tf.reshape(x, [-1,28,28,1])
# single layer convolution neural network
W1 = tf.Variable(tf.truncated_normal([3,3,1,3], stddev=1.0 / np.sqrt(25*5)))
b1 = tf.Variable(tf.zeros([3]))
W2 = tf.Variable(tf.truncated_normal([3,3,3,3], stddev=1.0/np.sqrt(25*5*5)))
b2 = tf.Variable(tf.zeros([3]))
W3 = tf.Variable(tf.truncated_normal([3,3,3,3], stddev=1.0/np.sqrt(25*5*5)))
b3 = tf.Variable(tf.zeros([3]))

W_fc1 = tf.Variable(tf.truncated_normal([283500,10], stddev=1.0/np.sqrt(5*5*64)))
b_fc1 = tf.Variable(tf.zeros([10]))
W_fc2 = tf.Variable(tf.random_normal([10, 2], stddev=1.0 / np.sqrt(5)))
b_fc2 = tf.Variable(tf.zeros([2]))

C1 = tf.nn.relu(tf.nn.conv2d(x,W1, strides = [1,1,1,1], padding = 'SAME')+b1)
P1 = tf.nn.max_pool(C1,ksize = [1,10,10,1],strides = [1,1,1,1], padding = 'SAME')
C2 = tf.nn.relu(tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME') +b2)
P2 = tf.nn.max_pool(C2, ksize = [1,10,10,1], strides = [1,1,1,1], padding = 'SAME')
C3 = tf.nn.relu(tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME') +b3)
P3 = tf.nn.max_pool(C3, ksize = [1,10,10,1], strides = [1,1,1,1], padding = 'SAME')
P3_flat = tf.reshape(P3, [-1, 283500])
H1 = tf.nn.relu(tf.matmul(P3_flat, W_fc1)+b_fc1)
H1_drop = tf.nn.dropout(H1, keep_prob)
Z = tf.matmul(H1_drop, W_fc2)+b_fc2
y = tf.nn.softmax(Z)
#print(y.get_shape())
#H1_drop = tf.nn.dropout(H1, keep_prob0)
#y = tf.nn.softmax(tf.matmul(P1_flat, W_fc1) + b_fc1)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

session.run(tf.global_variables_initializer())
train_acc = np.zeros(num_epochs)
train_loss =np.zeros(num_epochs)
test_acc = np.zeros(num_epochs)
for epochs in range(num_epochs):
    epoch_train_acc = np.zeros(num_batch)
    epoch_train_loss =np.zeros(num_batch)
    for i in range(num_batch):
        X_train, y_train = train_inputs(X_in_train, y2_train, vocabulary, batch_size, i)
        _, train_accuracy, loss = session.run([train_step, accuracy, cross_entropy],
                                              feed_dict={x: X_train, y_:y_train,keep_prob:keep_prob0})
        print("epochs:%d accuracy:%f, loss:%f" % (epochs, train_accuracy, loss))
        epoch_train_acc[i]= train_accuracy
        epoch_train_loss[i] =loss
    train_acc[epochs] = np.average(epoch_train_acc)
    train_loss[epochs]= np.average(epoch_train_loss)
    num_epochs_test = int(X_in_test.shape[0]/batch_size)
    test_acc_epoch = np.zeros(num_epochs_test)
    for test_epoch in range(num_epochs_test):
        X_test,y_test = test_inputs(X_in_test,y2_test, vocabulary, batch_size, test_epoch)
        test_accuracy = session.run([accuracy], feed_dict={x:X_test, y_:y_test, keep_prob:1.0})
        test_acc_epoch[test_epoch] = test_accuracy[0]
        print("Test Accuracy: %f" % test_accuracy[0])
    test_acc[epochs] = np.average(test_acc_epoch)
time2 = time.time()
time_elapsed = time2 - time1
print("Time: %f" % time_elapsed)
np.save("test_acc_2class.npy", test_acc)
np.save("train_acc_2class.npy", train_acc)
np.save("train_loss_2class.npy", train_loss)