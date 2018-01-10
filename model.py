import tensorflow as tf
def conv(x, w, b, stride, name):
    with tf.variable_scope('conv'):
        return tf.nn.conv2d(x, filter=w, strides=[1, stride, stride, 1], padding='SAME', name=name) + b

######## after 30k iterations (batch_size=64)
# with data augmentation (flip, brightness, contrast) ~81.0%
# without data augmentation 69.6%
def word2vec_conv(X, embedding_matrix, keep_prob, reuse=False):
    with tf.variable_scope('word2vec_conv'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        batch_size = tf.shape(X)[0]
        #K = 64
        #M = 128
        filter = [3,4,5]
        embedding_size = 300
        pooled = []
        for filter_size in filter:

            W1 = tf.get_variable('D_W1_'+str(filter_size), [filter_size, embedding_size, 1, 20], initializer=tf.contrib.layers.xavier_initializer())
            B1 = tf.get_variable('D_B1'+str(filter_size), [20], initializer=tf.constant_initializer())

            W2 = tf.get_variable('D_W2'+str(filter_size), [filter_size, 20, 20, 20], initializer=tf.contrib.layers.xavier_initializer())
            B2 = tf.get_variable('D_B2'+str(filter_size), [20], initializer=tf.constant_initializer())

            W3 = tf.get_variable('D_W3'+str(filter_size), [filter_size, 20, 20, 20], initializer=tf.contrib.layers.xavier_initializer())
            B3 = tf.get_variable('D_B3'+str(filter_size), [20], initializer=tf.constant_initializer())


            embedded_x = tf.nn.embedding_lookup(embedding_matrix, X)
            embedded_x = tf.expand_dims(embedded_x, -1)
            embedded_x = tf.cast(embedded_x, tf.float32)
            conv1 = conv(embedded_x, W1, B1, stride=2, name='conv1')
            bn1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1))
            P1 = tf.nn.max_pool(bn1, ksize = [1,150-5+1,1,1], strides = [1,1,1,1], padding = 'SAME')

            conv2 = conv(P1, W2, B2, stride=2, name='conv2')
            bn2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2))
            P2 = tf.nn.max_pool(bn2, ksize = [1,10,1,1], strides = [1,1,1,1], padding = 'SAME')

            conv3 = conv(P2, W3, B3, stride=2, name='conv3')
            bn3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv3)), keep_prob)
            P3 = tf.nn.max_pool(bn3, ksize = [1,10,1,1], strides = [1,1,1,1], padding = 'SAME')

            if filter_size  == 3:
                pooled =P3
            else:
                pooled = tf.concat([pooled, P3], 0)
            #pooled.append(P3)

            #W4 = tf.get_variable('D_W4', [5, embedding_size, 5, 5], initializer=tf.contrib.layers.xavier_initializer())
            #B4 = tf.get_variable('D_B4', [5], initializer=tf.constant_initializer())


            #W5 = tf.get_variable('D_W5', [5, embedding_size, 5, 5], initializer=tf.contrib.layers.xavier_initializer())
            #B5 = tf.get_variable('D_B5', [5], initializer=tf.constant_initializer())

            #W6 = tf.get_variable('D_W6', [5, 5, N, N], initializer=tf.contrib.layers.xavier_initializer())
            #B6 = tf.get_variable('D_B6', [N], initializer=tf.constant_initializer())

            #W7 = tf.get_variable('D_W7', [5, 5, N, N], initializer=tf.contrib.layers.xavier_initializer())
            #B7 = tf.get_variable('D_B7', [N], initializer=tf.constant_initializer())

            #W8 = tf.get_variable('D_W8', [5, 5, N, N], initializer=tf.contrib.layers.xavier_initializer())
            #B8 = tf.get_variable('D_B8', [N], initializer=tf.constant_initializer())

            #W9 = tf.get_variable('D_W9', [5, 5, N, N], initializer=tf.contrib.layers.xavier_initializer())
            #B9 = tf.get_variable('D_B9', [N], initializer=tf.constant_initializer())

        N = len(filter)*19*38*10*2
        W_fc1 = tf.get_variable('D_W10', [N, 5], initializer=tf.contrib.layers.xavier_initializer())
        B_fc1 = tf.get_variable('D_B10', [5], initializer=tf.constant_initializer())
        #W_fc1 = tf.get_variable('D_W10', [N, 5], initializer=tf.contrib.layers.xavier_initializer())
        #B_fc1 = tf.get_variable('D_B10', [5], initializer=tf.constant_initializer())


        #W_fc3 = tf.get_variable('D_W12', [10, 2], initializer=tf.contrib.layers.xavier_initializer())
        #B_fc3 = tf.get_variable('D_B12', [2], initializer=tf.constant_initializer())

            #conv4 = conv(P3, W4, B4, stride =1, name = 'conv4')
            #bn4 = tf.nn.relu(tf.contrib.layers.batch_norm(conv4))
            #P4 = tf.nn.max_pool(bn4, ksize = [1,150-5+1,1,1], strides = [1,1,1,1], padding = 'SAME')

            #conv5 = conv(P4, W5, B5, stride = 1, name = 'conv5')
            #bn5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv5))
            #P5 = tf.nn.max_pool(bn5, ksize = [1,150-5+1,1,1], strides = [1,1,1,1], padding = 'SAME')

            #conv6 = conv(bn5, W6, B6, stride = 2, name = 'conv6')
            #bn6 = tf.nn.relu(tf.contrib.layers.batch_norm(conv6))

            #conv7 = conv(bn6, W7, B7, stride = 2, name = 'conv7')
            #bn7 = tf.nn.relu(tf.contrib.layers.batch_norm(conv7))

            #conv8 = conv(bn7, W8, B8, stride=2, name='conv8')
            #bn8 = tf.nn.relu(tf.contrib.layers.batch_norm(conv8))

            #conv9 = conv(bn8, W9, B9, stride=2, name='conv9')
            #bn9 = tf.nn.relu(tf.contrib.layers.batch_norm(conv9))

            #conv10 = conv(bn8, W9, B9, stride=2, name='conv10')
            #bn10 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv7)), keep_prob)

            #pooled = tf.nn.max_pool(bn1,ksize=[1,150-5+1,1,1],strides=[1,1,1,1],padding='SAME')
            #bn7 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(pooled)), keep_prob)



        flat = tf.reshape(pooled,[batch_size, N])
        #bn6 = tf.nn.relu(tf.matmul(flat, W_fc1)+B_fc1)
        #drop6 = tf.nn.dropout(bn6, keep_prob)
        output = tf.matmul(flat, W_fc1) + B_fc1

        return tf.nn.softmax(output)
        #return output