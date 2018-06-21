import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
tf.reset_default_graph()

# One epoch = one forward pass and one backward pass of !all the training examples!
hm_epochs = 1
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    # Data comes in shape (batchsize, n_chunks, chunksize), so a batchsize amount of n_chunk by chunksize images in 2D arrays
    # Transpose the data, swap the first dimension with the second, the data is now of the form [n_chunks, batch_size, chunk_size]
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)
    # Finally the data is a list of arrays that contain lists of lists that have the a horizontal layer of pixels. 
    # E.g. [[(layer 1)[picture 1],[picture 2]], [(layer 2)[picture 1],[picture 2]]]
    # These is a function below to see this

    # rnn_size is the number of recurrences, the number of times it is passed through the lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    # outputs is a array of outputs of the chunk size, with the last output being the chunk that is the 
    # the last horizontal layer of pixels
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):

    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Amount of iterations, when this is done, the all training samples have been through separated by batch sized   
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # Data comes in batches of [1,2,3,4,5,6], but needs to be reshaped for a RNN
                epoch_x = epoch_x.reshape(batch_size,n_chunks,chunk_size)
                
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        # tf.argmax gives index of highest value, if they are equal, the "one-hot" is the same
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))
    

train_neural_network(x)

def data_transformation_visual():
    a = tf.placeholder(tf.float32)
    operation1 = tf.transpose(a, perm = [1,0,2])
    operation2 = tf.reshape(tf.transpose(a, perm = [1,0,2]), [-1, 5])
    operation3 = tf.split(tf.reshape(tf.transpose(a, perm = [1,0,2]), [-1, 5]), 5, 0)
    with tf.Session() as sess:
        print(sess.run(a, feed_dict = {a : [[[ 1.,  1.,  1.,  1.,  1.], 
                                                 [ 2.,  1.,  1.,  1.,  1.],
                                                 [ 3.,  1.,  1.,  1.,  1.],
                                                 [ 4.,  1.,  1.,  1.,  1.],
                                                 [ 5.,  1.,  1.,  1.,  1.]],
                                                 [[ 1., 2.,  2.,  2.,  2.], 
                                                 [ 2., 2.,  2.,  2.,  2.],
                                                 [  3., 2.,  2.,  2.,  2.],
                                                 [  4., 2.,  2.,  2.,  2.],
                                                 [ 5., 2.,  2.,  2.,  2.]]]}))
        print()
        print(sess.run(operation1, feed_dict = {a : [[[ 1.,  1.,  1.,  1.,  1.], 
                                                 [ 2.,  1.,  1.,  1.,  1.],
                                                 [ 3.,  1.,  1.,  1.,  1.],
                                                 [ 4.,  1.,  1.,  1.,  1.],
                                                 [ 5.,  1.,  1.,  1.,  1.]],
                                                 [[ 1., 2.,  2.,  2.,  2.], 
                                                 [ 2., 2.,  2.,  2.,  2.],
                                                 [  3., 2.,  2.,  2.,  2.],
                                                 [  4., 2.,  2.,  2.,  2.],
                                                 [ 5., 2.,  2.,  2.,  2.]]]}))
        print()
        print(sess.run(operation2, feed_dict = {a : [[[ 1.,  1.,  1.,  1.,  1.], 
                                                 [ 2.,  1.,  1.,  1.,  1.],
                                                 [ 3.,  1.,  1.,  1.,  1.],
                                                 [ 4.,  1.,  1.,  1.,  1.],
                                                 [ 5.,  1.,  1.,  1.,  1.]],
                                                 [[ 1., 2.,  2.,  2.,  2.], 
                                                 [ 2., 2.,  2.,  2.,  2.],
                                                 [  3., 2.,  2.,  2.,  2.],
                                                 [  4., 2.,  2.,  2.,  2.],
                                                 [ 5., 2.,  2.,  2.,  2.]]]}))
        print()
        print(sess.run(operation3, feed_dict = {a : [[[ 1.,  1.,  1.,  1.,  1.], 
                                                 [ 2.,  1.,  1.,  1.,  1.],
                                                 [ 3.,  1.,  1.,  1.,  1.],
                                                 [ 4.,  1.,  1.,  1.,  1.],
                                                 [ 5.,  1.,  1.,  1.,  1.]],
                                                 [[ 1., 2.,  2.,  2.,  2.], 
                                                 [ 2., 2.,  2.,  2.,  2.],
                                                 [  3., 2.,  2.,  2.,  2.],
                                                 [  4., 2.,  2.,  2.,  2.],
                                                 [ 5., 2.,  2.,  2.,  2.]]]}))


