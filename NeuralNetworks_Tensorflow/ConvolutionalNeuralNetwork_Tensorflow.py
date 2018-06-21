import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

"""
Feed forward:
Input > weight > hiddenlayer 1 (activationfunction) > weight > hiddenlayer 2 (activation function)
> weights > output layer

Compare output to intended output > cost function (cross entropy)

Back propogation:
Optimization function (optimizer) > minimize the cost (AdamOptimizer, Stochastic gradient descent, AdaGrad)
"""

n_classes = 10
batch_size = 128
# One epoch = one forward pass and one backward pass of !all the training examples!
hm_epochs = 1;
# To reduce overfitting and mimic the brain for dead neurons.
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder('float', [None, 784])
# Label
y = tf.placeholder('float')


def conv2d(x, W):
    # strides is how much pixels the convolutional window will move
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = "SAME")

def maxpool2d(x):
    # ksize = window size in form [batchnmr, height, width, channels], channels is 1 here, only gray in Mnist
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")

def convolutional_neural_network(x):
    # Convolution NN gets a 28 by 28 picture, there is a 5 by 5 filter that goes over the 28 by 28 picture which can be random
    # The filter can be set so that edges will be emphasized. The first layer will produce 32 outputs from the 28x28 
    # convulated picture by making use of pooling.
    # 5 by 5 convolution, 1 input (the picture), 32 features/outputs

    weights = {'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])),
               # 5 by 5 convolution, 32 inputs (previous layer), 64 outputs
               'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])),
               # 7 because of pooling with stride = 2 gives, 28/2/2= 7
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):

    prediction = convolutional_neural_network(x)
    # Softmaxes the predictions and labels (a one-hot map of the right digit) (softmax is a function so that the input gets normalized
    # and the sum is 1) and calculates the crossentropy between those. The cost is the mean of all the crossentropies. 
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Amount of iterations, when this is done, the all training samples have been through separated by batch sized   
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)               
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
    
train_neural_network(x)