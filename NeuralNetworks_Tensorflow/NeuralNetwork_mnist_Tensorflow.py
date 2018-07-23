import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

"""
Feed forward:
Input > weight > hiddenlayer 1 (activationfunction) > weight > hiddenlayer 2 (activation function)
> weights > output layer

Compare output to intended output > cost function (cross entropy)

Back propogation:
Optimization function (optimizer) > minimize the cost (AdamOptimizer, Stochastic gradient descent, AdaGrad)
"""

# Nodes in the hidden layers
n_nodes_hl1 = 500;
n_nodes_hl2 = 500;
n_nodes_hl3 = 500;

n_classes = 10
batch_size = 100

# Data, comes in matrix form [0, 28*28 = 784]
X = tf.placeholder('float', [None, 784])
# Label
y = tf.placeholder('float')

def neural_network_model(data):
    # Random weights for layer 1, with 784 being the input nodes and n_nodeshl1 the amount of nodes in the first hidden layer. 
    # So for every input node there is an associated hidden layer node with a number (weight) which is instantiated randomly
    # The bias gets added after all the input data is multiplied with weights and summed, so there are the amount of hidden layer nodes
    # of biases for the first hidden layer. This is so that when all data is 0 and no neurons fire, some might, 
    # so there is always some activity
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([784, n_nodes_hl1])), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
                      'biases' : tf.Variable(tf.random_normal([n_classes]))}

    # Multiply input data with hidden layer 1 weights, so that you get a tensor of shape [None, n_nodes_hl1].
    # Add the bias and you have a new input for the hidden layer 2
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # Activation function (fire or not)
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    # Softmaxes the predictions and labels (a one-hot map of the right digit) (softmax is a function so that the input gets normalized
    # and the sum is 1) and calculates the crossentropy between those. The cost is the mean of all the crossentropies. 
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # One epoch = one forward pass and one backward pass of !all the training examples!
    hm_epochs = 3
    
    with tf.Session() as sess:
        # Initialize the graph
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Amount of iterations, when this is done, the all training samples have been through separated by batch sized
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Gets a batch of samples
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y:epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "Completed out of", hm_epochs, "Loss:", epoch_loss)
            
        # tf.argmax gives index of highest value, if they are equal the "one-hot" is the same
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print('Accuracy', accuracy.eval({x:mnist.test.images, y : mnist.test.labels}))
    
    
train_neural_network(X)
    