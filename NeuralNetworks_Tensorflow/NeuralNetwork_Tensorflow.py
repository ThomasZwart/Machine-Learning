import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np

train_x, train_y, test_x, test_y = create_feature_sets_and_labels("pos.txt", "neg.txt")

"""
Feed forward:
Input > weight > hiddenlayer 1 (activationfunction) > weight > hiddenlayer 2 (activation function)
> weights > output layer

Compare output to intended output > cost function (cross entropy)

Back propogation:
Optimization function (optimizer) > minimize the cost (AdamOptimizer, Stochastic gradient descent, AdaGrad)
"""

# Nodes in the hidden layers
n_nodes_hl1 = 1000;
n_nodes_hl2 = 1000;
n_nodes_hl3 = 1000;

n_classes = 2
batch_size = 100

X = tf.placeholder('float', [None, len(train_x[0])])
# Label
y = tf.placeholder('float')


hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 
                  'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
                  'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
                  'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
                'biases' : tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):
    # Random weights for layer 1, with 784 being the input nodes and n_nodeshl1 the amount of nodes in the first hidden layer. 
    # So for every input node there is an associated hidden layer node with a number (weight) which is instantiated randomly
    # with a normal distribution. The bias gets added after all the input data is multiplied with weights and summed, 
    # so there are the amount of hidden layer nodes of biases for the first hidden layer. 
    # This is so that when all data is 0 and no neurons fire, some might, so there is always some activity

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
    hm_epochs = 1
    
    with tf.Session() as sess:
        # Initialize the graph
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Amount of iterations, when this is done, the all training samples have been through separated by batch sized        
            i = 0
            while i < len(train_x):
                # Gets a batch of samples
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print("Epoch", epoch + 1, "Completed out of", hm_epochs, "Loss:", epoch_loss)
            
        # tf.argmax gives index of highest value, if they are equal, the "one-hot" is the same
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print('Accuracy', accuracy.eval({x:test_x, y : test_y}))
    
    
train_neural_network(X)
    