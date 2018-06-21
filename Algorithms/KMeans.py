import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random as rd
import warnings
import math

class K_Means:
    def __init__(self, n_clusters = 2, max_iterations = 300, tolerance = 0.001):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = {}
        self.colors = int(n_clusters/2)*['r', 'b', 'k', 'c', 'g', 'y', 'purple']
        self.tolerance = tolerance
    
    def fit(self, data):
        # Data comes in np array of features, eg [[feature], [feature]]
        if len(data) < self.n_clusters:
            warnings.warn("Data is less then cluster amount")
            return 
        
        self.centroids = {}     
        clusters = {}
        # Takes n random numbers without duplicates
        randoms = rd.sample(range(len(data)), self.n_clusters)
        # n random centroids to start with        
        for i in range(len(randoms)):  
            self.centroids[i] = data[randoms[i]]
        for i in range(self.max_iterations):
            clusters = {}
            for i in range(self.n_clusters):
                clusters[i] = []
            # Calculate clusters
            for feature in data:
                best_centroid = 0;
                # Fills the cluster dictionary, for every centroid there will be a list of corresponding features
                for centroid in self.centroids:
                    if np.linalg.norm(np.array(feature) - np.array(self.centroids[best_centroid])) > np.linalg.norm(np.array(feature) - np.array(self.centroids[centroid])):
                        best_centroid = centroid
                clusters[best_centroid].append(feature)
                   
            # To check for tolerance, if the changes are too small iterations will stop
            prev_centroids = dict(self.centroids)
            # Update centroids
            for centroid in self.centroids:               
                featurelist = clusters[centroid]
                # New centroid is the average vector of all the features associated with the current centroid
                new_centroid = np.average(np.array(featurelist), axis= 0)
                self.centroids[centroid] = new_centroid
            
            optimized = True
                
            # If there is no significant change, stop iteration, increases performance
            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                # The sum of the changes in every dimension needs to be higher than a tolerance value
                if math.fabs(np.sum(((current_centroid - original_centroid) / original_centroid) * 100.0)) > self.tolerance:
                    optimized = False    
            if optimized:
                break
        return self.centroids
  
    def predict(self, data):
        # recieves a numpy array of features and returns a list of predictions
        predictions = []
        # The closest centroid to the datapoint becomes its classification 
        for feature in data:
            best_centroid = 0;
            for centroid in self.centroids:             
                if np.linalg.norm(np.array(feature) - np.array(self.centroids[best_centroid])) > np.linalg.norm(np.array(feature) - np.array(self.centroids[centroid])):
                        best_centroid = centroid
            predictions.append(best_centroid)
        return predictions
    
    def visualize(self, data): 
        # Only 2D data
        if (len(data[0]) > 2):
            warnings.warn("To many dimensions to visualize")
            return
        
        # Size proportional to the dataset size
        datapointsize = 2500 / len(data)
        if (datapointsize > 200):
            datapointsize = 200
        if (datapointsize < 1):
            datapointsize = 1
        
        # Get the predictions of the data to visualize the right color        
        predictions = self.predict(data)
        for i in range(len(predictions)):
            plt.scatter(data[i][0], data[i][1], c = self.colors[predictions[i]], s = datapointsize)            
        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], marker = '*', s = 150, c = self.colors[centroid])        
        plt.show()
 
def create_random_dataset(amount, min_x, max_x, min_y, max_y):
    dataset = []
    for i in range(amount):        
        dataset.append([rd.uniform(min_x, max_x), rd.uniform(min_y, max_y)])
    return np.array(dataset)

def test():
    data = create_random_dataset(233, 0, 10, 0, 10)   
    clf = K_Means(n_clusters = 17, max_iterations = 300, tolerance = 0.001)
    clf.fit(data)
    clf.visualize(data)
 
test()
    








