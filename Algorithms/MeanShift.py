import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import warnings
from sklearn.datasets.samples_generator import make_blobs

### Disclaimer: Just a simple implementation, works slow and not very well
class Mean_Shift:
    def __init__(self, radius = None, radius_norm_step = 100):
        self.radius = radius
        self.colors = 10*['r', 'b', 'k', 'c', 'g', 'y', 'purple']
        self.radius_norm_step = radius_norm_step
        
    def fit(self, data):
        
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
            
        centroids = {}
        
        # with mean shift all data points start as centroids
        for i in range(len(data)):
            centroids[i] = data[i]
            
        # Weights list
        weights = [i for i in range(self.radius_norm_step)][::-1] # reverses the list      
    
        # Untill convergence
        while True:
            # Every centroid will get updated in here
            new_centroids = []

            # For all centroids
            for i in centroids:
                # To store all features in the bandwidth of the centroid
                in_bandwidth = []
                centroid = centroids[i]
            
                # Store the features in the list
                # The bandwidth is dynamic, every point will get a weight based on how far it is from the centroid
                # and the higher the weight the more times it will be added in the bandwidth list.
                # The new centroid will be the average of that bandwidth list
                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.00000001
                    weight_index = int(distance/self.radius) # How many radius steps from centroid
                    # Weight index bigger than the weight list
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    
                    # Less radius steps away from centroid is higher weight, so it gets added more to the 
                    # bandwidth list where the average will come from
                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth += to_add
                    
                # New centroid becomes the average of all the features in the bandwidth
                new_centroid = np.average(in_bandwidth, axis = 0)
                new_centroids.append(tuple(new_centroid))
                
            # When everything converges we find that some are identical copies, 
            # this way the algorithm finds how many clusters there are by itsself
            uniques = sorted(list(set(new_centroids)))
            
            to_pop = []
            
            # If 2 centroids are 1 step away from eachother, pop one out.
            for i in uniques:
                if i in to_pop: pass # we're not inspecting centroids in radius of i since i will be popped
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius and ii not in to_pop: # skipping already-added centroids
                        to_pop.append(ii)

            for i in to_pop:
                uniques.remove(i)
            
            prev_centroids = dict(centroids)
            
            centroids = {}
            
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            
            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break
        
        self.centroids = centroids
    
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
    

X, y = make_blobs(n_samples = 50, centers=3, n_features = 2)
clf = Mean_Shift()
clf.fit(X)

clf.visualize(X)

                
                
                
                
            
        