import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random as rd
import pickle
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        self.vis = True
        
    # Train
    def fit(self, data):
        self.data = data
        # {||w||: [w,b]}
        opt_dict = {}
        
        # Omdat dit niet uitmaakt voor de magnitude maar wel voor de constraint moet je ze op alle 4 testen
        transforms = [[1,1], [-1,1], [-1, -1], [1, -1]]
              
        self.max_feature_value = 0;
        self.min_feature_value = 0;
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    if feature < self.min_feature_value:
                        self.min_feature_value = feature
                    if feature > self.max_feature_value:
                        self.max_feature_value = feature   
                        
                    
        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]
        
        # Expensive 
        b_range_multiple = 2
        # We dont need to take as small of steps with b, so we multiple the step size for b
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10        
        
        for step in step_sizes:
            # Begin hoog en langzaam ||w|| laten dalen
            w = np.array([latest_optimum, latest_optimum])
            # We can do this because convex
            optimized = False
            # Zodra het optimale punt is gevonden gaat de stapgrootte omlaag
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value * b_range_multiple), 
                                   self.max_feature_value*b_range_multiple, step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # Weakest link in svm fundament, we will test this w and b against all
                        # of the data set to see if it holds the contraint
                        for yi in self.data:
                            for xi in self.data[yi]:
                                # Yi(Xi*w + b >= 1)                             
                                if not (yi * (np.dot(w_t,xi) + b)) >= 1:
                                    found_option = False
                                    break
                        # if it holds the constraint it is an option, so it will be added to the dictionary 
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
                        
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # w wordt weer op latest optimum gezet, die gaat klein beetje omhoog
            latest_optimum = opt_choice[0][0] + step*2
        

    def predict(self, features):
        # Sign (x*w + b)
        if self.visualization and self.vis:
            self.vis = False      
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1) 
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker = '*', color = self.colors[classification])
        return classification

    def visualize(self):
        if self.visualization and self.vis:
            self.vis = False
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1) 

        [[self.ax.scatter(x[0], x[1], s = 100, color = self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        # hyperplane = x * w + b
        # v = x*w +b
        # Positive support vector = 1, negative support vector = -1
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]
        
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        
        # The positive support vector
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)      
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')
        
        # The negative support vector
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)      
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')
        
        # The decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)      
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')
        
        plt.show()


data_dict = {-1:[], 1:[]}

# generate dataset
for i in range(50):
    x = rd.randint(0,15)
    data_dict[-1].append([x, rd.randint(x, 15)])
for i in range(50):
    x = rd.randint(0,14)
    data_dict[1].append([x, rd.randint(-5, x-5)])


svm = Support_Vector_Machine()
svm.fit(data_dict)
svm.visualize()






