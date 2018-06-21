#m = (x.mean * y.mean - (x*y).mean)/((x.mean)^2 - (x^2).mean)
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(amount, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(amount):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype= np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = (mean(xs) * mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
    b = mean(ys) - mean(xs) * m
    return m, b

# De squared error van de mean is proportioneel aan de variantie
# Dit is 1 - (squared error regressie lijn / squared error gem y)
    
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    # Code werkt ineens niet... 
    #squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    #squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))
    # Vervangende code
    squared_error_regr = 0
    squared_error_y_mean = 0
    for i in range(len(ys_line)):
        squared_error_regr += (ys_line[i] - ys_orig[i]) ** 2
        squared_error_y_mean += (y_mean_line[i] - ys_orig[i]) ** 2    
    return 1 - (squared_error_regr/squared_error_y_mean)

xs, ys = create_dataset(40, 40, 4, correlation = 'pos')
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m*x)+b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# scatter plot
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()



