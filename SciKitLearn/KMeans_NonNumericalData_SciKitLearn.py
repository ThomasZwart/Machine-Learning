from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name', 'pclass', 'fare', 'parch', 'embarked'], 1, inplace=True)
df.convert_objects(convert_numeric = True)
# waardes die NaN zijn worden 0
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    """ For machine learning all data needs to be numerical, so this function converts the non numerical to numerical data """
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        # Dictionary is e.g. {Female: 0, Male: 1} zo werkt de functie hieronder
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # Dataframe column to list
            column_contents = df[column].values.tolist()
            # Make a set from the list, so duplicates are removed
            dataset = set(column_contents)
            # Fill the dictionary
            i = 0
            for item in dataset:              
                text_digit_vals[item] = i
                i += 1
            
            # map takes a function and an input list and outputs an output list based on the function with the list keyword
            df[column] = list(map(convert_to_int, df[column]))
    return df
            
df = handle_non_numerical_data(df)


X = np.array(df.drop(['survived'], 1).astype(float))
# Essential
X = preprocessing.scale(X)
y = np.array(df['survived'])

# SkLearn Kmeans
clf = KMeans(n_clusters = 2)
# Own Kmeans
#clf = OwnKMeans.K_Means(n_clusters = 2)
clf.fit(X)

# For a single prediction, input a list within a list eg, [[feature]]
single_prediction = clf.predict([X[1]])
# All predictions, returns a list where every datapoint is matched to a cluster number
predictdata = clf.predict(X)

# one way to calculate accuracy
correct2 = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct2 += 1
print('Accuracy: ', correct2/len(X))

# another way to do it
correct = 0;
total = 0;
for i in range(len(predictdata)):
    # If the cluster numbers of data match exactly with the deaths, we have found that the data is divided by survival
    if (predictdata[i] == y[i]):
        correct += 1
    total += 1

print('Accuracy: ', correct/total)
    
