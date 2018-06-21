from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
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
clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

# De column cluster group en row (iloc) i wordt labels[i]

for i in range(len(X)) :
    original_df['cluster_group'].iloc[i] = labels[i]
  
n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    # Takes all rows where cluster group is equal to i
    temp_df = original_df[(original_df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

