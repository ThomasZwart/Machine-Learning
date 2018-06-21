#own nearestneighbours
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total votings")
        
    distances = []
    
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    
    # geeft de votes in een lijst van [class, class, ...]
    votes = [i[1] for i in sorted(distances)[:k]]
    # count die votes lijst en geeft een lijst met tuples van [(class, freq), (class, freq),...]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# alles een float maken
full_data = df.astype(float).values.tolist()
# shuffle de data
random.shuffle(full_data)

# aanmaken data sets
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# Vullen data sets
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print(k_nearest_neighbors(train_set, [10,10,2,3,4,2,3,4,2]))
        
print("Accuracy: ", correct/total)
    

