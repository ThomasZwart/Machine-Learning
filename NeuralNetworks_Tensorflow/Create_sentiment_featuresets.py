from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            # All lines in an array
            contents = f.readlines()
            # Add all the tokenized words to the lexicon list
            for line in contents[:hm_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # Counter gives a dictionary of frequency per words
    w_counts = Counter(lexicon)
    final_lexicon = []
    
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
           final_lexicon.append(w)
    return final_lexicon
            
def sample_handling(sample, lexicon, classification):
    """ Creates a featureset of a sample input of the form [[[0,0,1,0 ..., 0, 1], [0, 1]], [[next feature],[class]]], 
    where the first list in a feature is the data and every 1 corresponds to a word in the lexicon. 
    Classification is [0,1] for negative and [1,0] for a positive sentiment."""
        
    featureset = []
    
    with open (sample, "r") as file:
        contents = file.readlines()

        # Every line will be a feature
        for line in contents [:hm_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            # Make an empty feature
            features = np.zeros(len(lexicon))
            for word in current_words:
                # If the word is in the lexicon, the index where it is will be set to 1 in the feature
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    # Classification is in form [1,0] and [0,1] because the neural network needs a one-hot to and performs a tf.argmax()
    # So it is based on index
    features += sample_handling(pos, lexicon, [1,0])
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)
    
    features = np.array(features)
    
    testing_size = int(test_size*len(features))
    
    # Gets all the features with the [:,0] notation untill testing size
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels("pos.txt", "neg.txt")
    with open("sentiment_set.pickle", 'wb') as f:
        pickle.dump([train_x * 10 , train_y * 10 , test_x * 10 , test_y * 10], f)
    







