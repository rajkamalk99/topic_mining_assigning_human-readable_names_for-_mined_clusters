import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import MeanShift
import os
import math

# initialising the lemmatizer
lemmatizer = WordNetLemmatizer()


# function for creating the lexicon
def create_lexicon(text_file):
    lexicon = []

    with open(text_file, 'r') as f:
        # reading the contents of the file
        contents = f.readlines()
        for l in contents:
            # tokenising the words 
            all_words = word_tokenize(l.lower())
            lexicon += list(all_words)
    # lemmatising the words 
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_count = Counter(lexicon)
    l2 = []
    for w in w_count:
        # removing the more common words whose presence won't make any difference so that lexicon size won't be much larger
        if  100 > w_count[w] > 20:
            l2.append(w)
    return l2   

# function for creating data out of a file to fit into a meanshift classifier

def create_fitting_data(lexicon_file, main_file):
    featureset = []
    lexicon = create_lexicon(lexicon_file)
    # finding the size of the file  
    with open(main_file, 'r') as f:
        statinfo = os.stat(main_file)
        size = statinfo.st_size
        i = 82
        while(i < size):
            contents = f.read(i)
            if contents == "":
                break
            i +=82
            # tokenising and lemmatising the words 
            all_words = word_tokenize(contents)
            all_words = [lemmatizer.lemmatize(i) for i in all_words]
            # creating a numpy array of length of size of the lexicon
            features = np.zeros(len(lexicon))
            for word in all_words:
                if word in lexicon:
                    index = lexicon.index(word)
                    # incrementing the corresponding index according to ,at which index ,is the current word found in the lexicon
                    features[index] +=1 
            # converting this array to a list and append to the featureset
            features = list(features)
            featureset.append(features)
    return featureset

X = create_fitting_data('/home/raj/Documents/AI/ML/Fathom/total.txt', '/home/raj/Documents/AI/ML/Fathom/total.txt')
# creating the MeanShift classifier
ms = MeanShift(bandwidth=16)
# fitting the data
ms.fit(X)
labels = ms.labels_
n_clusters_ = len(np.unique(labels))
cluster_centers = ms.cluster_centers_
print("\n The total number of topics in all the videos are  :", n_clusters_, "\n")

# function for creating the cluster centers of a text file and predicting those centre's with the passed meanshift classifier

def create_cluster_centers_and_predict(lexicon_file, main_ms_classifier):
    files = os.listdir('./')
    for f in files:
        if f.lower()[-3:] == 'txt':
            # creating the featureset and fitting the data of the file with a meanshift classifier
            file_name  = f.lower()
            featureset = create_fitting_data(lexicon_file, f)
            s_ms = MeanShift(bandwidth=5)
            s_ms.fit(featureset)
            cluster_centers = s_ms.cluster_centers_
            print("The text file ", file_name, " contains ", len(cluster_centers), " topics\n")
            topic_counter =0
            for centre in cluster_centers:
                topic_counter +=1
                # reshaping and predicting the centre with the passed Meanshift classifier
                instance = np.reshape(centre, (1, -1))
                predicted = main_ms_classifier.predict(instance)
                predicted = int(predicted)
                print("The topic ", topic_counter, " in the text file ", file_name, " is the ", predicted, "th topic in the list of total number of topics\n")



create_cluster_centers_and_predict('/home/raj/Documents/AI/ML/Fathom/total.txt', ms)
print("\n Assigning Human-readable names to labels\n")
# now finding which vector belongs to which cluster

cluster_vector = {}

for i in range(n_clusters_):
    cluster_vector[i] = []

for i in X:
    i = np.reshape(i, (1, -1))
    predicted = ms.predict(i)
    predicted = int(predicted)
    i = np.reshape(i, (1,-1))
    k = []
    for m in i:
        for j in m:
            k.append(j)
    cluster_vector[predicted].append(k)
# cluster_vector contains the label as the key and vectors belonging to that label as values


# function for finding the frequency of a term i.e  which is the number of times a vector appears in a cluster

def tf(vector, cluster):
    return cluster.count(vector) / len(cluster)

# n_containing returns the number of clusters  containing the vector.

def n_containing(vector, cluster_vector):
    sum =0
    for cluster in cluster_vector:
        if vector in cluster_vector[cluster]:
            sum +=1
    return sum

# idf function computes 'inverse document frequency' which measures how common a word is among all the clusters

def idf(vector, cluster_vector):
    return math.log(len(cluster_vector) / (1 + n_containing(vector, cluster_vector)))

# function tfidf computes the TF-IDF score. It's the product of tf and idf.

def tfidf(vector, cluster, cluster_vector):
    return tf(vector, cluster) * idf(vector, cluster_vector)

# finding which vector has the highest TF-IDF in a cluster

quat_final = {}

for cluster in cluster_vector:
    max_vector = [] 
    max_score = 0
    for vector in cluster_vector[cluster]:
        score = (tfidf(vector, cluster_vector[cluster], cluster_vector))
        if score > max_score:
            max_score = score 
            max_vector.append(vector)
    quat_final[cluster] = max_vector[-1]

lexicon = create_lexicon('/home/raj/Documents/AI/ML/Fathom/total.txt')


# after finding vector with highest TF-IDF score, transformimg that vector to text by mapping with lexicon

semi_final = {}
for label in quat_final:
    title = []
    for i in quat_final[label]:
        if i > 0.0:
            i = int(i)
            word = lexicon[i]
            title.append(word)
    if len(title) == 0:
        word = "This title in not in the lexicon"
        title.append(word)
        semi_final[label] = title
    else:
        semi_final[label] = np.unique(title)

# now creating a dictionary with labels as keys and titles(formed by mapping highest score vector with lexicon) as values

final = {}
for label in semi_final:
    title = ""
    for i in semi_final[label]:
        title += " "+i
    final[label] = title


for label in final:
    print("\nThe title for the label ", label, " is ", final[label],"\n")
