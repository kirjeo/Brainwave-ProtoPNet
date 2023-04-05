import numpy as np
import pandas as pd
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from numba import jit, cuda

df=pd.read_csv('../datasets/split_datasets/main_train_split_80_threshold_4.csv')
print(df.head())

df_test=pd.read_csv('../datasets/split_datasets/main_test_split_20_for_threshold_4.csv')
print(df_test.head())

#preparing train data
data_array=df.iloc[:,:2].values
dataset=data_array.T
sequences=dataset[1]
species=dataset[0]
print("Num unique train species: {}".format(np.unique(species).shape))

#preparing test data
test_array=df_test.iloc[:,:2].values
test_dataset=test_array.T
test_sequences=test_dataset[1]
test_species=test_dataset[0]
print("Num unique test species: {}".format(np.unique(test_species).shape))

# changing labels into indices if necessary
uniq_species=np.unique(species)
classes= np.unique(species, return_inverse=True)[1]
print(uniq_species)
print(classes)

#get all possible kmers for AGTC i.e for all sequences
# @jit
def get_all_kmers(k):
    all_kmers=[]
    for i in product('agtc', repeat = k):
        all_kmers.append(''.join(i))
    return all_kmers

#get all kmers for the sequence
# @jit
def list_kmers(seq, k):
    kmers = []
    # Calculate how many kmers of length k there are
    if len(seq)<200:
        num_kmers=len(seq)-k+1
    else:
        #only for the first 200 letters of the sequence
        num_kmers = len(seq[:200]) - k + 1 
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = seq[i:i+k]
        kmers.append(kmer)
    return kmers

# mandel_gpu = cuda.jit(restype=uint32, argtypes=[f8, f8, uint32], device=True)(mandel)
#create 2d array: for every sequence, array of counts of every kmer
# @jit(target_backend='cuda')  
def create_feature_table(sequences,k):
    feature_table=[]
    for seq in sequences:
        cv = CountVectorizer(vocabulary=(get_all_kmers(k))) #lists counts of all words
        #representing sequences as sentences with words that are kmers i.e all kmers separated by space
        features=np.asarray(cv.fit_transform([(" ".join(list_kmers(seq,k)))]).toarray())  
        features=features.flatten().tolist()
        feature_table.append(features)
    return np.asarray(feature_table)

#multinomial naive bayes
NB = MultinomialNB()
Tree = DecisionTreeClassifier()
RF = RandomForestClassifier()
GB = GradientBoostingClassifier()
LR = LogisticRegression()

kmers= [3]
ft=[]
for i in kmers:
    ft.append(create_feature_table(sequences, i))

for j in range(len(kmers)):
    NB.fit(ft[j], species)
    Tree.fit(ft[j], species)
    RF.fit(ft[j], species)
    GB.fit(ft[j], species)
    LR.fit(ft[j], species)
    #predictions with test data
    ft_test=create_feature_table(test_sequences, kmers[j])
    predicted=NB.predict(ft_test)
    #accuracy with test data
    print('Naive Bayes accuracu with {} mer: {}'.format(kmers[j], 100*NB.score(ft_test, test_species)))
    print('Decision tree accuracu with {} mer: {}'.format(kmers[j], 100*Tree.score(ft_test, test_species)))
    print('Random forest accuracu with {} mer: {}'.format(kmers[j], 100*RF.score(ft_test, test_species)))
    print('Gradient boosted accuracu with {} mer: {}'.format(kmers[j], 100*GB.score(ft_test, test_species)))
    print('Logistic Reg accuracu with {} mer: {}'.format(kmers[j], 100*LR.score(ft_test, test_species)))

# ft_test=create_feature_table(test_sequences,8)


#accuracy with test data

#correctly predicted species and number of correct predictions
# correct_species=[]
# correct=0
# for i in range(len(test_species)):
#     if predicted[i]==test_species[i]:
#         correct_species.append((test_species[i]))
#         correct+=1

# print(correct)

# np.unique(np.asarray(correct_species))

# #accuracy with train data
# NB.score(ft_3, species)

# len(np.unique(np.asarray(correct_species)))