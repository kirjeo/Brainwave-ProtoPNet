#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# 
# 
# 
# #### Helpful sites
# 1) https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/ <br>
# 2) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# # 1. Imports

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# # 2. Converting the tsv file to dataframe

df=pd.read_csv("./datasets/mitofish.12S.Sep2021.tsv", sep='\t', header=0)

df[:50]


# # 3. Only Selecting the Sequence and Species from the Dataset

filter_df=df[['Family','Sequence']]

filter_df.head()


# ### 3.1 Adding the Length of DNA sequence column to the Dataframe

filter_df['Sequence_length']= df['Sequence'].apply(lambda x: len(x))

filter_df.head()


# ### 3.3 Total number of Family Classes
unique_classes = df['Family'].nunique()
print("The total number of samples are: ", len(filter_df))
print("The total number of unique classes are ", unique_classes)

filter_df['Family'].value_counts()


# ### 4) Translating DNA sequence into sequence of world
# 
def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

'''
filter_df['words'] = filter_df.apply(lambda x: getKmers(x['Sequence']), axis=1)



filter_df.head()


filter_df['sentence']=filter_df['words'].apply(lambda x: ' '.join(x))



filter_df.head()


# ### 5) Convert a collection of text documents to a matrix of token counts.



corpus=list(filter_df['sentence'])



vectorizer=CountVectorizer()
X=vectorizer.fit_transform(corpus)


# ### 5.1 Visualizing all our feature array


features=vectorizer.get_feature_names()

print("The length of the feature is: ", len(features))


# ## Second Technique
# ### 5.3 Creating a one-hot encoder for each and every feature array and mapping the DNA sequence with it


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(features)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# ### 5.4 Creating a python dictionary for mapping the feature with its respective one hot encoder

feature_encoder_mapper = dict(zip(features, onehot_encoded))


# ### 5.5 Mapping dna sequence with its respective one hot encoder

def dna_mapper(sequence,mapper):
    encoded_dna=[]
    for feature in sequence:
        encoded_dna.append(mapper[feature])
    return encoded_dna



my_seq=filter_df['words'][0]

encoded_seq=dna_mapper(my_seq,feature_encoder_mapper)


# ## 5.6 Only selecting sequence that have senquence length less than 3,000 



new_df=filter_df.loc[filter_df['Sequence_length'] <= 500]

new_df.head()




unique_classes = new_df['Family'].nunique()
print("The total number of samples are: ", len(new_df))
print("The total number of unique classes are ", unique_classes)


# ## Creating a array of size 850 that has all zeros 



zero_arr=np.zeros((850,), dtype=int)


# ## Replacing a zero array if the length of the sequence does not meet the maximum length 



max_seq_length=max(new_df.Sequence_length)




entire_dataset=[]

for i in range(len(new_df)):
    my_seq=new_df['words'].iloc[i]
    encoded_seq=dna_mapper(my_seq,feature_encoder_mapper)
    if len(encoded_seq)<max_seq_length:
        for i in range(len(encoded_seq),max_seq_length):
            encoded_seq.append(zero_arr)
    entire_dataset.append(encoded_seq)
            


# ## Implementing train test split



from sklearn.model_selection import train_test_split

X = entire_dataset
y = new_df['Family']

# Choose your test size to split between training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)'''
