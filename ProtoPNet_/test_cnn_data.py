#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings

from torch.nn.modules.activation import MultiheadAttention
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# In[2]:


df=pd.read_csv("./datasets/mitofish.12S.Sep2021.tsv", sep='\t', header=0)


# In[3]:


df=df[['Family','Sequence']]


# In[5]:



df['Sequence_length']= df['Sequence'].apply(lambda x: len(x))


# In[7]:


df=df.loc[df['Sequence_length'] <= 500]


# In[9]:


def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


# In[10]:



df['words'] = df.apply(lambda x: getKmers(x['Sequence']), axis=1)


# In[12]:


df['sentence']=df['words'].apply(lambda x: ' '.join(x))


# In[15]:


corpus=list(df['sentence'])
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(corpus)
features=vectorizer.get_feature_names()


# In[18]:


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(features)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# In[20]:


feature_encoder_mapper = dict(zip(features, onehot_encoded))


# In[21]:


def dna_mapper(sequence,mapper):
    encoded_dna=[]
    for feature in sequence:
        encoded_dna.append(mapper[feature])
    return encoded_dna


# In[25]:


zero_arr=np.zeros((344,), dtype=int)


# In[27]:


max_seq_length=max(df.Sequence_length)


# In[29]:


entire_dataset=[]

for i in range(len(df)):
    my_seq=df['words'].iloc[i]
    encoded_seq=dna_mapper(my_seq,feature_encoder_mapper)
    if len(encoded_seq)<max_seq_length:
        for i in range(len(encoded_seq),max_seq_length):
            encoded_seq.append(zero_arr)
    entire_dataset.append(encoded_seq)


# In[31]:


from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from model import My_CNN  

X = entire_dataset
y = df['Family']

# Choose your test size to split between training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
train_batch_size = 100
test_batch_size = 100
num_epochs = 2
print(locals())
'''
print(locals())

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

train_set = TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, 
    shuffle=False, num_workers=4, pin_memory=False)

test_set = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, 
    shuffle=False, num_workers=4, pin_memory=False)

n_classes = df['Family'].nunique()
cnn = My_CNN(n_classes=n_classes)
optimizer = torch.optim.Adam
optimizer_specs = \
[{'params': cnn.conv1.parameters(), 'lr':1, 'weight_decay': 1e-3},
 {'params': cnn.hiddens.parameters(), 'lr':1},
 {'params': cnn.last_layer.parameters(), 'lr':1},
]
optimizer = torch.optim.Adam(optimizer_specs)

for epoch in range(num_epochs):
    n_correct = 0
    n_examples = 0

    for sample, label in train_loader:
        print(sample.shape)
        output = cnn(sample)

        _, predicted = torch.max(output.data, 1)
        n_examples += label.size(0)
        n_correct += [predicted == label].sum().item()

        loss = torch.nn.functional.cross_entropy(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch {} train accuracy: {}%".format(epoch, 100 * n_correct/n_examples))
    n_correct = 0
    n_examples = 0

    for sample, label in test_loader:
        print(sample.shape)
        output = cnn(sample)

        _, predicted = torch.max(output.data, 1)
        n_examples += label.size(0)
        n_correct += [predicted == label].sum().item()
    print("Epoch {} test accuracy: {}%".format(epoch, 100 * n_correct/n_examples))
'''