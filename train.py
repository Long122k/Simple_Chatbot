from nltk.util import transitive_closure
from nltk_process import bag_words, tokenize, stem
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json') as f:
    intents = json.load(f)

tags = []
words = []
tag_word = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern)
        words.extend(word)
        tag_word.append((tag, word))

ignore_word = ['?', ',', '.', '!']
words = [stem(word) for word in words if word not in ignore_word]
words = sorted(set(words))
tags = sorted(set(tags))


#set data train
X_train = []
Y_train = []

for (pattern_sentence, tag) in tag_word:
    bag = bag_words(pattern_sentence, words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)#cross entropy

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 8
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print(X_train)