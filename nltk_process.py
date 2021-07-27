import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
# nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):#interested, interesting -> interest
    return PorterStemmer().stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    sentence = ["Good", "morning"]
    all_words =    ["hello", "hi", "good", "thank", "you", "morning"]
    bag_of_words = [   0,      0,      1,      0,      0,      1    ]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype= np.float32)
    for idx,word in enumerate(words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag



