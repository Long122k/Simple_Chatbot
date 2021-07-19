import nltk
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return PorterStemmer().stem(word.lower())

def bag_words(tokenize_sentences, words):
    return 0

sen = "Hello how are you?"
list_sen = ["interested", "insteresting"]

# print(tokenize(sen))
print([stem(i) for i in list_sen])