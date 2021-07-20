from nltk_process import tokenize, stem
import json

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

print(words)