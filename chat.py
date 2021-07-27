import torch
import torch.nn as nn
import json
import random
from model import NeuralNet
from nltk_process import bag_of_words, tokenize

devide = torch.device('cuda' if torch.cuda.is_available() else "cpu")

with open('intents.json') as f:
    intents = json.load(f)

file = "data.pth"
data = torch.load(file)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(devide)
model.load_state_dict(model_state)
model.eval()

bot_name = "My bot"
print("Print 'quit' if you want to exit")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(devide)

    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim= 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents'] :
            if tag == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: Sorry, i do not understand')