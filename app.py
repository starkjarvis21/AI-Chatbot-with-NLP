# -*- coding: utf-8 -*-
# Import necessary libraries
import random
import numpy as np
import pickle
import json
import streamlit as st
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model("chatbot_model.h5")
data_file = open("intents.json").read()
intents = json.loads(data_file)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Define chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize and generate bag of words
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    # Predict the class of the sentence
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

# Streamlit App
def chatbot_response(msg):
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
    return res

# Streamlit UI
st.title("AI Chatbot")

# Input field for the user message
user_input = st.text_input("Enter your message:")

# When the user submits a message, process it
if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input)
        st.write(f"Chatbot: {response}")
