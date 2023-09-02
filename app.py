#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load the model and tokenizer
MODEL_NAME = "Ahmedhisham/social_bias_Bert"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

st.title("DistillBERT social Classifier")

# Take user input
user_input = st.text_input("Enter text to classify:")

if user_input:
    # Tokenize and predict
    encoded_input = tokenizer.encode_plus(
        user_input,
        truncation=True,
        padding='max_length',
        max_length=128, 
        return_tensors='tf'
    )
    output = model(encoded_input['input_ids'])
    logits = output.logits
    predicted_label_index = logits.numpy().argmax(1)[0]
    
    label_map = {0: 'race', 1: 'gender', 2: 'culture', 3: 'victim'}
    predicted_label = label_map[predicted_label_index]
    
    st.write(f"Predicted category: {predicted_label}")

