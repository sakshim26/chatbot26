import os
import json
import random
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the vectorizer and the model
with open("vectorizer.pkl", "rb") as file:
    vectorizer = joblib.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = joblib.load(file)

with open("chatbot_model.pkl", "rb") as file:
    clf = joblib.load(file)

# Load intents file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Define the chatbot function
def chatbot(input_text):
    input_text_vectorized = vectorizer.transform([input_text])  # Transform user input
    predicted_class_index = clf.predict(input_text_vectorized)[0]  # Predict tag index
    predicted_tag = label_encoder.inverse_transform([predicted_class_index])[0]  # Map to tag

    for intent in intents:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])  # Select a random response
            return response

# Streamlit application
def main():
    st.title("AI Chatbot")
    st.write("Start a conversation by typing below. Type 'bye' to exit the chat.")

    user_input = st.text_input("You:", placeholder="Type your message here...")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=150, max_chars=None, key="chatbot_response")

        if response.lower() in ["goodbye", "bye"]:
            st.write("Thank you for chatting! Have a great day!")
            st.stop()

if __name__ == "__main__":
    main()
