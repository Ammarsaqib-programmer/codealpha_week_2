
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# FAQ DATASET
# -----------------------------
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "You can return any product within 30 days of purchase."
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your email."
    },
    {
        "question": "Do you offer cash on delivery?",
        "answer": "Yes, we offer cash on delivery in selected cities."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact customer support at support@example.com."
    },
    {
        "question": "What payment methods are accepted?",
        "answer": "We accept credit cards, debit cards, PayPal, and bank transfers."
    }
]

# -----------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Preprocess FAQ questions
faq_questions = [preprocess_text(faq["question"]) for faq in faqs]

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)

# -----------------------------
# FIND BEST MATCH FUNCTION
# -----------------------------
def get_best_answer(user_question):
    # Preprocess user question
    processed_question = preprocess_text(user_question)

    # Convert to vector
    user_vector = vectorizer.transform([processed_question])

    # Compute cosine similarity
    similarity = cosine_similarity(user_vector, faq_vectors)

    # Get best match index
    best_match_index = similarity.argmax()

    # Get similarity score
    score = similarity[0][best_match_index]

    # Threshold check
    if score < 0.2:
        return "Sorry, I could not understand your question."

    return faqs[best_match_index]["answer"]

# -----------------------------
# CHATBOT UI USING TKINTER
# -----------------------------
def send_message():
    user_message = user_input.get()

    if user_message.strip() == "":
        return

    # Display user message
    chat_area.insert(tk.END, "You: " + user_message + "\n")

    # Get bot response
    bot_response = get_best_answer(user_message)

    # Display bot response
    chat_area.insert(tk.END, "Bot: " + bot_response + "\n\n")

    # Clear input box
    user_input.delete(0, tk.END)

# Create main window
window = tk.Tk()
window.title("FAQ Chatbot")
window.geometry("500x500")

# Chat display area
chat_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=20)
chat_area.pack(padx=10, pady=10)

# Input field
user_input = tk.Entry(window, width=40)
user_input.pack(side=tk.LEFT, padx=10, pady=10)

# Send button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack(side=tk.LEFT)

# Run application
window.mainloop()