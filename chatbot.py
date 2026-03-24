import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')

# Read data file
with open('data.txt', 'r', encoding='utf8') as f:
    corpus = f.read().lower()

sent_tokens = nltk.sent_tokenize(corpus)

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "hey", "greetings")
GREETING_RESPONSES = ["Hello!", "Hi there!", "Hey!", "How can I help you?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# Text preprocessing
def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Chatbot response
def chatbot_response(user_input):
    sent_tokens.append(user_input)
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    tfidf = vectorizer.fit_transform(sent_tokens)
    similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    if score == 0:
        return "Sorry, I don't understand that."
    else:
        return sent_tokens[idx]

# Main loop
print("AI Chatbot: Hello! Type 'bye' to exit.")

while True:
    user_input = input("You: ").lower()

    if user_input == 'bye':
        print("AI Chatbot: Goodbye! 👋")
        break

    greet = greeting(user_input)
    if greet:
        print("AI Chatbot:", greet)
    else:
        print("AI Chatbot:", chatbot_response(user_input))
