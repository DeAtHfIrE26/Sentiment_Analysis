from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import speech_recognition as sr
import pyttsx3

# Expanded training dataset
training_data = [
    ("I love this product", "positive"),
    ("This is the best purchase I've ever made", "positive"),
    ("I'm so happy with this", "positive"),
    ("The product quality is excellent", "positive"),
    ("The service was quick and good", "positive"),
    ("I am very happy with the service", "positive"),
    ("Just now I got the rewards for my work", "positive"),
    ("The delivery was quick and efficient", "positive"),
    ("This product exceeded my expectations", "positive"),
    ("I hate this product", "negative"),
    ("This is the worst purchase I've ever made", "negative"),
    ("I'm so disappointed with this", "negative"),
    ("The product quality is terrible", "negative"),
    ("The service was slow and bad", "negative"),
    ("I am very unhappy with the service", "negative"),
    ("Just now I faced issues with my order", "negative"),
    ("The delivery was delayed", "negative"),
    ("This product did not meet my expectations", "negative"),
    ("It's okay, nothing special", "neutral"),
    ("Not bad, could be better", "neutral"),
    ("It's an average product", "neutral"),
    ("The delivery was quick, but the product is not great", "neutral"),
    ("I need help with this", "neutral"),
    ("How can I add this to my resume?", "neutral"),
    ("I have mixed feelings about this product", "neutral"),
    ("The service was neither good nor bad", "neutral"),
    ("The product is neither good nor bad", "neutral"),
    ("I don't feel strongly about this", "neutral"),
    # New training data related to education
    ("The lecture was very informative", "positive"),
    ("I enjoyed the online course", "positive"),
    ("I don't understand this topic", "neutral"),
    ("The exam was very difficult", "negative"),
    ("The professor is very helpful", "positive"),
    # New training data related to news and current affairs
    ("The recent policy changes are beneficial", "positive"),
    ("I am concerned about the economic downturn", "negative"),
    ("The news coverage was biased", "negative"),
    ("I am indifferent to the election results", "neutral"),
    ("The new technology has great potential", "positive"),
    # New training data related to technology
    ("This software is very user-friendly", "positive"),
    ("I am experiencing issues with this app", "negative"),
    ("The update improved performance significantly", "positive"),
    ("I am not impressed with the new features", "negative"),
    ("The interface is clean and intuitive", "positive"),
    # New training data related to general knowledge
    ("I learned something new today", "positive"),
    ("I find this topic boring", "negative"),
    ("The book was very enlightening", "positive"),
    ("I don't have any opinion on this", "neutral"),
    ("The documentary was well-made", "positive")
]


# Preprocess and vectorize the text data
def preprocess_data(training_data):
    texts, labels = zip(*training_data)
    return texts, labels

texts, labels = preprocess_data(training_data)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(texts, labels)

# Function to predict sentiment
def predict_sentiment(text):
    return model.predict([text])[0]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get("message")
    sentiment = predict_sentiment(user_input)
    response = {
        'message': user_input,
        'sentiment': sentiment
    }
    return jsonify(response)

# Voice assistant functions
recognizer = sr.Recognizer()

@app.route('/voice_input', methods=['GET'])
def voice_input():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            sentiment = predict_sentiment(command)
            response = {
                'message': command,
                'sentiment': sentiment
            }
            return jsonify(response)
        except sr.UnknownValueError:
            return jsonify({"error": "Sorry, I did not understand that."})
        except sr.RequestError:
            return jsonify({"error": "Could not request results from the speech recognition service."})

if __name__ == "__main__":
    app.run(debug=True)