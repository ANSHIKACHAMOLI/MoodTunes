import os
import numpy as np
import pandas as pd
from zipfile import ZipFile
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib
from flask import Flask, request, render_template, redirect, url_for, session

# Download NLTK data
nltk.download('stopwords')

# Set up Spotify API credentials
SPOTIPY_CLIENT_ID = '9a0dc7b9c0e045fe91fb42f3df9b89d8'
SPOTIPY_CLIENT_SECRET = 'b3231df9cb4244048e8049a07a99bd4d'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# Authenticate with Spotify API
sp_oauth = SpotifyOAuth(client_id='9a0dc7b9c0e045fe91fb42f3df9b89d8',
                        client_secret='b3231df9cb4244048e8049a07a99bd4d',
                        redirect_uri='http://localhost:8888/callback',
                        scope='user-library-read')

# Paths for processed data
processed_data_path = 'processed_data.pkl'
vectorizer_path = 'count_vectorizer.pkl'
classifier_path = 'emotion_classifier_model.pkl'

# Preprocess text data
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

if not os.path.exists(processed_data_path) or not os.path.exists(vectorizer_path) or not os.path.exists(classifier_path):
    # Load dataset
    dataset = 'archive (3).zip'
    with ZipFile(dataset, 'r') as zip:
        zip.extractall()
        print("Dataset is extracted")

    ds = pd.read_csv('text.csv')

    # Process the dataset
    corpus = [preprocess_text(text) for text in ds['text']]
    print("Preprocessing done.")

    # Vectorize text data
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = ds.iloc[:, -1].values
    print("Vectorizing is done.")

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.02, random_state=2)
    print("Data is split.")

    # Train emotion classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=2)
    classifier.fit(X_train, Y_train)

    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')

    # Save the processed data, classifier, and vectorizer
    joblib.dump((X_train, X_test, Y_train, Y_test), processed_data_path)
    joblib.dump(cv, vectorizer_path)
    joblib.dump(classifier, classifier_path)
else:
    # Load the processed data, classifier, and vectorizer
    X_train, X_test, Y_train, Y_test = joblib.load(processed_data_path)
    cv = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    print("Processed data, vectorizer, and classifier loaded.")

# Define functions for predicting emotion and recommending songs
def predict_emotion(text):
    text_processed = preprocess_text(text)
    text_vectorized = cv.transform([text_processed])
    prediction = classifier.predict(text_vectorized)
    emotion_labels = ['Sad', 'Happy', 'Fear', 'Angry', 'Fear', 'Disgust']
    emotion = emotion_labels[prediction[0]]
    return emotion

def get_bollywood_song_recommendations(emotion):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='9a0dc7b9c0e045fe91fb42f3df9b89d8',
                                                   client_secret='b3231df9cb4244048e8049a07a99bd4d',
                                                   redirect_uri='http://localhost:8888/callback'))
    emotion_to_keywords = {
        'Happy': 'Bollywood Top Happy Songs',
        'Sad': 'Bollywood Cheerful Lofi Songs',
        'Angry': 'Bollywood Calm Lofi',
        'Fear': 'Bollywood Bhagwan Bhakti',
        'Surprise': 'Bollywood Surprise',
        'Disgust': 'Bollywood Motivational'
    }
    keywords = emotion_to_keywords.get(emotion, 'Bollywood')
    results = sp.search(q=keywords, type='track', limit=5)
    songs = [
        {
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "url": track['external_urls']['spotify']
        }
        for track in results['tracks']['items']
    ]
    return songs

# Create Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') # Replace with a real secret key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('home'))

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    emotion = predict_emotion(user_input)
    recommended_songs = get_bollywood_song_recommendations(emotion)

    if emotion == 'Happy':
        message = "Let me suggest you some songs that align with your mood."
    elif emotion == 'Sad':
        message = "I observe you are not feeling your best today. Let me suggest you some songs to cheer you up."
    elif emotion == 'Angry':
        message = "Oh you are brewing today ! Take a chill pill . Let me suggest some songs to help calm yourself."
    elif emotion == 'Fear':
        message = "Don't be scared . Believe in almighty . Let me suggest some songs to make you fight your fear back. "
    else:
        message = "Let me suggest some songs for you !"

    return render_template('result.html', emotion=emotion, songs=recommended_songs, message=message)

if __name__ == '__main__':
    app.run(debug=True, port=8888)


