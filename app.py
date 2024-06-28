from flask import Flask, request, jsonify
import pandas as pd
import json
import re
import matplotlib
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from markupsafe import Markup, escape
from dotenv import load_dotenv
import os
from waitress import serve

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
DEBUG = os.getenv('DEBUG')
CLIENT_URL = os.getenv('CLIENT_URL')

matplotlib.use('Agg')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = SECRET_KEY

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load datasets
with open('data/album-song-lyrics.json') as f:
    album_song_lyrics = json.load(f)

# Function to preprocess lyrics by stemming
def preprocess_lyrics(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    return ' '.join(stemmer.stem(word) for word in words if word.lower() not in stop_words)

data = []
for album in album_song_lyrics:
    album_code = album['Code']
    album_title = album['Title']
    album_year = album['Year']
    for song in album['Songs']:
        track_number = song['TrackNumber']
        track_title = song['Title']
        for lyric in song['Lyrics']:
            data.append({
                'AlbumCode': album_code,
                'AlbumTitle': album_title,
                'TrackNumber': track_number,
                'TrackTitle': track_title,
                'AlbumYear': album_year,
                'Order': lyric['Order'],
                'SongPart': lyric['SongPart'],
                'Text': lyric['Text'],
                'PreprocessedText': preprocess_lyrics(lyric['Text'])  # Preprocess the text
            })

lyrics_df = pd.DataFrame(data)

album_css_classes = {
    "Taylor Swift": "taylor-swift",
    "Fearless": "fearless",
    "Speak Now": "speak-now",
    "Red": "red",
    "1989": "nineteen-eighty-nine",
    "Reputation": "reputation",
    "Lover": "lover",
    "Folklore": "folklore",
    "Evermore": "evermore",
    "Midnights": "midnights",
    "The Tortured Poets Department": "the-tortured-poets-department"
}

# Function to highlight and make words clickable
def highlight_and_make_clickable(text, query, stemmed_word):
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
        'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
        'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'can', 'will', 'just', 'don', 'should', 'now'
    }

    def make_word_clickable(word):
        if word.lower() not in stopwords:
            return f'<a href="{CLIENT_URL}/lyricmatcher?query={word}" class="clickable-word" onclick="searchWord(\'{escape(word)}\')">{escape(word)}</a>'
        return f'<span class="non-clickable-word">{escape(word)}</span>'

    word_regex = re.compile(rf'\b{stemmed_word}\w*\b', re.IGNORECASE)
    highlighted_text = ' '.join([
        f'<span class="highlight">{make_word_clickable(word)}</span>'
        if word_regex.match(stemmer.stem(re.sub(r'[^\w\s]', '', word))) else make_word_clickable(word)
        for word in text.split()
    ])

    return Markup(highlighted_text)

# Function to find and group lyrics based on a stemmed search word
def find_lyrics_by_stemmed_word(word):
    stemmed_word = stemmer.stem(word)
    word_regex = re.compile(rf'\b{stemmed_word}\w*\b', re.IGNORECASE)
    matches = lyrics_df[lyrics_df['PreprocessedText'].str.contains(word_regex)].copy()
    matches['HighlightedText'] = matches['Text'].apply(lambda x: highlight_and_make_clickable(x, word, stemmed_word))

    grouped = matches.groupby(['AlbumTitle', 'TrackTitle', 'AlbumYear']).agg({
        'Text': lambda x: list(x),
        'HighlightedText': lambda x: list(x)
    }).reset_index()
    grouped['AlbumClass'] = grouped['AlbumTitle'].apply(lambda x: album_css_classes.get(x, ''))
    grouped['Lyrics'] = grouped['Text'].apply(lambda x: ' '.join(x))
    return grouped

@app.route('/similar_lyrics', methods=['GET'])
def similiar_lyrics():
    query = request.args.get('query')
    query = re.sub(r'[^\w\s]', '', query) if query else ''
    results = []
    if query:
        results = find_lyrics_by_stemmed_word(query)
        results_dict = results.to_dict('records')
        grouped_results = {}
        for record in results_dict:
            album = record['AlbumTitle']
            album_class = record['AlbumClass']
            if album not in grouped_results:
                grouped_results[album] = { 'class': album_class, 'tracks': [] }
            grouped_results[album]['tracks'].append(record)

        album_order = [
            "Taylor Swift",
            "Fearless",
            "Speak Now",
            "Red",
            "1989",
            "Reputation",
            "Lover",
            "Folklore",
            "Evermore",
            "Midnights",
            "The Tortured Poet's Department",
        ]
        grouped_results = sorted(grouped_results.items(),
                    key=lambda x: album_order.index(x[0]) if x[0] in album_order else len(album_order)
        )
        print(f"Results for query '{query}': {results_dict}")
    else:
        print("No query provided")
        grouped_results = []

    return jsonify(grouped_results)

if __name__ == '__main__':
    if DEBUG:
        app.run(debug=True)
    else:
      serve(app, port=10081)
