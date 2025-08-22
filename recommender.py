# recommender.py
import pandas as pd

# Load dataset and clean column names
df = pd.read_csv('movie_dataset.csv')
df.columns = df.columns.str.strip()  # Strip spaces from headers

def get_available_languages():
    return sorted(df['language'].dropna().unique())

def recommend_movies(emotion, language=None, top_n=5):
    # Emotion-genre mapping
    genre_filter = {
        'happy': ['Comedy', 'Romance', 'Musical'],
        'sad': ['Drama', 'Romance', 'Biography'],
        'lonely': ['Drama', 'Romance', 'Adventure'],
        'fear': ['Horror', 'Thriller', 'Mystery'],
        'surprise': ['Mystery', 'Thriller', 'Sci-Fi']
    }

    matches = df[df['emotion'] == emotion]

    # Filter by language (if user specified)
    if language:
        matches = matches[matches['language'].str.lower() == language.lower()]

    # Filter by emotion-appropriate genres
    if emotion in genre_filter:
        matches = matches[matches['genre'].isin(genre_filter[emotion])]

    top_rated = matches.sort_values(by='rating', ascending=False)
    return top_rated[['title', 'genre', 'language', 'rating', 'review']].head(top_n)
