### Mood2Movie — Enhanced Version (Up to Feature 4)
# Includes: Transformer-based Emotion Detection, Better Filtering, OMDb API Poster Fetch, Multilingual Input, and Movie Metadata (Platform, Cast, Crew)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
from langdetect import detect
from googletrans import Translator
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Setup rich console
console = Console()

# Load movie dataset
df = pd.read_csv(r"C:\Users\Harshitha TN\AppData\Local\Programs\Python\Python312\movie recommendation system\movie_dataset.csv")

# Add new metadata columns if missing
for col in ["platform", "director", "producer", "hero", "heroine"]:
    if col not in df.columns:
        df[col] = ""

# Load transformer model for emotion detection
MODEL = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Translator for multilingual input
translator = Translator()

# Emotion-to-Genre Mapping
genre_filter = {
    'happy': ['Comedy', 'Romance', 'Musical'],
    'sad': ['Drama', 'Romance', 'Biography'],
    'lonely': ['Drama', 'Romance', 'Adventure'],
    'fear': ['Comedy', 'Sci-Fi', 'Mystery'],
    'surprise': ['Mystery', 'Thriller', 'Horror']
}

# Normalize model emotion labels to expected ones
emotion_alias = {
    'joy': 'happy',
    'happiness': 'happy',
    'anger': 'fear',
    'sadness': 'sad',
    'fear': 'fear',
    'surprise': 'surprise',
    'lonely': 'lonely',
    'neutral': 'happy'
}

def detect_emotion(text):
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, dest='en').text
    except:
        pass
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    ranking = scores.argsort()[::-1]
    top_emotion = model.config.id2label[ranking[0]].lower()
    return emotion_alias.get(top_emotion, top_emotion)

def fetch_omdb_metadata(title, year=None):
    api_key = "ca21149d"
    try:
        url = f"http://www.omdbapi.com/?t={title}&type=movie&apikey={api_key}"
        if year:
            url += f"&y={year}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True":
                return {
                    "Poster":data.get("Poster", "N/A"),
                    "Director":data.get("Director", ""),
                    "Writer": data.get("Writer", ""),
                    "Actors": data.get("Actors", "")
                }
    except:
        pass
    return {"Poster": "N/A", "Director": "", "Writer": "", "Actors": ""}


def recommend_movies(emotion, language=None, top_n=5):
    matches = df[df['emotion'].str.lower() == emotion.lower()]

    if language:
        matches = matches[matches['language'].str.lower() == language.lower()]

    if emotion in genre_filter:
        matches = matches[matches['genre'].isin(genre_filter[emotion])]

    if matches.empty and emotion in genre_filter:
        matches = df[df['emotion'].str.lower() == emotion.lower()]
        if language:
            matches = matches[matches['language'].str.lower() == language.lower()]

    top_rated = matches.sort_values(by='rating', ascending=False).head(top_n)
    return top_rated

def main():
    console.print("\n🎬 [bold cyan]Welcome to MoodMatcher![/bold cyan]")
    feeling = input("How are you feeling today?\n> ")
    emotion = detect_emotion(feeling)
    console.print(f"\nDetected emotion: [bold yellow]{emotion.upper()}[/bold yellow]")

    languages = sorted(df['language'].dropna().unique())
    console.print(f"\nAvailable languages: {', '.join(languages)}")
    user_lang = input("Choose your preferred movie language [or press Enter to see all]:\n> ")
    if user_lang.strip() == "":
        user_lang = None

    recommendations = recommend_movies(emotion, user_lang)

    if recommendations.empty:
        console.print("😞 [bold red]Sorry, we couldn't find any movies matching that mood and language.[/bold red]")
    else:
        console.print("\n [bold green]Recommended Movies:[/bold green]\n")
        for _, row in recommendations.iterrows():
            meta = fetch_omdb_metadata(row['title'])
            actors = meta['Actors'].split(", ")
            hero = actors[0] if actors else row['hero']
            heroine = actors[1] if len(actors) > 1 else row['heroine']
            platform = row['platform'] or "Netflix/Prime/Other"
            panel = Panel.fit(
                f"[bold]{row['title']}[/bold] ({row['genre']} - {row['language']})\n"
                f"Rating: {row['rating']}\n"
                f"Review: {row['review']}\n"
                f"Director:{meta['Director']}\nHero: {hero}, Heroine: {heroine}\n"
                f"Poster:{meta['Poster']}",
                title=row['title'],
                border_style="cyan"
            )
            console.print(panel)

if __name__ == '__main__':
    main()
