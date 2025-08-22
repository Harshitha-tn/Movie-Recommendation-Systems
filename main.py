# main.py
from emotion_detector import detect_emotion
from recommender import recommend_movies, get_available_languages

def main():
    print("🎬 Welcome to Mood2Movie!")
    user_input = input("How are you feeling today?\n> ")

    # Display available languages
    available_languages = get_available_languages()
    print("\nAvailable languages:", ", ".join(available_languages))

    language = input("Choose your preferred movie language [or press Enter to see all]:\n> ").strip()

    if language and language.lower() not in [lang.lower() for lang in available_languages]:
        print(f"\n⚠️ '{language}' is not in the available list. Showing recommendations from all languages instead.")
        language = None

    # Detect emotion
    emotion = detect_emotion(user_input)
    print(f"\nDetected emotion: **{emotion.upper()}**")

    # Show recommendations
    print("\n🎥 Recommended Movies:\n")
    recommendations = recommend_movies(emotion, language=language if language else None)
    if recommendations.empty:
        print("😞 Sorry, no movies found for that mood and language.")
    else:
        for idx, row in recommendations.iterrows():
            print(f"- {row['title']} ({row['genre']} - {row['language']}) - {row['rating']}\n  \"{row['review']}\"\n")

if __name__ == "__main__":
    main()
