# emotion_detector.py
from textblob import TextBlob

def detect_emotion(user_input):
    text = user_input.lower()
    if any(word in text for word in ['happy', 'joy', 'glad']):
        return 'happy'
    elif any(word in text for word in ['sad', 'down', 'depressed']):
        return 'sad'
    elif any(word in text for word in ['fear', 'anxious', 'scared']):
        return 'fear'
    elif any(word in text for word in ['surprised', 'unexpected', 'shock']):
        return 'surprise'
    elif any(word in text for word in ['lonely', 'alone', 'isolated']):
        return 'lonely'
    else:
        # Fallback: use sentiment polarity
        sentiment = TextBlob(user_input).sentiment.polarity
        if sentiment > 0.2:
            return 'happy'
        elif sentiment < -0.2:
            return 'sad'
        else:
            return 'lonely'
