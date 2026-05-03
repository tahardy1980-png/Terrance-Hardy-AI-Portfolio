from preprocess import preprocess_text
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Sample topic keywords (light integration of your topic model concept)
TOPIC_KEYWORDS = {
    "Technology / AI": ["ai", "machine", "learning", "data", "model", "tech"],
    "Politics": ["election", "government", "policy", "vote", "political"],
    "Sports": ["game", "team", "score", "championship", "season"]
}

def detect_topic(tokens):
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}

    for token in tokens:
        for topic, keywords in TOPIC_KEYWORDS.items():
            if token in keywords:
                scores[topic] += 1

    best_topic = max(scores, key=scores.get)

    return best_topic if scores[best_topic] > 0 else "General"

def analyze_text(text):
    # Step 1: preprocess
    tokens = preprocess_text(text)

    # Step 2: sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)

    if sentiment_score["compound"] >= 0.05:
        sentiment = "Positive"
    elif sentiment_score["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Step 3: topic detection
    topic = detect_topic(tokens)

    # Step 4: output report
    return {
        "clean_text": tokens,
        "sentiment": sentiment,
        "scores": sentiment_score,
        "topic": topic
    }

if __name__ == "__main__":
    sample_text = "AI and machine learning are transforming technology and data science"

    result = analyze_text(sample_text)

    print("\n--- NEWSBOT ANALYSIS ---")
    print("Clean Text:", result["clean_text"])
    print("Sentiment:", result["sentiment"])
    print("Topic:", result["topic"])
    print("Scores:", result["scores"])
