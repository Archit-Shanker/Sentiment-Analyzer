import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')


def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    return blob.sentiment.polarity


def analyze_sentiment_vader(text):
    """Analyze sentiment using NLTK's VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound']


def sentiment_analyzer(text):
    """Combine TextBlob and VADER results"""
    textblob_score = analyze_sentiment_textblob(text)
    vader_score = analyze_sentiment_vader(text)
    average_score = (textblob_score + vader_score) / 2

    if average_score >= 0.05:
        return "Positive"
    elif average_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
print(sentiment_analyzer(""))