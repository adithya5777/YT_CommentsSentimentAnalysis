import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob

df = pd.read_csv("Full Comments.csv")

sia = SentimentIntensityAnalyzer()

nrc_lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                          sep='\t', names=["word", "emotion", "association"])

# Preprocessing
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')


def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())

    filtered_tokens = [lemmatizer.lemmatize(
        token) for token in tokens if token not in stopwords]

    return ' '.join(filtered_tokens)

# Function to get the dominant emotion for a comment
def get_dominant_emotion(comment):
    emotion_scores = {emotion: 0 for emotion in set(nrc_lexicon["emotion"])}

    words = nltk.word_tokenize(comment)
    for word in words:
        word_emotion = nrc_lexicon[(nrc_lexicon["word"] == word) & (
            nrc_lexicon["association"] == 1)]
        if not word_emotion.empty:
            for emotion in word_emotion["emotion"]:
                emotion_scores[emotion] += 1

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    return dominant_emotion


df["ProcessedComment"] = df["Comment"].apply(preprocess_text)
df["Emotion"] = df["ProcessedComment"].apply(get_dominant_emotion)

# Sentiment analysis using TextBlob for improved negation handling


def get_sentiment(comment):
    blob = TextBlob(comment)
    sentiment = blob.sentiment.polarity
    return "positive" if sentiment >= 0 else "negative"


df["Sentiment"] = df["Comment"].apply(get_sentiment)

# Store comments with different emotions in separate CSV files
unique_emotions = set(df["Emotion"])
for emotion in unique_emotions:
    filename = f"{emotion}_comments.csv"
    emotion_comments = df[df["Emotion"] == emotion]
    emotion_comments.to_csv(filename, index=False)
    print(f"Comments with {emotion} emotion saved in {filename}")

# Calculate the sum of comments in the filtered CSV files
total_comments_filtered = 0
for emotion in unique_emotions:
    filename = f"{emotion}_comments.csv"
    emotion_comments = pd.read_csv(filename)
    total_comments_filtered += len(emotion_comments)

