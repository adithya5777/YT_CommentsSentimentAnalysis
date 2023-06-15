import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# # Download necessary resources from NLTK
# nltk.download('punkt')
# nltk.download('vader_lexicon')

# Load the comments from the CSV file
df = pd.read_csv("Full Comments.csv")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the NRC Emotion Lexicon
nrc_lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                          sep='\t', names=["word", "emotion", "association"])

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


# Perform emotion analysis on each comment
df["Emotion"] = df["Comment"].apply(get_dominant_emotion)

# Store comments of different emotions in separate CSV files
emotions = df["Emotion"].unique()
for emotion in emotions:
    filtered_comments = df[df["Emotion"] == emotion]
    filename = f"{emotion}_comments.csv"
    filtered_comments.to_csv(filename, index=False)
    print(f"Comments with {emotion} emotion saved in {filename}")
