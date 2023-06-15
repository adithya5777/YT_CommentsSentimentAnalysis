import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the comments from the CSV file
df = pd.read_csv("Full Comments.csv")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the NRC Emotion Lexicon
nrc_lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                          names=["word", "emotion", "association"], sep='\t')

# Create a dictionary to map words to emotions
word_emotion_map = {}
for _, row in nrc_lexicon.iterrows():
    word = row["word"]
    emotion = row["emotion"]
    association = row["association"]
    if word in word_emotion_map:
        word_emotion_map[word].append((emotion, association))
    else:
        word_emotion_map[word] = [(emotion, association)]

# Function to get the dominant emotion for a comment


def get_dominant_emotion(comment):
    emotion_scores = {emotion: 0 for emotion in set(nrc_lexicon["emotion"])}

    words = nltk.word_tokenize(comment)
    for word in words:
        if word in word_emotion_map:
            for emotion, association in word_emotion_map[word]:
                emotion_scores[emotion] += association

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
