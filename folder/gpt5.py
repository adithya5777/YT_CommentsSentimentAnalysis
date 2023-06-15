import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the NRC-Emotion-Lexicon files
emotion_lexicon_senselevel_file = 'NRC-Emotion-Lexicon-Senselevel-v0.92.txt'
emotion_lexicon_wordlevel_file = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

emotion_lexicon = {}

# Load Senselevel Lexicon
with open(emotion_lexicon_senselevel_file, 'r') as f:
    for line in f:
        word, emotion, value = line.strip().split('\t')
        if word in emotion_lexicon:
            emotion_lexicon[word].add(emotion)
        else:
            emotion_lexicon[word] = {emotion}

# Load Wordlevel Lexicon
with open(emotion_lexicon_wordlevel_file, 'r') as f:
    for line in f:
        word, emotion, value = line.strip().split('\t')
        if word in emotion_lexicon:
            emotion_lexicon[word].add(emotion)
        else:
            emotion_lexicon[word] = {emotion}


# Preprocess comments
def preprocess_comment(comment):
    # Remove non-alphanumeric characters and convert to lowercase
    comment = re.sub(r'[^a-zA-Z0-9\s]', '', comment).lower()
    return comment


# Load comments from CSV
comments_df = pd.read_csv('Full Comments.csv')

# Preprocess comments
comments_df['Comment'] = comments_df['Comment'].apply(preprocess_comment)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    comments_df['Comment'], comments_df['Emotion'], test_size=0.2, random_state=42)

# Predict emotions for each comment


def predict_emotion(comment):
    emotions = {'anger': 0, 'anticipation': 0, 'disgust': 0,
                'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}

    for word in comment.split():
        if word in emotion_lexicon:
            for emotion in emotion_lexicon[word]:
                emotions[emotion] += 1

    # Return the emotion with the maximum count
    return max(emotions, key=emotions.get)


# Apply prediction function to train set
y_train_predicted = X_train.apply(predict_emotion)

# Calculate accuracy on train set
train_accuracy = accuracy_score(y_train, y_train_predicted)
print(f'Train Accuracy: {train_accuracy}')

# Apply prediction function to test set
y_test_predicted = X_test.apply(predict_emotion)

# Calculate accuracy on test set
test_accuracy = accuracy_score(y_test, y_test_predicted)
print(f'Test Accuracy: {test_accuracy}')

# Fit the model on the entire dataset
full_comments_predicted = comments_df['Comment'].apply(predict_emotion)

# Store the comments with their corresponding emotions in separate CSV files
comments_df['Predicted Emotion'] = full_comments_predicted
grouped_comments = comments_df.groupby('Predicted Emotion')

for emotion, group in grouped_comments:
    group.to_csv(f'{emotion}_comments.csv', index=False)
