import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Define the path to the input CSV file
input_file = 'Full Comments.csv'

# Create a folder for the sentiment analysis results
output_folder = 'Sentiment Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the input CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Initialize the sentiment analysis models
textblob_model = TextBlob
vader_model = SentimentIntensityAnalyzer()

# Perform sentiment analysis and store the results in separate DataFrames
positive_comments_textblob = pd.DataFrame(columns=['Username', 'Comment'])
negative_comments_textblob = pd.DataFrame(columns=['Username', 'Comment'])
positive_comments_vader = pd.DataFrame(columns=['Username', 'Comment'])
negative_comments_vader = pd.DataFrame(columns=['Username', 'Comment'])

for index, row in df.iterrows():
    username = row['Username']
    comment = row['Comment']

    # Sentiment analysis using TextBlob
    textblob_sentiment = textblob_model(comment).sentiment.polarity

    # Sentiment analysis using VADER
    vader_sentiment = vader_model.polarity_scores(comment)['compound']

    if textblob_sentiment >= 0:
        # Positive sentiment according to TextBlob
        positive_comments_textblob = pd.concat([positive_comments_textblob,
                                                pd.DataFrame({'Username': [username], 'Comment': [comment]})],
                                               ignore_index=True)
    else:
        # Negative sentiment according to TextBlob
        negative_comments_textblob = pd.concat([negative_comments_textblob,
                                                pd.DataFrame({'Username': [username], 'Comment': [comment]})],
                                               ignore_index=True)

    if vader_sentiment >= 0:
        # Positive sentiment according to VADER
        positive_comments_vader = pd.concat([positive_comments_vader,
                                             pd.DataFrame({'Username': [username], 'Comment': [comment]})],
                                            ignore_index=True)
    else:
        # Negative sentiment according to VADER
        negative_comments_vader = pd.concat([negative_comments_vader,
                                             pd.DataFrame({'Username': [username], 'Comment': [comment]})],
                                            ignore_index=True)

# Save the positive and negative comments for each model in separate CSV files
textblob_output_folder = os.path.join(output_folder, 'TextBlob')
vader_output_folder = os.path.join(output_folder, 'VADER')

if not os.path.exists(textblob_output_folder):
    os.makedirs(textblob_output_folder)

if not os.path.exists(vader_output_folder):
    os.makedirs(vader_output_folder)

positive_comments_textblob.to_csv(os.path.join(
    textblob_output_folder, 'positive.csv'), index=False)
negative_comments_textblob.to_csv(os.path.join(
    textblob_output_folder, 'negative.csv'), index=False)
positive_comments_vader.to_csv(os.path.join(
    vader_output_folder, 'positive.csv'), index=False)
negative_comments_vader.to_csv(os.path.join(
    vader_output_folder, 'negative.csv'), index=False)

print("Sentiment analysis completed. Results are stored in separate folders for each model.")