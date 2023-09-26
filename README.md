
# Youtube Comments Sentiment Analysis

This project focuses on sentiment analysis of YouTube comments VADER (Valence Aware Dictionary and Sentiment Reasoner) model. The project involves collecting comments from a YouTube video through web scraping and storing them in a CSV file. The VADER model is then applied to assign sentiment scores to the comments, categorizing them as positive or negative based on predefined thresholds. Separate CSV files are created for positive and negative comments for detailed analysis. The results reveal the distribution of sentiments and provide insights into user opinions. This research showcases the practical application of sentiment analysis in understanding YouTube user sentiments, offering valuable insights for future research in sentiment analysis techniques.


## Process

#### Stage 1: Web Scraping
Process of Using Bots to Extract Comments from the Youtube
Website.

#### Stage 2: Sentiment Analysis
Identifying the Emotional tone behind the Comment text.

#### Stage 3: Result and Visualization

##  Design & Implementation

The sentiment analysis and emotion analysis tasks are implemented using various libraries and techniques in Python. The NLTK library is utilized for sentiment analysis, specifically the SentimentIntensityAnalyzer class, which applies a lexicon-based approach to determine sentiment scores. The NRC Emotion Lexicon is employed for emotion analysis, using a DataFrame to map words to different emotions.


![Image](https://raw.githubusercontent.com/adithya5777/YT_analyzer/blob/main/arch.png)

Preprocessing steps, such as lowercase conversion, tokenization, stopword removal, and lemmatization, were applied to clean and standardize the text data before analysis. These steps were carried out using functions from NLTK and TextBlob libraries. To facilitate the analysis, the comments were stored in a pandas DataFrame for efficient manipulation and processing. Separate CSV files were generated to categorize the comments based on sentiment (positive or negative) and dominant emotion.
Overall, the implementation involved integrating various NLP libraries, applying preprocessing techniques, and leveraging lexicons and pre-trained models to perform sentiment and emotion analysis on the collected comments. The resulting CSV files provide valuable insights into sentiment and emotions expressed by users in the dataset.

## Deployment

To host the website on Local Machine:

```bash
  >> python app.py
```

