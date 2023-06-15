import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('Full Comments.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a single string
    text = ' '.join(tokens)
    return text


# Apply preprocessing to the 'comment' column
data['comment'] = data['comment'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['comment'], data['sentiment'], test_size=0.2, random_state=42)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectors, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test_vectors)

# Create separate DataFrames for each sentiment category
sentiments = set(data['sentiment'])
for sentiment in sentiments:
    # Filter the data for the specific sentiment
    filtered_data = data[data['sentiment'] == sentiment]
    # Save the comments to a CSV file
    filtered_data.to_csv(f'{sentiment}_comments.csv', index=False)

# Evaluate the model
print(classification_report(y_test, y_pred))
