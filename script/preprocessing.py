import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin', on_bad_lines='skip', header=None)
    df = df[[0, 5]]
    df.rename(columns={0: 'Sentiment', 5: 'Text'}, inplace=True)
    df['Sentiment'] = df['Sentiment'].replace(4, 1)
    return df

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
url_mention_pattern = re.compile(r"(?:\@|https?\://)\S+|[^\w\s#]")

def cleansing_tweet(tweet):
    tweet = tweet.lower()
    tweet = url_mention_pattern.sub('', tweet)
    tokens = tweet.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    processed_tweet = ' '.join(lemmatized_tokens)
    return processed_tweet

def preprocess_data(df):
    df['tweet_clean'] = df['Text'].apply(cleansing_tweet)
    return df

def vectorize_text(X_train, X_test, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def save_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    # Load data
    filepath = "../data/raw/training.1600000.processed.noemoticon.csv"
    df = load_data(filepath)

    # Preprocess data
    df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['tweet_clean'], df['Sentiment'], test_size=0.2, random_state=42)

    # Vectorize text using TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_text(X_train, X_test)

    # Save TF-IDF model and data
    save_pickle(tfidf_vectorizer, '../model/tfidf_vectorizer.pkl')
    save_pickle((X_train_tfidf, y_train), '../data/proses/train_tfidf.pkl')
    save_pickle((X_test_tfidf, y_test), '../data/proses/test_tfidf.pkl')


