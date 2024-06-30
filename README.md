
# Twitter Sentiment Project

## 1. Project Overview

This project aims to perform sentiment analysis on tweets using a neural network model. The process involves several stages, including text cleaning, feature extraction using TF-IDF, and future neural network model training. We have also set a criterion that the model will only be saved if it achieves an accuracy above 70%.

The folder structure of this project is as follows:

- **data**: Contains raw data files and pickle files from the vectorizer process.
- **model**: Contains pickle files of the trained AI model.
- **notebook**: Contains the `eda.ipynb` file where data understanding, insight exploration, and optimal method iterations are conducted.
- **script**: Contains Python scripts (`preprocessing.py`, `training.py`) that are the final results of notebook iterations with modular functions.

## 2. Dataset Overview

The dataset used in this project is a pre-processed tweet dataset that does not contain context for column names, only numbers 0 to 5. Upon examining the data, it was concluded that only columns 0 and 5 can be used.

- **Column 0**: The target for sentiment prediction, where the original values are 0 for positive and 4 for negative. In this case, we changed the value 4 to 1 to make it a binary classification problem (0 and 1).
- **Column 5**: The content of the tweets from various Twitter users.

The dataset contains 1,600,000 tweet entries.
The dataset you can download at : https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477


## 3. Text Cleansing Preprocessing

The text cleaning process involves several steps:
- Converting text to lowercase.
- Removing URLs, mentions, and non-alphanumeric characters except for hashtags.
- Tokenizing text based on spaces.
- Lemmatizing tokens.
- Removing stopwords and short tokens.

The preprocessing code is stored in the `preprocessing.py` file, which contains the following functions:
- `load_data(filepath)`: Loads data from the CSV file.
- `cleansing_tweet(tweet)`: Cleans the tweet text.
- `preprocess_data(df)`: Applies the cleaning function to the entire dataset.
- `split_train_test(df)`: Splits the data into training and testing sets with an 80-20 ratio.

## 4. EDA

Exploratory Data Analysis (EDA) helps in understanding the distribution of sentiment data and cleaning text. After text cleaning, we can see the distribution of words and the frequency of their occurrence in positive and negative tweets. EDA is conducted to ensure the data is ready for modeling. Detailed EDA can be found in the `eda.ipynb` file in the `notebook` folder.

## 5. Text Vectorizer

TF-IDF is used to convert text into feature vectors that can be used by machine learning models. TF-IDF (Term Frequency-Inverse Document Frequency) is a method to transform a collection of texts into a matrix of features, considering the frequency of words and how common the words are in the collection.

### Mathematical Formula

- **Term Frequency (TF)**: Measures how frequently a word appears in a document. The formula is:

  $$
  \text{TF}(t, d) = \frac{\text{Number of occurrences of t in d}}{\text{Total number of words in d}}
  $$

- **Inverse Document Frequency (IDF)**: Measures how important a word is in the entire collection of documents. The formula is:

  $$\text{IDF}(t, D) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents containing t}} \right)$$

- **TF-IDF**: Combines both measures to assign a weight to words, so that words that frequently appear in a document but are rare in the collection get a higher weight. The formula is:

  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$


This process involves:
- `TfidfVectorizer` from scikit-learn to convert text into feature vectors.
- Saving the TF-IDF model and the vectorized data into pickle files.

The code for this process is encapsulated in the following functions:
- `vectorize_text(X_train, X_test, max_features)`: Converts text into TF-IDF features.
- `save_pickle(obj, filename)`: Saves objects into pickle files.

## 6. Next Step

### Model Training

The next step is to train a neural network model using PyTorch. The training process involves:
- Loading the pre-processed data from pickle files.
- Converting the data into PyTorch tensors.
- Training the neural network model with training data.
- Evaluating the model using test data.
- Saving the model only if the accuracy is above 70%.

The training code is stored in the `training.py` file, which contains the function:
- `train_neural_network(X_train, y_train, X_test, y_test, num_epochs, batch_size)`: Trains and evaluates the neural network model.

### Deployment and Inference

After the model is trained and saved, the next step is to use the model for prediction and further evaluation. This process will include:
- Loading the trained model from the pickle file.
- Making predictions on new data.
- Evaluating the model's performance on real-time data.

A `main.py` file will be created to handle this process, ensuring the pipeline runs smoothly from preprocessing to model inference.

---

With this pipeline, we can efficiently and accurately perform sentiment analysis on tweets, leveraging the power of neural networks and effective text preprocessing techniques.

