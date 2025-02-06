SENTIMENT ANALYSIS

COMPANY : CODTECH IT SOLUTIONS 
NAME : JASVANTH VARMA MUPPALA 
INTERN ID : CT08ORY 
DOMAIN : DATA ANALYSIS 
DURATION : 4 WEEKS 
MENTOR : NEELA SANTOSH

PROJECT DESCRIPTION : SENTIMENT ANALYSIS ON MOVIE REVIEWS USING NLP

This project demonstrates the process of performing sentiment analysis on textual data (movie reviews) using Natural Language Processing (NLP) techniques. The objective is to classify the sentiment of movie reviews into positive or negative categories based on the text content. Here's an overview of what the project involves:

Key Steps:
Data Preprocessing:

The dataset contains movie reviews with associated sentiment labels (positive or negative).
The text data is preprocessed by:
Converting the text to lowercase to maintain uniformity.
Removing punctuation to eliminate unnecessary symbols that may interfere with analysis.
Tokenizing the text into individual words (tokens) and removing common words (stopwords) that don’t contribute much to the meaning (e.g., “the,” “is,” etc.).
Feature Extraction:

The cleaned text data is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency). This method helps capture the importance of words in the context of the entire corpus, making the data suitable for machine learning models.
Model Training:

A Logistic Regression model is trained on the transformed text data to learn the relationship between the features (processed words) and the sentiment labels (positive/negative).
Model Evaluation:

The model's performance is evaluated using the test data, and metrics such as accuracy and the classification report are used to measure how well the model performs in predicting the sentiment of new reviews.
Model and Vectorizer Saving:

The trained model and TF-IDF vectorizer are saved using Pickle for future use, enabling easy deployment and prediction on new data.
Tools & Libraries Used:
pandas: For handling and processing the dataset.
nltk: For tokenization and stopword removal.
scikit-learn: For machine learning, feature extraction, and model evaluation.
Logistic Regression: As the classifier for sentiment analysis.
Pickle: For saving the trained model and vectorizer.
This project showcases how text data can be effectively processed and analyzed to extract useful insights, with applications in sentiment analysis for reviews, feedback, and social media monitoring. The goal of the project is to create an accurate model that can predict sentiment from textual data, providing valuable insights for businesses and organizations to gauge customer opinions and improve their offerings.
