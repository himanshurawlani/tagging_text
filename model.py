"""
Module to create text classification module and perform training.
"""

import os
import logging
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

LOGGER = logging.getLogger("tagging_text_training")


def label_vectorizer(labels):
    """
    Encode target labels with value between 0 and n_classes-1.

    Arguments:
        labels {list}: List of labels

    Returns:
        lb_output {list}: Array output after converting labels to integers
        lb_classes {list}: List of classes with index corresponding to its label
    """
    try:
        LOGGER.info("Encoding target labels to integers...")
        LOGGER.debug("len(labels): %d", len(labels))

        # Make sure artifacts directory is created
        os.makedirs("artifacts", exist_ok=True)

        # Initialize the encoder
        lb = LabelEncoder()
        lb_output = lb.fit_transform(labels)

        # Save the labels list
        with open("artifacts/labels.pickle", "wb") as f:
            pickle.dump(lb.classes_, f)

        LOGGER.info("Number of classes in labels: %d", len(lb.classes_))
        LOGGER.info("Target labels created")

        return lb_output, lb.classes_
    except:
        LOGGER.exception("Failed to encode labels")
        raise Exception("Failed to encode labels") from None


def tfidf_vectorizer(texts, debug=False):
    """Create TF-IDF vectors

    Arguments:
        texts {list}: Texts to convert to TF-IDF vectors

    Returns:
        NumPy Array: TF-IDF vectors
    """

    try:
        LOGGER.info("Creating TF-IDF vectors...")
        LOGGER.debug("len(texts): %d", len(texts))

        # Make sure artifacts directory is created
        os.makedirs("artifacts", exist_ok=True)

        # Create vectorizer object
        vectorizer = TfidfVectorizer(
            stop_words=None,
            use_idf=True,
            ngram_range=(1, 2),
            lowercase=True,
            analyzer="word",
            max_features=10000,
        )

        # Fit and create TFIDF vectors
        tfidf_output = vectorizer.fit_transform(texts)

        if debug:
            # Save the feature names to file
            with open("data/tf_vocab.txt", "w") as f:
                for feature_name in vectorizer.get_feature_names():
                    f.write(feature_name + "\n")

        # Save the model to artifacts
        with open("artifacts/vectorizer.pickle", "wb") as f:
            pickle.dump(vectorizer, f)

        LOGGER.info("Number of words in TF-IDF: %d", len(vectorizer.vocabulary_))
        LOGGER.info("TF-IDF vectors created")

        return tfidf_output.toarray()

    except:
        LOGGER.exception("Failed to create TF-IDF vectors")
        raise Exception("Failed to create TF-IDF vectors") from None


def count_vectorizer(texts, debug=False):
    """Create Count vectors

    Arguments:
        texts {list}: Texts to convert to Count vectors

    Returns:
        NumPy Array: Count vectors
    """
    try:
        LOGGER.info("Creating count vectors...")
        LOGGER.debug("len(texts): %d", len(texts))

        # Make sure artifacts directory is created
        os.makedirs("artifacts", exist_ok=True)

        vectorizer = CountVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            analyzer="word",
            max_features=10000,
            lowercase=True,
        )

        # Fit and create TFIDF vectors
        tfidf_output = vectorizer.fit_transform(texts)

        if debug:
            # Save the feature names to file
            with open("data/cv_vocab.txt", "w") as f:
                for feature_name in vectorizer.get_feature_names():
                    f.write(feature_name + "\n")

        # Save the model to artifacts
        with open("artifacts/vectorizer.pickle", "wb") as f:
            pickle.dump(vectorizer, f)

        LOGGER.info(
            "Number of words in Count vectorizer: %d", len(vectorizer.vocabulary_)
        )
        LOGGER.info("Count vectors created")

        return tfidf_output.toarray()
    except:
        LOGGER.exception("Failed to create Count vectors")
        raise Exception("Failed to create Count vectors") from None


def train_classifier(X, y, classes):
    """
    Create a naive bayes classifier model.

    Arguments:
        X {list}: List of vectorized input training text
        y {list}: List of integer labels for training
        classes {list}: List of class names corresponding to integer labels

    Returns:
        MultinomialNB: Naive bayes classifier created with the specified parameters.
    """
    try:
        LOGGER.info("Training text classifier...")

        # Make sure artifacts directory is created
        os.makedirs("artifacts", exist_ok=True)

        NBclassifier = MultinomialNB(alpha=1, fit_prior=True, class_prior=None)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=2, shuffle=True
        )

        # Train the classifier
        NBclassifier.fit(X_train, y_train)
        y_pred = NBclassifier.predict(X_test)

        # Classification metrics
        # TODO: Create a CSV report and save it with artifacts
        train_report = classification_report(y_test, y_pred)
        print("\n Accuracy: ", accuracy_score(y_test, y_pred))
        print("Classes", classes)
        print("\nClassification Report")
        print("======================================================")
        print("\n", train_report)

        # Save the model to artifacts
        with open("artifacts/classifier.pickle", "wb") as f:
            pickle.dump(NBclassifier, f)

        LOGGER.info("Text classifier training completed")

        return NBclassifier
    except:
        LOGGER.exception("Failed to create classifier model")
        raise Exception("Failed to create classifier model") from None
