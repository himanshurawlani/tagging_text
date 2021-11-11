"""
Main module to load and process dataset,
load models to train and create mlflow predictor.
"""

import shutil

import mlflow

from preprocess import preprocess_dataset
from model import label_vectorizer, tfidf_vectorizer, count_vectorizer, train_classifier
from predict import TextClassificationWrapper
import logger

LOGGER = logger.create_logger("tagging_text_training", "./logs")


def train(train_df):
    """
    Module to train text classifier on given dataset

    Arguments:
        train_df {DataFrame}: A pandas DataFrame consisting of input text
                              and corresponding labels
    """
    try:
        # Load the processed conversations
        texts = train_df["text"].tolist()
        labels = train_df["label"].tolist()

        # Transform the labels to integers
        y, classes = label_vectorizer(labels)

        # Load the vectorizer model and transform the input text to vectors
        # X = tfidf_vectorizer(texts)
        X = count_vectorizer(texts)

        # Load the classifier
        NBclassifier = train_classifier(X, y, classes)

    except:
        LOGGER.exception("Failed to train model")
        raise Exception("Failed to train model") from None


def create_mlflow_model():
    """
    Function to create mlflow servable model
    """
    # Path where the servable model file will be created
    model_path = "./servable"

    # Remove the model if it already exists
    shutil.rmtree(model_path, ignore_errors=True)

    # Path to pre-trained artifacts
    artifacts = {
        "vectorizer": "./artifacts/vectorizer.pickle",
        "classifier": "./artifacts/classifier.pickle",
        "labels": "./artifacts/labels.pickle",
    }

    # Environment file
    conda_file = "./conda.yaml"

    # Path to python scripts to be included in the servable
    code_path = ["./preprocess.py", "./logger.py"]

    # Create custom mlflow python model
    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=TextClassificationWrapper(),
        artifacts=artifacts,
        conda_env=conda_file,
        code_path=code_path,
    )


def main():
    """
    Main stub to load dataset, start training and save mlflow artifacts
    """
    # Initialize input variables
    labels_txt_path = "./tagging_test/metadata/mapping_conv_topic.train.txt"
    conversations_txt_path = "./tagging_test"
    output_path = "./data"

    # Load dataset
    train_df, test_df = preprocess_dataset(
        labels_txt_path=labels_txt_path,
        conversations_txt_path=conversations_txt_path,
        output_path=output_path,
    )

    # Run training
    train(train_df)

    # Create mlflow servable model
    create_mlflow_model()


if __name__ == "__main__":
    main()
