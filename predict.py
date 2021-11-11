"""
Inference API for tagging text
"""

import pickle
import mlflow

import preprocess
import logger

LOGGER = logger.create_logger("tagging_text_inference", "./logs")


class TextClassificationWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow wrapper for Tagging text
    """

    def __init__(self):
        """
        TaggingTextWrapper constructor
        """
        pass

    def load_context(self, context):
        """
        Load context for mlflow.pyfunc.PythonModel

        Arguments:
            context {PythonModelContext}: Context for mlflow.pyfunc.PythonModel
        """
        # Load text vectorizer
        self.vectorizer_path = context.artifacts["vectorizer"]
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load text classifier
        self.classifier_path = context.artifacts["classifier"]
        with open(self.classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

        # Load labels list
        self.labels_path = context.artifacts["labels"]
        with open(self.labels_path, "rb") as f:
            self.labels = pickle.load(f)

        LOGGER.info("Loaded model context")

    def predict(self, context, model_input):
        """
        Custom predict function for PythonModel

        Arguments:
            context {PythonModelContext}: Context for mlflow.pyfunc.PythonModel
            model_input {DataFrame}: Input for prediction

        Returns:
            list: Predicted classes
        """
        try:
            LOGGER.info("Loading model input...")
            # Load the text from the model input
            input_text = model_input["text"].tolist()

            LOGGER.info("Processing input text...")
            preprocessed_texts = []
            # Loop over input text
            for text in input_text:
                # Preprocess text
                preprocessed_texts.append(preprocess.preprocess_conversation(text))

            LOGGER.info("Vectorizing input text...")
            # Vectorize the text
            X = self.vectorizer.transform(preprocessed_texts)

            LOGGER.info("Running text classifier...")
            # Get classifier predictions
            predictions = self.classifier.predict(X)

            LOGGER.info("Processing model predictions...")
            predicted_labels = []
            for prediction in predictions:
                predicted_labels.append(self.labels[prediction])

            LOGGER.info("Inference completed successfully")
            return {
                "STATUS": "SUCCESS",
                "MESSAGE": "Inference completed successfully",
                "RESULTS": {"Text": input_text, "Prediction": predicted_labels},
            }
        except:
            LOGGER.exception("Inference failed")
            return {
                "STATUS": "FAILED",
                "MESSAGE": "Inference failed. Please try again or contact support.",
            }
