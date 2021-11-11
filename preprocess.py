"""
Module to process conversations in TXT file
"""

import os
import re
import logging
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
LOGGER = logging.getLogger("tagging_text_training")


def preprocess_text(text, remove_tags=True, remove_stopwords=False, lemmatize=False):
    """Apply basic preprocessing on texts

    Arguments:
        text {str}: Text on which you need to apply basic preprocessing
        remove_tags {bool}: Whether to remove tags or not while preprocessing (default: {True})
        remove_stopwords {bool}: Whether to remove stopwords or not while preprocessing
                                    (default: {False})

    Returns:
        text {str}: Lemmatized string with removed tags and stopwords
    """

    try:

        if lemmatize:
            lemmatizer = WordNetLemmatizer()

        if remove_tags:
            # Lower text
            text = text.lower()

            # Replace "nan"
            text = text.replace("nan", "")

            # Substitute IP
            text = re.sub(
                (
                    r"(?:(?:2(?:[0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9]|(\*?))\.){3}(?:(?:2([0-4]"
                    r"[0-9]|5[0-5])|[0-1]?[0-9]?[0-9]|(\*))?)"
                ),
                " {ip} ",
                text,
            )

            # Substitute date
            text = re.sub(
                (
                    r"((\d\d?(st|nd|rd|th)?([\-\./])?)\s?(((january|jan)|(february|feb)|(march"
                    r"|mar)|(april|apr)|may|(june|jun)|(july|jul)|(august|aug)|(september|sept)"
                    r"|(october|oct)|(november|nov)|(december|dec)))((\s)?([\-\./])?(,)?(\s)?"
                    r"(\d\d\d?\d?))?)|(([\d]+)([\-])([\d]+)([\-])([\d]+))|(((january|jan)|"
                    r"(february|feb)|(march|mar)|(april|apr)|may|(june|jun)|(july|jul)|(august|"
                    r"aug)|(september|sept)|(october|oct)|(november|nov)|(december|dec))([\s\-]"
                    r")(|([\d]+){1,2}([\s\-]|\,))([\d]+){4})|(january|february|march|april|may|"
                    r"june|july|august|september|october|november|december)|(([\d]+)([\.])([\d]"
                    r"+)([\.])([\d]+))|(([\d]+)([/])([\d]+)([/])([\d]+))"
                ),
                " {date} ",
                text,
            )

            # Remove HTML tags
            text = re.sub(
                (
                    r"(<script(\s|\S)*?<\/script>)|(<style(\s|\S)*?<\/style>)|(<!--(\s|\S)*?-->"
                    r")|(<\/?(\s|\S)*?>)"
                ),
                " ",
                text,
            )

            # Substitute Email
            text = re.sub(
                (
                    r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-"
                    r"9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
                ),
                " {email} ",
                text,
            )

            # Substitute path
            text = re.sub(
                (
                    r'([a-zA-Z]:\\(((?![<>:"/\\|?*])[a-zA-Z0-9])+((?![ .])\\)?)*)|(\s\/[A-z0-9-'
                    r"_+]*)(\/([A-z0-9-_+]*)\/*[A-z0-9-_+]+)"
                ),
                " {path} ",
                text,
            )

            # Substitute URL
            text = re.sub(
                (
                    r"([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|"
                    r"[--:\w?@%&+~#=]+)?"
                ),
                " {url} ",
                text,
            )

            # Substitute time
            text = re.sub(
                (
                    r"((^(([0-1]?\d)|(2[0-3]))(:|\.|)?[0-5][0-9]$)|(^((0?[1-9])|(1[0-2]))(:|\.|"
                    r")([0-5][0-9])( ||,)([aA]|[pP])[mM]$)|(^([aA]|[pP])[mM]( |,|)((0?[1-9])|(1"
                    r"[0-2]))(|:|\.)([0-5][0-9])$)|([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?"
                    r")|(\d\d?\s?((am|AM|pm|PM)))"
                ),
                " {time} ",
                text,
            )

            # Substitute floating point numbers
            text = re.sub(
                r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?",
                "{number}",
                text,
            )

            # Subtitute alphanumeric numbers
            text = re.sub(
                r"(\S+{number}\S*)|(\S*{number}\S+)", " {alphanumeric} ", text
            )

            # Lower text
            text = text.lower()

        # Add white space before and after punctuation
        text = re.sub(r'([`~@$^&\\;\'<>.,!?()#"|=\-:/_*\[\]+%])', r" ", text)

        # Convert multiple white spaces to one white space
        text = re.sub(r" {2,}", " ", text)

        if remove_stopwords:
            stop_words = []
            with open("stop_words.txt", "r") as stop_words_file:
                stop_words = stop_words_file.read().split("\n")
            # Remove stop words
            text_temp = ""
            for word in text.split(" "):
                if word not in stop_words:
                    if lemmatize:
                        word = lemmatizer.lemmatize(word)
                    text_temp += word + " "
            # Strip texts of white spaces
            text = text_temp.strip()

        return text
    except:
        LOGGER.exception("Failed to apply preprocessing on texts")
        raise Exception("Failed to apply preprocessing on texts") from None


def preprocess_conversation(text):
    """
    Process the conversation transcript text files.
    Remove speakers and time stamps and concatenate the text.
    Remove stopwords and

    Arguments:
        text {str}: String from a conversation txt file

    Returns:
        preprocessed_text {str}: Processed and simplified string
    """
    try:
        # Split lines and remove speaker and timestamps
        lines = text.split("\n")
        processed_lines = []
        for i, line in enumerate(lines):
            lines[i] = line.split(" ", 3)
            if len(lines[i]) == 4:
                processed_lines.append(lines[i][3])
        conversation_text = " ".join(processed_lines)

        # Perform basic text pre-processing
        preprocessed_text = preprocess_text(
            conversation_text, remove_tags=False, remove_stopwords=True, lemmatize=True
        )

        return preprocessed_text
    except:
        LOGGER.exception("Failed to apply preprocessing on conversations")
        raise Exception("Failed to apply preprocessing on conversations") from None


def preprocess_dataset(labels_txt_path, conversations_txt_path, output_path):
    """
    Get the conversation text and the labels and store them in a CSV file.
    It performs preprocessing of the conversation text.

    Arguments:
        labels_txt_path {str}: Path to labels txt
        conversations_txt_path {str}: Path to folder containing conversation files
        output_path {str}: Path where meta.csv file is to be created

    Returns:
        train_df {pd.DataFrame}: DataFrame consisting of preprocessed text and
                                 corresponding labels
        test_df {pd.DataFrame}: DataFrame consisting of preprocessed text only
    """
    try:
        LOGGER.info("Reading and processing labels...")
        # Read labels file line by line
        with open(labels_txt_path, "r") as f:
            lines = f.readlines()
        # Split filename and label name
        labels = {}
        for i, line in enumerate(lines):
            line = line.replace("\n", "").replace('"', "").split(" ", 1)
            labels[int(line[0])] = str(line[1])

        LOGGER.info(f"Reading and processing input data...")
        # Create train and test splits
        train = {"text": [], "label": []}
        test = {"text": []}
        # Read input files in the directory
        input_files = os.listdir(conversations_txt_path)
        for filename in input_files:
            if filename.endswith(".txt"):
                _, file_number, _ = filename.split(".")
                file_number = int(file_number)

                # Read file
                with open(os.path.join(conversations_txt_path, filename)) as f:
                    input_text = f.read()
                # Preprocess text
                preprocessed_text = preprocess_conversation(input_text)

                # Check if the label is present
                if labels.get(file_number, None):
                    train["text"].append(preprocessed_text)
                    train["label"].append(labels[file_number])
                else:
                    test["text"].append(preprocessed_text)

        LOGGER.info(f"Saving processed input data...")
        os.makedirs(output_path, exist_ok=True)
        # Save train CSV
        train_df = pd.DataFrame.from_dict(train)
        train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)

        # Save test CSV
        test_df = pd.DataFrame.from_dict(test)
        test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)

        return train_df, test_df
    except:
        LOGGER.exception("Failed to convert data to CSV")
        raise Exception("Failed to convert data to CSV") from None
