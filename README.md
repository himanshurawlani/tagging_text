# Tagging text

This repository contains code to classify transcripts of conversation into one of the 6 classes: ['Bank Bailout', 'Budget', 'Credit Card', 'Family Finance', 'Job Benefits', 'Taxes']

### Setting up python environment

Create a conda environment and then install packages defined in requirements.txt

```
$ conda create -n <env_name> python==3.8.5
$ pip install -r requirements.txt
```

### Getting the dataset

Place the `tagging_test` dataset in the home directory of the project:

```
tagging_text
├── logs
└── tagging_test
    └── metadata
├── inference.ipynb
├── logger.py
├── model.py
├── predict.py
├── preprocess.py
├── requirements.txt
├── stop_words.txt
└── train.py
```

### Training

The `train.py` script contains functions for running training pipeline. This script performs the following tasks:
1. Load the dataset
2. Pre-process the datatset
3. Save the pre-processed dataset to `data` directory
4. Build text vectorizers
5. Build and train text classification model (Multinomial Naive Bayes in this case)
6. Save the trained vectorizers and model to `artifacts` directory
7. Create mlflow model using the training artifacts

```
$ python train.py
```

Once the training is completed you'll be presented with the metrics. These are the metrics that I got (NOTE: The training metrics would differ on different machines.):

```
Accuracy:  0.9444444444444444
Classes ['Bank Bailout' 'Budget' 'Credit Card' 'Family Finance' 'Job Benefits' 'Taxes']

Classification Report
======================================================

               precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       1.00      0.75      0.86         4
           2       0.92      0.92      0.92        13
           3       1.00      0.95      0.98        22
           4       0.94      1.00      0.97        17
           5       0.88      1.00      0.94        15

    accuracy                           0.94        72
   macro avg       0.79      0.77      0.78        72
weighted avg       0.93      0.94      0.94        72
```

Once the training is completed an `artifacts` folder would be created containing pickle files. We use these files to create our Mlflow inference API to handle our inference request via REST endpoint. The training script automatically creates an mlflow artifact inside `servable` folder which would be used to start our Gunicorn server.

### Starting Inference API

The Mlflow model is saved inside `servable` folder. We use the following command to start our Gunicorn server:

```
$ mlflow models serve -m servable -p 2343 --no-conda
```

### Inference

The Jupyter notebook contains the code to send a request, containing the text payload, to our inference API. You will also be able to see the logs of the inference in `logs/app.log` file.

You can also run inference from command line by creating a CSV file. This CSV file should contain `text` column with input text in subsequent rows. Use the following command to run inference from CLI

```
$ mlflow models predict -m servable -i data/test.csv -t csv --no-conda
```

The output would be printed to stdout. If you want to redirect the output to a JSON file you can append `-o output.json` flag to the above command.
