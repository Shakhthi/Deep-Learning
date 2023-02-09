# Helper functions for my NLP Projects
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import os
import pathlib
import datetime
import zipfile
import random

# To unzip the zipfile which contains dataset
def unzip_data(filename):
  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extractall()
  zip_ref.close()

# create a TensorBoard callback (functionized because we need one for each model)
def tensorboard_callback(dir_name, experiment_name):
  
  logs_directory = dir_name+"/"+experiment_name+"/"+datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
  
  TensorBoard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_directory)
  print(f"Saving TensorBoard logfiles to: {logs_directory}")
  
  return TensorBoard_callback
 
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def model_evaluation(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

