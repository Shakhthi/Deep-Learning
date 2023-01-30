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
