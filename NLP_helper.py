# Helper functions for my NLP Projects
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import os
import pathlib
import zipfile
import random

# To unzip the zipfile which contains dataset
def unzip_data(filename):
  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extractall()
  zip_ref.close()
