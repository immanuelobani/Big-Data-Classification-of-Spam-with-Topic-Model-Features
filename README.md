# Big Data Classification of Spam Using Topic Model Features

## Overview

This project implements a machine-learning pipeline for the detection of spam in text data. The pipeline includes preprocessing of text data, feature extraction, model training, and evaluation. Several machine learning models including Naive Bayes, Support Vector Machines and Neural Networks are explored and evaluated.

## Prerequisites

The project requires the following Python libraries:

from imblearn.over_sampling import SMOTE
import spacy
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from gensim import corpora, models
#import openai
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gensim
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

- imblearn: For over-sampling using SMOTE and RandomOverSampler.
- spacy: For advanced Natural Language Processing.
- pandas: For data manipulation and analysis.
- tqdm: For providing progress bars to loops.
- gensim: For unsupervised topic modelling and natural language processing.
- nltk: For natural language processing tasks such as tokenization and stopwords.
- scikit-learn: For various machine learning models and metrics.
- tensorflow: For constructing deep learning models.
- matplotlib and seaborn: For visualizing data and results.
- BERTopic: For creating topics using a BERT-based model.

## Data Preprocessing

Data preprocessing includes tokenizing the text, converting it to lowercase, and removing non-letter characters. Also handled missing data from the dataset.

## Feature Extraction

Used the following methods for feature extraction:

- `CountVectorizer` for converting text data into a matrix of token counts.
- `LatentDirichletAllocation` (LDA) for topic modelling.
- `BERTopic` for topic modelling using BERT embeddings.

## Model Training

Split the data into training and testing sets and train multiple models:

- `MultinomialNB`: Naive Bayes classifier for multinomial models.
- `SVC`: Support Vector Machine classifier.
- `Sequential` from tensorflow.keras: For constructing Neural Network models.

## Model Evaluation

Models are evaluated based on accuracy, precision, recall, and F1 score. Also used ROC curves for assessing the performance of binary classifiers.

## Usage

To run the project:

1. Ensure that you have all the required libraries installed.
2. Load your dataset.
3. Run the preprocessing script to clean and prepare your data.
4. Execute the feature extraction script to transform your data.
5. Train the models using the training script.
6. Evaluate the models with the provided evaluation metrics.

## Visualisation

Results and data exploration are visualised using `matplotlib` and `seaborn` to create plots and confusion matrices.

## Hyperparameter Tuning

For models that do not achieve satisfactory performance, hyperparameter tuning is conducted using grid search or other optimization techniques.

## Contribution

Contributions to this project are Welcome. Please fork the repo and submit a pull request.


## Contact

Immanuel Obani
immanuelobani@gmail.com
