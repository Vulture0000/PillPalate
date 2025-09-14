import numpy as np 
import pandas as pd
from sklearn.model_selection import test_train_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df_not_cleaned = pd.read_csv('./Dataset/as7265x_synthetic_dataset.csv')

# print(df.head)

df = df_not_cleaned.fillna(0) 

# print(df.isnull().sum())

