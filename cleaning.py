import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df_not_cleaned = pd.read_csv('./Dataset/as7265x_synthetic_dataset.csv')

# print(df.head)

df = df_not_cleaned.fillna(0) 

# print(df.isnull().sum())

y = df['Element']
X = df.drop('Element',axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
    
# print(f"Accuracy: {accuracy}")   0.725
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

print(rf_classifier.predict([[98.95068832025882,135.84941798075576,138.93605837051044,92.38612056735225,100.2425859818214,129.2386983642949,122.50271569480486,134.67846395415157,118.88398882917232,155.53397549641298,155.3570025471478,151.44061666145896,161.68921964567286,149.73014967027146,123.45116945403703,156.9255010298216,134.70250356467153,166.99926873787106]]))