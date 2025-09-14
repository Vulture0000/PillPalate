import numpy as np 
import pandas as pd

df = pd.read_csv('./Dataset/as7265x_synthetic_dataset.csv')

print(df.head)

df_clean = df.fillna(0)