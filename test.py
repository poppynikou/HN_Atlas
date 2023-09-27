import pandas as pd 
import os
current_directory = os.getcwd()
preprocessing_data = current_directory +'/preprocessing.csv'
slices = pd.read_csv(preprocessing_data)

print(slices['Min_y'].iloc[1])