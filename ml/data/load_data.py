# %%  data preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from PIL import Image

# %% load data
data_set_path=r"C:\Users\SyedShaistaTabassum\Capstone project\pneumonia-detection-local\external-dataset\Data_Entry_2017.csv"
data_set = pd.read_csv(data_set_path)
data_set.head()
data_set.describe()
data_set.info()

#%% NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories:  
'''-- 1) Atelectasis 
-- 2) Cardiomegaly 
-- 3) Effusion
-- 4) Infiltration
-- 5) Mass
-- 6) Nodule
-- 7) Pneumonia
-- 8) Pneumothorax
-- 9) Consolidation
-- 10) Edema
-- 11)Emphysema
-- 12) Fibrosis
-- 13)Pleural_Thickening
-- 14 Hernia'''

# %% check for missing values
categories_P= data_set.loc[data_set['Finding Labels'].str.contains('Pneumonia')]
categories_P
categories_P.describe()


# %%

sns.countplot(data=categories_P, x='Finding Labels')
plt.xticks(rotation=90)
plt.title('Distribution of Pneumonia Cases')
plt.show()
# %%
