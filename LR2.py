

import pandas as pd
import numpy as np

df=pd.read_csv("E:\Gradutation-Project\python-AI\Toddler Autism dataset July 2018.csv")


#df.head()

#df.info()

#df.isnull().any()

#delete non contributing columns
df.drop('Case_No',axis=1,inplace=True)
df.drop('Qchat-10-Score',axis=1,inplace=True)
df.drop('Who completed the test',axis=1,inplace=True)
df.rename(columns={"Age_Mons":"Age_Month","Class/ASD Traits ": "Class/ASD"},
          inplace=True)


#df['Sex'].unique()

df_cat=df.select_dtypes(object)

df_one_encode=df['Ethnicity']

df_cat.drop('Ethnicity',axis=1,inplace=True)

df_num=df.select_dtypes('int64')



#df_cat['Ethnicity'].value_counts()



#-----------------------label Encoding-------



from sklearn.preprocessing import LabelEncoder
 
le=LabelEncoder()


for col in df_cat:
   df_cat[col]=le.fit_transform(df_cat[col])
    
    
#for col in df_cat:
  #  unique_values = df_cat[col].unique()
  #  print(f"Column: {col}")
  #  print(f"Unique Values: {unique_values}")
  #  print()
    
    

    
#df.info()
    
    
#df_one_encode.head()


#-----------------------oneHotEncoding-------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


coltrans=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0])],
                           remainder='passthrough')

df_one_encode=np.array(coltrans.fit_transform(df_one_encode))














    
    