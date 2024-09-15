# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:37:39 2023

@author: Muhammad-Ahmed
"""

 
import pandas as pd


data=pd.read_csv('E:\Gradutation-Project\python-AI\Toddler_dataset.csv')

#data.head()

#data.info()

#data.isnull().any()

#delete non contributing columns
data.drop('Case_No',axis=1,inplace=True)
data.drop('Qchat-10-Score',axis=1,inplace=True)
data.drop('Who completed the test',axis=1,inplace=True)
data.rename(columns={"Age_Mons":"Age_Month","Class/ASD Traits ": "Class/ASD"},
          inplace=True)




#data.isnull().sum()

m=(data.dtypes=='object')

object_cols=list(m[m].index)

#object_cols

#print('unique Values',data['Sex'].unique())

#print('unique Values',data['Ethnicity'].unique())

#print('unique Values',data['Jaundice'].unique())

#print('unique Values',data['Family_mem_with_ASD'].unique())

#print('unique Values',data['Class/ASD'].unique())



from sklearn.preprocessing import LabelEncoder,OneHotEncoder


#-----------------------oneHotEncoding-------


object_cols=['Ethnicity']

oh_encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)


oh_data=pd.DataFrame(oh_encoder.fit_transform(data[object_cols]))



oh_new=data.drop(object_cols,axis=1)

oh_train=pd.concat([oh_new,oh_data],axis=1)


y=oh_train['Class/ASD']

x_old=oh_train.drop('Class/ASD',axis=1)

#x.info()


#-----------------------label Encoding-------



lb_encoder=LabelEncoder()

y=lb_encoder.fit_transform(y)

y

feature_names=['Sex','Jaundice','Family_mem_with_ASD']

x_data=pd.DataFrame(oh_train[feature_names])

for col in x_data:
   x_data[col]=lb_encoder.fit_transform(x_data[col])



#oh_train.info()

del_oh_train=x_old.drop(['Sex','Jaundice','Family_mem_with_ASD'],axis=1)

x_old=pd.concat([del_oh_train,x_data],axis=1)

#x.info()

#x.info()

#x_old.info()

#----------------train/test split---------------

x=x_old.iloc[:,:].values


from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)



#----------------------Normalization--------------
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)



#--------------Analyising------------------

# test-size=.262

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
predict_clf2=model.predict(x_test)






from sklearn.metrics import accuracy_score,confusion_matrix


accuracy_RF=accuracy_score(y_test, predict_clf2)

conf_RF=confusion_matrix(y_test, predict_clf2)






new_data_sample = [[1, 1, 1, 1, 1, 1, 1,0,0,0,28,0,0,0,0,1,0,0,0,0,0,0,1,1,1]]
                   


new_data_sample_scaled = sc.transform(new_data_sample)

# Predict the class/ASD for the new data
new_data_sample_pred = model.predict(new_data_sample_scaled)

# Convert the predicted value back to its original label
new_data_sample_pred_label = lb_encoder.inverse_transform(new_data_sample_pred)

# Print the predicted class/ASD value

print('\n****************************************************\n')
print("Predicted Class/ASD for the new data sample:")

#print(new_data_sample_pred_label[0])

print(new_data_sample_pred_label)

print('\n****************************************************\n')
print('Accuracy_Score using Decission Tree is: %.2f'
     %( accuracy_RF *100),'%')
print('\nconfusion matrix: \n',conf_RF)
print('\n****************************************************\n')




























