# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 07:45:43 2023

@author: Muhammad-Ahmed
"""



import pandas as pd


#getting the dataset file
df=pd.read_csv("E:\Gradutation-Project\python-AI\Toddler_dataset.csv")


#################------------preprocessing------------###############
#df.head()
#print(df.shape)
#print(df.columns)

#delete non contributing columns
df1=df.drop('Case_No',axis=1)
df1=df1.drop('Qchat-10-Score',axis=1)
df1=df1.drop('Who completed the test',axis=1)
df1= df1.rename(columns={"Age_Mons":"Age_Month","Class/ASD Traits ": "Class/ASD"})
#print(df1.columns)
#df1.shape
#df1.info()



#no null values






#-----------------------oneHotEncoding-------


df1=pd.get_dummies(df1,columns=["Ethnicity"])

#df1.columns

order=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
       'Age_Month', 'Sex', 'Jaundice', 'Family_mem_with_ASD',
       'Ethnicity_Hispanic', 'Ethnicity_Latino', 'Ethnicity_Native Indian',
       'Ethnicity_Others', 'Ethnicity_Pacifica', 'Ethnicity_White European',
       'Ethnicity_asian', 'Ethnicity_black', 'Ethnicity_middle eastern',
       'Ethnicity_mixed', 'Ethnicity_south asian'
       , 'Class/ASD']



df1=df1[order]

#df1.head()



#-----------------------label Encoding-------


from sklearn.preprocessing import LabelEncoder
 
le=LabelEncoder()


for col in df1:
    df1[col]=le.fit_transform(df1[col])
    
    
#df1.info()

    
    
#-----------------------Train And Test-------


from sklearn.model_selection import train_test_split    

x=df1.drop(columns=["Class/ASD"])
y=df1["Class/ASD"]

#y.info()





X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.3,random_state=42)



#-----------Encoding----------



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

#X_train.info()


X_train['Age_Month']=sc.fit_transform(X_train[['Age_Month']])

X_test['Age_Month']=sc.fit_transform(X_test[['Age_Month']])



#X_test['Age_Month'].unique()



#--------------- Modeling------------------


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,Y_train)

y_pred=model.predict(X_test)




from sklearn.metrics import accuracy_score,confusion_matrix


accuracy=accuracy_score(Y_test,y_pred)

cnf_matrix=confusion_matrix(Y_test,y_pred)

#print("\n {} \n".format(cnf_matrix))

print("\n accuracy={} \n".format(accuracy))





























