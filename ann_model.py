# ANN model
# Designed by: Oğuz SUSAM, Ph.D. Naval Architecture and Marine Engineering
#              Ezgi DEMİR, Ph.D. Industrial Engineering
#              Batuhan ATASOY, Ph.D. Mechatronics Engineering

#%% Inporting Libraries
import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%% Opening Dataframe
def open_file(filename):
    dataframe=pd.read_csv(filename)
    return dataframe

#%% Seperating Dataframe
def seperate(dataframe,percent):
    independent=dataframe.iloc[:,:-1].values
    dependent=dataframe.iloc[:,-1].values
    xtrain,xtest,ytrain,ytest=train_test_split(independent,dependent,test_size=percent,shuffle=True,random_state=None)
    return xtrain,xtest,ytrain,ytest

#%% Splitting Input and Output Data
def split(dataframe):
    independent=dataframe.iloc[:,:-1].values
    dependent=dataframe.iloc[:,-1].values
    return independent,dependent

#%% Building ANN Model
def ann_model(dataframe,dataframe2,dataframe3,percent):
    
    x_train,x_test,y_train,y_test=seperate(dataframe,percent)
    
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    
    act_func='relu'
    
    model=Sequential()
    model.add(Dense(units=10,input_dim=len(x_train[0]),kernel_initializer = 'uniform',activation=act_func))
    model.add(Dense(units=10,input_dim=10,kernel_initializer = 'uniform',activation=act_func))
    model.add(Dense(units=10,input_dim=10,kernel_initializer = 'uniform',activation=act_func))
    model.add(Dense(units=1,input_dim=10,kernel_initializer = 'uniform',activation='relu'))
    model.compile(optimizer='adam',loss='mse',metrics=['mse'])
    model.fit(x_train,y_train,batch_size=16,epochs=2000)

    results=model.predict(x_test)
    results_new=np.column_stack((y_test,results))
    results_new=pd.DataFrame(results_new,columns=['Test Value','Results']).sort_values('Test Value') 
    #mse=model.history['mean_squared_error']
    
    x0_1,y0_1=split(dataframe2)
    x0_2,y0_2=split(dataframe3)
    results2=model.predict(x0_1)
    results2_new=np.column_stack((y0_1,results2))
    results2_new=pd.DataFrame(results2_new,columns=['Test Value','Results'])
    
    results3=model.predict(x0_2)
    results3_new=np.column_stack((y0_2,results3))
    results3_new=pd.DataFrame(results3_new,columns=['Test Value','Results']) 
    
    return results_new,results2_new,results3_new

#%% Main Program
df=open_file('encoded_data1.csv')
df1=open_file('encoded_data_0.1.csv')
df2=open_file('encoded_data_0.2.csv')
x_train,x_test,y_train,y_test=seperate(df,0.2)
results,y01,y02=ann_model(df,df1,df2,0.2)

