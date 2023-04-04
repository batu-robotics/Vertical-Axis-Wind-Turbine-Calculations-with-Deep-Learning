import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

def open_file(filename):
    dataframe=pd.read_csv(filename)
    return dataframe

def plot(dataframe,index):
    head=['Independent','Dependent']
    data1=dataframe.iloc[:,index].values
    data2=dataframe.iloc[:,-1].values
    
    new_df=pd.DataFrame(np.column_stack((data1,data2)),columns=head).sort_values(head[0])
    plt.figure()
    sns.regplot(new_df.iloc[:,0].values,new_df.iloc[:,1].values)

    
    plt.grid(True)
    plt.show(new_df.iloc[:,0].values,new_df.iloc[:,1].values)
    
    # plt.figure()
    # sns.distplot(new_df.iloc[:,0].values)
    # plt.grid(True)
    # plt.show()
    
    return new_df

df=open_file('encoded_data1.csv')

df1=plot(df,0)
df2=plot(df,1)
df3=plot(df,2)
df4=plot(df,3)
df5=plot(df,4)