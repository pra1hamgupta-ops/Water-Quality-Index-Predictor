import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


data = pd.read_csv('non_null_data.csv', dtype= np.float64)

columns = np.array(data.columns)
values = [8.5, 500, 500,250,45,250,100,75,50,200,10]
who = np.array(values)
weight = np.array([4,4,1,5,5,5,3,3,3,2,2])
weight_sum = weight.sum()


WQI = []
for i in range(0,len(data)):
    wqi = 0
    for j in range(0,len(columns)):
        W = (weight[j]/weight_sum)
        qi = (data.iloc[i][columns[j]]/who[j])
        wqi = wqi + W*qi*100
    WQI.append(wqi)
        
WQI = pd.Series(np.array(WQI))
data['WQI'] = WQI

scaling = MinMaxScaler()

scaled_values = np.array(scaling.fit_transform(data))

for i in range(0,len(data)):
    for j in range(0,len(columns)):
        data.at[i,columns[j]] = scaled_values[i][j]


model = Sequential()
model.add(Dense(15,input_dim = 11, activation = 'relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')

y = data['WQI']
X = data.drop('WQI', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model.fit(X_train,y_train,epochs = 150, batch_size = 10)

model.save('my_model.h5')


