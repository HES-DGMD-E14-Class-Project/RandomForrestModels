import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))
# print("data_dict variable",data_dict )


data = np.array(data_dict['data'])
data2 = []

for index in range(len(data)):
    if len(data[index]) != 42:
        data[index] = data[index][:42]
    data2.append(data[index][:42])
        
print(type(data))

for item in data2:
    if len(item) != 42:
        print(len(item))
        print(item)

data2 = np.array(data2)        

print("This is the data variable's shape", data.shape)

labels = np.array(data_dict['labels']) 

x_train, x_test, y_train, y_test = train_test_split(data2, labels, test_size=0.2, shuffle=True, stratify=labels)
# print(x_train)

model = RandomForestClassifier()

print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print(f"x {type(x_train)} y {type(y_train)}")


model.fit(x_train, y_train)
print("After fitting")

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()








