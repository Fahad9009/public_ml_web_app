import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv("diabetes.csv")      # 1.load dataset

# print(diabetes_dataset.head())

# print(diabetes_dataset.describe())      #ig to check mean and std for StandardScaler

# print(diabetes_dataset["Outcome"].value_counts())   # 2.count target values

X = diabetes_dataset.drop(columns= "Outcome", axis=1)       # 3. X has all the feautures(columns) except "Outcome"
Y = diabetes_dataset["Outcome"]                             #    Y has "Outcome" which is target column

# print(X)      #To check like i said in point 3
# print(Y)

scaler = StandardScaler()   # 4. All the values of different feautures are like one feautures has 100 and other has 0.6
# standardscaler is a preprocessing step not a model
scaler.fit(X)
standarized_data = scaler.transform(X)

X = standarized_data            # 5. X and Y Updated X has standardized.
Y = diabetes_dataset["Outcome"]

# NOW SPLITTING THE DATA INTO TRAINING AND TESTING DATA...

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)
# print(standarized_data.std())

model = svm.SVC(kernel='linear')            #model used in this project : svm

model.fit(X_train, Y_train)

#Accuracy Score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print("Accuracy Score of the training data", training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print("Accuracy Score of the testing data", test_data_accuracy)




import pickle

filename = 'diabetes_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))



input_data = (6,148,72,35,0,33.6,0.627,50)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if(prediction[0] == 0):
    print("The person is non diabetic")
else:
    print("Person is diabetic")
