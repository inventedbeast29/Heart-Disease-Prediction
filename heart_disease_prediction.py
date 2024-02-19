# libraries used numpy, pandas, sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reading CSV data
heart_data = pd.read_csv('heart_disease.csv')

# printing first 5 rows of the dataset
heart_data.head()

# printing last 5 rows of the dataset
heart_data.tail()

# showing number of rows and columns of the dataset
heart_data.shape

# printing information about the dataset
heart_data.info()

# checking the missing value
heart_data.isnull().sum()

# showing statistical measures
heart_data.describe()

# checking the distribution od target variable
heart_data['target'].value_counts()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# training our model
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy of trained data model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy of test data model
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)

# taking the user input
age = int(input("Enter age >> "))
sex = int(input("Enter sex (M:1/F:0) >> "))
chest_pain = int(input("Enter chest pain (0/1) >> "))
restBP = int(input("Enter hypertension/rest BP >> ")) 
cholestrol = int(input("enter chlestrol >> "))
blood_sugar = int(input("Enter blood sugar >> "))
ecg = int(input("Enter ECG(electrocardiogram) value >> "))
max_heart_rate = int(input("Enter max heart rate >> "))
exercise_induced_angina = int(input("Enter exercise induced angina(0/1) >> "))
old_peak = float(input("Enter old peak >> "))
slope = float(input("Enter slope >> "))
coronary_artery = int(input("Enter coronary calcium scan >> "))
thalassemia = int(input("Enter thalassemia >> "))

input_data = (age,sex,chest_pain,restBP,cholestrol,blood_sugar,ecg,max_heart_rate,exercise_induced_angina,old_peak,slope,coronary_artery,thalassemia)
input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
# print(prediction)

if (prediction[0]== 0):
  print('---------The Person does NOT have a HEART DISEASE---------')
else:
  print('---------The Person has HEART DISEASE---------')