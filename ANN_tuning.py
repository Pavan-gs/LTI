import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv('cust_churning.csv')

dataset.describe()

dataset['Location'].value_counts()
sns.countplot(dataset['Left'])
dataset['Left'].value_counts()
pd.crosstab(dataset['Left'],dataset['Location'])

sns.countplot(dataset['Location'], hue=dataset['Left'])
sns.countplot(dataset['Gender'], hue=dataset['Left'])

dataset.isna().sum()
dataset.info()

df.info()
df.describe().transpose()
df.columns

sns.boxplot(x="Exited", y = 'CreditScore', data=df, hue="Gender")
sns.countplot(x="Exited", data=df)

sns.countplot(x="Exited", data=df, hue='Gender')
sns.countplot(x="Gender", data=df)
sns.countplot(x="Geography", data=df)

'''
X = dataset.iloc[:, 1:11].values
y = dataset['Left'].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
'''

x = df.iloc[:,1:11]
y = df['Left']


d = pd.get_dummies(x[['Gender','Location']], drop_first=True)

x.drop(['Gender','Location'], inplace = True, axis = 1)

x = pd.concat([d,x], axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout

model= Sequential()

model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# model.add(Dropout(p = 0.1))

# Adding the second hidden layer
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# model.add(Dropout(p = 0.1))

# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

y_test.head(5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
(1551+136)/2000

#Evaluating a ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def fn():
    model= Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn= fn,  batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = model, X=X_train, y=y_train,cv = 10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
help(cross_val_score)

# Improving the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Dropout regularisation technique


def fn(optimizer):
    model = Sequential()
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

model = KerasClassifier(build_fn = fn)
parameters = {"batch_size" : [25,35],
              "epochs" : [100,200],
              "optimizer" : ["adam","rmsprop"]}

grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = "accuracy", cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
help(GridSearchCV)










