import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.utils import shuffle

#mysql connection
connection = pymysql.connect(host=HOST,
                             user=USER,
                             password=PASSWORD,
                             db=DB,
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor)

#query data
sql_script= """X"""
query = pd.read_sql(sql_script, connection)
data = query
print(data.head())
print(data.shape)
print(data.columns.values)
print(data.isnull().any())

#rank price within a user
data['price_rank'] = ''
data['price_rank'] = data.groupby(['id'])['price'].rank(method='dense', ascending=True)

#check for colinearity
corr = data.corr()
corr = (corr)
ax = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
ax.show()

#preprocessing
data = shuffle(data)

#remove unwanted columns
data = data.iloc[:, [0,1,2,3,5,6,7,8,11,12,13,14,15,16]]
X = data.iloc[:, [0,1,2,7,9,11]]
y = data.iloc[:, 12]
y = y.fillna(0)

#encode categorical variable
le = LabelEncoder()
X.iloc[:, 4] = le.fit_transform(X.iloc[:, 4].values)

#convert to numpy
Xn = X.values
yn = y.values

#standardise
Xn_std = preprocessing.scale(Xn)

#train test split
X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.2, random_state=2018)

#logistic regression implementation
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

#model performance
lr_score_train = lr.score(X_train, y_train)*100
lr_score_test = lr.score(X_test, y_test)*100
print('Performance of model on training data: %s' % (lr_score_train))
print('Performance of model on test data: %s' % (lr_score_test))

#weights of model
lr_weights = lr.coef_
print(lr_weights)

#support vector machine - linear
svm = SVC(kernel='linear', C=1.0, random_state=0, verbose=True)
svm.fit(X, y)
#model performance
svm_linear_score_train = svm.score(X_train, y_train)*100
svm_linear_score_test = svm.score(X_test, y_test)*100
print('Performance of model on training data: %s' % (svm_linear_score_train))
print('Performance of model on test data: %s' % (svm_linear_score_test))

#weights of model
svm_linear_weights = svm.coef_
print(svm_linear_weights.T)

#mlp
mlp = MLPClassifier(hidden_layer_sizes = (100,2),
                    activation = 'logistic',
                    solver = 'adam',
                    alpha = 0.0001,
                    batch_size = 'auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_iter=1000,
                    shuffle=True,
                    verbose=True)

mlp.fit(X_train, y_train)

#model performance
scoretrain = mlp.score(X_train, y_train)
print(scoretrain)
scoretest = mlp.score(X_test, y_test)
print(scoretest)

#keras
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=80, batch_size=50)

#model performance
mlp_train_score = model.evaluate(X_train, y_train_categ, batch_size=50)
mlp_test_score = model.evaluate(X_test, y_test_categ, batch_size=50)
print('Performance of model on training data: %s' % (mlp_train_score[1]*100))
print('Performance of model on test data: %s' % (mlp_test_score[1]*100))


