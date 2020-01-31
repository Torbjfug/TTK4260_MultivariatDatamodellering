import pandas as pd
import sys
import numpy as np

filepath = 'Data/PowerProduction/'

X = pd.read_csv(filepath + 'X.csv', sep=';')
Y = pd.read_csv(filepath + 'Y.csv', sep=';')
print(len(X))

one_hot_columns = ['IsDayBin', 'Month']

X = X.join(Y['WindPower'])

for col in one_hot_columns:
    one_hot = pd.get_dummies(X[col], prefix=col)
    X = X.join(one_hot)
    X = X.drop(col, axis=1)

X = X.drop('Id', axis=1)

test_percent = 0.1
X_test = X.sample(round(len(X)*test_percent))
X = X.drop(X_test.index)


val_mask = X["Year"] == 2018
X_train = X[~val_mask]
X_val = X[val_mask]


Y_val = X_val['WindPower']
Y_test = X_test['WindPower']
Y_train = X_train['WindPower']

X_test = X_test.drop(['WindPower', 'Year'], axis=1)
X_val = X_val.drop(['WindPower', 'Year'], axis=1)
X_train = X_train.drop(['WindPower', 'Year'], axis=1)


X_train.to_csv(filepath+'X_train.csv', sep=',')
X_test.to_csv(filepath+'X_test.csv', sep=',')
X_val.to_csv(filepath+'X_val.csv', sep=',')

Y_train.to_csv(filepath+'Y_train.csv', sep=',', header=True)
Y_val.to_csv(filepath+'Y_val.csv', sep=',', header=True)
Y_test.to_csv(filepath+'Y_test.csv', sep=',', header=True)
