import pandas as pd


filepath = 'Data/PowerProduction/'

X = pd.read_csv(filepath + 'X.csv', sep=';')
Y = pd.read_csv(filepath + 'Y.csv', sep=';')

print(X.head())


X["Month"] = X["Month"].astype('category')
#X = pd.get_dummies(X, "Month")
#X = pd.get_dummies(X, "IsDayBin")
print(X.head())


test_mask = X["Year"] == 2018
X_train = X[~test_mask]
X_test = X[test_mask]
Y_train = Y[~test_mask]
Y_test = Y[test_mask]

print(X_train["Year"].tail())
print(X_test["Year"].head())



X_train.to_csv(filepath+'X_train.csv',sep=';')
X_test.to_csv(filepath+'X_test.csv',sep=';')


