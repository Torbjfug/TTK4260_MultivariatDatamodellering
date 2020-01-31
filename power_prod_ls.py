import numpy as np
import pandas as pd
import sklearn


def rmse(y_hat, y):
    return np.sqrt(np.mean((y_hat-y)**2))


def RSS(y_hat, y):
    return np.linalg.norm(y_hat-y)


def FUV(y_hat, y):
    return RSS(y_hat, y)/np.var(y_hat-y)


def r_squared(y_hat, y):
    return 1-FUV(y_hat, y)


def fit(y_hat, y):
    return 100*(1-np.sqrt(rmse(y_hat, y)/np.var(y)))


def mad(y_hat, y):
    return np.mean(np.abs(y-y_hat))


filepath = 'Data/PowerProduction/'

X_train = pd.read_csv(filepath + 'X_train.csv', sep=',', index_col=0)
Y_train = pd.read_csv(filepath + 'Y_train.csv', sep=',', index_col=0)
X_val = pd.read_csv(filepath + 'X_val.csv', sep=',', index_col=0)
Y_val = pd.read_csv(filepath + 'Y_val.csv', sep=',', index_col=0)

# X_mask = [:, :n]
n = 13

theta = np.linalg.inv(
    X_train.values[:, :n].T@X_train.values[:, :n])@X_train.values[:, :n].T@Y_train.values

Y_hat = X_val.values[:, :n]@theta
rmse1 = rmse(Y_hat, Y_val.values)
r_squared1 = r_squared(Y_hat, Y_val.values)
rss1 = RSS(Y_hat, Y_val.values)
fuv1 = FUV(Y_hat, Y_val.values)
fit1 = fit(Y_hat, Y_val.values)
mad1 = mad(Y_hat, Y_val.values)


print(np.mean(Y_hat))
print("RMSE: ", rmse1)
print("RSS: ", rss1)
print("FUV: ", fuv1)
print("R_sq:", r_squared1)
print("FIT: ", fit1)
print("MAD:", mad1)
