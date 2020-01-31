import numpy as np
import pandas as pd
import sklearn


def rmse(y_hat, y):
    return np.sqrt(np.mean((y_hat-y)**2))


def RSS(y_hat, y):
    return np.linalg.norm(y_hat-y, 2)


def FVU(y_hat, y):
    return RSS(y_hat, y)/np.var(y)


def r_squared(y_hat, y):
    return 1-FVU(y_hat, y)


def fit(y_hat, y):
    return 100*(1-np.sqrt(FVU(y_hat, y)))


def mad(y_hat, y):
    return np.mean(np.abs(y-y_hat))


def statistical_indecies(y_hat, y):
    indecies = {}
    indecies['rmse'] = rmse(y_hat, y)
    indecies['r_squared'] = r_squared(y_hat, y)
    indecies['rss'] = RSS(y_hat, y)
    indecies['fvu'] = FVU(y_hat, y)
    indecies['fit'] = fit(y_hat, y)
    indecies['mad'] = mad(y_hat, y)
    return indecies


filepath = 'Data/PowerProduction/'

X_train = pd.read_csv(filepath + 'X_train.csv', sep=',', index_col=0)
Y_train = pd.read_csv(filepath + 'Y_train.csv', sep=',', index_col=0)
X_val = pd.read_csv(filepath + 'X_val.csv', sep=',', index_col=0)
Y_val = pd.read_csv(filepath + 'Y_val.csv', sep=',', index_col=0)

# X_mask = [:, :n]

drops = {}
thetas = {}
idecies = {}
# drops['all'] = []
drops['month'] = ['Month_1', 'Month_2',
                  'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8',
                  'Month_9', 'Month_10', 'Month_11', 'Month_12']
drops['isDay'] = ['IsDayBin_Day', 'IsDayBin_Night']
drops['allCat'] = ['Month_1', 'Month_2',
                   'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8',
                   'Month_9', 'Month_10', 'Month_11', 'Month_12', 'IsDayBin_Day', 'IsDayBin_Night']
drops['all'] = X_train.keys().drop(['WindSpeed', 'IrrDirect', 'IrrDiffuse',                                         'Temperature', 'Percipitation',
                                    'SnowFlow', 'AirDensity',
                                    'CloudCover', 'IsDayBin_Day', 'IsDayBin_Night', 'Month_1', 'Month_2',
                                    'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8',
                                    'Month_9', 'Month_10', 'Month_11', 'Month_12'])
# n = 11 drops all catecoricals, 13 only drops months
n = 13

for key in drops:
    print(key)

    phi = X_train.drop(drops[key], axis=1).values
    A = phi.T@phi
    if np.linalg.det(A) > 5e-1:
        theta = np.linalg.inv(phi.T@phi)@phi.T@Y_train.values
    else:
        theta = np.linalg.pinv(phi.T@phi)@phi.T@Y_train.values
    phi_val = X_val.drop(drops[key], axis=1).values
    y_hat = phi_val@theta
    thetas[key] = theta
    index = statistical_indecies(y_hat, Y_val.values)
    idecies[key] = index

    for k in index:
        print(k, index[k])
    print()

for key in thetas:
    print(key)
    print(thetas[key])
print(X_train.keys())
