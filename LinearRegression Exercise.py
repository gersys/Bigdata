# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:08:35 2019

@author: SangWon Jung
"""

import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score

# data load
mat = np.genfromtxt('Real estate valuation data set.csv', 
                    delimiter = ',', encoding = 'latin-1')
# load feature
feat = np.genfromtxt('Real estate valuation data set.csv', 
                    delimiter = ',', dtype = str, skip_footer= mat.shape[0] -1)
feat = list(feat[1:-1])  #feature meaning column

print(mat)
print("----------------------------")
print(feat)
print("----------------------------")
    
# head row, column 제거   
mat = mat[1:,1:]
#data shuffle
np.random.seed(0)
np.random.shuffle(mat)
#feature, label로 분리
X = mat[:, :6]
Y = mat[:, 6]

# train, validation, test set으로 분리
N = mat.shape[0]
train_N = int(N * 0.6)
val_N = int(N*0.8)
#feature maxtrix 분리
train_X = X[:train_N, :]
val_X = X[train_N:val_N, :]
test_X = X[val_N:, :]
#label maxtrix 분리
train_Y = Y[:train_N]
val_Y = Y[train_N:val_N]
test_Y = Y[val_N:]   

#normalization
scaler = preprocessing.StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

#Least squares
regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)
pred_Y= regr.predict(test_X)
mse = mean_squared_error(pred_Y, test_Y)
rmse = np.power(mse, 0.5)
print("Least Squares RMSE : ", rmse)
print("Least Squares variance : ", r2_score(test_Y, pred_Y))
print(feat)
print("Least Squares coef : ", regr.coef_)
print("")



#lasso
alphas = [0.01, 0.1, 0.5, 1, 5, 10, 100]
mse_list = []
#finetune the alpha using the validation set
""" -------- Your Answer -------- """
for alpha in alphas:
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(train_X, train_Y)
    prediction = lasso.predict(val_X)
    mse= mean_squared_error(val_Y,prediction)
    mse_list.append(mse)


""" ----------------------------- """
idx = mse_list.index(min(mse_list))
print("Lasso best alpha : ", alphas[idx])
#evaluation
""" -------- Your Answer -------- """
lasso = linear_model.Lasso(alpha=alphas[idx])
lasso.fit(train_X, train_Y)
pred_Y = lasso.predict(test_X)
""" ----------------------------- """
mse = mean_squared_error(pred_Y, test_Y)
rmse = np.power(mse, 0.5)
print("Lasso RMSE : ", rmse)
print("Lasso variance : ", r2_score(test_Y, pred_Y))
print(feat)
print("Lasso coef : ", lasso.coef_)
print("")

#Ridge
#finetune the alpha using the validation set
alphas = [0.01, 0.1, 0.5, 1, 5, 10, 100]
mse_list = []
""" -------- Your Answer -------- """
for alpha in alphas:
    ridge= linear_model.Ridge(alpha=alpha)
    ridge.fit(train_X, train_Y)
    pred_Y = ridge.predict(val_X)
    mse = mean_squared_error(pred_Y, val_Y)
    mse_list.append(mse)
""" ----------------------------- """
idx = mse_list.index(min(mse_list))
print("Ridge alpha : ", alphas[idx])
#evaluation
""" -------- Your Answer -------- """
ridge = linear_model.Ridge(alpha=alphas[idx])
ridge.fit(train_X, train_Y)
y_pred = ridge.predict(test_X)
""" ----------------------------- """
mse = mean_squared_error(y_pred, test_Y)
rmse = np.power(mse, 0.5)
print("Ridge RMSE : ", rmse)
print("Ridge variacne : ", r2_score(test_Y, y_pred))
print(feat)
print("Ridge coef : ", ridge.coef_)