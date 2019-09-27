import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = np.loadtxt("weight-height.csv", delimiter = ',', skiprows = 1, usecols = (1,2))
X = data[:int(data.shape[0]/2),0]
y = data[:int(data.shape[0]/2),1]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size = 0.33, random_state = 42)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
X_val = X_val.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_val = y_val.reshape(-1,1)

alphas = [0.0001, 0.001, 0.01, 0.1, 1]
mseList = []

for alpha in alphas:
    ridge = linear_model.Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    prediction = ridge.predict(X_val)
    mse = mean_squared_error(y_val, prediction)
    mseList.append(mse)
    
idx = mseList.index(min(mseList))

# Create linear regression object
ridge = linear_model.Ridge(alphas[idx])

# Train the model using the training sets
ridge.fit(X_train, y_train)

# Make predictions using the testing set
prediction = ridge.predict(X_test)

# The best alpha
print("Best alpha: \n", alphas[idx])
# The coefficients
print('Coefficients: \n', ridge.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, prediction))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, prediction))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, prediction, color='blue', linewidth=3)
plt.xlabel("height")
plt.ylabel("weight")

plt.xticks(())
plt.yticks(())

plt.show()
