import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = np.loadtxt("weight-height.csv", delimiter = ',', skiprows = 1, usecols = (1,2))

print(data)
print(data.shape[0]/2)
X = data[:int(data.shape[0]/2),0] #use only male's data
y = data[:int(data.shape[0]/2),1] #use only male's label

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
prediction = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
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
