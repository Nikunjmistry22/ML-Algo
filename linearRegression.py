import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Taking only one features of diabetes to test
diabetes=datasets.load_diabetes()
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-20:]

diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-20:]

model=linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predict=model.predict(diabetes_X_test)

print("mean square error: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predict))
print("Weigh: ",model.coef_)
print("Intercept: ",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predict)
plt.show()