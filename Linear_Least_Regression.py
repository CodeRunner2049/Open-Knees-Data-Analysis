import h5py
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model

class linear_regression ():

    def __init__(self, data):
        self.x_data = data[0]
        self.y_data = data[1]

    def regression (self):
        #Looping through table columns and creating regression equations
        for (columnName, columnData) in self.y_data.iteritems():
            X = self.x_data
            Y = self.y_data[columnName]

            #Regression done with sklearn
            regr = linear_model.LinearRegression()
            regr.fit(X, Y)

            print(columnName + " Intercept: " + str(regr.intercept_))
            print(columnName + " Coefficients: " + str(regr.coef_))
