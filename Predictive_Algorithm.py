import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class predictive_algorithm ():

    def __init__(self, x_df, y_df):
        self.x_data = x_df
        self.y_data = y_df
        self.list_of_regressions = []
        self.split_training_test_sets()

    def split_training_test_sets ():
        x_msk = np.random.rand(len(self.x_data)) < 0.75
        y_msk = np.random.rand(len(self.y_data)) < 0.75

        self.x_training_data = self.x_data[msk]
        self.x_test_data = self.x_data[~msk]

        self.y_training_data = self.y_data[msk]
        self.y_test_data = self.y_data[~msk]

    def do_linear_regression(self):
        print("Generating a regression... This may take a moment")

        #Looping through table columns and creating regression objects
        #then appending to regression array
        for y_columnName in self.y_data.columns:
            self.list_of_regressions.append(linear_regression(self.x_data, self.y_data, y_columnName))

    def get_regression_list(self):
        return self.list_of_regressions

class linear_regression (predictive_algorithm):

    def __init__(self, x_data, y_data, current_column):
        super(linear_regression, self).__init__(x_data, y_data)
        self.column = current_column
        self.define_regression()

    def define_regression (self):
        X = self.x_data
        y = self.y_data[self.column]

        #Regression done with sklearn
        self.regr = linear_model.LinearRegression()
        self.regr.fit(X, y)

        #Generate prediction array
        self.pred = self.regr.predict(X)

        #Calculate the mean squared error
        self.ms_error = mean_squared_error(y, self.pred)

        #Print out the equation
        self.str_equation = str(self.column) + ' = ' + ''.join([str(coef) + ' * ' + str(x_columnName) + ' + '
                                for x_columnName, coef in zip(X.columns, self.regr.coef_)]) + str(self.regr.intercept_)
        print(self.str_equation)
        print('Mean squared error: ' + str(self.ms_error) + '\n')
