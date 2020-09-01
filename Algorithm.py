import pandas as pd
import numpy as np
import statistics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class algorithm ():

    def __init__(self, x_df, y_df):
        self.x_data = x_df
        self.y_data = y_df
        self.list_of_regressions = []
        self.regression_coef_dict = {}
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.25, random_state=0)

    def do_linear_regression(self):
        print("Generating a regression... This may take a moment")

        #Looping through table columns and creating regression objects
        #then appending to regression array
        for y_columnName in self.y_data.columns:
            lr = linear_regression(self.x_data, self.y_data[y_columnName])
            self.list_of_regressions.append(lr)
            #self.regression_coef_dict[y_columnName] = {x_col:coef for x_col, coef in zip(self.x_train.columns, lr.regr.coef)}

    def do_std_devation (self, data):
        return statistics.stdev(data)

    def get_regression_list(self):
        return self.list_of_regressions

class linear_regression (algorithm):

    def __init__(self, x, y):
        super(linear_regression, self).__init__(x, y)
        self.define_regression()

    def define_regression (self):
        self.y_columnName = self.y_data.name

        #Regression done with sklearn
        self.regr = linear_model.LinearRegression()
        self.regr.fit(self.x_train, self.y_train)

        #Generate prediction array from the testing data
        self.pred = self.regr.predict(self.x_test)
        self.actual_vrs_pred = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.pred})

        #Calculate the mean squared error
        self.ms_error = mean_squared_error(self.y_test, self.pred)

        #TODO Add coefficient of correlation

        #Calculate the coefficient of determination: 1 is the perfect precition
        self.r2_score = r2_score(self.y_test, self.pred)

        #Print out the equation
        self.str_equation = str(self.y_columnName) + ' = ' + ''.join([str(coef) + ' * ' + str(x_columnName) + ' + '
                                for x_columnName, coef in zip(self.x_train.columns, self.regr.coef_)]) + str(self.regr.intercept_)
        print(self.str_equation)
        print('Mean squared error: ' + str(self.ms_error))
        print('Coefficient of Determination: ' + str(self.r2_score) + '\n')
