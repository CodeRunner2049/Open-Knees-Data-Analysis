import pandas as pd
import tensorflow as tf
import numpy as np
import statistics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class algorithm():

    def __init__(self, x_df, y_df):
        self.x_data = x_df
        self.y_data = y_df
        self.list_of_regressions = []
        self.list_of_neural_networks = []
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data,
                                                                                test_size=0.25, random_state=0)

    def do_linear_regression(self):
        print("Generating a regression... This may take a moment")

        # Looping through kinematics data columns and creating regression objects
        # then appending to regression array
        for y_columnName in self.y_data.columns:
            lr = linear_regression(self.x_data, self.y_data[y_columnName])
            self.list_of_regressions.append(lr)

    def generate_neural_networks(self):
        print("Generating neural networks... This may take a moment")

        # Loop through kinematics data columns and create nn object and append to nn array
        for y_columnName in self.y_data.columns:
            nn = neural_network(self.x_data, self.y_data[y_columnName])
            self.list_of_neural_networks.append(nn)

    def do_std_devation(self, data):
        return statistics.stdev(data)

    def get_regression_list(self):
        return self.list_of_regressions

    def get_neural_network_list(self):
        return self.list_of_neural_networks


class linear_regression(algorithm):

    def __init__(self, x, y):
        super(linear_regression, self).__init__(x, y)
        self.define_regression()

    def define_regression(self):
        # Name of column of the data regression is predicting
        self.y_columnName = self.y_data.name

        # Regression done with sklearn
        self.regr = linear_model.LinearRegression()
        self.regr.fit(self.x_train, self.y_train)

        # Generate prediction array from the testing data
        self.pred = self.regr.predict(self.x_test)
        self.actual_vrs_pred = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.pred})

        # Calculate the mean squared error
        self.ms_error = mean_squared_error(self.y_test, self.pred)

        # Calculate the coefficient of determination: 1 is the perfect precition
        self.r2_score = r2_score(self.y_test, self.pred)

        # Print out the equation
        self.str_equation = str(self.y_columnName) + ' = ' + ''.join([str(coef) + ' * ' + str(x_columnName) + ' + '
                                                                      for x_columnName, coef in
                                                                      zip(self.x_train.columns,
                                                                          self.regr.coef_)]) + str(self.regr.intercept_)
        print(self.str_equation)
        print('Mean squared error: ' + str(self.ms_error))
        print('Coefficient of Determination: ' + str(self.r2_score) + '\n')


class neural_network(algorithm):
    # Constructor for the neural network class
    def __init__(self, x, y):
        super(neural_network, self).__init__(x, y)
        self.construct_nn()

    def get_compiled_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=6),
            tf.keras.layers.Dense(units=6),
            tf.keras.layers.Dense(1)
        ])
        # Compile the model (currently using gradiant descent and mean squared error)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def construct_nn(self):
        # Name of column of the data regression is predicting
        self.y_columnName = self.y_data.name
        print(self.y_columnName + "neural network:")

        # Generate the dataset used for the predicting
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train.values, self.y_train.values))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test.values, self.y_test.values))

        # Shuffle and batch the dataset
        train_dataset = train_dataset.shuffle(len(self.x_train)).batch(1)
        test_dataset = test_dataset.shuffle(len(self.x_test)).batch(1)

        # Fit a model from the training data and test its accuracy against testing data
        model = self.get_compiled_model()
        model.fit(train_dataset, epochs=6)
        self.test_loss = model.evaluate(test_dataset)
        print(self.test_loss)
