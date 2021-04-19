import pandas as pd
import numpy as np
import os
import tensorflow as tf
import statistics
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras import callbacks
from keras.models import Sequential
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def train_test_neural_networks(kinetics_data, kinematics_data):
    """Loops through the 6 kinematics channels and builds 6 neural networks"""

    # Store the neural networks
    all_neural_networks = []

    # Loop through the kinematics channels and build a neural network with all kinetics channels as input
    for channel in kinematics_data.columns:
        print("Generating neural network for " + channel + " column...")
        all_neural_networks.append(nonlinear_neural_network(kinetics_data, kinematics_data[channel], channel))

    return all_neural_networks


def nonlinear_neural_network(kinetics, kinematics, channel):
    """Build a neural network net with 6 input nodes, 2 hidden layers, and 1 output node """

    # change the directory to the current director
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    #var_names = list(kinematics)
    kinetics = kinetics.to_numpy()
    kinematics = kinematics.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(kinetics, kinematics, test_size=0.25, random_state=6)

    # New sequential network structure.
    model = Sequential()

    # Input layer with input dimension 6 and hidden layer i with 100 neurons.
    model.add(Dense(100, activation="tanh", input_dim=6))
    # Hidden layer j with 50 neurons plus activation layer.
    model.add(Dense(50, activation="tanh"))
    # Hidden layer k with 28 neurons.
    model.add(Dense(28, activation="tanh"))
    # Output Layer
    model.add(Dense(1, activation="linear"))

    # Model is derived and compiled using mean square error as loss
    # # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

    early_stopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience=5,restore_best_weights=True)

    # Fit the model
    np.random.seed(6)
    model.fit(x_train, y_train, epochs=5000, batch_size=100,  verbose=2,
              validation_data=(x_test, y_test), callbacks=early_stopping)
    # Calculate predictions
    PredTestSet = model.predict(x_train)
    PredValSet = model.predict(x_test)

    # Save predictions
    np.savetxt("trainresults.csv", PredTestSet, delimiter=",")
    np.savetxt("valresults.csv", PredValSet, delimiter=",")

    # Plot actual vs predition for training set
    TestResults = np.genfromtxt("trainresults.csv", delimiter=",")

    # # evaluate the model
    # scores = model.evaluate(X1, Y1, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Compute R-Square value for training set
    TestR2Value = r2_score(y_train, TestResults)
    TestMSEValue = mean_squared_error(y_train, TestResults)
    print("Training Set R-Square=", TestR2Value)
    print("Training Set MSE=", TestMSEValue)

    # Plot actual vs prediction for validation set
    ValResults = np.genfromtxt("valresults.csv", delimiter=",")

    # Compute R-Square value for validation set
    ValR2Value = r2_score(y_test, ValResults)
    ValmseValue = mean_squared_error(y_test, ValResults)

    print("Validation Set R-Square=", ValR2Value)
    print("Validation Set MSE=", ValmseValue)
    model.save(r'OK_Data_Graphs/NN_Models/'+'_Flexion_flipped'+str(channel))

    plt.plot(y_train,TestResults,'ro')

    plt.xlabel('Actual', size = 24)
    plt.ylabel('Predicted', size = 24)
    plt.title(r'Training set: Flexion_flipped'+str(channel)+'\n\nR^2= '+str(TestR2Value)+', RMSE='+str(np.sqrt(TestMSEValue)), size=24)
    # plt.title(r'Training set: Flexion_flipped'+'\n\nR^2= ' +str(TestR2Value)+ ', RMSE= '+ str(np.sqrt(TestMSEValue)), size = 24)
    plt.savefig(r'OK_Data_Graphs/NN_Models/Training_Flexion_flipped'+str(channel), dpi="figure", bbox_inches="tight")
    # plt.savefig(r'dat/NN_Models/Figures/Training_Flexion_flipped')
    plt.show()

    plt.plot(y_test,ValResults,'ro')
    plt.xlabel('Actual', size = 24)
    plt.ylabel('Predicted', size = 24)
    plt.title('Validation set: Flexion_flipped'+str(channel)+'\n\nR^2= '+str(ValR2Value)+',RMSE= '+str(np.sqrt(ValmseValue)), size=24)
    plt.savefig(r'OK_Data_Graphs/NN_Models/Validation_Flexion_flipped'+str(channel), dpi="figure", bbox_inches="tight")

    # plt.title('Validation set: Flexion_flipped'+'\n\nR^2= ' +str(ValR2Value)+ ',RMSE= '+ str(np.sqrt(ValmseValue)), size = 24)
    # plt.savefig('dat/NN_Models/Figures/Validation_Flexion_flipped')
    plt.show()

    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=3)
    # y_train = y_train.to_frame()
    # y_test = y_test.to_frame()
    #
    # # New sequential network structure.
    # model = Sequential()
    #
    # # Input layer with dimension 6 and hidden layer i with 128 neurons.
    # model.add(Dense(128, input_dim=6, activation='relu'))
    # # Dropout of 20% of the neurons and activation layer.
    # model.add(Dropout(.2))
    # model.add(Activation("linear"))
    # # Hidden layer j with 64 neurons plus activation layer.
    # model.add(Dense(64, activation='relu'))
    # model.add(Activation("linear"))
    # # Hidden layer k with 64 neurons.
    # model.add(Dense(64, activation='relu'))
    # # Output Layer.
    # model.add(Dense(1))
    #
    # # Model is derived and compiled using mean square error as loss
    # # function, accuracy as metric and gradient descent optimizer.
    # model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    #
    # # Training model with train data. Fixed random seed:
    # np.random.seed(3)
    # model.fit(X_train, y_train, epochs=256, batch_size=2, verbose=2)
    #
    # # Predict response variable with new data
    # predicted = model.predict(X_test)
    #
    # # Plot in blue color the predicted data and in green color the
    # # actual data to verify visually the accuracy of the model.
    # pyplot.plot(predicted, color="blue")
    # pyplot.plot(y_test[channel].tolist(), color="green")
    # fig.savefig(r"C:\Users\techteam\Documents\Open-Knees-Presentation-Material\", dpi="figure", bbox_inches=bbox)
    # pyplot.show()

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
            nn = neural_network(self.x_data, self.y_data[y_columnName], 1)
            self.list_of_neural_networks.append(nn)

    def generate_one_column_nn (self, optimizer, y_columnName):
        print("Generating neural network for " + y_columnName + " column")
        for y_columnName in self.y_data.columns:
            print(y_columnName)
        nn = neural_network(self.x_data, self.y_data[y_columnName], 1, optimizer)
        self.list_of_neural_networks.append(nn)

    def generate_singular_nn(self):
        nn = neural_network(self.x_data, self.y_data, 6)

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
    def __init__(self, x, y, ouput_layers, optimizer):
        super(neural_network, self).__init__(x, y)
        self.construct_nn(ouput_layers, optimizer)

    def get_compiled_model(self, output_layers, optimizer):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=6),
            tf.keras.layers.Dense(units=6),
            tf.keras.layers.Dense(output_layers)
        ])
        # Compile the model (currently using gradiant descent and mean squared error)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

    def construct_nn(self, output_layers, optimizer):
        # Name of column of the data regression is predicting
        self.y_columnName = self.y_data.name
        print(self.y_columnName + " neural network:")

        # Generate the dataset used for the predicting
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train.values, self.y_train.values))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test.values, self.y_test.values))

        # Shuffle and batch the dataset
        train_dataset = train_dataset.shuffle(len(self.x_train)).batch(1)
        test_dataset = test_dataset.shuffle(len(self.x_test)).batch(1)

        # Fit a model from the training data and test its accuracy against testing data
        model = self.get_compiled_model(output_layers, optimizer)
        self.fit = model.fit(train_dataset, epochs=10, verbose=2)
        #print(self.fit.history)
        self.test_loss = model.evaluate(test_dataset, verbose=2)
        print("test loss, test acc:", self.test_loss)

    #def construct_singular_NN (self):
