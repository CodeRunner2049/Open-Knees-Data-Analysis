from nptdms import TdmsFile
from collections import defaultdict
from Algorithm import algorithm
import numpy as np
import pandas as pd
import copy
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class file_opener:
    """Constructor for the the file_opener class. Initializes the data dictionaries and the directory name"""

    def __init__(self, file_or_directory, directory, pkg_name):
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_pkg_name = pkg_name
        if (file_or_directory == 1):
            self.file_dict = self.build_file_dict(directory)
            # Build the x & y dataframes from the presented files
            self.x_dataframe, self.y_dataframe = self.files_to_dataframes(self.file_dict)
            self.write_HDF5(self.x_dataframe, self.y_dataframe)
            self.print_files_with_deviation()
        elif (file_or_directory == 2):
            self.x_dataframe, self.y_dataframe = self.read_HDF5(directory)
        else:
            ValueError("invalid input please restart")
        # Ask the user if they would like to write to hdf5 or continue
        # Generate class attribute algorithm imported from algorithm class (drop filename column)
        #self.algorithm = algorithm(self.x_dataframe, self.y_dataframe)

    def ask_h5(self):
        while True:
            try:
                h5_inp = str(input("Would you like to graph your data? (Y/N): "))
                if not h5_inp.lower() == 'y' or h5_inp.lower() == 'n':
                    raise ValueError
                elif h5_inp.lower() == 'y':
                    return True
                    break
                elif h5_inp.lower() == 'n':
                    return False
                else:
                    raise ValueError
            except ValueError:
                print("Not a valid input please input again")

    def verify_file(self, file):

        """Verify if the file has empty columns"""

        with TdmsFile.open(file) as contents:
            try:
                for kinetics in contents["State.JCS Load"].channels():
                    kinetics[0:1]
                for kinematics in contents["State.JCS Load"].channels():
                    kinematics[0:1]
            except KeyError:
                return False

            return True

    def build_file_dict(self, directory):
        index = 0
        file_dict = defaultdict(dict)
        for root, dirs, files in os.walk(directory, topdown=True):
            for filename in files:
                print(filename)
                if filename.endswith("_main_processed.tdms") and self.verify_file(os.path.join(root, filename)) is True:
                    print(filename)
                    file_dict[index] = {'root': root, 'filename': filename}
                    index += 1;
        return file_dict

    def print_files_with_deviation(self):
        self.clear_deprecated_files()
        for index, root_file in self.file_dict.items():
            filename = root_file['filename']
            x_chunk = self.x_dataframe.xs(filename, level=1, axis=0, drop_level=False)
            y_chunk = self.y_dataframe.xs(filename, level=1, axis=0, drop_level=False)
            print("file " + str(index) + ": " + str(root_file['filename']) + "\n")
            # [print(x_columnName + " std_deviation: " +
            #        str(self.algorithm.do_std_devation(x_chunk[x_columnName]))) for x_columnName in
            #  x_chunk.columns.values]
            # [print(y_columnName + " std_deviation: " +
            #        str(self.algorithm.do_std_devation(y_chunk[y_columnName]))) for y_columnName in
            #  y_chunk.columns.values]
            print('\n')

    def split_data(self, file_inp):
        file_list = file_inp.split(",")
        int_list = []
        for i in file_list:
            int_list.append(int(i))
        return int_list

    def prune_files(self, split, pruner):
        # If the split is zero pass the pruning and return temp, currently same as files
        if split == 0:
            pass
        # If the split is 1 only keep the files that are specified in the pruner, delete the rest
        elif split == 1:
            for index in list(self.file_dict):
                if not index in pruner:
                    del self.file_dict[index]
        # If the split is 2 remove the files specified in the pruner
        elif split == 2:
            for index in pruner:
                del self.file_dict[index]
        return self.file_dict

    def clear_deprecated_files(self):
        for index in list(self.file_dict):
            if '(deprecated)' in self.file_dict[index]['filename']:
                del self.file_dict[index]

    def clean_dataframes(self, x_df, y_df):
        assert isinstance(x_df, pd.DataFrame)
        assert isinstance(y_df, pd.DataFrame)
        indexes_to_drop = []
        for (x_columnName, x_columnData), (y_columnName, y_columnData) in zip(x_df.iteritems(), y_df.iteritems()):
            indexes_to_drop.extend(x_df[x_df[x_columnName].isnull()].index.tolist())
            indexes_to_drop.extend(y_df[y_df[y_columnName].isnull()].index.tolist())
        return x_df.drop(indexes_to_drop), y_df.drop(indexes_to_drop)

    def build_dataframe(self, x_data, y_data, filename):

        """Writes x and y data parameters as pandas dataframes. If one of the dataframes is empty return two empty
        dataframes"""

        x_df = pd.DataFrame(x_data, columns=[k for k in x_data.keys()])
        y_df = pd.DataFrame(y_data, columns=[k for k in y_data.keys()])
        new_x_df, new_y_df = self.clean_dataframes(x_df, y_df)
        if new_x_df.empty == True or new_y_df.empty == True:
            for index in list(self.file_dict):
                if self.file_dict[index]['filename'] == filename:
                    self.file_dict[index]['filename'] += " (deprecated)"
            return None, None
        return new_x_df, new_y_df

    def files_to_dataframes(self, file_list):
        print("Reading files and loading data... This may take a moment")
        x_df, y_df = [list(df) for df in
                      zip(*[self.read_TDMS(root_file['root'], root_file['filename']) for root_file in
                            file_list.values()])]
        # x_df, y_df = list(filter(None, x_df)), list(filter(None, y_df))
        new_x_df, new_y_df = self.clean_dataframes(pd.concat(x_df), pd.concat(y_df))
        # Reset the indexes to be numbered from 0 to len(new_x_df)
        new_x_df.reset_index(drop=True, inplace=True)
        new_y_df.reset_index(drop=True, inplace=True)
        # Append a second index that sorts the dataframe by the filename
        new_x_df.set_index(['filename'], inplace=True, append=True, drop=True)
        new_y_df.set_index(['filename'], inplace=True, append=True, drop=True)
        return new_x_df, new_y_df

    def update_dataframes(self, pruner_inp, exclusionary_list):
        self.prune_files(pruner_inp, exclusionary_list)
        self.x_dataframe, self.y_dataframe = self.files_to_dataframes(self.file_dict)
        self.print_files_with_deviation()
        print("Your linear regressions have changed to: ")
        self.algorithm = algorithm(self.x_dataframe, self.y_dataframe)
        self.algorithm.do_linear_regression()

    def write_HDF5(self, x_df, y_df):

        """Writes x and y dataframes into an HDF5 file"""

        print("Pushing data to hdf5 file... This may take a moment")
        hdf5_dir = os.path.join(self.current_directory, "hdf5_files")
        h5_pkg = os.path.join(hdf5_dir, self.data_pkg_name)
        if not os.path.exists(h5_pkg):
            os.mkdir(h5_pkg)
        x_hdf5_path, y_hdf5_path = h5_pkg + "\\" + self.data_pkg_name + "_x_hierarchical_data.h5", h5_pkg + "\\" + self.data_pkg_name + "_y_hierarchical_data.h5"
        x_df.to_hdf(x_hdf5_path, key='x_data', mode='w')
        y_df.to_hdf(y_hdf5_path, key='y_data', mode='w')

    def write_HDF5(self, kinetics, kinematics):
        """Writes kinetics and kinematics dataframes into an HDF5 file"""

        print("Pushing data to hdf5 file... This may take a moment")

        # Create a hdf5 directory and add the h5 files there
        hdf5_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hdf5_files")
        if not os.path.exists(hdf5_dir):
            os.mkdir(hdf5_dir)
        h5_pkg = os.path.join(hdf5_dir, self.data_pkg_name)
        if not os.path.exists(h5_pkg):
            os.mkdir(h5_pkg)
        x_hdf5_path = h5_pkg + "\\" + self.data_pkg_name + "_kinetics_hierarchical_data.h5"
        y_hdf5_path = h5_pkg + "\\" + self.data_pkg_name + "_kinematics_hierarchical_data.h5"

        # Store the dataframes in the hdf5 files
        kinetics.to_hdf(x_hdf5_path, key='x_data', mode='w')
        kinematics.to_hdf(y_hdf5_path, key='y_data', mode='w')

    def read_HDF5(self, directory):
        for root, dirs, files in os.walk(directory, topdown=True):
            for file in files:
                filepath = os.path.join(root, file)
                if (filepath).endswith("x_hierarchical_data.h5"):
                    x_h5_path = filepath
                elif (filepath).endswith("y_hierarchical_data.h5"):
                    y_h5_path = filepath
                else:
                    raise ValueError("something went wrong, please try another directory")
        x_df = pd.read_hdf(x_h5_path, 'x_data')
        y_df = pd.read_hdf(y_h5_path, 'y_data')
        return x_df, y_df


    def read_TDMS(self, root, filename):
        file = os.path.join(root, filename)
        print(file)
        with TdmsFile.open(file) as tdms_file:
            x_data, y_data = defaultdict(list), defaultdict(list)
            # Generate the x & y data as a dictionary for each tdms file
            for x_ch, y_ch in zip(tdms_file["State.JCS Load"].channels(), tdms_file["State.Knee JCS"].channels()):
                x_data[str(x_ch.name) + '(' + str(x_ch.properties['NI_UnitDescription']) + ')'].extend(
                    np.array(x_ch[:]))
                y_data[str(y_ch.name) + '(' + str(y_ch.properties['NI_UnitDescription']) + ')'].extend(
                    np.array(y_ch[:]))
                x_data_len, y_data_len = len(x_ch[:]), len(y_ch[:])
            # Remove the file from the file_dict if it is empty
            if x_data_len == 0 and y_data_len == 0:
                for index in list(self.file_dict):
                    if self.file_dict[index]['filename'] == filename:
                        self.file_dict[index]['filename'] += " (deprecated)"
            # Create a new column that contains the filename of data
            x_data['filename'] = np.full(x_data_len, filename)
            y_data['filename'] = np.full(y_data_len, filename)
            # close the file
            tdms_file.close()
            return self.build_dataframe(x_data, y_data, filename)

    def export_legend(self, legend, filename):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    def graph_data(self, path):
        # Uses matplotlib to graph given data
        print("Generating plots... This may take a moment")

        plt.close('all')
        font = {'family': 'calibri',
                'weight': 'bold',
                'size': 12, }
        fig, axs = plt.subplots(len(self.x_dataframe.columns.values), len(self.y_dataframe.columns.values),
                                figsize=(25, 15))
        x_data, y_data = self.x_dataframe, self.y_dataframe

        for root_file in self.file_dict.values():
            filename = root_file['filename']
            x_chunk = x_data.xs(filename, level=1, axis=0, drop_level=False)
            y_chunk = y_data.xs(filename, level=1, axis=0, drop_level=False)
            for y_idx, y_col in enumerate(y_chunk.columns.values):
                for x_idx, x_col in enumerate(x_chunk.columns.values):
                    axs[y_idx, x_idx].scatter(x_chunk[x_col], y_chunk[y_col], s=0.7, label=filename)

        leg = plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        for ax in axs.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        # Set column labels
        for x_idx, x_col in enumerate(x_data.columns.values):
            axs[-1, x_idx].set_xlabel(x_col, fontdict=font)
        for y_idx, y_col in enumerate(y_data.columns.values):
            axs[y_idx, 0].set_ylabel(y_col, fontdict=font)

        plt.tight_layout()
        fig.canvas.set_window_title(self.data_pkg_name)
        image_path = path + "_graphs.png"
        legend_path = path + "_graphs_legend.png"
        plt.savefig(image_path, format="png", dpi=300, bbox_inches="tight")
        self.export_legend(leg, legend_path)
        # plt.show()
        plt.close()

    def graph_regression_error(self, path):
        print("Generating error plots... This may take a moment")
        plt.close('all')
        font = {'family': 'calibri',
                'weight': 'bold',
                'size': 12, }
        fig, axs = plt.subplots(1, len(self.y_dataframe.columns.values), figsize=(30, 40))
        x_data, y_data = self.algorithm.x_test, self.y_dataframe

        for root_file in self.file_dict.values():
            filename = root_file['filename']
            for idx, (x_col, y_col) in enumerate(zip(x_data.columns.values, y_data.columns.values)):
                y_regr = self.algorithm.list_of_regressions[idx].actual_vrs_pred
                x_chunk = x_data.xs(filename, level=1, axis=0, drop_level=False)
                y_chunk = y_regr.xs(filename, level=1, axis=0, drop_level=False)
                y_pred_error = [y - y_prime for y, y_prime in zip(y_chunk['Actual'], y_chunk['Predicted'])]
                axs[idx].scatter(x_chunk[x_col], y_pred_error, s=0.7, label=filename)

        leg = plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        plt.tight_layout()
        fig.canvas.set_window_title(self.data_pkg_name + ' Regression Errors')
        regerror_path = path + "_regression_error_graphs.png"
        # plt.savefig(regerror_path, format="png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Plot generation complete check your current working directory for graph images and the hdf5 file")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))
    hdf5_dir = os.path.join(current_directory, "hdf5_files")
    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)
    image_dir = os.path.join(current_directory, "OK_Data_Graphs")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    pkg_dir = input("Enter in the directory (filepath) of the Open Knee data that you would like to analyze: ")
    pkg_name = re.search("joint_mechanics-oks\d{3}", pkg_dir).group(0)
    fo = file_opener(1, pkg_dir, pkg_name)
    # while True:
    #     try:
    #         file_or_directory = int(input("Would you like to read from a directory or from a stored h5 file? (1 for directory, "
    #                   "2 for h5)"))
    #         if not file_or_directory in range(1, 3):
    #             raise ValueError
    #         elif file_or_directory == 1:
    #             pkg_dir = input("Enter in the directory (filepath) of the Open Knee data that you would like to analyze: ")
    #             pkg_name = re.search("joint_mechanics-oks\d{3}", pkg_dir).group(0)
    #             fo = file_opener(1, pkg_dir, pkg_name)
    #             break
    #         elif file_or_directory == 2:
    #             h5_pkgs = defaultdict(dict)
    #             index = 0
    #             for root, dirs, files in os.walk(hdf5_dir, topdown=True):
    #                 for subdir in dirs:
    #                     h5_pkgs[index] = (str(subdir), os.path.join(root, subdir))
    #                     index += 1
    #             print(h5_pkgs)
    #             val = int(input("Which experiment would you like to analyze (enter the number associated ie. 3)\n"))
    #             pkg_name, pkg_dir = h5_pkgs[val][0], h5_pkgs[val][1]
    #             fo = file_opener(2, pkg_dir, pkg_name)
    #             break
    #         else:
    #             raise ValueError
    #     # except ValueError:
    #     #     print("Not a valid input please input again")
    fo.algorithm.do_linear_regression()
    while True:
        try:
            pruner_inp = int(input("Would you like to include or exclude certain data?: (enter 0 to skip/enter 1 to "
                                   "include/enter 2 to exclude): "))
            if not pruner_inp in range(0, 3):
                raise ValueError
            if pruner_inp == 0:
                break
            elif pruner_inp == 1:
                exclusionary_list = fo.split_data(input("Enter a comma seperated list of files to include (eg. 2, 4, 7): "))
                fo.update_dataframes(pruner_inp, exclusionary_list)
                break
            elif pruner_inp == 2:
                exclusionary_list = fo.split_data(input("Enter a comma seperated list of files to exclude (eg. 2, 4, 7): "))
                fo.update_dataframes(pruner_inp, exclusionary_list)
                break
            else:
                raise ValueError
        except ValueError:
            print("Not a valid input please input again")

    #Loop to check for user input to graph data
    image_dir = os.path.join(current_directory, "OK_Data_Graphs")
    image_path = image_dir + "\\" + pkg_name
    while True:
        try:
            graph_inp = str(input("Would you like to graph your data? (Y/N): "))
            if not graph_inp.lower() == 'y' or graph_inp.lower() == 'n':
                raise ValueError
            elif graph_inp.lower() == 'y':
                fo.graph_data(image_path)
                fo.graph_regression_error(image_path)
                break
            elif graph_inp.lower() == 'n':
                break
            else:
                raise ValueError
        except ValueError:
            print("Not a valid input please input again")

    fo.algorithm.generate_neural_networks()
    neural_networks = fo.algorithm.get_neural_network_list()
    for nn in neural_networks:
        print(nn.y_columnName + " neural network mean_squared error and accuracy: " + str(nn.test_loss))
    #".\Open Knees File Visualization\joint_mechanics-oks009\joint_mechanics-oks00\TibiofemoralJoint\KinematicsKinetics"
    #D:\Mourad\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics


