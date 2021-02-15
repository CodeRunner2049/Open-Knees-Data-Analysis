from File_Opener import file_opener
from collections import defaultdict
import os
import re

def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    hdf5_dir = os.path.join(current_directory, "hdf5_files")
    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)
    image_dir = os.path.join(current_directory, "OK_Data_Graphs")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    while True:
        try:
            file_or_directory = int(input("Would you like to read from a directory or from a stored h5 file? (1 for directory, "
                      "2 for h5)"))
            if not file_or_directory in range(1, 3):
                raise ValueError
            elif file_or_directory == 1:
                pkg_dir = input("Enter in the directory (filepath) of the Open Knee data that you would like to analyze: ")
                pkg_name = re.search("joint_mechanics-oks\d{3}", pkg_dir).group(0)
                fo = file_opener(1, pkg_dir, pkg_name)
                break
            elif file_or_directory == 2:
                h5_pkgs = defaultdict(dict)
                index = 0
                for root, dirs, files in os.walk(hdf5_dir, topdown=True):
                    for subdir in dirs:
                        h5_pkgs[index] = (str(subdir), os.path.join(root, subdir))
                        index += 1
                print(h5_pkgs)
                val = int(input("Which experiment would you like to analyze (enter the number associated ie. 3)\n"))
                pkg_name, pkg_dir = h5_pkgs[val][0], h5_pkgs[val][1]
                fo = file_opener(2, pkg_dir, pkg_name)
                break
            else:
                raise ValueError
        except ValueError:
            print("Not a valid input please input again")
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
    #".\Open Knees File Visualization\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics"
    #D:\Mourad\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics

main()
