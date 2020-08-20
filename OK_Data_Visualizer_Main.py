from File_Opener import file_opener
from Algorithm import algorithm
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
    data_dir = input("Enter in the directory (filepath) of the Open Knee data that you would like to analyze: ")
    pkg_name = re.search("joint_mechanics-oks\d{3}", data_dir).group(0)
    fo = file_opener(data_dir, pkg_name)
    fo.algorithm.do_linear_regression()
    while True:
        try:
            pruner_inp = int(input("Would you like to include or exclude certain data?: (enter 0 to skip/enter 1 to include/enter 2 to exclude): "))
            if not pruner_inp in range(0, 3):
                raise ValueError
            exclusionary_list = []
            if pruner_inp == 0:
                break
            elif pruner_inp == 1:
                exclusionary_list = fo.split_data(input("Enter a comma seperated list of files to include (eg. 2, 4, 7): "))
                fo = file_opener(data_dir, pkg_name, fo.prune_files(pruner_inp, exclusionary_list))
                fo.algorithm.do_linear_regression()
            elif pruner_inp == 2:
                exclusionary_list = fo.split_data(input("Enter a comma seperated list of files to exclude (eg. 2, 4, 7): "))
                fo = file_opener(data_dir, pkg_name, fo.prune_files(pruner_inp, exclusionary_list))
                fo.algorithm.do_linear_regression()
            else:
                raise ValueError
        except ValueError:
            print("Not a valid input please input again")
        else:
            break

    #Loop to check for user input to graph data
    while True:
        try:
            graph_inp = str(input("Would you like to graph your data? (Y/N): "))
            if not graph_inp.lower() == 'y' or graph_inp.lower() == 'n':
                raise ValueError
            elif graph_inp.lower() == 'y':
                fo.graph_data()
            elif graph_inp.lower() == 'n':
                break
            else:
                raise ValueError
        except ValueError:
            print("Not a valid input please input again")
        else:
            break
    #".\Open Knees File Visualization\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics"
    #D:\Mourad\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics

main()
