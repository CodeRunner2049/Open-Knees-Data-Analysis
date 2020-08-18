from File_Opener import fileOpener
from Linear_Least_Regression import linear_regression
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
    fo = fileOpener(data_dir, pkg_name)
    files = fo.print_files()
    hdf5_path = hdf5_dir + "\\" + pkg_name + "_hierachical_data.hdf5"
    image_path = image_dir + "\\" + pkg_name + "_graphs.png"
    legend_path = image_dir + "\\" + pkg_name + "_graphs_legend.png"
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
            elif pruner_inp == 2:
                exclusionary_list = fo.split_data(input("Enter a comma seperated list of files to exclude (eg. 2, 4, 7): "))
            else:
                raise ValueError
        except ValueError:
            print("Not a valid input please input again")
        else:
            break
    #".\Open Knees File Visualization\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics"
    #D:\Mourad\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics
    fo.readTDMS(fo.prune_files(pruner_inp, files, exclusionary_list))
    lr = linear_regression(fo.readHDF5(hdf5_path))
    lr.regression()

main()
