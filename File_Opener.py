from nptdms import TdmsFile
from collections import defaultdict
from Algorithm import algorithm
import numpy as np
import pandas as pd
import os
import itertools
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class file_opener ():

    """Constructor for the the file_opener class. Initializes the data dictionaries and the directory name"""
    def __init__(self, directory, pkg_name, pruned_files = None):
        self.current_directory =  os.path.dirname(os.path.realpath(__file__))
        self.data_pkg_name = pkg_name
        self.directory = directory
        if pruned_files is None:
            self.file_dict = self.build_file_dict()
        else:
            self.file_dict = pruned_files
        #Here self.file_specific_data, self.StateJCSLoad, self.StateKneeJCS, self.load_properties
        #, and self.knee_properties are all instantiated
        self.read_TDMS(self.file_dict)
        self.x_dataframe, self.y_dataframe = self.build_dataframe(self.StateJCSLoad, self.StateKneeJCS)
        #Here self.hdf5_path is instantiated
        self.write_HDF5(self.x_dataframe, self.y_dataframe)

        self.algorithm = algorithm(self.x_dataframe, self.y_dataframe)
        self.print_files_with_deviation()


    def build_file_dict(self):
        index = 0
        file_dict = {}
        for root, dirs, files in os.walk(self.directory, topdown=True):
            for filename in files:
                if(filename.endswith("main_processed.tdms")):
                    file_dict[index] = {'root': root, 'filename' : filename}
                    index+=1;
        return file_dict

    def print_files_with_deviation (self):
        for index, root_file in self.file_dict.items():
            print("file " + str(index) + ": " + str(root_file['filename']))

    def split_data(self, file_inp):
        file_list = file_inp.split(",")
        int_list = []
        for i in file_list:
            int_list.append(int(i))
        return int_list

    def prune_files(self, split, pruner):
        #If the split is zero pass the pruning and return temp, currently same as files
        if split == 0:
            pass
        #If the split is 1 only keep the files that are specified in the pruner, delete the rest
        elif split == 1:
            for index in list(self.file_dict):
                if not index in pruner:
                    del self.file_dict[index]
        #If the split is 2 remove the files specified in the pruner
        elif split == 2:
            for index in pruner:
                del self.file_dict[index]
        return self.file_dict

    def clean_dataframe (self, x_df, y_df):
        assert isinstance(x_df, pd.DataFrame)
        assert isinstance(y_df, pd.DataFrame)
        #x_NaN_indexes, y_NaN_indexes = x_df[x_df.isna().any(axis=0)].index.tolist(), y_df[y_df.isna().any(axis=0)].index.tolist()
        indexes_to_drop = []
        for (x_columnName, x_columnData), (y_columnName, y_columnData) in zip(x_df.iteritems(), y_df.iteritems()):
            indexes_to_drop.extend(x_df[x_df[x_columnName].isnull()].index.tolist())
            indexes_to_drop.extend(y_df[y_df[y_columnName].isnull()].index.tolist())
        return x_df.drop(indexes_to_drop), y_df.drop(indexes_to_drop)

    """Writes x and y data parameters as pandas dataframes"""
    def build_dataframe(self, x_data, y_data):
        x_df = pd.DataFrame(x_data, columns=[k for k in x_data.keys()], dtype = 'float64')
        y_df = pd.DataFrame(y_data, columns=[k for k in y_data.keys()], dtype = 'float64')
        return self.clean_dataframe(x_df, y_df)

    """Writes x and y dataframes into an HDF5 file"""
    def write_HDF5 (self, x_df, y_df):
        print("Pushing data to hdf5 file... This may take a moment")
        print(x_df)
        print(y_df)
        hdf5_dir = os.path.join(self.current_directory, "hdf5_files")
        self.hdf5_path = hdf5_dir + "\\" + self.data_pkg_name + "_hierachical_data.hdf5"
        store = pd.HDFStore(self.hdf5_path, "w")
        store['StateJCSLoad'] = x_df
        store['StateKneeJCS'] = y_df
        store.close()

    def read_HDF5(self, hdf5_path):
        x_df = pd.read_hdf(hdf5_path, 'StateJCSLoad')
        y_df = pd.read_hdf(hdf5_path, 'StateKneeJCS')
        return x_df, y_df

    def read_TDMS (self, files):
        self.file_specific_data, self.StateJCSLoad, self.StateKneeJCS = defaultdict(list), defaultdict(list), defaultdict(list)
        self.load_properties, self.knee_properties = defaultdict(list), defaultdict(list)
        print("Loading Data.... This may take a moment")
        for index, root_file in files.items():
            with TdmsFile.open(os.path.join(root_file['root'], root_file['filename'])) as tdms_file:
                file_specific_StateJCSLoad = {}
                file_specific_StateKneeJCS = {}
                #Loop through the channels in StateJCSLoad and StateKneeJCS simulatneously
                for (c, d, e) in zip(tdms_file["State.JCS"].channels(), tdms_file["State.Knee JCS"].channels(), tdms_file["State.JCS Load"].channels()):
                    #Initialize the variables for each loading channel (name, data, properties)
                    StateJCSLoad_ChName, StateJCSLoad_data, StateJCSLoad_properties = e.name, np.array(e[:]), e.properties
                    #Extend the data points in the channel to the class attribute StateJCSLoad (dictionary)
                    self.StateJCSLoad[StateJCSLoad_ChName].extend(StateJCSLoad_data)
                    #Set the file-specific StateJCSLoad dictionary equal to the data in the file
                    file_specific_StateJCSLoad[StateJCSLoad_ChName] = StateJCSLoad_data
                    #Set the properties for each loading channel (column, units)
                    self.load_properties['column'].append(StateJCSLoad_ChName)
                    self.load_properties['units'].append(StateJCSLoad_properties['NI_UnitDescription'])

                    #Initialize the variables for each kinematics channel (name, data, properties)
                    StateKneeJCS_ChName, StateKneeJCS_data, StateKneeJCS_properties = d.name, np.array(d[:]), d.properties
                    #Extend the data points in the channel to the class attribute StateKneeJCS (dictionary)
                    self.StateKneeJCS[StateKneeJCS_ChName].extend(StateKneeJCS_data)
                    #Set the file-specific StateKneeJCS dictionary equal to the data in the file
                    file_specific_StateKneeJCS[StateKneeJCS_ChName] = StateKneeJCS_data
                    #Set the properties for each kinematics channel (column, units)
                    self.knee_properties['column'].append(StateKneeJCS_ChName)
                    self.knee_properties['units'].append(StateKneeJCS_properties['NI_UnitDescription'])

                self.file_specific_data[root_file['filename']] = {'StateJCSLoad' : file_specific_StateJCSLoad,
                                                                        'StateKneeJCS' : file_specific_StateKneeJCS}

        return self.file_specific_data, self.StateJCSLoad, self.StateKneeJCS

    def export_legend(self, legend, filename):
        fig = legend.figure
        fig.canvas.draw()
        bbox=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    def graph_data (self):
        #Uses matplotlib to graph given data
        print("Generating plots... This may take a moment")
        
        plt.close('all')
        font = {'family': 'calibri',
                'weight': 'bold',
                'size': 12,}
        fig, axs = plt.subplots(6, 6, figsize=(25, 15))


        for filename, data in self.file_specific_data.items():
            if filename.endswith("main_processed.tdms"):
                filename = filename[:-20]
            x, y = 0, 0
            for (knee_column_name, knee_column_data) in data.get('StateKneeJCS').items():
                x = 0
                for (load_column_name, load_column_data) in data.get('StateJCSLoad').items():
                    #print("(" + str(y) + ", " + str(x) + ") load column data: " + str(load_column_data) + "\n knee column data " + str(knee_column_data))
                    axs[y, x].scatter(load_column_data, knee_column_data, s=0.7, label=filename)
                    x = x+1
                y = y+1

        leg=plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        for ax in axs.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        for i in range(6):
            axs[-1, i].set_xlabel(self.load_properties['column'][i] + "(" + self.load_properties['units'][i] + ")", fontdict=font)
            axs[i, 0].set_ylabel(self.knee_properties['column'][i]+ "(" + self.knee_properties['units'][i] + ")", fontdict=font)
        plt.tight_layout()
        fig.canvas.set_window_title('')
        image_dir = os.path.join(self.current_directory, "OK_Data_Graphs")
        image_path = image_dir + "\\" + self.data_pkg_name + "_graphs.png"
        legend_path = image_dir + "\\" + self.data_pkg_name + "_graphs_legend.png"
        plt.savefig(image_path, format="png", dpi=300, bbox_inches="tight")
        self.export_legend(leg, legend_path)
        #plt.show()
        plt.close()
        print("Plot generation complete check your curreny working directory for graph images and the hdf5 file")
