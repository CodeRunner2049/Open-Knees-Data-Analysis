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

#K nearest neighbor outlier detection

class file_opener ():

    """Constructor for the the file_opener class. Initializes the data dictionaries and the directory name"""
    def __init__(self, directory, pkg_name):
        self.current_directory =  os.path.dirname(os.path.realpath(__file__))
        self.data_pkg_name = pkg_name
        self.directory = directory
        self.file_dict = self.build_file_dict()
        #Build the x & y dataframes from the presented files
        self.x_dataframe, self.y_dataframe = self.files_to_dataframes(self.file_dict)
        #Here self.hdf5_path is instantiated
        self.write_HDF5(self.x_dataframe, self.y_dataframe)
        #Generate class attribute algorithm imported from algorithm class (drop filename column)
        self.algorithm = algorithm(self.x_dataframe, self.y_dataframe)
        #self.print_files_with_deviation()

    def build_file_dict(self):
        index = 0
        file_dict = defaultdict(dict)
        for root, dirs, files in os.walk(self.directory, topdown=True):
            for filename in files:
                if(filename.endswith("main_processed.tdms")):
                    file_dict[index] = {'root': root, 'filename': filename}
                    index+=1;
        return file_dict

    def print_files_with_deviation (self):
        for index, root_file in self.file_dict.items():
            filename = root_file['filename']
            x_chunk = self.x_dataframe.xs(filename, level=1, axis=0, drop_level=False)
            y_chunk = self.y_dataframe.xs(filename, level=1, axis=0, drop_level=False)
            print("file " + str(index) + ": " + str(root_file['filename']))
            [print(self.algorithm.do_std_devation(x_chunk[x_columnName])) for x_columnName in x_chunk.columns.values]
            [print(self.algorithm.do_std_devation(y_chunk[y_columnName])) for y_columnName in y_chunk.columns.values]

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

    def clean_dataframes (self, x_df, y_df):
        assert isinstance(x_df, pd.DataFrame)
        assert isinstance(y_df, pd.DataFrame)
        indexes_to_drop = []
        for (x_columnName, x_columnData), (y_columnName, y_columnData) in zip(x_df.iteritems(), y_df.iteritems()):
            indexes_to_drop.extend(x_df[x_df[x_columnName].isnull()].index.tolist())
            indexes_to_drop.extend(y_df[y_df[y_columnName].isnull()].index.tolist())
        return x_df.drop(indexes_to_drop), y_df.drop(indexes_to_drop)

    """Writes x and y data parameters as pandas dataframes"""
    def build_dataframe(self, x_data, y_data):
        x_df = pd.DataFrame(x_data, columns=[k for k in x_data.keys()])
        y_df = pd.DataFrame(y_data, columns=[k for k in y_data.keys()])
        return self.clean_dataframes(x_df, y_df)

    def files_to_dataframes(self, file_list):
        print("Reading files and loading data... This may take a moment")
        x_df, y_df = [list(df) for df in
                zip(*[self.read_TDMS(root_file['root'], root_file['filename']) for root_file in file_list.values()])]
        new_x_df, new_y_df = self.clean_dataframes(pd.concat(x_df), pd.concat(y_df))
        #Reset the indexes to be numbered from 0 to len(new_x_df)
        new_x_df.reset_index(drop=True, inplace=True)
        new_y_df.reset_index(drop=True, inplace=True)
        #Append a second index that sorts the dataframe by the filename
        new_x_df.set_index(['filename'], inplace=True, append=True, drop=True)
        new_y_df.set_index(['filename'], inplace=True, append=True, drop=True)
        return new_x_df, new_y_df

    def update_dataframes (self, pruner_inp, exclusionary_list):
        self.prune_files(pruner_inp, exclusionary_list)
        self.files_to_dataframes(self.file_dict)
        self.print_files_with_deviation()
        print("Your linear regressions have changed to: ")
        self.algorithm.do_linear_regression()

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

    def read_TDMS (self, root, filename):
        with TdmsFile.open(os.path.join(root, filename)) as tdms_file:
            x_data, y_data = defaultdict(list), defaultdict(list)
            #Generate the x & y data as a dictionary for each tdms file
            for x_ch, y_ch in zip(tdms_file["State.Knee JCS"].channels(), tdms_file["State.JCS Load"].channels()):
                x_data[str(x_ch.name) + '(' + str(x_ch.properties['NI_UnitDescription']) + ')'].extend(np.array(x_ch[:]))
                y_data[str(y_ch.name) + '(' + str(y_ch.properties['NI_UnitDescription']) + ')'].extend(np.array(y_ch[:]))
                x_data_len, y_data_len = len(x_ch[:]), len(y_ch[:])
            #Remove the file from the file_dict if it is empty
            if x_data_len == 0 and y_data_len == 0:
                del self.file_dict[[index for index, root_file in self.file_dict.items() if root_file['filename'] == filename]]
            #Create a new column that contains the filename of data
            x_data['filename'] = np.full(x_data_len, filename)
            y_data['filename'] = np.full(y_data_len, filename)
            #close the file
            tdms_file.close()
            return self.build_dataframe(x_data, y_data)

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
        fig, axs = plt.subplots(len(self.x_dataframe.columns.values), len(self.y_dataframe.columns.values), figsize=(25, 15))
        x_data, y_data = self.x_dataframe, self.y_dataframe

        # for filename, data in self.file_specific_data.items():
        #     if filename.endswith("main_processed.tdms"):
        #         filename = filename[:-20]
        #     x, y = 0, 0
        #     for (knee_column_name, knee_column_data) in data.get('StateKneeJCS').items():
        #         x = 0
        #         for (load_column_name, load_column_data) in data.get('StateJCSLoad').items():
        #             #print("(" + str(y) + ", " + str(x) + ") load column data: " + str(load_column_data) + "\n knee column data " + str(knee_column_data))
        #             axs[y, x].scatter(load_column_data, knee_column_data, s=0.7, label=filename)
        #             x = x+1
        #             y = y+1

        for root_file in self.file_dict.values():
            filename = root_file['filename']
            x_chunk = x_data.xs(filename, level=1, axis=0, drop_level=False)
            y_chunk = y_data.xs(filename, level=1, axis=0, drop_level=False)
            for y_idx, y_col in enumerate(y_chunk.columns.values):
                for x_idx, x_col in enumerate(x_chunk.columns.values):
                    axs[y_idx, x_idx].scatter(x_chunk[x_col], y_chunk[y_col], s=0.7, label=filename)

        leg=plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        for ax in axs.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        #Set column labels
        for x_idx, x_col in enumerate(x_data.columns.values):
            axs[-1, x_idx].set_xlabel(x_col, fontdict=font)
        for y_idx, y_col in enumerate(y_data.columns.values):
            axs[y_idx, 0].set_ylabel(y_col, fontdict=font)

        plt.tight_layout()
        fig.canvas.set_window_title(self.data_pkg_name)
        image_dir = os.path.join(self.current_directory, "OK_Data_Graphs")
        image_path = image_dir + "\\" + self.data_pkg_name + "_graphs.png"
        legend_path = image_dir + "\\" + self.data_pkg_name + "_graphs_legend.png"
        plt.savefig(image_path, format="png", dpi=300, bbox_inches="tight")
        self.export_legend(leg, legend_path)
        #plt.show()
        plt.close()
        print("Plot generation complete check your curreny working directory for graph images and the hdf5 file")
