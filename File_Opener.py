from nptdms import TdmsFile
import numpy as np
import pandas as pd
import os
import itertools
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class fileOpener ():
    def __init__(self, directory, pkg_name):
        #Constructor for the the fileOpener class. Initializes the data dictionaries and the directory name
        self.data_pkg_name = pkg_name
        self.directory = directory
        self.current_directory =  os.path.dirname(os.path.realpath(__file__))

    def writeHDF5 (self, x_data, y_data):
        #Writies all the data as datasets in a HDF5 file
        print("Pushing data to hd5f file... This may take a moment")
        hdf5_dir = os.path.join(self.current_directory, "hdf5_files")
        hdf5_path = hdf5_dir + "\\" + self.data_pkg_name + "_hierachical_data.hdf5"
        store = pd.HDFStore(hdf5_path, "w")
        x_df = pd.DataFrame(x_data, columns=[k for k in x_data.keys()], dtype = 'float64')
        y_df = pd.DataFrame(y_data, columns=[k for k in y_data.keys()], dtype = 'float64')
        self.clean_dataset(x_df, y_df)
        print(x_df)
        print(y_df)
        store['StateJCSLoad'] = x_df
        store['StateKneeJCS'] = y_df
        #store.append('StateJCSLoad', x_df, data_columns= x_df.columns, min_itemsize={'values': len(x_df)})
        #store.append('StateKneeJCS', y_df, data_columns= y_df.columns, min_itemsize={'values': len(y_df)})
        store.close()

    def readHDF5(self, hdf5_path):
        x_df = pd.read_hdf(hdf5_path, 'StateJCSLoad')
        y_df = pd.read_hdf(hdf5_path, 'StateKneeJCS')
        return (x_df, y_df)

    def print_files(self):
        index = 0
        file_dict = {}
        for root, dirs, files in os.walk(self.directory, topdown=True):
            for filename in files:
                if(filename.endswith("main_processed.tdms")):
                    print("file " + str(index) + ": " + filename)
                    file_dict[index] = (root, filename)
                    index+=1;
        return file_dict

    def prune_files(self, split, files, pruner):
        temp = files.copy()
        if split ==0:
            files=files
        elif split == 1:
            for index, filename in files.items():
                if not index in pruner:
                    del temp[index]
        elif split == 2:
            for index in pruner:
                del temp[index]
        pruned_files = []
        for index, filename in temp.items():
            pruned_files.append(filename)
        return pruned_files

    def split_data(self, file_inp):
        file_list = file_inp.split(",")
        int_list = []
        for i in file_list:
            int_list.append(int(i))
        return int_list

    def readTDMS (self, pruned_files):
        knee_data = {}
        StateJCSLoad = {}
        StateKneeJCS = {}
        load_column_properties, knee_column_properties = {'column':[], 'units':[]}, {'column':[], 'units':[]}
        print("Loading Data.... This may take a moment")
        for root_file in pruned_files:
            with TdmsFile.open(os.path.join(root_file[0], root_file[1])) as tdms_file:
                temp_StateJCSLoad = {}
                temp_StateKneeJCS = {}
                for (c, d, e) in zip(tdms_file["State.JCS"].channels(), tdms_file["State.Knee JCS"].channels(), tdms_file["State.JCS Load"].channels()):
                    StateJCSLoad_ChName, StateJCSLoad_data, StateJCSLoad_properties = e.name, np.array(e[:]), e.properties
                    if StateJCSLoad_ChName in StateJCSLoad:
                        StateJCSLoad[StateJCSLoad_ChName].extend(StateJCSLoad_data)
                    else:
                         StateJCSLoad[StateJCSLoad_ChName] = [x for x in StateJCSLoad_data]
                    temp_StateJCSLoad[StateJCSLoad_ChName] = StateJCSLoad_data
                    load_column_properties['column'].append(StateJCSLoad_ChName)
                    load_column_properties['units'].append(StateJCSLoad_properties['NI_UnitDescription'])
                    #print(StateJCSLoad_properties)
                    #print(StateJCSLoad_ChName + " data:\n" + ','.join(str(x) for x in StateJCSLoad_data))

                    StateKneeJCS_ChName, StateKneeJCS_data, StateKneeJCS_properties = d.name, np.array(d[:]), d.properties
                    if StateKneeJCS_ChName in StateKneeJCS:
                        StateKneeJCS[StateKneeJCS_ChName].extend(StateKneeJCS_data)
                    else:
                         StateKneeJCS[StateKneeJCS_ChName] = [x for x in StateKneeJCS_data]
                    temp_StateKneeJCS[StateKneeJCS_ChName] = StateKneeJCS_data
                    knee_column_properties['column'].append(StateKneeJCS_ChName)
                    knee_column_properties['units'].append(StateKneeJCS_properties['NI_UnitDescription'])
                    #print(StateKneeJCS_properties)
                    #print(StateKneeJCS_ChName + " data:\n" + ','.join(str(x) for x in StateKneeJCS_data))

                knee_data[root_file[1]] = {'StateJCSLoad' : temp_StateJCSLoad, 'StateKneeJCS' : temp_StateKneeJCS}
        self.writeHDF5(StateJCSLoad, StateKneeJCS)
        self.graphData(knee_data, load_column_properties, knee_column_properties)

    def export_legend(self, legend, filename):
        fig = legend.figure
        fig.canvas.draw()
        bbox=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    def graphData (self, data, x_properties, y_properties):
        #Uses matplotlib to graph given data
        plt.close('all')
        font = {'family': 'calibri',
                'weight': 'bold',
                'size': 12,}
        fig, axs = plt.subplots(6, 6, figsize=(25, 15))


        for filename, knee_data in data.items():
            if filename.endswith("main_processed.tdms"):
                filename = filename[:-20]
            x, y = 0, 0
            for (knee_column_name, knee_column_data) in knee_data.get('StateKneeJCS').items():
                x = 0
                for (load_column_name, load_column_data) in knee_data.get('StateJCSLoad').items():
                    #print("(" + str(y) + ", " + str(x) + ") load column data: " + str(load_column_data) + "\n knee column data " + str(knee_column_data))
                    axs[y, x].scatter(load_column_data, knee_column_data, s=0.7, label=filename)
                    x = x+1
                y = y+1

        print("Generating plots... This may take a moment")
        leg=plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        #leg=axs[0, 5].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        for ax in axs.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        for i in range(6):
            axs[-1, i].set_xlabel(x_properties['column'][i] + "(" + x_properties['units'][i] + ")", fontdict=font)
            axs[i, 0].set_ylabel(y_properties['column'][i]+ "(" + y_properties['units'][i] + ")", fontdict=font)
        plt.tight_layout()
        fig.canvas.set_window_title('joint_mechanics-oks009_graphs')
        image_dir = os.path.join(self.current_directory, "OK_Data_Graphs")
        image_path = image_dir + "\\" + self.data_pkg_name + "_graphs.png"
        legend_path = image_dir + "\\" + self.data_pkg_name + "_graphs_legend.png"
        plt.savefig(image_path, format="png", dpi=300, bbox_inches="tight")
        self.export_legend(leg, legend_path)
        #plt.show()
        plt.close()
        print("Plot generation complete check your curreny working directory for graph images and the hdf5 file")
