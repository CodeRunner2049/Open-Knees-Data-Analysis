from nptdms import TdmsFile
import numpy as np
import re
import os
import itertools
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class fileOpener ():
    def __init__(self, directory, pkg_name):
        #Constructor for the the fileOpener class. Initializes the data dictionaries and the directory name
        self.knee_data = {}
        self.data_pkg_name = pkg_name
        self.directory = directory

    def writeHDF5 (self):
        #Writies all the data as datasets in a HDF5 file
        with h5py.File("KinematicsData.hdf5", "w") as f:
            for k, v in self.knee_data.items():
                f.create_dataset(k, data = np.array(v))

    def readTDMS (self):
        index = 0
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                if(filename.endswith("main_processed.tdms")):
                    print("file " + str(index) + ": " + filename)
                    index+=1;
                    with TdmsFile.open(os.path.join(root, filename)) as tdms_file:
                        StateJCSLoad = {}
                        StateKneeJCS = {}
                        for (c, d, e) in zip(tdms_file["State.JCS"].channels(), tdms_file["State.Knee JCS"].channels(), tdms_file["State.JCS Load"].channels()):
                            StateKneeJCS_ChName, StateKneeJCS_data, StateKneeJCS_properties = d.name, np.array(d[:]), d.properties
                            StateKneeJCS[StateKneeJCS_ChName] = {'properties': StateKneeJCS_properties, 'data': StateKneeJCS_data}
                            #print(StateKneeJCS_properties)
                            #print(StateKneeJCS_ChName + " data:\n" + ','.join(str(x) for x in StateKneeJCS_data))

                            StateJCSLoad_ChName, StateJCSLoad_data, StateJCSLoad_properties = e.name, np.array(e[:]), e.properties
                            StateJCSLoad[StateJCSLoad_ChName] = {'properties': StateJCSLoad_properties, 'data': StateJCSLoad_data}
                            #print(StateJCSLoad_properties)
                            #print(StateJCSLoad_ChName + " data:\n" + ','.join(str(x) for x in StateJCSLoad_data))

                        self.knee_data[filename] = {'x' : StateJCSLoad, 'y' : StateKneeJCS}

    def export_legend(self, legend, filename):
        fig = legend.figure
        fig.canvas.draw()
        bbox=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    def graphData (self):
        #Uses matplotlib to graph given data
        plt.close('all')
        font = {'family': 'calibri',
                'weight': 'bold',
                'size': 12,}
        fig, axs = plt.subplots(6, 6, figsize=(25, 15))
        load_column_properties, knee_column_properties = {'column':[], 'units':[]}, {'column':[], 'units':[]}
        print("Loading Data.... This may take a moment")

        for filename, knee_data in self.knee_data.items():
            if filename.endswith("main_processed.tdms"):
                filename = filename[:-20]
            x, y = 0, 0
            for (knee_column_name, knee_column_data) in knee_data.get('y').items():
                x = 0
                knee_column_properties['column'].append(knee_column_name)
                knee_column_properties['units'].append(knee_column_data['properties']['NI_UnitDescription'])

                for (load_column_name, load_column_data) in knee_data.get('x').items():
                    load_column_properties['column'].append(load_column_name)
                    load_column_properties['units'].append(load_column_data['properties']['NI_UnitDescription'])
                    #print("(" + str(y) + ", " + str(x) + ") load column data: " + str(load_column_data) + "\n knee column data " + str(knee_column_data))
                    axs[y, x].scatter(load_column_data['data'], knee_column_data['data'], s=0.7, label=filename)
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
            axs[-1, i].set_xlabel(load_column_properties['column'][i] + "(" + load_column_properties['units'][i] + ")", fontdict=font)
            axs[i, 0].set_ylabel(knee_column_properties['column'][i]+ "(" + knee_column_properties['units'][i] + ")", fontdict=font)
        plt.tight_layout()
        fig.canvas.set_window_title('joint_mechanics-oks009_graphs')
        plt.savefig('.\\OK Data Graphs\\' + self.data_pkg_name + '_graphs.png')
        self.export_legend(leg, ".\\OK Data Graphs\\" + self.data_pkg_name + "_graphs_legend.png")
        plt.show()
        plt.close()

def main():
    hdf5_dir = os.path.join('.\hdf5_files')
    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)
    image_dir = os.path.join('.\OK Data Graphs')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    data_dir = input("Enter in the directory (filepath) of the Open Knee data that you would like to analyze: ")
    pkg_name = re.search("joint_mechanics-oks\d{3}", data_dir)
    FO = fileOpener(data_dir, pkg_name)
    #".\Open Knees File Visualization\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics"
    #D:\Mourad\joint_mechanics-oks009\joint_mechanics-oks009\TibiofemoralJoint\KinematicsKinetics
    FO.readTDMS()
    FO.graphData()
    #FO.writeHDF5()

main()
