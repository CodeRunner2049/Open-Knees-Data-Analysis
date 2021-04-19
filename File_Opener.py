from nptdms import TdmsFile
from collections import defaultdict
from Algorithm import train_test_neural_networks
import numpy as np
import pandas as pd
import copy
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
try:
    import configparser as cp
except:
    import configparser as cp


class Group:
    def __init__(self, name, channels, data, units, time, processed=False):
        self.name = name
        self.channels = channels
        self.data = data
        self.units = units
        self.time = time
        self.processed = processed


def tdms_contents(file_directory, file):
    """
    This function extracts all the information and data from the tdms file. returns the data as a list of Group objects
    """

    tdms_file = TdmsFile.read(os.path.join(file_directory, file))
    all_groups = tdms_file.groups()

    # prints list of groups and extracts channels from tdms file

    # collect groups objects in list
    groups = []

    for grp in all_groups:
        # grp = tdms_file[gn.name]
        channels = grp.channels()

        # creates empty lists to place information later
        channel_list = []
        channel_data_list = []
        channel_unit_list = []
        channel_time_list = []

        # extracts information from each channel
        for channel in channels:
            try:
                channel_label = channel.properties["NI_ChannelName"]
                channel_units = channel.properties['NI_UnitDescription']
                channel_data = channel.data
                channel_time = channel.time_track()

                # creates lists for plotting
                channel_data_list.append(channel_data)
                channel_list.append(channel_label)
                channel_unit_list.append(channel_units)
                channel_time_list.append(channel_time)

                new_group = Group(grp.name, channel_list, channel_data_list, channel_unit_list, channel_time_list)
                groups.append(new_group)
            except:
                break

    return groups


def find_average_data(group, channel):
    dt = np.asarray(group.data[channel])
    ave = np.nanmean(dt)

    return ave


def get_kinetics_kinematics(groups):
    """ extracts the relevant groups for kinetics and kinematics processing"""

    kinetics_group = None
    kinematics_group = None

    for g in groups:
        if g.name == 'State.JCS Load':
            kinetics_group = g
        elif g.name == 'State.Knee JCS':
            kinematics_group = g
        else:
            pass

    return kinetics_group, kinematics_group


def get_desired_kinetics(groups):
    desired_kinetics_group = None
    for g in groups:
        if g.name == 'Kinetics.JCS.Desired':
            desired_kinetics_group = g
            break
        else:
            pass

    return desired_kinetics_group


def crop_index(group, cutoff_value, cutoff_channel, baseline = 0.0):
    """ crop the data such that on the desired channel the data does not exceed the cutoff value in either direction"""

    # i = group.channels.index(cutoff_channel)
    i=cutoff_channel
    dt = group.data[i]
    keep_idx = np.where(np.logical_and(dt >= baseline-cutoff_value, dt <= baseline + cutoff_value))[0]

    keep_idx = np.asarray(keep_idx)

    return keep_idx


def crop_data(group, cropping_index):
    """ crop all the data in the group by the cropping index"""
    cropped_data = []
    cropped_time = []

    for D in np.asarray(group.data):
        cd = D[cropping_index]
        cropped_data.append(cd)

    for T in np.asarray(group.time):
        ct = T[cropping_index]
        cropped_time.append(ct)

    # make changes to the group
    group.data = cropped_data
    group.time = cropped_time
    group.processed = True


def cut_data(group, cutting_index):
    """ delete all the data in the group by the given by the cutting index"""

    data = np.asarray(group.data)
    cut_data = np.delete(data, cutting_index)

    # for D in np.asarray(group.data):
    #     cd = D[cutting_index]
    #     cropped_data.append(cd)

    time = np.asarray(group.time)
    cut_time = np.delete(time, cutting_index)

    # make changes to the group
    group.data = cut_data
    group.time = cut_time
    group.processed = True


def write_HDF5(kinetics, kinematics, pkg_name):
    """Writes kinetics and kinematics dataframes into an HDF5 file"""

    print("Pushing data to hdf5 file... This may take a moment")

    # Create a hdf5 directory and add the h5 files there
    hdf5_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hdf5_files")
    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)
    h5_pkg = os.path.join(hdf5_dir, pkg_name)
    if not os.path.exists(h5_pkg):
        os.mkdir(h5_pkg)
    x_hdf5_path = h5_pkg + "\\" + pkg_name + "_x_hierarchical_data.h5"
    y_hdf5_path = h5_pkg + "\\" + pkg_name + "_y_hierarchical_data.h5"

    # Store the dataframes in the hdf5 files
    kinetics.to_hdf(x_hdf5_path, key='x_data', mode='w')
    kinematics.to_hdf(y_hdf5_path, key='y_data', mode='w')

def resampling_index(group, resampling_channel, increment, range, start=0.0):
    # chan_idx = group.channels.index(resampling_channel)
    chan_idx = resampling_channel
    chan_data = group.data[chan_idx]

    if increment > 0.0:
        resampling_increments = np.arange(start, np.max(chan_data) + increment, increment)
    else:
        resampling_increments = np.arange(start, np.min(chan_data) + increment, increment)

    indices = []

    for a in resampling_increments:
        a_ind = np.where(np.abs(chan_data - a) < range)[0]

        indices.append(a_ind)

    return indices, resampling_increments


def resample_data(group, resamp_idx, resamp_intervals):
    """ given the indices of the data points included for each resampled point, take the average of the data points.
    add the resampled data to the group and return the new group"""

    data_resampled = []
    data = np.asarray(group.data)
    for i in resamp_idx:
        di = data[:, i]
        di_ave = np.average(di, axis=1)
        data_resampled.append(di_ave)

    data_resampled = np.asarray(data_resampled)
    data_resampled = data_resampled.T  # to get it in the same format as the initial data list
    data_resampled = data_resampled.tolist()

    group.data = data_resampled
    group.time = [resamp_intervals] * len(data)
    group.processed = True


def generate_dataframes(group, x_label):
    """Generate the dataframes from the partitioned kinematics and kinetics data"""

    if group is None:
        return None

    df = pd.DataFrame()
    df[x_label] = group.time[0]
    for i in range(6):
        df[group.channels[i] + ' [' + group.units[i] + ']'] = group.data[i]
    return df


def get_NaN_indices(group):
    """Get all NaN indexes from the channel data"""

    data = np.asarray(group.data)

    # Tuple channel number and index location of NaN data points
    indices_to_drop = list(np.argwhere(data == 0))
    indices_to_drop = [index[1] for index in indices_to_drop]

    # Nan_tuples = list(map(tuple, np.where(np.isnan(data))))
    # indices_to_drop = []
    # for index, tuple in enumerate(Nan_tuples):
    #     indices_to_drop.append(tuple[2])

    # To use when sorting through a dataframe

    # var = data[channel].isnull()
    # NaN_indeces = data[data[channel].isnull()].index.tolist()

    return indices_to_drop


def clean_dataframes(kinetics_data, kinematics_data):
    """Clean the NaN indexes from the kinematics and kinetics data"""

    assert isinstance(kinetics_data, pd.DataFrame)
    assert isinstance(kinematics_data, pd.DataFrame)

    kinetics_data_copy = kinetics_data.copy(deep=True)
    kinematics_data_copy = kinematics_data.copy(deep=True)
    indices_to_drop = []

    # Make sure the only channels present are the kinetics and kinematics data
    kinetics_channels = ['JCS Load Lateral Drawer [N]', 'JCS Load Anterior Drawer [N]', 'JCS Load Distraction [N]',
                         'JCS Load Extension Torque [Nm]', 'JCS Load Varus Torque [Nm]',
                         'JCS Load External Rotation Torque [Nm]']
    kinematics_channels = ['Knee JCS Medial [mm]', 'Knee JCS Posterior [mm]', 'Knee JCS Superior [mm]',
                           'Knee JCS Flexion [deg]', 'Knee JCS Valgus [deg]', 'Knee JCS Internal Rotation [deg]']

    # Check the kinetics data for null indices and append the indexes to the dropping array

    for x_columnName, x_columnData in kinetics_data_copy.iteritems():
        if x_columnName not in kinetics_channels:
            kinetics_data_copy = kinetics_data_copy.drop([x_columnName], axis=1)
        else:
            indices_to_drop.extend(kinetics_data_copy[kinetics_data_copy[x_columnName].isnull()].index.tolist())

    # Check the kinematics data for null indices and append the indexes to the dropping array

    for y_columnName, y_columnData in kinematics_data_copy.iteritems():
        if y_columnName not in kinematics_channels:
            kinematics_data_copy = kinematics_data_copy.drop(y_columnName, axis=1)
        else:
            indices_to_drop.extend(kinematics_data_copy[kinematics_data_copy[y_columnName].isnull()].index.tolist())

    # Drop the indices from the dropping array

    kinetics_data_copy = kinetics_data_copy.drop(indices_to_drop, axis=0)
    kinematics_data_copy = kinematics_data_copy.drop(indices_to_drop, axis=0)

    return kinetics_data_copy, kinematics_data_copy


def find_hold_indices(group, channel):
    """ find the indices where the desired kinetics load was held steady on the channel"""

    data = np.asarray(group.data[channel])
    time = group.time[channel]
    tol = 1e-3

    change = np.abs(data[:-1] - data[1:])
    small_changes = np.where(change < tol)[0]  # where the data is stable - ie force was held
    increments = np.abs(small_changes[:-1] - small_changes[1:])  # how many indices between the stable points
    large_increments = np.where(increments > 20)[0]  # where the increment jumps

    if large_increments.size == 0:
        neg_end = data.size - 1
        pos_end = 0
    else:
        # this gives the index for the data point at the start and end of each flat region
        index_end = small_changes[large_increments] + 1
        index_start = small_changes[large_increments + 1] + 1

        # final "start" point is not relevant - it is when the curve returns to zero after loading is completed
        index_start = index_start[:-1]

        # create a "fake" start point for the first end point - this is the zero load
        # check length of other hold period
        other_len = index_end[1] - index_start[0]
        index_start = np.insert(index_start, 0, index_end[0] - other_len)

        # split the indices in half - positive and negative loading
        idx_split = int(len(index_end) / 2)
        pos_end = index_end[:idx_split]
        neg_end = index_end[idx_split:]
        pos_start = index_start[:idx_split]
        neg_start = index_start[idx_split:]

        pos_tuples = list(zip(pos_start, pos_end))
        neg_tuples = list(zip(neg_start, neg_end))

    # return just the end points. to return the start and end points for each flat zone, return the tuples instead
    return pos_end, neg_end


def laxity_processing_2(groups, tdms_directory):
    """use the desired kinematics channels to filter the kinematics and kinetics data"""

    kinetics_group, kinematics_group = get_kinetics_kinematics(groups)
    desired_kinetics_group = get_desired_kinetics(groups)

    # find the average flexion angle in the data
    flexion_angle = find_average_data(kinematics_group, 3)
    rounded_flexion = int(
        round(flexion_angle, -1))  # round to the nearest 10 - this is the 'intended' flexion, use for naming files

    # plot_groups("Laxity_{}deg_Kinetics_desired_raw".format(rounded_flexion), desired_kinetics_group, 'Time (ms)', tdms_directory,show_plot=False)

    # save the raw data as csv and png file
    # plot_groups("Laxity_{}deg_Kinematics_raw".format(rounded_flexion), kinematics_group, 'Time (ms)',
    #             tdms_directory, show_plot=False)
    # plot_groups("Laxity_{}deg_Kinetics_raw".format(rounded_flexion), kinetics_group, 'Time (ms)', tdms_directory,
    #             show_plot=False)

    Loading_channels = [4, 5, 1]
    channel_nickname = ['VV', 'EI', 'AP']
    all_laxity_kinematics_dfs = []
    all_laxity_kinetics_dfs = []

    for n, chan in enumerate(Loading_channels):

        # use the desired kinetics to find the indices of the data points at the end of the "flat" regions
        # where forces were held steady
        pos_index, neg_index = find_hold_indices(desired_kinetics_group, chan)

        for c, index in enumerate([pos_index, neg_index]):
            # make copies of the data so we don't make changes to the original data at every loop
            kinetics_group_copy = copy.deepcopy(kinetics_group)
            kinematics_group_copy = copy.deepcopy(kinematics_group)

            # extract the data from the kinematics and kinetics
            crop_data(kinetics_group_copy, index)
            crop_data(kinematics_group_copy, index)

            # set the x axis to the applied load (actual)
            kinetics_group_copy.time = [kinetics_group_copy.data[chan]] * len(kinetics_group_copy.data)
            kinematics_group_copy.time = [kinetics_group_copy.data[chan]] * len(kinematics_group_copy.data)

            channel_units = kinetics_group_copy.units[chan]

            # these are the results that have undergone some processing and will be pushlished for data representation step
            # plot
            # plot_groups('Laxity_{}deg_'.format(rounded_flexion) + channel_nickname[n] + str(
            #     c + 1) + '_TibiaKinetics_in_TibiaCS_experiment', kinetics_group_copy,
            #             'Applied Load (' + channel_units + ')', tdms_directory, show_plot=False)
            # plot_groups('Laxity_{}deg_'.format(rounded_flexion) + channel_nickname[n] + str(
            #     c + 1) + '_kinematics_in_JCS_experiment', kinematics_group_copy,
            #             'Applied Load (' + channel_units + ')', tdms_directory, show_plot=False)

            # continue with remaining processing steps for our team's workflow

            # apply experiment offsets to the kinematics data
            # apply_offsets(kinematics_group_copy, experiment_offsets)
            #
            # # report the axes in the right handed coordinate system we defined.
            # change_kinematics_reporting(kinematics_group_copy)
            # change_kinetics_reporting(kinetics_group_copy)  # this is external loads on tibia in tibia cs
            #
            # # report the kinetics as external loads applied to femur in tibia coordinate system
            # kinetics_tibia_to_femur(kinetics_group_copy, kinematics_group_copy, chan)
            #
            # # apply model offsets - Note this is done AFTER changing the signs of the data to register with model outputs.
            # apply_offsets(kinematics_group_copy, -model_offsets)

            # # Collect the indices across all channels where NaN values exist
            # kinetics_NaN_indices = get_NaN_indices(kinetics_group_copy)
            # kinematics_NaN_indices = get_NaN_indices(kinematics_group_copy)
            # print(kinetics_NaN_indices)
            # print(kinematics_NaN_indices)
            #
            # # Clear NaN indices from the given channel including NaN indices from other channels
            #
            # cut_data(kinetics_group_copy, kinetics_NaN_indices)
            # cut_data(kinematics_group_copy, kinematics_NaN_indices)

            # processed data this will be used to generate models replicating experiment
            all_laxity_kinetics_dfs.append(
                generate_dataframes(kinetics_group_copy, 'Applied Load (' + channel_units + ')'))
            all_laxity_kinematics_dfs.append(
                generate_dataframes(kinematics_group_copy, 'Applied Load (' + channel_units + ')'))

    laxity_kinetics_df = pd.concat(all_laxity_kinetics_dfs)
    laxity_kinematics_df = pd.concat(all_laxity_kinematics_dfs)

    return laxity_kinetics_df, laxity_kinematics_df


def passive_flexion_processing_2(groups, tdms_directory):
    kinetics_group, kinematics_group = get_kinetics_kinematics(groups)

    # plot_groups("Passive_Flexion_Kinematics_Raw", kinematics_group, 'Flexion Angle (deg)', tdms_directory,
    #             show_plot=False)
    # plot_groups("Passive_Flexion_Kinetics_Raw", kinetics_group, 'Flexion Angle (deg)', tdms_directory, show_plot=False)

    # separate flexion and extension data
    # find max data point, anything before is flexion, anything after is extension
    flexion_kinematics = kinematics_group.data[3]
    max_flex_idx = np.argmax(flexion_kinematics)

    # ext_crop_idx = np.arange(max_flex_idx, len(flexion_kinematics)) # in case we need the extension points for something

    flex_crop_idx = np.arange(0, max_flex_idx)

    # use only the flexion data
    crop_data(kinetics_group, flex_crop_idx)
    crop_data(kinematics_group, flex_crop_idx)

    # resample at 5 degree increments by averageing each channel where flexion angle is within range degrees
    resampling_channel = 3
    increments = 5.0
    range = 0.5

    resamp_idx, resamp_intervals = resampling_index(kinematics_group, resampling_channel, increments, range)
    resample_data(kinematics_group, resamp_idx, resamp_intervals)
    resample_data(kinetics_group, resamp_idx, resamp_intervals)

    # # Collect the indices across all channels where NaN values exist
    # kinetics_NaN_indices = get_NaN_indices(kinetics_group)
    # kinematics_NaN_indices = get_NaN_indices(kinematics_group)
    # print(kinetics_NaN_indices)
    # print(kinematics_NaN_indices)
    #
    #  # Clear NaN indices from the given channel including NaN indices from other channels
    #
    # cut_data(kinetics_group, kinetics_NaN_indices)
    # cut_data(kinematics_group, kinematics_NaN_indices)

    # files for data representation
    # plot_groups("Passive_Flexion_Kinematics_in_JCS_experiment", kinematics_group, 'Flexion Angle (deg)',
    #             tdms_directory, show_plot=False)
    # plot_groups("Passive_Flexion_TibiaKinetics_in_TibiaCS_experiment", kinetics_group, 'Flexion Angle (deg)',
    #             tdms_directory, show_plot=False)

    # continue processing for our workflow
    # apply experiment offsets to the kinematics data (add)
    # apply_offsets(kinematics_group, experiment_offsets)
    #
    # # report the axes in the same direction as the model reporting :
    # # positive directions are - extension, adduction (varus), external
    # change_kinematics_reporting(kinematics_group)
    # change_kinetics_reporting(kinetics_group)
    #
    # # report the kinetics as external loads applied to femur in tibia coordinate system
    # kinetics_tibia_to_femur(kinetics_group, kinematics_group)
    #
    # # apply model offsets (subtract) - Note this is done AFTER changing the signs of the data to register with model outputs.
    # apply_offsets(kinematics_group, -model_offsets)

    # save the data as pandas dataframes - this is what will be used to model experimental conditions
    flexion_kinematics_df = generate_dataframes(kinematics_group, 'Flexion Angle (deg)')
    flexion_kinetics_df = generate_dataframes(kinetics_group, 'Flexion Angle (deg)')

    return flexion_kinetics_df, flexion_kinematics_df


def combined_processing(groups, tdms_directory):
    """process combined loading files. find data point where valgus = 10 Nm, internal = 5 Nm, anterior = 0"""

    kinetics_group, kinematics_group = get_kinetics_kinematics(groups)
    desired_kinetics_group = get_desired_kinetics(groups)

    # find the average flexion angle in the data
    flexion_angle = find_average_data(kinematics_group, 3)
    rounded_flexion = int(round(flexion_angle, -1))  # round to the nearest 10 - this is the 'intended' flexion, use for naming files

    # save the raw data as csv and png file
    # plot_groups("Combined_{}deg_Kinetics_desired_raw".format(rounded_flexion), desired_kinetics_group, 'Time (ms)', tdms_directory,show_plot=False)
    # plot_groups("Combined_{}deg_Kinematics_raw".format(rounded_flexion), kinematics_group, 'Time (ms)', tdms_directory,show_plot=False)
    # plot_groups("Combined_{}deg_Kinetics_raw".format(rounded_flexion), kinetics_group, 'Time (ms)', tdms_directory,show_plot=False)

    # find indices for cropping the data where valgus = 10 Nm, internal = 5 Nm
    # channel 1- Anterior Drawer
    # channel 4 - varus torque
    # channel 5 - external torque
    cutoff = 1.0e-08
    cropping_chans = [4,5,1]
    cropping_baselines = [-10,-5,0]

    # crop using desired kinetics
    for idx, chan in enumerate(cropping_chans):
        cropping_idx = crop_index(desired_kinetics_group, cutoff, chan, baseline=cropping_baselines[idx])
        crop_data(desired_kinetics_group, cropping_idx)
        crop_data(kinetics_group, cropping_idx)
        crop_data(kinematics_group, cropping_idx)

    # take the endpoint of whatever data is left
    cropping_idx = [-1]
    crop_data(kinetics_group, cropping_idx)
    crop_data(kinematics_group, cropping_idx)

    # plot and save the cropped data
    # plot_groups("Combined_{}deg_Kinematics_cropped".format(rounded_flexion), kinematics_group, 'Time (ms)',
    #             tdms_directory, show_plot=False)
    # plot_groups("Combined_{}deg_Kinetics_cropped".format(rounded_flexion), kinetics_group, 'Time (ms)', tdms_directory,
    #             show_plot=False)

    # return the final data so we can combine the different flexion angles into one data set
    kinetics_df = generate_dataframes(kinetics_group, 'Time (ms)')
    kinematics_df = generate_dataframes(kinematics_group, 'Time (ms)')

    return kinetics_df, kinematics_df


def process_tdms_files(file_directory):
    # sort through files, label them as the state file or tdms file
    tdms_files = []
    State_file = None
    os.chdir(file_directory)
    for root, dirs, files in os.walk(file_directory, topdown=True):
        for file in files:
            # if file.endswith('.cfg'):
            #     State_file = file
            if file.endswith('main_processed.tdms'):
                tdms_files.append((root, file))
            else:
                pass

    # calculate experiment offsets and model offsets - return both in mm, deg
    # experiment_offsets = get_offsets(State_file, file_directory)
    # model_offsets = find_model_offsets(ModelProperties)

    # print(experiment_offsets)
    # print(model_offsets)

    # store all the data together for plotting/visual analysis purpose
    all_data = []
    read_files = []
    kinetics_dfs = []
    kinematics_dfs = []
    partitioned_kinematics_dfs = []
    partitioned_kinetics_dfs = []

    # process the data in each of the tdms files
    for root_file in tdms_files:
        root, file = root_file[0], root_file[1]
        groups = tdms_contents(root, file)
        if len(groups) == 0:
            continue
        kinetics_group, kinematics_group = get_kinetics_kinematics(groups)
        kinetics_df = generate_dataframes(kinetics_group, 'Time (ms)')
        kinematics_df = generate_dataframes(kinematics_group, 'Time (ms)')
        kinetics_df, kinematics_df = clean_dataframes(kinetics_df, kinematics_df)
        kinetics_dfs.append(kinetics_df)
        kinematics_dfs.append(kinematics_df)
        print(file)

        # processing for passive flexion file:
        if 'passive flexion' in file.lower():
            # kinetics_group, kinematics_group = get_kinetics_kinematics(groups)
            # all_data.append((kinetics_group, kinematics_group))

            # this processing script was used in initial knee hub calibration, but we found a better way to do it.
            # passive_flexion_processing(groups, experiment_offsets, model_offsets,  file_directory)

            flexion_kinetics_df, flexion_kinematics_df = passive_flexion_processing_2(groups, file_directory)

            # Append the filename to the read files directory
            read_files.append(file)

            # Set the file directory as a secondary index for the dataframe (for graphing purposes)
            flexion_kinetics_df.set_index(np.full(flexion_kinetics_df.shape[0], file),
                                          inplace=True, append=True, drop=True)
            flexion_kinematics_df.set_index(np.full(flexion_kinematics_df.shape[0], file),
                                            inplace=True, append=True, drop=True)

            partitioned_kinetics_dfs.append(flexion_kinetics_df)
            partitioned_kinematics_dfs.append(flexion_kinematics_df)


        elif 'laxity' in file.lower():

            # kinetics_group, kinematics_group = get_kinetics_kinematics(groups)
            # all_data.append((kinetics_group, kinematics_group))

            # this processing script was used in initial knee hub calibration, but we found a better way to do it.
            # laxity_processing(groups, experiment_offsets, model_offsets,  file_directory)

            laxity_kinetics_df, laxity_kinematics_df = laxity_processing_2(groups, file_directory)

            # Append the filename to the read files directory
            read_files.append(file)

            # Set the file directory as a secondary index for the dataframe (for graphing purposes)

            laxity_kinetics_df.set_index(np.full(laxity_kinetics_df.shape[0], file),
                                          inplace=True, append=True, drop=True)
            laxity_kinematics_df.set_index(np.full(laxity_kinetics_df.shape[0], file),
                                          inplace=True, append=True, drop=True)

            partitioned_kinetics_dfs.append(laxity_kinetics_df)
            partitioned_kinematics_dfs.append(laxity_kinematics_df)

            # pass
        """elif 'combined' in file.lower():

            combined_kinetics_df, combined_kinematics_df = combined_processing(groups, file_directory)

            # Append the filename to the read files directory
            read_files.append(file)

            # Set the file directory as a secondary index for the dataframe (for graphing purposes)

            combined_kinetics_df.set_index(np.full(combined_kinetics_df.shape[0], file),
                                         inplace=True, append=True, drop=True)
            combined_kinematics_df.set_index(np.full(combined_kinetics_df.shape[0], file),
                                           inplace=True, append=True, drop=True)

            all_kinetics_dfs.append(combined_kinetics_df)
            all_kinematics_dfs.append(combined_kinematics_df)"""

    return pd.concat(partitioned_kinetics_dfs), pd.concat(partitioned_kinematics_dfs), \
           pd.concat(kinetics_dfs), pd.concat(kinematics_dfs), read_files



def plot_data(kinetics_data, kinematics_data, files):
    """Plot the columns of the kinetics and kinematics data in a 6x6 graph matrix"""

    # Uses matplotlib to graph given data
    print("Generating plots... This may take a moment")

    plt.close('all')
    font = {'family': 'calibri',
            'weight': 'bold',
            'size': 12, }
    fig, axs = plt.subplots(len(kinetics_data.columns.values), len(kinematics_data.columns.values),
                            figsize=(25, 15))
    x_data, y_data = kinetics_data, kinematics_data

    for file in files:
        x_chunk = x_data.xs(file, level=1, axis=0, drop_level=False)
        y_chunk = y_data.xs(file, level=1, axis=0, drop_level=False)
        for y_idx, y_col in enumerate(y_chunk.columns.values):
            for x_idx, x_col in enumerate(x_chunk.columns.values):
                axs[y_idx, x_idx].scatter(x_chunk[x_col], y_chunk[y_col], s=0.7, label=file)

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
    #fig.canvas.set_window_title(self.data_pkg_name)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    #legend_path = path + "_graphs_legend.png"
    plt.savefig(r"OK_Data_Graphs/kinetics-kinematics-graphs.png", format="png", dpi="figure", bbox_inches="tight")
    #self.export_legend(leg, legend_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # main(sys.argv[-1])

    # tdms_directory = '/home/schwara2/Documents/Open_Knees/knee_hub/oks003/calibration/DataProcessing/'
    # Model_Properties = '/home/schwara2/Documents/Open_Knees/knee_hub/oks003/calibration/Registration/model/ModelProperties.xml'
    # laxity and passive flexion processing for calibration

    # all tdms file must be in the tdms_directory, not in subfolders. State file must also be in the same folder
    tdms_directory = r"C:\Users\techteam\Documents\Open-Knees-Data-Analysis\Knee Experiments\joint_mechanics-oks006\TibiofemoralJoint\KinematicsKinetics"
    pkg_name = re.search("joint_mechanics-oks\d{3}", tdms_directory).group(0)
    # Model_Properties = "C:\\Users\schwara2\Documents\Open_Knees\oks003_calibration\Registration\ModelProperties.xml"

    # Generate partition dataframes for the kinetics and kinematics data and clean null indicies
    partitioned_kinetics_df, partitioned_kinematics_df, kinetics_data, kinematics_data, read_files =\
        process_tdms_files(tdms_directory)

    # Store the full datasets as hdf5 files
    write_HDF5(kinetics_data, kinematics_data, pkg_name)

    # kinetics_data, kinematics_data = clean_dataframes(kinetics_data, kinematics_data)
    partitioned_kinetics_df, partitioned_kinetics_df = clean_dataframes(partitioned_kinetics_df, partitioned_kinematics_df)

    # Add a numerical index to the cleaned dataframes
    partitioned_kinetics_df.set_index(np.arange(0, len(partitioned_kinetics_df.index)), append=True, inplace=True)
    partitioned_kinetics_df.set_index(np.arange(0, len(partitioned_kinetics_df.index)), append=True, inplace=True)

    # Plot the data using the cleaned dataframes
    #plot_data(kinetics_data, kinematics_data, read_files)
    plot_data(partitioned_kinetics_df, partitioned_kinetics_df, read_files)

    # Train and test prediction neural networks
    train_test_neural_networks(kinetics_data, kinematics_data)
    #neural_networks = train_test_neural_networks(partitioned_kinetics_df, partitioned_kinetics_df)


class file_opener:
    """Constructor for the the file_opener class. Initializes the data dictionaries and the directory name"""

    def __init__(self, file_or_directory, directory, pkg_name):
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.data_pkg_name = pkg_name
        if (file_or_directory == 1):
            self.file_dict = self.build_file_dict(directory)
            # Build the x & y dataframes from the presented files
            self.x_dataframe, self.y_dataframe = self.files_to_dataframes(self.file_dict)
            self.print_files_with_deviation()
        elif (file_or_directory == 2):
            self.x_dataframe, self.y_dataframe = self.read_HDF5(directory)
        else:
            ValueError("invalid input please restart")
        # Ask the user if they would like to write to hdf5 or continue
        if (self.ask_h5):
            self.write_HDF5(self.x_dataframe, self.y_dataframe)
        # Generate class attribute algorithm imported from algorithm class (drop filename column)
        self.algorithm = algorithm(self.x_dataframe, self.y_dataframe)

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

    def build_file_dict(self, directory):
        index = 0
        file_dict = defaultdict(dict)
        for root, dirs, files in os.walk(directory, topdown=True):
            for filename in files:
                if filename.endswith("main_processed.tdms"):
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
            [print(x_columnName + " std_deviation: " +
                   str(self.algorithm.do_std_devation(x_chunk[x_columnName]))) for x_columnName in
             x_chunk.columns.values]
            [print(y_columnName + " std_deviation: " +
                   str(self.algorithm.do_std_devation(y_chunk[y_columnName]))) for y_columnName in
             y_chunk.columns.values]
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
        with TdmsFile.open(os.path.join(root, filename)) as tdms_file:
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
        c

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
