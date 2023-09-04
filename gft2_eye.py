"""
functions for processing eye tracker

@author: giuliano giari, giuliano.giari@gmail.com
"""

import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from astropy.stats import kuiper_two
from gft2_utils import load_mat, resample_data, euclidean_distance, nearest
from gft2_preprocessing import read_epochs
from scipy.io import loadmat
from scipy import stats, signal
from scipy.stats import gaussian_kde


def edf2asc(sub_id, ses_id, i_block, i_trl, opt_local):
    """
    Convert edf files to asc (requires the sr-research sdk https://www.sr-support.com/thread-13.html)
    full description of the eyelink files can be found at https://www.sr-research.com/support/thread-7675.html
    """
    if ses_id != 'meg':
        edf_fname = f"{opt_local['bidsPath']}eye/{sub_id.split('sub-')[1]}_{ses_id[0]}/{sub_id.split('sub-')[1]}_b{i_block}_t{i_trl}.edf"
    else:
        edf_fname = f"{opt_local['bidsPath']}eye/{sub_id.split('sub-')[1]}/{sub_id.split('sub-')[1]}_b{i_block}_t{i_trl}.edf"
    # substitute the data format with asc
    asc_fname = edf_fname.replace('.edf', '.asc')
    # convert
    if not os.path.exists(asc_fname):
        os.system(f"edf2asc -res -vel {edf_fname}")

    return asc_fname


def parse_eye_events(asc_fname, event_id, eye_id):
    """
    Parse eye events from the asc file
    :return:
    """
    # open the asc file
    f = open(asc_fname, 'r').read().splitlines()
    if event_id == 'msg':
        cols = ['time', 'msg']
        event_str = 'MSG'
    elif event_id == 'saccade':
        cols = ['event_id', 'start_time', 'end_time', 'duration', 'start_x', 'start_y', 'end_x', 'end_y', 'amp', 'peak_vel']
        event_str = f'ESACC {eye_id}'
    elif event_id == 'blink':
        cols = ['event_id', 'start_time', 'end_time', 'duration']
        event_str = f'EBLINK {eye_id}'
    elif event_id == 'fixation':
        cols = ['event_id', 'start_time', 'end_time', 'duration', 'avg_x', 'avg_y', 'avg_pupil']
        event_str = f'EFIX {eye_id}'
    # initialize dict
    event_dict = {k: [] for k in cols}
    # retrieve events in the asc file
    for i, l in enumerate(f):
        if l.startswith(event_str):
            # split the line into its components
            l_split = l.split('\t')
            if event_id == 'msg':
                # append the time
                event_dict['time'].append(l_split[1].split(' ')[0])
                # append the message
                event_dict['msg'].append(' '.join(l.split('\t')[1].split(' ')[1:]))
            else:
                # split the first element into event_id and start_time
                event_dict['event_id'].append(event_str)
                event_dict['start_time'].append(float(l_split[0].split(event_str)[1]))
                # append the rest of the elements
                for j, k in enumerate(cols[2:]):
                    try:
                        event_dict[k].append(float(l_split[j+1]))
                    except ValueError:
                        assert l_split[j+1].strip() == '.'
                        event_dict[k].append(0)
    df = pd.DataFrame(event_dict)
    # save the dataframe
    df.to_csv(asc_fname.replace('.asc', f'_{event_id}.csv'), index=False)
    return df


def load_asc(asc_fname, opt_local):
    """
    load asc file and return only the data of interest, starting from the synctime trigger
    this is based on info found in the eyelink manual
    """
    # open the asc file and retrieve the starting sample
    f = open(asc_fname, 'r').read().splitlines()
    samples_ind = [l.split('\t')[0].isdigit() for l in f] # if the first line is digit then its a sample
    if f"sub-{asc_fname.split('_b')[0][-12:]}" in opt_local['exp_2']['subj_list']:
        # we have recorded only the right eye
        samples_cols = ['time', 'xpr', 'ypr', 'psr', 'xvr', 'yvr', 'xr', 'yr']
    else:
        samples_cols = ['time', 'xpl', 'ypl', 'psl', 'xpr', 'ypr', 'psr', 'xvl', 'yvl', 'xvr', 'yvr', 'xr', 'yr']
    """
    from the Eyelink manual
    <time>     timestamp in milliseconds
    <xpl>, <ypl>     left-eye X and Y position data
    <xpr>, <ypr>     right-eye X and Y position data
    <psl>     left pupil size (area or diameter)
    <psr>     right pupil size (area or diameter)  
    <xvl>, <yvl>     left-eye instantaneous velocity (degrees/sec)
    <xvr>, <yvr>     right-eye instantaneous velocity (degrees/sec)
    <xr>, <yr>     X and Y resolution (positionunits/degree)
    """
    df_samples = pd.read_csv(asc_fname, delim_whitespace=True, skiprows=np.where(np.logical_not(samples_ind))[0],
                             names=samples_cols, usecols=np.arange(len(samples_cols)), header=None, dtype=np.float64,
                             na_values='.').fillna(0)
    # the token '.' represents a missing value (e.g. when blinking) thus we substitute it with zeros

    # get the events dfs
    df_list = [parse_eye_events(asc_fname, event_id, opt_local['eye_id'][0].title())
               for event_id in ['saccade', 'blink', 'fixation']]
    # add an event column to the df_samples dataframe
    df_samples['event'] = np.nan
    # add the events to the df_samples dataframe
    for df in df_list:
        for i, row in df.iterrows():
            df_samples.loc[(df_samples.time >= row.start_time) & (df_samples.time <= row.end_time), 'event'] = row.event_id
    # find the trigger of the eye tracker
    trigger_time = int([x[1]['time'] for x in parse_eye_events(asc_fname, 'msg', []).iterrows()
                        if x[1]['msg'] == 'SYNCTIME'][0])
    trigger_ind = np.where(df_samples.time.values == trigger_time)[0][0]
    # find the length of the trial according to the trigger
    # cut the time series using the trigger
    return df_samples.drop(np.arange(trigger_ind)).reset_index(drop=True)


def find_lag(a, b, show_figure=False):
    """
    Find lag (i.e. time delay) between eye-tracker edf file and meg recorded eye tracker by maximizing cross-correlation
    between x-axis time series
    #a = eye_data_meg.copy()
    #b = eye_data_edf.copy()
    """

    if len(a) != len(b):
        m = min([len(a), len(b)])
        a = a[:m]
        b = b[:m]

    assert sum(np.isnan(a)) == 0, 'a contains nan values'
    assert sum(np.isnan(b)) == 0, 'b contains nan values'

    # demean
    a -= a.mean()
    b -= b.mean()
    # cross correlation and lags
    out = signal.correlate(a, b, mode='full')
    lags = signal.correlation_lags(len(a), len(a), mode='full')
    lag = lags[np.abs(out).argmax()] #- (len(a)-1)]
    
    if show_figure:
        # shift and cut the data up to the length of the meg data
        # https://stackoverflow.com/questions/2150108/efficient-way-to-rotate-a-list-in-python
        b_lag = np.roll(b, lag)[:len(a)]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[1].plot(lags, out)
        ax[1].set_xlabel('Lag (samples)')
        ax[1].set_ylabel('Correlation')
        ax[1].vlines(lag, ax[1].get_ylim()[0], ax[1].get_ylim()[1], color='r', ls='--')
        ax[1].set_title('Cross-correlation \n Eye is ' + str(lag) + ' samples after meg')
        line_list = []
        line_list.append(ax[0].plot(np.arange(0, len(b))/1000, b, color='gray', alpha=.5, label='asc eye data'))
        line_list.append(ax[0].plot(np.arange(0, len(b))/1000, b_lag, c='g', alpha=.5, label='asc eye data realigned'))
        ax[0].set_ylabel('Gaze position (pixel)')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title('Eye time series')
        ax1 = ax[0].twinx()
        line_list.append(ax1.plot(np.arange(0, len(b))/1000, a, color='r', alpha=.5, label='meg eye data'))
        ax1.set_ylabel('Gaze position (volt)')
        ax[0].legend(line_list[0]+line_list[1]+line_list[2], [l_[0].get_label() for l_ in line_list], loc='best')
        ax[0].set_xlim(0, 10)
        plt.tight_layout()

    return lag


def eye_resample(df, opt_local):
    """
    resample the eye tracker data to match the stimuli presentation rate (120 Hz)
    """
    # determine the resampling factor
    n_samples = opt_local['tmax'] * 1000
    n_samples_120 = int(opt_local['tmax'] / ( 1 / 120 ))
    resample_factor = n_samples / n_samples_120
    # initialize output
    eye_data_res = {k: [] for k in df.keys()}
    # loop over trials
    for trl_id in df.trl_id.unique():
        # get the data of this trial
        trl_df = df.loc[df['trl_id'] == trl_id]
        # resample the data columns
        for data_col in df.columns.difference(
                ['trl_id', 'times', 'angle', 'ang_res', 'trajectory', 'event']):
            data_res, times_res = resample_data(trl_df[data_col].values,
                                                trl_df['times'].values, resample_factor)
            assert len(data_res) == n_samples_120
            eye_data_res[data_col].extend(data_res)
        # add the text columns
        eye_data_res['times'].extend(times_res)
        eye_data_res['trl_id'].extend([trl_id] * n_samples_120)
        eye_data_res['ang_res'].extend(np.repeat(trl_df.ang_res.values[0], n_samples_120))
        # add the events column by finding the nearest labelled time point in the time series
        eye_data_res['event'].extend([trl_df.event.values[nearest(trl_df.times.values, t)] for t in times_res])
        eye_data_res['angle'].extend([trl_df.angle.values[nearest(trl_df.times.values, t)] for t in times_res])
        eye_data_res['trajectory'].extend([trl_df.trajectory.values[nearest(trl_df.times.values, t)] for t in times_res])
    # return a dataframe with resampled data
    return pd.DataFrame.from_dict(eye_data_res)


def eye_preprocess(sub_id, ses_id, opt_local, show_figure=False):
    """
    Preprocess eye-tracker data
    """

    eye_data_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_eye-data.csv.gz"

    if not os.path.exists(eye_data_fname):
        # 1)
        # load the meg data
        epochs = read_epochs(sub_id, ses_id, opt_local, 'task')
        epochs.crop(0, round(epochs.times[-1]))
        if sub_id in opt_local['exp_2']['subj_list']: # are saved as left eye in the meg
            epochs.pick_channels([opt_local['eye_chan']["LX"]])
        else:
            epochs.pick_channels([opt_local['eye_chan'][f"{opt_local['eye_id'][0].title()}X"]])
        # load exp mat file
        if ses_id != 'meg':
            mat_exp = load_mat(f"{opt_local['behPath']}{sub_id.split('sub-')[1]}_{ses_id[0]}/{sub_id.split('sub-')[1]}_{ses_id[0]}_exp.mat")
        else:
            mat_exp = load_mat(f"{opt_local['behPath']}{sub_id.split('sub-')[1]}/{sub_id.split('sub-')[1]}_exp.mat")
        # initialize output
        eye_data = {k: [] for k in ['trl_id', 'event', 'times', 'trajectory', 'ang_res', 'trj_id', 'angle', 'is_flipped',
                                    f"xp{opt_local['eye_id'][0]}", f"yp{opt_local['eye_id'][0]}", f"ps{opt_local['eye_id'][0]}",
                                    f"xv{opt_local['eye_id'][0]}", f"yv{opt_local['eye_id'][0]}", "xr", "yr"]}
        # loop over blocks
        for i_block in range(1, 7):
            # take one trial
            for i_trl in range(1, 9):
                # take into account those participants that had technical problems (either eye tracker or meg)
                if (sub_id == 'sub-19961224AAMN' and ses_id == 'dots' and i_block == 6) or \
                    (sub_id == 'sub-19820925PEGA' and ses_id == 'lines' and i_block == 3 and i_trl == 5) or \
                    (sub_id == 'sub-19941010LCTY' and ses_id == 'lines' and i_block == 4 and
                     (i_trl == 6 or i_trl == 7 or i_trl == 8)) or \
                    (sub_id == 'sub-19870521NRIA' and ses_id == 'lines' and i_block == 6 and i_trl == 6) or \
                    (sub_id == 'sub-19940921BRFL' and ses_id == 'lines' and i_block == 1) or \
                    (sub_id == 'sub-19960711EEBN' and ses_id == 'lines' and i_block == 3) or \
                    (sub_id == 'sub-19960711EEBN' and ses_id == 'lines' and i_block == 4) or \
                        (sub_id == 'sub-20010614DMCA' and ses_id == 'dots' and i_block == 1):
                    continue
                # get eye tracker data saved along with meg data
                trl_ind = np.where(epochs.metadata.trl_id == f"b{i_block}_t{i_trl}")[0][0]
                x_meg = epochs[trl_ind].get_data().squeeze()
                # load and process the eye tracker data
                asc_fname = edf2asc(sub_id, ses_id, i_block, i_trl, opt_local.copy())
                trl_df = load_asc(asc_fname, opt_local.copy())
                # determine time lag of the eye tracker data
                lag = find_lag(x_meg, trl_df[f"xp{opt_local['eye_id'][0]}"].values.copy(), show_figure)
                # correct timings
                trl_df = trl_df.shift(lag, axis=0).dropna()
                assert len(trl_df) >= opt_local['tmax'] * epochs.info['sfreq']
                # add data to the dictionary
                for data_id in ['xp', 'yp', 'ps', 'xv', 'yv']:
                    eye_data[f"{data_id}{opt_local['eye_id'][0]}"].extend(
                        trl_df[f"{data_id}{opt_local['eye_id'][0]}"].values[:len(x_meg)])
                for data_id in ['xr', 'yr', 'event']:
                    eye_data[data_id].extend(trl_df[data_id].values[:len(x_meg)])
                # add time, trial id and angular resolution
                times = ( trl_df['time'].values[:len(x_meg)] - trl_df['time'].values[0] ) / 1000
                eye_data['times'].extend(times)
                eye_data['trl_id'].extend([f"b{i_block}_t{i_trl}"] * len(x_meg))
                eye_data['ang_res'].extend(list(epochs[trl_ind].event_id.values()) * len(x_meg))
                # make time axis for the trajectories
                trj_times = np.arange(0, 44000 + 167, 167) / 1000
                # add info on the trajectory
                trajectory = np.zeros(len(x_meg))
                for i, (onset, offset) in enumerate(zip(trj_times[:-1], trj_times[1:])):
                    trajectory[(times >= onset) & (times < offset)] = int(i + 1)
                eye_data['trajectory'].extend(trajectory.astype(int))
                # get the trajectory order
                order_list = mat_exp['stim'][f'block_{i_block}'][f'trl_{i_trl}']['order']
                # replicate through the number of cycles
                trj_list = list(order_list) * opt_local['n_cycles'][str(list(epochs[trl_ind].event_id.values())[0])]
                assert len(trj_list) == 264
                # add trajectory id to the df
                trj_id = np.zeros(len(trajectory))
                for i, x in enumerate(trj_list):
                    trj_id[trajectory == i + 1] = x
                eye_data['trj_id'].extend(trj_id.astype(int))
                # add the trajectory angle in 360°
                eye_data['angle'].extend(np.mod((trj_id.astype(int) - 1) * list(epochs[trl_ind].event_id.values())[0]
                                                - 10, 360))
                # add is_flipped, a boolean indicating whether the trajectory started in its original or flipped version
                # i.e., 180° rotation
                is_flipped = np.zeros(len(trajectory))
                for i, x in enumerate(np.concatenate(mat_exp['stim'][f'block_{i_block}'][f'trl_{i_trl}']['is_flipped'])):
                    is_flipped[trajectory == i + 1] = x
                eye_data['is_flipped'].extend(is_flipped.astype(bool))
        # transform to dataframe
        df = pd.DataFrame.from_dict(eye_data)
        # save csv
        df.to_csv(eye_data_fname, compression="gzip")
    else:
        df = pd.read_csv(eye_data_fname, sep=',', index_col=0)

    # 2) compute percentage fixation and remove trials with fixation lower than 80%
    perc_df = compute_percentage_fixation(sub_id, ses_id, df, opt_local)
    # remove trials in which participants have fixated for < 80% of the trial
    df = df.loc[(df['trl_id'].isin(perc_df.loc[(perc_df['fixation_perc'] >
                                                opt_local['fixation_thresh'])]['trl_id']))].reset_index(drop=True)

    # 3) remove time points with blinks. within trial
    blink_ind_list = []
    for trl_id in df.trl_id.unique():
        # get this trial data
        trl_df = df.loc[df['trl_id'] == trl_id]
        # find in csv the time points with blinks (indicate by EBLINK R)
        blink_ind = trl_df.loc[trl_df['event'].values == f"EBLINK {opt_local['eye_id'][0].upper()}"].index
        if blink_ind.size > 0:
            # split the blink indices into separate blinks
            blink_ind = np.split(blink_ind, np.where(np.diff(blink_ind) != 1)[0] + 1)
            # for each blink, include the time points before and after the blink
            for i, b in enumerate(blink_ind):
                blink_ind[i] = np.arange(min(b)-50, max(b)+50)
            blink_ind = np.unique(np.concatenate(blink_ind))
            # make sure that the indices are within the trial
            if any(blink_ind > trl_df.index[-1]):
                blink_ind = blink_ind[blink_ind < trl_df.index[-1]]
            if any(blink_ind < trl_df.index[0]):
                blink_ind = blink_ind[blink_ind > trl_df.index[0]]
            blink_ind_list.extend(blink_ind)
    # substitute the blinks with NaN except for the trl_id, event, times, trj_id, ang_res, trajectory columns
    df.loc[blink_ind_list, df.columns.difference(
        ['trl_id', 'event', 'times', 'trj_id', 'angle', 'is_flipped', 'ang_res', 'trajectory'])] = np.nan

    # 4) reorganize the angles based on "is_flipped", thus matching the actual presented angle 
    df.loc[df['is_flipped'].values, 'angle'] = np.mod(df.loc[df['is_flipped'].values, 'angle'].values + 180, 360)

    # 5) interpolate the missing values
    df = df.groupby('trl_id').apply(lambda x: x.interpolate(method='linear', limit_direction='both'))

    return df


def eye_correlation(sub_id, ses_id, opt_local):
    """
    Compute distance between eye-tracker/stimulus position and the fixation point (screen center).
    Then correlate these distances. This replicates the method in Wilming et al., 2018
    """
    assert ses_id == 'dots'

    # define the file name
    correlation_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_eye-corr-{corr}.h5"

    if not os.path.exists(correlation_fname):
        # preprocess or load eye tracker data and remove unnecessary columns
        df = eye_preprocess(sub_id, ses_id, opt_local)
        df.drop(['trj_id', 'is_flipped'], inplace=True, axis=1)
        # resample data to match the stimulus sampling rate
        eye_data = eye_resample(df, opt_local)
        # load the experiment mat file
        mat_exp = load_mat(f"{opt_local['behPath']}{sub_id.split('sub-')[1]}_{ses_id[0]}/{sub_id.split('sub-')[1]}_{ses_id[0]}_exp.mat")
        # initialize the output dictionary
        out_dict = {event_id: {k: [] for k in ['r']} for event_id in opt_local['ang_res']}
        # loop over blocks
        for i_block in range(1, 7):
            # load the block mat file
            block_fname = f"{opt_local['behPath']}{sub_id.split('sub-')[1]}_{ses_id[0]}/{sub_id.split('sub-')[1]}_BLOCK0{i_block}_{ses_id[0]}.mat"
            mat_block = loadmat(block_fname, struct_as_record=False, squeeze_me=True)['behav_experiment']
            # loop over trials
            for i_trl, mat_trl in enumerate(mat_block):
                # get this trial eye data
                eye_trl = eye_data.loc[eye_data['trl_id'] == f"b{i_block}_t{i_trl + 1}"]
                # determine positions that were not presented due to being in the fixation window
                dot_presented = mat_exp['stim']['dot_to_present'][f'ang_res{mat_trl.ang_res}']
                not_presented = np.zeros(max(dot_presented), dtype=bool)
                not_presented[np.setdiff1d(np.arange(min(dot_presented), max(dot_presented)), dot_presented)] = True
                not_presented = [not_presented] * 264
                # find the time points with blinks
                blink_ind = eye_trl['event'].isin([f"EBLINK {opt_local['eye_id'][0].title()}"]).values
                # remove from the eye time series the time points in which the dot was not present on screen or
                # there were blinks
                x_pos = eye_trl[f"xp{opt_local['eye_id'][0]}"].values
                y_pos = eye_trl[f"yp{opt_local['eye_id'][0]}"].values
                not_presented = np.reshape(not_presented, -1)
                x_eye = x_pos[np.logical_not(not_presented + blink_ind)]
                y_eye = y_pos[np.logical_not(not_presented + blink_ind)]

                if x_eye.size == 0:
                    r = np.nan
                else:
                    # get the x-y pixel coordinate of the stimuli at each time stamp; x == 0, y == 1
                    trj_mat = mat_exp['stim'][f'block_{i_block}'][f'trl_{i_trl + 1}']['trajectories']
                    x_dot = trj_mat[:, 0, :]
                    y_dot = trj_mat[:, 1, :]
                    # reshape to match the continuous eye time series
                    x_dot = np.reshape(x_dot, -1)
                    y_dot = np.reshape(y_dot, -1)
                    x_dot = x_dot[np.logical_not(not_presented + blink_ind)]
                    y_dot = y_dot[np.logical_not(not_presented + blink_ind)]
                    # compute distance between fixation point (screen center) and: 1) eye position, 2) dot position
                    eye_pos = [euclidean_distance([mat_exp['ptb']['x_center'], mat_exp['ptb']['y_center']],
                                                   [x_eye[tp], y_eye[tp]])
                                for tp in range(len(x_dot))]
                    stim_pos = [euclidean_distance([mat_exp['ptb']['x_center'], mat_exp['ptb']['y_center']],
                                                   [x_dot[tp], y_dot[tp]])
                                for tp in range(len(x_dot))]
                    # compute correlation
                    r, _ = stats.pearsonr(eye_pos, stim_pos)
                # collect output
                event_id = mat_trl.ang_res
                out_dict[event_id]['r'].append(r)

        # save the output as h5
        f = h5py.File(correlation_fname, 'w')
        [f.create_dataset(name=f'/{k}/r', data=v['r']) for k, v in out_dict.items()]
        f.close()

    else: # load the output
        f = h5py.File(correlation_fname, 'r')
        out_dict = {k: list(np.array(f.get(name=f'/{k}/r'))) for k in f.keys()}
        f.close()
    return out_dict


def compute_percentage_fixation(sub_id, ses_id, csv, opt_local):
    """
    Compute percentage of time that eye is within the fixation window
    """

    out_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_percentage.csv"

    if not os.path.exists(out_fname):

        out = {k: [] for k in ['sub_id', 'ses_id', 'trl_id', 'ang_res', 'fixation_perc']}
        for trl_id in csv.trl_id.unique():
            # get data of this trial
            trl_data = csv.loc[csv['trl_id'] == trl_id]
            # get eye position
            x_eye = trl_data['xpr'].values
            y_eye = trl_data['ypr'].values
            # check if the eye is within the fixation window
            # in fixation is now a boolean array
            in_fixation = ((opt_local['fixWin'][0] < x_eye) & (x_eye < opt_local['fixWin'][2])) & \
                          ((opt_local['fixWin'][1] < y_eye) & (y_eye < opt_local['fixWin'][3]))
            # compute proportion
            trl_perc = sum(in_fixation) / len(in_fixation)
            # store output
            out['sub_id'].append(sub_id)
            out['ses_id'].append(ses_id)
            out['trl_id'].append(trl_id)
            out['ang_res'].append(trl_data.ang_res.values[0])
            out['fixation_perc'].append(trl_perc)
        # make dataframe and save as csv
        out_df = pd.DataFrame.from_dict(out)
        out_df.to_csv(out_fname)
    else:
        out_df = pd.read_csv(out_fname, index_col=0)
    return out_df


def _realign_eye_data(df, mat, ses_id, trl_id, opt_local):
    """ realign eye data to the start of the trajectory 1. adapted from gft2_utils.py """
    # define the max length of trials according to cycle time in angular resolution
    max_len = 42 if df.ang_res[0] == 30 else 40
    # get trial information
    order_list = mat['stim'][f'block_{trl_id.split("_")[0][1]}'][f'trl_{trl_id.split("_")[1][1]}']['order']
    if ses_id == 'meg':
        trj_list = list(order_list) * opt_local['n_cycles'][str(df.ang_res[0])] * 2
    else:
        trj_list = list(order_list) * opt_local['n_cycles'][str(df.ang_res[0])]
    assert len(trj_list) == 264
    # get the indices of the trj_id trajectory
    trj_id_ind = np.where(np.array(trj_list) == 1)[0]
    # here +1 is to account for python base-0 indexing
    t0 = ((min(trj_id_ind) + 1) * opt_local['baserate_ms']) / 1000
    tEnd = (max(trj_id_ind) * opt_local['baserate_ms']) / 1000
    # now check that this trial length matches the expected length, if not we add or remove some samples
    if (tEnd - t0) != max_len:
        # here is done using + since the difference in the second term is negative
        tEnd += (max_len - (tEnd - t0))
    # check
    assert tEnd - t0 - max_len < 1e-10
    # crop the csv to retain only the data based on time
    return df.loc[(df['times'] >= t0) & (df['times'] < tEnd)].reset_index(drop=True)


def realign_to_dots(sub_id, ses_id, opt_local):
    """
    Extract time window of eye position centered around dot presentation
    """
    out_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_desc-dotOnset.csv.gz"
    if not os.path.exists(out_fname):
        # load the eye tracker data
        eye_df = eye_preprocess(sub_id, ses_id, opt_local, show_figure=False)
        # load the experiment mat file
        mat_exp = load_mat(f"{opt_local['behPath']}{sub_id.split('sub-')[1]}_{ses_id[0]}/{sub_id.split('sub-')[1]}_{ses_id[0]}_exp.mat")
        # initialize output
        out_df = pd.DataFrame()
        # loop over blocks
        for trl_id in eye_df.trl_id.unique():
            # get block number and trial number from trl_id
            i_block = int(trl_id.split('_')[0].split('b')[1])
            i_trl = int(trl_id.split('_')[1].split('t')[1])
            # load the block mat file
            block_fname = f"{opt_local['behPath']}{sub_id.split('sub-')[1]}_{ses_id[0]}/{sub_id.split('sub-')[1]}_BLOCK0{i_block}_{ses_id[0]}.mat"
            mat_block = loadmat(block_fname, struct_as_record=False, squeeze_me=True)['behav_experiment']
            # get the data of this block/trial
            eye_trl = eye_df.loc[(eye_df['trl_id'] == trl_id)].reset_index(drop=True)
            # loop over dots
            for i_dot in range(2):
                # get the timing of appearance of this dot
                dot_sample = np.round(opt_local['baserate_ms'] * mat_exp['stim'][f'block_{i_block}'][f'trl_{i_trl}']['dot_trj'][i_dot] +
                                      mat_exp['stim'][f'block_{i_block}'][f'trl_{i_trl}']['dot_ind'][i_dot] * 8.3)
                # get the data one second before and one second after the dot presentation
                dot_trl = eye_trl.loc[(eye_trl['times'] >= (dot_sample-1000) / 1000) & \
                                      (eye_trl['times'] < (dot_sample+1000) / 1000)].reset_index(drop=True)
                # calculate distance from the dot
                dot_pos = mat_exp['stim'][f'block_{i_block}'][f'trl_{i_trl}']['dot_pos'][i_dot]
                dot_dist = np.array([euclidean_distance(dot_pos, [dot_trl[f"xp{opt_local['eye_id'][0]}"].values[tp],
                                                                  dot_trl[f"yp{opt_local['eye_id'][0]}"].values[tp]])
                                     for tp in range(len(dot_trl))])
                # update time axis
                times = np.arange(-1, 1, 0.001)
                # drop times columns in dot_trl df
                dot_trl = dot_trl.drop(columns=['times'])
                # add the data to the dataframe
                dot_trl.loc[:, 'times'] = times
                dot_trl.loc[:, 'dist'] = dot_dist
                dot_trl.loc[:, 'dot_id'] = i_dot
                # concatenate with the previous data
                out_df = pd.concat([out_df, dot_trl])
        # save the data
        out_df.to_csv(out_fname, index=False, compression='gzip')


def _compute_heatmap(x, y, remove_nans=True, downsampling_factor=1):
    """
    create a heatmap from the x and y coordinates
    """
    # substitute 0s with nans
    if remove_nans:
        x = x[np.logical_not(np.isnan(x))]
        y = y[np.logical_not(np.isnan(y))]

    grid = np.array([1440, 1080]) # size of the screen
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    # create a multidimensional grid covering the whole image. these will be the points at which the kde is evaluated
    X, Y = np.mgrid[0:grid[0]:downsampling_factor, 0:grid[1]:downsampling_factor]
    positions = np.vstack([X.ravel(), Y.ravel()])
    # create a gaussian kernel density estimate
    kernel = gaussian_kde(np.vstack([x, y]))
    hmap = np.reshape(kernel.evaluate(positions).T, X.shape)
    return hmap


def compute_heatmaps(sub_id, ses_id, opt_local):
    """
    Compute heatmaps for each trajectory
    """
    out_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_{opt_local['hmap_method']}_" \
                f"{opt_local['hmap_downsampling']}-heatmaps.h5"
    out_dict = {str(ang_res): {} for ang_res in opt_local['ang_res']}
    if not os.path.isfile(out_fname):
        # open output h5 file
        f = h5py.File(out_fname, 'w')
        # load the preprocessed eye tracker data
        eye_df = eye_preprocess(sub_id, ses_id, opt_local)
        # transform angles to 180° degree space
        eye_df['angle'] = np.mod(eye_df['angle'], 180)
        # loop over angular resolutions
        for ang_res in opt_local['ang_res']:
            # get data of this angular resolution
            ang_df = eye_df.loc[(eye_df['ang_res'] == ang_res)].reset_index(drop=True)
            for i, angle in enumerate(ang_df.angle.unique()):
                # get eye position
                x = ang_df.loc[(ang_df['angle'] == angle), f"xp{opt_local['eye_id'][0]}"].values
                y = ang_df.loc[(ang_df['angle'] == angle), f"yp{opt_local['eye_id'][0]}"].values
                # create heatmap of eye positions
                heatmap = _compute_heatmap(x, y, remove_nans=True)
                f.create_dataset(f"{ang_res}/{angle}", data=heatmap)
                # store in output dictionary
                out_dict[str(ang_res)][str(angle)] = heatmap
    else:
        f = h5py.File(out_fname, 'r')
        for ang_res in f.keys():
            for angle in f[str(ang_res)].keys():
                out_dict[ang_res][angle] = f[ang_res][angle][:]
    f.close()
    # return sorted dictionary by key in ascending order
    return {ang_res: dict(sorted(out_dict[ang_res].items(), key=lambda x: int(x[0]))) for ang_res in out_dict.keys()}


def correlate_heatmaps(sub_id, ses_id, opt_local):
    """
    Correlate heatmaps of different angular resolutions
    :return:
    """
    from scipy import stats, spatial
    import itertools
    # compute or load the heatmaps
    heatmaps_dict = compute_heatmaps(sub_id, ses_id, opt_local)
    # create output dictionary
    out_dict = {k: [] for k in ['sub_id', 'ang_res', 'ang_1', 'ang_2', 'ang_diff', 'hmap_r']}
    for ang_res in opt_local['ang_res']:
        # select the central fixation window
        in_fixation = slice(int(opt_local['fixWin'][0]), int(opt_local['fixWin'][2])), \
                      slice(int(opt_local['fixWin'][1]), int(opt_local['fixWin'][3]))
        heatmaps_array = np.array([heatmaps_dict[str(ang_res)][str(angle)][in_fixation[0], in_fixation[1]]
                                   for angle in heatmaps_dict[str(ang_res)].keys()])
        # flatten the heatmaps and compute the correlation matrix
        r = 1 - spatial.distance.pdist(heatmaps_array.reshape([heatmaps_array.shape[0], -1]), 'correlation')
        # compute the angular difference between all pairs of angles
        # and select the upper triangle of the resulting matrix
        angles_list = [int(x) for x in heatmaps_dict[str(ang_res)].keys()]
        angles_pair = np.array(list(itertools.combinations(angles_list, 2)))
        ang_diff = np.mod(np.abs(angles_pair[:, 0] - angles_pair[:, 1]), 180)
        # transform the angles difference to be between 0 and 90
        ang_diff[ang_diff > 90] = 180 - ang_diff[ang_diff > 90]
        # store output
        out_dict['sub_id'].extend([sub_id]*len(r))
        out_dict['hmap_r'].extend(r)
        out_dict['ang_diff'].extend(ang_diff)
        out_dict['ang_1'].extend(angles_pair[:, 0])
        out_dict['ang_2'].extend(angles_pair[:, 1])
        out_dict['ang_res'].extend([ang_res]*len(r))
    #
    df = pd.DataFrame.from_dict(out_dict)
    df.to_csv(f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_hmap_corr.csv", index=False)


def compare_angle_distributions(sub_id, ses_id, opt_local):
    """
    Compute gaze angle at each time point and compare distributions for the different trajectories
    :return:
    """
    # load the data
    eye_df = eye_preprocess(sub_id, ses_id, opt_local)
    # remove fixations outside the fixation window
    in_fixation = slice(int(opt_local['fixWin'][0]), int(opt_local['fixWin'][2])), \
                  slice(int(opt_local['fixWin'][1]), int(opt_local['fixWin'][3]))
    eye_df = eye_df.loc[(eye_df['xpr'] >= in_fixation[0].start) & (eye_df['xpr'] <= in_fixation[0].stop) &
                        (eye_df['ypr'] >= in_fixation[1].start) & (eye_df['ypr'] <= in_fixation[1].stop)]
    # compute angle of each fixation
    eye_df['trj_angle'] = np.mod(eye_df['angle'], 180)
    xCenter = 1440 // 2
    yCenter = 1080 // 2
    # compute angle of the eyes with respect to the center of the screen
    angles = np.arctan2(eye_df['ypr'] - yCenter, eye_df['xpr'] - xCenter)
    # convert values in the range [0, 2pi]
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    eye_df['eye_angle'] = angles
    # create output dictionary
    out_dict = {k: [] for k in ['sub_id', 'ang_res', 'ang_1', 'ang_2', 'ang_diff', 'D']}
    # loop over angular resolutions
    for ang_res in opt_local['ang_res']:
        # get data of this angular resolution
        ang_df = eye_df.loc[(eye_df['ang_res'] == ang_res)].reset_index(drop=True)
        angles_pair = np.array(list(itertools.combinations(sorted(ang_df.trj_angle.unique()), 2)))
        for trj_angle1, trj_angle2 in angles_pair:
            # get eye position
            alpha1 = ang_df.loc[(ang_df['trj_angle'] == trj_angle1)].reset_index(drop=True)['eye_angle'].values
            alpha2 = ang_df.loc[(ang_df['trj_angle'] == trj_angle2)].reset_index(drop=True)['eye_angle'].values
            # kuiper test
            D, fpp = kuiper_two(alpha1, alpha2)
            # store output
            out_dict['sub_id'].append(sub_id)
            out_dict['ang_res'].append(ang_res)
            out_dict['ang_1'].append(trj_angle1)
            out_dict['ang_2'].append(trj_angle2)
            ang_diff = np.mod(np.abs(trj_angle1 - trj_angle2), 180)
            if ang_diff > 90:
                # transform the angles difference to be between 0 and 90
                out_dict['ang_diff'].append(180 - ang_diff)
            else:
                out_dict['ang_diff'].append(float(ang_diff))
            out_dict['D'].append(D)
    df = pd.DataFrame.from_dict(out_dict)
    df.to_csv(f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_hist_D.csv", index=False)
    # plot the correlation for the different angular resolutions
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    for i, ang_res in enumerate([15, 30]):
        df_ang = df.loc[(df['ang_res'] == ang_res)]
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_ang['D'], df_ang['ang_diff'])
        # plot the linear regression model
        ax[i].scatter(df_ang['D'], df_ang['ang_diff'])
        ax[i].plot(df_ang['D'], intercept + slope * df_ang['D'], 'r', label=f'{ang_res}°')
    plt.setp(ax, xlabel='Correlation', ylabel='Angular difference')
    plt.suptitle(f"{sub_id} - {ses_id} {ang_res}° p:{p_value:.4f}")
    fig.savefig(f"{opt_local['figPath']}{sub_id}_ses-{ses_id}_angleD_corr.png")

    