"""
utility functions
@author: giulianogiari
"""

import h5py
import logging
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import warnings
from scipy.io import loadmat, matlab


def nearest(array, value):
    """ return the index of closest value in array """
    return np.abs(np.asarray(array) - value).argmin()


def resample_data(data_array, data_times, resample_factor=4, verbose=False):
    """
    resample the data to match the resample factor.
    if resample factor is int this is done by taking one data point every resample_factor.
    else, if resample factor is a float, resampling is done through interpolation [1].
    this assumes proper filtering has been applied
    [1] https://stackoverflow.com/questions/29085268/resample-a-numpy-array
    """
    if verbose: print('resampling data')
    if isinstance(resample_factor, int):
        time_inds = slice(0, None, resample_factor)
        times_res = data_times[time_inds]
        data_res = data_array[..., time_inds]
    elif isinstance(resample_factor, float):
        time_inds = np.linspace(0, data_times[-1], round(len(data_times) / resample_factor))
        times_res = np.interp(time_inds, data_times, data_times)
        data_res = np.interp(time_inds, data_times, data_array)

    return data_res, times_res



def split_segments_array(data_array, data_times, data_class, noverlap, seg_len, expected_seg):
    """
    Split the data time series.
    noverlap: amount of overlap, in seconds
    seg_len: length of each segment, in seconds
    """
    # check the dimensions of the input
    if data_class == mne.source_estimate.VolSourceEstimate or \
            data_class == mne.evoked.EvokedArray or \
            data_class == mne.source_estimate.SourceEstimate or data_class == mne.io.fiff.raw.Raw:
        assert data_array.ndim == 2 and data_array.shape[-1] == len(data_times)
    elif data_class == mne.epochs.EpochsArray or data_class == mne.epochs.EpochsFIF:
        assert data_array.ndim == 3 and data_array.shape[-1] == len(data_times)

    # adjust the input to samples and compute some values
    sfreq = 1 / np.diff(data_times)[0]
    noverlap = int(sfreq * noverlap)
    nperseg = int(seg_len * sfreq)
    step = int(nperseg - noverlap)
    # now check the segments
    n_seg = len(np.arange(0, data_times[-1] * sfreq, step).astype(int))
    if n_seg - expected_seg != 0:
        warnings.warn(f"{n_seg-expected_seg}, 'segments are too short")

    # split the time series into smaller segments
    # this is adapted from scipy.signal.spectral._fft_helper line 1896 (version 1.6.1)
    # https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/signal/spectral.py#L1870
    shape = data_array.shape[:-1] + (n_seg, nperseg)
    strides = data_array.strides[:-1] + (step * data_array.strides[-1], data_array.strides[-1])
    if data_class == mne.source_estimate.VolSourceEstimate or \
            data_class == mne.evoked.EvokedArray or \
            data_class == mne.source_estimate.SourceEstimate or data_class == mne.io.fiff.raw.Raw:
        result = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides).transpose(1, 0, 2)
    elif data_class == mne.epochs.EpochsArray or mne.epochs.EpochsFIF:
        result = np.concatenate(np.lib.stride_tricks.as_strided(data_array,
                                                                shape=shape, strides=strides).transpose(0, 2, 1, 3), 0)
    # check dimensions
    if data_class == mne.source_estimate.VolSourceEstimate or data_class == mne.evoked.EvokedArray or data_class == mne.io.fiff.raw.Raw:
        assert result.shape[0] == n_seg, 'different number of segments than requested'
    elif data_class == mne.epochs.EpochsArray or mne.epochs.EpochsFIF:
        assert result.shape[0] == n_seg * data_array.shape[0], 'different number of segments than requested'
    assert result.ndim == 3, 'data array has more than 3 dimensions'
    return result


def significance_bar(t, p, x, y, ax, lw=1):
    """
    Plot significance bar
    https://stackoverflow.com/questions/37707406/python-seaborn-how-to-add-significance-bars-and-asterisks-to-boxplots
    """
    # check the procedure went fine
    assert isinstance(t, (int, float))
    assert isinstance(p, (int, float))

    if p <= 0.001:
        sig = '***'
    elif p <= 0.01:
        sig = '**'
    elif p <= 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    ax.plot(x, y, 'k', lw=lw)
    ax.text(np.mean(x), y[1], sig, ha='center', va='bottom', color='k')
    # ax.set_ylim((0, y[0]+y[0]/2))


def extend_ylim(ax):
    """
    Add a bit of space to the y axis limits of a plot.
    """
    ylim = ax.get_ylim()
    air = np.diff(ylim)[0] * 0.1
    # ax.set_ylim(ylim[0] - air, ylim[1] + air)
    return air


def load_mat(filename):
    """
    taken from
    https://stackoverflow.com/questions/48970785/complex-matlab-struct-mat-file-read-by-python
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


def realign_to_trj(data_object, trj_id, opt_local):
    """
    Realign the single trials to one trajectory, returning the realigned data array
    """
    logging.getLogger('mne').info(f'realigning trials in time to trajectory {trj_id}')
    # define the max length of trials according to cycle time in angular resolution
    max_len = 42 if list(data_object.event_id.keys())[0] == 'ang_res_30' else 40

    out = []
    for i_trl in range(len(data_object)):
        # get the data of this trial
        trl = data_object[i_trl]
        # get the id of each trajectory, by multiplying the order of presentation for the number of cycles
        # of this angular resolution, coded in .events[0, -1]
        order_list = [int(x.replace('[', '').replace(']', '').strip())
                      for x in trl.metadata['order'].values[0][0].split(' ')
                      if x.replace('[', '').replace(']', '').strip()]
        if data_object.filename.split('ses-')[-1].split('_')[0] == 'meg':
            trj_list = order_list * opt_local['n_cycles'][str(trl.events[0, -1])] * 2
        else:
            trj_list = order_list * opt_local['n_cycles'][str(trl.events[0, -1])]
        assert len(trj_list) == 264
        # get the indices of the trj_id trajectory
        trj_id_ind = np.where(np.array(trj_list) == trj_id)[0]
    
        # here +1 is to account for python base-0 indexing
        t0 = ( (min(trj_id_ind)+1) * opt_local['baserate_ms'] ) / 1000
        tEnd = ( max(trj_id_ind) * opt_local['baserate_ms'] ) / 1000

        # now check that this trial length matches the expected length, if not we add or remove some samples
        if (tEnd - t0) != max_len:
            # here is done using + since the difference in the second term is negative
            tEnd += (max_len - (tEnd - t0))
        # check
        assert tEnd - t0 == max_len
        # crop the time series to match the expected length
        trl.crop(t0, tEnd)
        # store the individual trials
        out.append(trl.get_data().squeeze()[..., : int(max_len * data_object.info['sfreq']) ])
    # return an epochs object
    out = mne.EpochsArray(np.array(out), data_object.info, events=data_object.events, tmin=0, baseline=None,
                          event_id=data_object.event_id, reject=None, proj=False, metadata=data_object.metadata)
    out.reject = data_object.reject
    return out


def make_metadata(sub_id, ses_id, opt_local):
    """
    Create metadata (i.e. pandas dataframes) to be added to the epochs
    """
    meta_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_metadata.csv"

    if not os.path.exists(meta_fname):
        # initialize
        order_list, trl_id_list = [], []

        df = pd.DataFrame()

        # add info about the order of the trajectories from the mat file created during the experiment and
        # unique trial identifier, to be used to match with eye tracker edf file
        if ses_id != 'meg':
            mat = load_mat(
                f"{opt_local['bidsPath']}beh/{sub_id.split('-')[-1]}_{ses_id[0]}/{sub_id.split('-')[-1]}_{ses_id[0]}_exp.mat")
        else:
            mat = load_mat(f"{opt_local['bidsPath']}beh/{sub_id.split('-')[-1]}/{sub_id.split('-')[-1]}_exp.mat")

        for k in mat['stim'].keys():
            if k.startswith('block_'):
                if (sub_id == 'sub-19961224AAMN' and ses_id == 'dots' and k.startswith('block_6')) or \
                    (sub_id == 'sub-19940921BRFL' and ses_id == 'lines' and k.startswith('block_1')) or \
                    (sub_id == 'sub-19960711EEBN' and ses_id == 'lines' and k.startswith('block_3')) or \
                    (sub_id == 'sub-19960711EEBN' and ses_id == 'lines' and k.startswith('block_4')) or \
                        (sub_id == 'sub-20010614DMCA' and ses_id == 'dots' and k.startswith('block_1')):
                    # technical issues
                    continue
                else:
                    for trl_id, trl_dict in mat['stim'][k].items():
                        order_list.append(trl_dict['order'])
                        trl_id_list.append(f"b{k.split('block_')[-1]}_t{trl_id.split('_')[-1]}")
        # add it to the dataframe
        df = df.assign(order=order_list, trl_id=trl_id_list)
        df['order'].apply(pd.Series)
        # df.set_index('trl_id', inplace=True)
        # save
        df.to_csv(meta_fname)
    else:  # load it
        df = pd.read_csv(meta_fname, index_col=0, header=[0, 1])

    return df


def assert_this(condition, message):
    """
    Assert condition and save output message to log
    :param condition: bool
    :param message: str
    """
    try:
        assert condition
    except:
        logging.getLogger('mne').error(message)
        raise AssertionError(message)


def euclidean_distance(q, p):
    """
    euclidean distance in 2d
    https://en.wikipedia.org/wiki/Euclidean_distance#Two_dimensions
    """
    q = np.asarray(q)
    p = np.asarray(p)
    return np.sqrt(np.sum((q - p) ** 2))


def raincloud(df, x, y, ax, palette, order, orientation='v', jitter=.2, outliers=None):
    """
    raincloud plot using seaborn
    """
    # violin plot
    ax = sns.violinplot(data=df, x=x, y=y, axes=ax, palette=palette, order=order, inner=None, orient=orientation, cut=0,
                        saturation=.85)
    # remove half of the violin
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        if orientation == 'h':
            y0 -= .01
            height /= 2
        elif orientation == 'v':
            x0 -= .01
            width /= 2
        violin.set_clip_path(plt.Rectangle((x0, y0), width, height, transform=ax.transData))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # box plot
    sns.boxplot(data=df, x=x, y=y, dodge=True, ax=ax, showfliers=False, saturation=1, color='k', width=0.2,
                boxprops={'zorder': 3, 'facecolor': 'none'}, linewidth=1, order=order)
    # define the number of collections before adding the dots
    old_len_collections = len(ax.collections)
    # plot individual data points as dots
    if outliers is None:
        sns.stripplot(data=df, x=x, y=y, jitter=jitter, dodge=False, size=4, ax=ax, palette=palette, alpha=.8,
                      edgecolor='k', linewidth=1, legend=False, order=order)
    else:
        # remove outliers from the dataframe
        df_out = df.loc[outliers]
        df_clean = df.loc[~outliers]
        # plot separately outliers as diamonds and the rest as dots
        sns.stripplot(data=df_out, x=x, y=y, jitter=jitter, dodge=False, size=4, ax=ax, palette=palette, alpha=.8,
                      edgecolor='k', linewidth=1, legend=False, order=order, marker='d')
        sns.stripplot(data=df_clean, x=x, y=y, jitter=jitter, dodge=False, size=4, ax=ax, palette=palette, alpha=.8,
                      edgecolor='k', linewidth=1, legend=False, order=order)
    # move
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([jitter, 0]))
    ax.set_xlim(xlim)
    return ax

    