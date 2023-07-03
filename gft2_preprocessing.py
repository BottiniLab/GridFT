"""
functions for preprocessing
@author: giulianogiari
"""

import glob
import h5py
import logging
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import warnings
from gft2_utils import make_metadata, assert_this, split_segments_array, realign_to_trj
from scipy.stats import kurtosis

# to avoid figures to pop up
matplotlib.use('Agg')


def process_diode(raw_, sub_id, ses_id, i_run, opt_local):
    """
    Correct event timing using photodiode
    """
    # get the photodiode data
    diode_data = raw_.get_data(picks=raw_.ch_names.index('MISC008')).squeeze()
    # low pass at 40 hz, to account for high frequency noise
    diode_data = mne.filter.filter_data(diode_data, raw_.info['sfreq'], l_freq=None, h_freq=15)
    # initialize samples indices
    samples = np.arange(0, len(diode_data))
    # diode goes from gray to black, then continues white/black for the trajectories
    # here we invert the values so that the first change is positive
    diode_data = np.abs(diode_data - diode_data.max())
    # plt.plot(diode_data)
    # subject specific "baseline" noise, to account for diodes displacement
    diode_min, diode_max = np.amin(diode_data[10:500]), np.amax(diode_data[10:500])
    # create a strict binary mask: 1 == stimulus present
    mask = diode_data >= diode_max
    # find out when it changes to get the onset
    onsets = np.nonzero(np.diff(mask) > 0)[0]
    # find events from triggers
    events_ = mne.find_events(raw_, stim_channel='STI101', min_duration=.005, shortest_event=1, output='onset')
    """
    plt.figure()
    plt.plot(mask)
    plt.plot(diode_data)
    plt.plot(onsets, [diode_max] * len(onsets), '.r')
    plt.vlines(events_[:, 0]-raw_.first_samp, 0 , 1, 'k', lw=2)
    """
    # keep only the events coded also using the diode (here we exclude the rest period start since it is not task)
    start_trg = mne.pick_events(events_, exclude=[253, 254, 199, 99, 150])
    # initialize the output
    start_correct = start_trg.copy()
    trj_list = []
    # initialize the figure
    fig, ax = plt.subplots(len(start_trg), 1, dpi=150, sharey=True, figsize=(15, 8))
    # then correct each event
    for i_event, start_event in enumerate(start_trg):
        # 1) find trial start
        # get the onset of the trial according to meg trigger
        meg_trg_start = start_event[0] - raw_.first_samp
        # define trial samples
        trl_ind = np.arange(meg_trg_start - 100, meg_trg_start + opt_local['tmax'] * 1000)
        # now find the onsets that are:
        # - after the start according to the meg trigger
        # - within the tolerance value (in samples)
        start_ind = np.nonzero(np.logical_and(onsets - meg_trg_start >= 0,
                                              (onsets - meg_trg_start) <= opt_local['diode_tolerance']))[0]

        if sub_id == 'sub-19981201VRII' and i_event == 7 and i_run == '06' and ses_id == 'lines':
            # something weird happened in this trial, we take 167 ms before the onset of the first trajectory
            start_correct[i_event, 0] = onsets[np.isclose(onsets, meg_trg_start, atol=200).nonzero()[0].max()] - \
                                        opt_local['baserate_ms']
        elif sub_id == 'sub-19940921BRFL' and any([i_run == '02', i_run == '06']) and ses_id == 'lines':
            # something weird happened in this trial, we take 167 ms before the onset of the first trajectory
            start_correct[i_event, 0] = onsets[np.isclose(onsets, meg_trg_start, atol=200).nonzero()[0].max()] - \
                                        opt_local['baserate_ms']
        elif sub_id == 'sub-19960711EEBN' and i_event == 0 and i_run == '06' and ses_id == 'lines':
            # something weird happened in this trial, we take 167 ms before the onset of the first trajectory
            start_correct[i_event, 0] = onsets[np.isclose(onsets, meg_trg_start, atol=200).nonzero()[0].max()] - opt_local['baserate_ms']
        # this participant diode is noisy
        elif sub_id == 'sub-20011101DBBU' and any([i_run == '02' and i_event == 0,  i_run == '03' and i_event == 6,
                                                   i_run == '04' and i_event == 0, i_run == '04' and i_event == 1,
                                                   i_run == '05' and i_event == 0, i_run == '04' and i_event == 3]):
            start_correct[i_event, 0] = onsets[max(np.isclose(onsets, meg_trg_start, atol=160).nonzero()[0]) + 1] - opt_local['baserate_ms']
        else:  # take the minimum of these onsets
            start_correct[i_event, 0] = onsets[min(start_ind)]
        # plot
        ax[i_event].plot(samples[trl_ind], diode_data[trl_ind])
        ax[i_event].vlines(samples[meg_trg_start], diode_data.min(), diode_data.max(), 'r', label='meg trigger')
        ax[i_event].vlines(samples[start_correct[i_event, 0]], diode_data.min(), diode_data.max(), 'g', label='corrected')
        ax[i_event].set_xlim(meg_trg_start - 50, meg_trg_start + 500)
    # add back the first sample
    start_correct[:, 0] += raw_.first_samp
    # display corrected timings
    delta_t = (start_correct[:, 0] - start_trg[:, 0]) / raw_.info['sfreq']
    logging.getLogger('mne').info(f"Photodiode correction: +{np.mean(delta_t):2.2f} ± {np.std(delta_t):2.2f} s")
    plt.suptitle(f"photodiode correction: +{np.mean(delta_t):2.2f}±{np.std(delta_t):2.2f} s")
    ax[0].legend()
    ax[-1].set_xlabel('Time (samples)')
    plt.setp(ax, xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_run-{i_run}_diode.png", dpi=300)
    # final checks
    assert_this(len(start_correct) == opt_local['n_trls'] // opt_local['n_blocks'], \
                'different number of photodiode events than expected')
    # return the corrected timings
    return start_correct, trj_list


def make_epochs(sub_id, ses_id, opt_local):
    """
    Cut the continuous time series into epochs based on timings corrected with photodiode
    """
    # initialize logging
    mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                     output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)

    # define file names
    task_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_epo.fif.gz"

    if not os.path.exists(task_fname):

        # get the files of this participant that we want to process
        raw_fname = sorted(glob.glob(f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-{ses_id}_raw.fif.gz"))[0]

        # load preprocessed and concatenated data
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        # get the metadata
        metadata = make_metadata(sub_id, ses_id, opt_local.copy())

        # create epochs arguments
        epochs_kwargs = {'baseline': None, 'decim': 1,  # decim==1 corresponds to no-downsampling
                         'preload': False, 'proj': False, 'detrend': None, 'reject': None}

        # load the events
        events = mne.read_events(f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-{ses_id}_eve.fif")
        task_events = mne.pick_events(events, exclude=[150])

        # create epochs of the task
        epochs = mne.Epochs(raw, tmin=opt_local['tmin'], tmax=opt_local['tmax'], events=task_events,
                            metadata=metadata, reject_by_annotation=False, **epochs_kwargs,
                            event_id={lab: trg for lab, trg in opt_local['event_id'].items() if trg in task_events[:, 2]})
        epochs.save(task_fname, overwrite=True)


def meg_preprocess(sub_id, ses_id, opt_local):
    """
    Apply filters and extract timings. Then concatenate and save the data
    sub_id: string, subject identifier
    ses_id: string, session identifier
    opt_local: dictionary, options as found in gft2_config
    """
    # initialize logging
    mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                     output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)

    # define file names
    task_fname = f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-{ses_id}_raw.fif.gz"
    event_fname = f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-{ses_id}_eve.fif" # these will store the corrected timings
    noise_fname = f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-noise_raw.fif.gz"

    # get the files of this participant that we want to process
    # here take into account those participants with which we had technical issues
    raw_fnames_list = sorted(glob.glob(f"{opt_local['maxPath']}/{sub_id}_ses-{ses_id}*_proc-tsss_meg.fif"))
    if sub_id == 'sub-19940921BRFL' and ses_id == 'lines' or sub_id == 'sub-20010614DMCA' and ses_id == 'dots':
        assert_this(len(raw_fnames_list) == 6, f"{sub_id}_ses-{ses_id} has {len(raw_fnames_list)} runs")
    elif sub_id == 'sub-19960711EEBN' and ses_id == 'lines':
        assert_this(len(raw_fnames_list) == 5, f"{sub_id}_ses-{ses_id} has {len(raw_fnames_list)} runs")
    else:
        assert_this(len(raw_fnames_list) == opt_local['n_blocks']+1,
                    f"{sub_id}_ses-{ses_id} has {len(raw_fnames_list)} runs")

    # remove run06 of this participants session as it is very noisy
    if sub_id == 'sub-19961224AAMN' and ses_id == 'dots':
        logging.getLogger('mne').warning('removing run-06 of sub-19961224AAMN_ses-dots')
        raw_fnames_list = [raw_fname for raw_fname in raw_fnames_list if "run-06" not in raw_fname]

    # main loop over blocks
    data_dict = {f'{k}_list': [] for k in ['events', 'raw', 'onsets']}
    for raw_fname in raw_fnames_list:
        # extract the run id and check if the file already exists
        task_name = raw_fname.split('task-')[1].split('_')[0]
        if task_name == 'noise':
            if os.path.exists(noise_fname):
                logging.getLogger('mne').info(f'{sub_id} {ses_id} empty data already preprocessed')
                continue
        else:
            if os.path.exists(task_fname):
                logging.getLogger('mne').info(f'{sub_id} {ses_id} task data already preprocessed')
                return

        # load max-filtered data
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        # extract and correct event timings and id
        if task_name != 'noise':
            # extract and correct event timings and id
            i_run = raw_fname.split('run-')[1][:2]
            # correct trial onset timings with recorded photodiode
            # the trigger recorded in the meg is usually earlier than the actual start of the trial
            events, trj_times = process_diode(raw, sub_id, ses_id, i_run, opt_local.copy())
            # save events run by run
            mne.write_events(f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-{ses_id}_run-{i_run}_eve.fif", events,
                             overwrite=True)
            # store the task data
            data_dict['events_list'].append(events)

        # keep meg, eog and misc (eye-tracker) channels
        # some participants have EOG062 in EEG064 due to machine problems, here we change this
        if 'EEG064' in raw.info['ch_names']:
            raw.set_channel_types({'EEG064': 'eog'})
            raw.rename_channels({'EEG064': 'EOG062'})
        raw.pick_types(meg=True, eog=True, eeg=False, stim=False, misc=True, ref_meg=False)

        # filter
        raw.filter(opt_local['hp'], opt_local['lp'], method='fir', filter_length='auto', phase='zero',
                   fir_window='hamming', fir_design='firwin')  # default parameters in mne 0.24

        # if task data store them in a list to later concatenate them. if empty save it directly
        if task_name == 'noise':
            raw.save(noise_fname, overwrite=True)
        else:
            data_dict['raw_list'].append(raw)

    # put all runs together and save them
    # data are now stored in chronological order, so that data.get_data()[0] corresponds to run1 trial1.
    raw, events = mne.concatenate_raws(data_dict['raw_list'], events_list=data_dict['events_list'])
    raw.save(task_fname, overwrite=True)
    mne.write_events(event_fname, events)

    # save also the timing of the corrected events
    logging.getLogger('mne').info(f"saving {opt_local['trjPath']}{sub_id}_ses-{ses_id}_timings.h5 ...")
    with h5py.File(f"{opt_local['trjPath']}{sub_id}_ses-{ses_id}_timings.h5", 'w') as hf:
        [hf.create_dataset(name=f"/run-0{i_run+1}/events", data=events) for i_run, events in enumerate(data_dict['events_list'])]


def segments_rejection(sub_id, ses_id, opt_local):
    """
    Compute metrics and save csv
    """
    # check if the files exist
    if not len(glob.glob(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_desc-*-segments_rejection.csv")) == 2:
        # check if the segmented epochs exist
        if not len(glob.glob(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_desc-*-seg-epo.fif.gz")) == 2:
            # load data
            epochs = read_epochs(sub_id, ses_id, opt_local, task='task', segments=False)
            # remove baseline
            epochs.crop(0, opt_local['tmax'])

        # for each angular resolution
        for ang_res in opt_local['ang_res']:
            print(f"processing {ang_res}° angular resolution")
            metrics_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_desc-{ang_res}-segments_rejection.csv"

            if not os.path.exists(metrics_fname):
                # -------------------------- load the segmented data or segment them now ----------------------------- #
                # from now on we will do the analyses only on the segments rather than individual trials
                if os.path.exists(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_desc-{ang_res}-seg-epo.fif.gz"):
                    epochs_event = read_epochs(sub_id, ses_id, opt_local, segments=True, ang_res=ang_res, reject=False)
                elif 'epochs' in locals():
                    epochs_event = make_segments_epochs(sub_id, ses_id, 'task', ang_res, opt_local, save=False,
                                                        epochs_event=realign_to_trj(epochs[f"ang_res_{ang_res}"],
                                                                                    opt_local['starting_trj'], opt_local
                                                                                    ))
                # initialize output dictionary
                out_dict = dict()
                out_dict['trl_id'] = list(epochs_event.metadata.trl_id.astype('string').values)
                out_dict['bad_segment'] = np.zeros(len(out_dict['trl_id']), dtype=bool)
                out_dict['bad_description'] = [[]] * len(epochs_event)
                out_dict['inspected'] = np.zeros(len(out_dict['trl_id']), dtype=bool)

                # ----------------------------- exclude trials based on eye tracker ---------------------------------- #
                # here we work with the original long trials and exclude all the segments that correspond to the trial
                # load the eye tracker data
                eye_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_percentage.csv"
                eye_csv = pd.read_csv(eye_fname, index_col=0)
                # keep trials of this angular resolution
                eye_csv = eye_csv.loc[eye_csv.ang_res == ang_res]
                # check if there is any trial that has been already excluded
                for trl_id in list(set(epochs_event.metadata.trl_id.astype('string').values) - set(eye_csv.trl_id)):
                    logging.getLogger('mne').info(f"setting trial {trl_id} as bad due to missing eye tracker data")
                    for seg_ind in np.nonzero(epochs_event.metadata.trl_id.values == trl_id)[0]:
                        out_dict['bad_segment'][seg_ind] = True
                        out_dict['inspected'][seg_ind] = True
                        out_dict['bad_description'][seg_ind] = 'missing eye-tracker'

                # identify outliers as those trials that are below the fixation threshold we set
                for i_row, row in eye_csv.iterrows():
                    if row.fixation_perc < opt_local['fixation_thresh']:
                        logging.getLogger('mne').info(f"setting trial {row.trl_id} "
                                                      f"as bad due to fixation time below threshold")
                        for seg_ind in np.nonzero(epochs_event.metadata.trl_id.values == row.trl_id)[0]:
                            out_dict['bad_segment'][seg_ind] = True
                            out_dict['inspected'][seg_ind] = True
                            out_dict['bad_description'][seg_ind] = 'fixation time below threshold'
                # define the channels to plot later for inspection
                picks = ['meg', 'eog', 'misc']
                data_for_metric = epochs_event.copy().pick_types(meg=True).get_data()
                group_by = 'position'
                
                # ------------------------------------- visual segment rejection ------------------------------------- #
                # compute metrics of the individual segments (variance/kurtosis over time), pick the outliers and plot
                # for further visual inspection
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                for i_metric, (metric, metric_id) in enumerate(zip([np.var(data_for_metric, axis=-1),
                                                                    kurtosis(data_for_metric, axis=-1)],
                                                                    ['variance', 'kurtosis'])):
                    # average across channels
                    metric = metric.mean(1)
                    # compute mean and sd
                    m = np.mean(metric)
                    sd = np.std(metric)
                    # plot
                    ax[i_metric].plot(np.logical_not(out_dict['bad_segment']).nonzero()[0],
                                      metric[np.logical_not(out_dict['bad_segment'])], '.b')
                    ax[i_metric].hlines(m+sd*2, ax[i_metric].get_xlim()[0], ax[i_metric].get_xlim()[1], ls='--', color='k')
                    ax[i_metric].set_title(metric_id)
                    # check if there is any outlier
                    outlier = metric > m + sd * 2
                    # if any is found, plot the time series for visual inspection
                    if any(outlier):
                        for seg_ind in outlier.nonzero()[0]:
                            # check if this segment has been already excluded or was already inspected
                            if not out_dict['inspected'][seg_ind]:
                                out_dict['inspected'][seg_ind] = True
                                # make this segment a separate object
                                epochs_trl = epochs_event[seg_ind].copy()
                                epochs_trl.plot(group_by=group_by, block=True, decim=4, title=metric_id,
                                                picks=picks)
                                # if during the plot we have marked this epoch as bad, the corresponding object will
                                # have length of zero. if this is the case we mark the epoch as bad in the csv
                                if len(epochs_trl) == 0:
                                    out_dict['bad_segment'][seg_ind] = True
                                    out_dict['bad_description'][seg_ind] = metric_id
                                    print(f"segment {seg_ind} marked as bad")
                plt.setp(ax, xlabel='segment number')
                fig.savefig(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_"
                            f"desc-{ang_res}_scatter.png", dpi=300)
                # save output as dataframe
                pd.DataFrame.from_dict(out_dict).to_csv(metrics_fname)


def make_segments_epochs(sub_id, ses_id, task, ang_res, opt_local, save=False, epochs_event=None):
    """ 
    split the data time series into segments and save as epochs object 
    """

    # initialize logging
    mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                     output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)

    epochs_split_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-{task}_desc-{ang_res}-seg-epo.fif.gz"

    if not os.path.exists(epochs_split_fname):
        if not epochs_event:
            # define file names
            epochs = read_epochs(sub_id, ses_id, opt_local, task='task')
            # remove baseline
            epochs.crop(0, None)
            # get this event data and realign
            epochs_event = realign_to_trj(epochs[f"ang_res_{ang_res}"], opt_local['starting_trj'], opt_local)

        logging.getLogger('mne').info(f'splitting data time series {ang_res}')

        # split the data time series
        data_array = split_segments_array(epochs_event.get_data(), epochs_event.times,
                                          epochs_event.__class__, noverlap=opt_local['seg_overlap'],
                                          seg_len=opt_local['seg_info'][str(ang_res)]['seg_len'],
                                          expected_seg=opt_local['seg_info'][str(ang_res)]['n_seg'])
        # create metadata
        metadata = pd.DataFrame.from_dict({'trl_id': np.repeat(epochs_event.metadata.trl_id.values.astype(str),
                                                               opt_local['seg_info'][str(ang_res)]['n_seg'])})
        # extract the event values
        event_value = np.stack([np.arange(0, data_array.shape[0]),
                                np.zeros(data_array.shape[0], dtype=int),
                                np.repeat(list(epochs_event.event_id.values()), data_array.shape[0])], 1)
        # create the epochs object and pass on the global rejection thresholds
        epochs_split = mne.EpochsArray(data_array, epochs_event.info, event_value, tmin=epochs_event.tmin,
                                       metadata=metadata,
                                       event_id=epochs_event.event_id, reject=None, baseline=None, proj=False)
        if save:
            logging.getLogger('mne').info(f'saving {epochs_split_fname}')
            epochs_split.save(epochs_split_fname, overwrite=True)

        return epochs_split
    else:
        warnings.warn(f'{sub_id} ses {ses_id} already segmented and saved. load the segmented data instead')


def read_epochs(sub_id, ses_id, opt_local, task='task', segments=False, ang_res=None, reject=True):
    """
    read epochs and (optionally) remove bad trials
    """

    if segments:
        assert ang_res is not None
        epochs_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-{task}_desc-{str(ang_res)}-seg-epo.fif.gz"
        epochs = mne.read_epochs(epochs_fname)
        # load the rejection csv and drop trials marked as bad
        reject_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_desc-{ang_res}-segments_rejection.csv"
        if os.path.exists(reject_fname):
            csv = pd.read_csv(reject_fname, index_col=0)
            assert all(csv.trl_id == epochs.metadata.trl_id)
            epochs.metadata = csv
        else:
            warnings.warn(f"{reject_fname} does not exist.")
            epochs.metadata = pd.concat([epochs.metadata,
                                         pd.DataFrame.from_dict({'bad_segment': np.zeros(len(epochs), dtype=bool)})],
                                        axis=1)
        if reject:
            epochs.drop(epochs.metadata.bad_segment.values)
    else:
        epochs_fname = f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-{task}_epo.fif.gz"
        epochs = mne.read_epochs(epochs_fname, preload=True)
        if task == 'task':
            metadata = make_metadata(sub_id, ses_id, opt_local)
            epochs.metadata = metadata

    return epochs

