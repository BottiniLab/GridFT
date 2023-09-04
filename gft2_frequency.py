"""
functions for frequency analysis

@author: giuliano giari, giuliano.giari@gmail.com
"""

import glob
import h5py
import logging
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
from gft2_preprocessing import make_segments_epochs, read_epochs
from gft2_utils import nearest, realign_to_trj, assert_this
from h5io import read_hdf5
from joblib import Parallel, delayed
from mne.io.meas_info import Info
from mne.time_frequency import write_tfrs
from mne.utils.numerics import _freq_mask
from scipy.signal import spectrogram
from tqdm import tqdm


def pick_frequency(freqs_, foi):
    """
    get frequency index from frequency array
    """
    foi_ind = nearest(freqs_, foi)
    assert_this(np.isclose(freqs_[foi_ind], foi, atol=np.diff(freqs_)[0]), 'no frequency matches the selection')
    return foi_ind


def compute_fft_epochs(sub_id, ses_id, opt_local):
    """
    Compute FFT on Epochs/Evoked objects returning a custom TFR object
    """
    # initialize logging
    mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                     output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)

    for task in opt_local['frq_do']:
        # define file names
        frq_fname = f"{opt_local['frqPath']}{sub_id}_ses-{ses_id}_task-{task}_desc-" \
                    f"{opt_local['frq_avg_trl']}-{opt_local['frq_out_fft']}_frq.h5"

        if not os.path.exists(frq_fname):
            # check if the segmented epochs exist
            if not len(glob.glob(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_desc-*-seg-epo.fif.gz")) == 2:
                # load data
                epochs = read_epochs(sub_id, ses_id, opt_local)
                # remove baseline
                epochs.crop(0, opt_local['tmax'])
            # for each event type, i.e. angular resolution
            out_list = list()
            for ang_res in opt_local['ang_res']:
                # get this event data
                if os.path.exists(
                        f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_desc-{ang_res}-seg-epo.fif.gz"):
                    epochs_event = read_epochs(sub_id, ses_id, opt_local, segments=True, ang_res=ang_res, reject=opt_local['reject_trls'])
                elif 'epochs' in locals():
                    epochs_event = make_segments_epochs(sub_id, ses_id, 'task', ang_res, opt_local, save=False, reject=opt_local['reject_trls'],
                                                        epochs_event=realign_to_trj(epochs[f"ang_res_{ang_res}"],
                                                                                    opt_local['starting_trj'], opt_local
                                                                                    ))
                epochs_event.pick_types(meg=True)
                logging.getLogger('mne').info(f'computing FFT for event {ang_res}°')
                # compute fft on the single trials
                logging.getLogger('mne').info('computing FFT on individual trials')
                spctrm = np.concatenate([compute_fft_array(trial, epochs_event.info['sfreq'], 
                                                           opt_local['frq_n_seg'],
                                                           opt_local.copy(),
                                        average=False, verbose=False)[0]
                                        for trial in epochs_event.get_data()]).squeeze()
                events = np.tile(epochs_event.events[0, :], spctrm.shape[0]).reshape(spctrm.shape[0], 3)
                # check if the computation of the spectrum resulted in creating more segments
                if spctrm.shape[0] == len(epochs_event.events): trl_id = epochs_event.metadata['trl_id'].values
                else: trl_id = np.repeat(epochs_event.metadata['trl_id'].values,
                                         len(epochs_event.times) // opt_local['frq_n_seg'])
                metadata = pd.DataFrame({'events': events[0, -1], 'trl_id': trl_id})
                # get the frequencies in output
                freqs = compute_fft_array(epochs_event.get_data()[0, ...], epochs_event.info['sfreq'],
                                          opt_local['frq_n_seg'], opt_local.copy(),
                                          average=True, verbose=True)[1]
                # store each event_id in a separate mne object, given the different frequency resolutions
                out_list.append(Frequency(epochs_event.info,
                                          np.array(spctrm)[..., None], times=[0], freqs=freqs,
                                          metadata=metadata, events=events, event_id=epochs_event.event_id))
            # save
            logging.getLogger('mne').info(f"saving {frq_fname} ...")
            write_tfrs(frq_fname, out_list)


def compute_fft_array(x, sfreq, n_seg, opt_local, verbose=True):
    """
    Compute the frequency spectrum using FFT
    input data should be a 2D array of channels/vertices/voxels x time
    """
    assert_this(x.ndim == 2, f"time series data have {x.ndim} dimensions")

    if verbose: logging.getLogger('mne').info('Computing the frequency spectrum...')

    # check if the time series should be split and calculate the samples per segment
    if isinstance(n_seg, int):
        n_per_seg = int(x.shape[-1] / n_seg)
    elif n_seg == 'half':
        n_per_seg = x.shape[-1] // 2
    n_fft = n_per_seg

    if verbose: logging.getLogger('mne').info(f"Window size {n_per_seg} samples")

    # compute frequency spectrum using welch method
    # see https://mne.tools/dev/auto_tutorials/time-freq/plot_ssvep.html#sphx-glr-auto-tutorials-time-freq-plot-ssvep-py
    # here we use directly the spectrogram function from scipy which enables to get complex data as output
    freqs_, center_times, spctrm_ = spectrogram(x, nfft=n_fft, fs=sfreq, noverlap=opt_local['seg_overlap'],
                                                nperseg=n_per_seg, window=opt_local['frq_taper'],
                                                mode='complex', detrend='constant')
    # check the number of segments matches what is expected
    assert_this(len(center_times) == x.shape[-1] // n_per_seg, 'different number of segments than expected')
    # make sure the segments length match the expected duration
    assert_this(all([(t * sfreq + n_fft / 2) - (t * sfreq - n_fft / 2) == n_per_seg for t in center_times]),
                'returned segments of different duration than expected')
    # the output frequencies match the expected, here approximated to the integers
    assert_this(all([np.abs(freqs_[pick_frequency(freqs_, foi)] - foi) < 1e-10 for foi in np.arange(.5, 10, .5)]),
                'output frequencies do not match the expected ones')
    # now reorganize the data to output a matrix of n_segments x n_channels x n_freqs    
    seg_ind = int(np.where([shape == len(center_times) for shape in spctrm_.shape])[0])
    ch_ind = int(np.where([shape == x.shape[0] for shape in spctrm_.shape])[0])
    frq_ind = int(np.where([shape == len(freqs_) for shape in spctrm_.shape])[0])
    spctrm_ = spctrm_.transpose(seg_ind, ch_ind, frq_ind)
    # define the logical mask for output frequencies
    if any(opt_local['frq_foi_lims']):
        foi_mask = np.logical_and(freqs_ >= (opt_local['frq_foi_lims'][0] if opt_local['frq_foi_lims'][0] else min(freqs_)),
                                  freqs_ <= (opt_local['frq_foi_lims'][1] if opt_local['frq_foi_lims'][1] else max(freqs_)))
        freqs_ = freqs_[foi_mask]
        spctrm_ = spctrm_[..., foi_mask]
    # organize output
    if verbose: logging.getLogger('mne').info(f"returning {opt_local['frq_out_fft']}")
    return spctrm_, freqs_
    

class Frequency(mne.time_frequency.EpochsTFR):
    """
    Create a custom frequency class
    we use the inheritance property of python classes to leverage the
    mne epochsTFR class by faking 1 time point. 
    plus we add some functionalities, such as cropping and picking frequencies
    """

    def crop_freq(self, fmin=None, fmax=None):
        """
        cut frequency interval
        adapted from https://github.com/mne-tools/mne-python/blob/maint/0.23/mne/epochs.py#L1603
        """
        if fmin is None:
            fmin = min(self.freqs)

        if fmax is None:
            fmax = max(self.freqs)

        fmask = _freq_mask(self.freqs, sfreq=self.info['sfreq'], fmin=fmin, fmax=fmax)
        self.freqs = self.freqs[fmask]
        self._data = self._data[:, :, fmask, :]

        return self

    def get_frequency_index(self, freq):
        return nearest(self.freqs, freq)

    def pick_frequency(self, freq=None, index=None):
        """ return only the specified frequency. operates in place """
        if freq is not None:
            if isinstance(freq, (int, float)):
                freq = [freq]
            index = np.array([self.get_frequency_index(x) for x in freq])
        return Frequency(self.info,
                         self.data.squeeze()[None, :, index, None],
                         freqs=[self.freqs[index]], times=self.times,
                         events=self.events, event_id=self.event_id)


def read_frq(fname):
    """
    Read Frequency datasets from hdf5 file
    adapted from https://github.com/mne-tools/mne-python/blob/maint/1.0/mne/time_frequency/tfr.py#L2532
    """
    from mne.utils import _prepare_read_metadata
    tfr_data = read_hdf5(fname, title='mnepython', slash='replace')
    for k, tfr in tfr_data:
        tfr['info'] = Info(tfr['info'])
        tfr['info']._check_consistency()
        if 'metadata' in tfr:
            tfr['metadata'] = _prepare_read_metadata(tfr['metadata'])

    inst = Frequency
    out = [inst(**d) for d in list(zip(*tfr_data))[1]]
    return out


def compute_coh_array(spctrm_, freqs_, opt_local, verbose=True):
    """
    Compute phase coherence from complex data
    """
    if verbose: logging.getLogger('mne').info('Computing Inter-Trial Phase Coherence...')
    # check that the data have 3 dimensions and that the last corresponds to frequency
    if spctrm_.ndim > 3: spctrm_ = spctrm_.squeeze()
    assert_this(spctrm_.dtype == 'complex', 'complex data type is needed to compute ITPC')
    assert_this(spctrm_.ndim == 3, f"fourier spectrum has {spctrm_.ndim} dimensions")
    assert_this(spctrm_.shape[-1] == len(freqs_), 'last dimension does not correspond to frequencies')

    if verbose: logging.getLogger('mne').info(f"using {opt_local['coh_method']}")
    # https://link.springer.com/article/10.1007%2Fs10827-012-0424-6
    return (np.mean(np.cos(_phase(spctrm_)), 0) ** 2) + (np.mean(np.sin(_phase(spctrm_)), 0) ** 2)


def compute_coh_epochs(sub_id, ses_id, opt_local):
    """
    Compute phase coherence across trials segments on epochs objects
    """
    # initialize logging
    mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                     output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)

    for task in opt_local['coh_do']:
        # fft name
        frq_fname = f"{opt_local['frqPath']}{sub_id}_ses-{ses_id}_task-{task}_desc-" \
                    f"{opt_local['frq_avg_trl']}-{opt_local['frq_out_fft']}_frq.h5"
        coh_fname = f"{opt_local['frqPath']}{sub_id}_ses-{ses_id}_task-{task}_desc-" \
                    f"{opt_local['coh_method']}_coh_frq.h5"

        if not os.path.exists(coh_fname):
            # load data
            frq_list = read_frq(frq_fname)
            #
            out_list = list()
            for frq_event in frq_list:

                out = compute_coh_array(frq_event.data, frq_event.freqs, opt_local.copy())

                out_list.append(Frequency(frq_event.info, out[None, ..., None], freqs=frq_event.freqs,
                                          times=[0], events=[frq_event.events[0, -1]], event_id=frq_event.event_id))

            # save
            logging.getLogger('mne').info(f"saving {coh_fname} ...")
            write_tfrs(coh_fname, out_list)

