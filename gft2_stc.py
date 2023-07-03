"""
functions for source analysis
@author: giulianogiari
"""

import glob
import h5py
import logging
import mne
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from gft2_preprocessing import make_segments_epochs, read_epochs
from gft2_src import make_forward, make_morpher
from gft2_frequency import compute_fft_array, compute_coh_array
from gft2_utils import realign_to_trj, assert_this


def compute_inverse(info_, fwd_, data_cov_, noise_cov_, rank_, opt_local):
    """
    Compute inverse solution depending on source method
    """
    # compute beamformer weights
    inverse_operator_ = make_lcmv(info_, fwd_, data_cov_, reg=opt_local['stc_lambda'],
                                  noise_cov=noise_cov_, pick_ori=opt_local['stc_out'], reduce_rank=True,
                                  rank=rank_)
    return inverse_operator_


def compute_noise_covariance(sub_id, ses_id, opt_local):
    """
    Compute noise (and data, if needed) covariance
    """
    # load data
    noise_fname = f"{opt_local['prePath']}{sub_id}_ses-{ses_id}_task-noise_raw.fif.gz"
    noise_ = mne.io.read_raw_fif(noise_fname, preload=True).pick_types(meg=True, eog=False)

    # define rank of the covariance matrix
    logging.getLogger('mne').info('reducing the rank of the covariance matrix before source analysis')
    rank_ = mne.compute_rank(noise_, rank='info')

    # compute covariance
    noise_cov_ = mne.compute_raw_covariance(noise_, method=opt_local['stc_cov_method'], rank=rank_)
    return noise_cov_, rank_


def compute_fft_stc(sub_id, ses_id, opt_local):
    """
    Compute FFT on stc objects 
    """

    if not len(glob.glob(f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_"
                         f"{opt_local['stc_method']}_{opt_local['frq_avg_trl']}_*_fft_{opt_local['src_type']}.h5")) == 2:

        mne.set_log_level('info')
        mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                         output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)

        # check if the segmented epochs exist
        if not len(glob.glob(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_desc-*-seg-epo.fif.gz")) == 2:
            # load data
            epochs = read_epochs(sub_id, ses_id, opt_local, 'task')
            # remove baseline
            epochs.crop(0, opt_local['tmax'])

        # make or load forward solution
        fwd = make_forward(sub_id, ses_id, opt_local)
        make_morpher(sub_id, ses_id, opt_local)

        # compute noise covariance
        logging.getLogger('mne').info(f"using the {opt_local['stc_cov_data']} data for noise covariance computation")
        noise_cov, rank = compute_noise_covariance(sub_id, ses_id, opt_local)

        for ang_res in opt_local['ang_res']:

            if os.path.exists(f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_"
                              f"{opt_local['stc_method']}_{opt_local['frq_avg_trl']}_{ang_res}_fft_{opt_local['src_type']}.h5"):
                logging.getLogger('mne').info(f"{ang_res}° already processed")
                continue

            # get this event data
            if os.path.exists(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_desc-{ang_res}-seg-epo.fif.gz"):
                epochs_event = read_epochs(sub_id, ses_id, opt_local, segments=True, ang_res=ang_res, reject=True)
            elif 'epochs' in locals():
                epochs_event = make_segments_epochs(sub_id, ses_id, 'task', ang_res, opt_local, save=False,
                                                    epochs_event=realign_to_trj(epochs[f"ang_res_{ang_res}"],
                                                                                opt_local['starting_trj'], opt_local
                                                                                ))
            epochs_event.pick_types(meg=True)

            # compute data covariance
            data_cov = mne.compute_covariance(epochs_event.copy().apply_baseline((None, None)),
                                              method=opt_local['stc_cov_method'], rank=rank)
        
            # compute inverse solution
            inverse_operator = compute_inverse(epochs_event.info, fwd, data_cov, noise_cov, rank, opt_local)

            logging.getLogger('mne').info('computing source reconstruction')
            # and apply the spatial filter to them to reconstruct the time series as source level. this returns a
            # generator object that is easier to deal with given the huge file size of these data
            stc_time = apply_lcmv_epochs(epochs_event, inverse_operator, return_generator=True)
            # hack to get output thingies
            trl = apply_lcmv_epochs(epochs_event[0], inverse_operator, return_generator=False)[0]

            # compute the fft
            def _parallelize_fft(trl, opt_local):
                trl_spctrm = compute_fft_array(trl.data, trl.sfreq, opt_local['frq_n_seg'],
                    opt_local.copy(), average=False, verbose=True)[0]
                return trl_spctrm
                
            out = Parallel(n_jobs=opt_local['stc_n_jobs'])(delayed(_parallelize_fft)(trl, opt_local.copy())
                                                           for trl in stc_time)
            # prepare spctrm data for the output
            spctrm = np.stack(out, 0)
            # get output frequencies
            freqs = compute_fft_array(trl.data[0, ...] if trl.data.ndim == 3 else trl.data, trl.sfreq,
                                      opt_local['frq_n_seg'], opt_local.copy(), average=True, verbose=True)[1]
            # put the results in a h5 container and save
            stc_fname = f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_" \
                        f"{opt_local['stc_method']}_{opt_local['frq_avg_trl']}_{ang_res}_fft_" \
                        f"{opt_local['src_type']}.h5"
            write_stc(stc_fname, spctrm.squeeze(), sub_id, trl.vertices, freqs[0], np.diff(freqs)[0])
            

def write_stc(stc_fname, spctrm, sub_id, vertices, freqs_min, delta_freq):
    """
    Write stc data to stc_fname h5
    :param stc_fname:
    :param spctrm:
    :param sub_id:
    :param vertices:
    :param freqs_min:
    :param delta_freq:
    :return:
    """
    logging.getLogger('mne').info(f"saving {stc_fname} ...")
    with h5py.File(stc_fname, 'w') as f:
        for k, v in {'data': np.squeeze(spctrm),
                     'subject': sub_id,
                     'vertices': vertices,
                     'tmin': freqs_min,
                     'tstep': delta_freq}.items():
            if k == 'subject' or k.startswith('t'):
                f.create_dataset(name=k, data=v)
            else: f.create_dataset(name=k, data=v, compression='gzip', compression_opts=9)


def read_stc(stc_fname, return_object=True):
    """
    read stc h5 dataset
    adapted from https://github.com/mne-tools/mne-python/blob/maint/1.0/mne/time_frequency/tfr.py#L2532
    :param stc_fname:
    :return:
    """
    logging.getLogger('mne').info(f"loading {stc_fname} ...")
    # determine the data type
    if 'vector' in stc_fname and not 'snr' in stc_fname and not 'coh' in stc_fname:
        if 'vol' in stc_fname: inst = mne.VolVectorSourceEstimate
        else: inst = mne.VectorSourceEstimate
    else:
        if 'vol' in stc_fname: inst = mne.VolSourceEstimate
        else: inst = mne.SourceEstimate
    
    with h5py.File(stc_fname, 'r') as f:
        out = dict()
        for k, v in f.items():
            if k == 'vertices': out[k] = list(np.array(f.get(k, v)))
            # https://groups.google.com/g/h5py/c/42oh2kyXVGs
            elif k == 'subject': out[k] = f[k].asstr()[()]
            else: out[k] = np.array(f.get(k, v))
    
    if return_object:
        return inst(**out)
    else:
        # from https://github.com/mne-tools/mne-python/blob/maint/1.0/mne/source_estimate.py#L818
        out['times'] = out['tmin'] + (out['tstep'] * np.arange(out['data'].shape[-1]))
        return out, inst


def compute_coh_stc(sub_id, ses_id, opt_local):
    """
    Compute coherence at source level
    """

    for ang_res in opt_local['ang_res']:

        if not glob.glob(f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_"
                         f"{opt_local['stc_method']}_{ang_res}_coh_{opt_local['src_type']}.h5"):

            mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-{ses_id}/{sub_id}_ses-{ses_id}_log.log",
                             output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)


            # get the data of this angular resolution
            fft_fname = f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_" \
                        f"{opt_local['stc_method']}_trl_{ang_res}_fft_{opt_local['src_type']}.h5"
            stc_dict, stc_type = read_stc(fft_fname, return_object=False)

            logging.getLogger('mne').info('computing coherence at source level...')
            
            coh = compute_coh_array(stc_dict['data'], stc_dict['times'], opt_local.copy())

            # save output
            coh_fname = f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_" \
                        f"{opt_local['stc_method']}_{ang_res}_coh_{opt_local['src_type']}.h5"
            write_stc(coh_fname, coh, sub_id, stc_dict['vertices'], stc_dict['tmin'], stc_dict['tstep'])

