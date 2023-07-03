 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  12 10:03:13 2021

@author: giulianogiari

========================================================================================================================
# mark bad channels in raw continuous data using mne-python visualization tool.
# this outputs a text file, meant to be used with maxfilter in the meg server and updates the bids dataset
# this script is meant to be called as a function

# options
# $1 subject_id
# $2 session_id
# e.g. /Volumes/ProjectData/MEGRID/gft2/prg/analysis/util_markBadChannels.py 19870707JNWL dots
========================================================================================================================

"""

import os
import sys
import glob
import getpass
import warnings
import mne_bids
import numpy as np
from datetime import datetime
from gft2_config import opt


# ----------------------------------------  define the main functions ------------------------------------------------ #
def check_badfile(bids_path):
    """ 
    Check if the bad channel text file already exists. If so load it 
    """
    # check if the empty room text file already exists
    bad_filename = f"{opt['badPath']}{bids_path.basename}_bads.txt"
    if os.path.exists(bad_filename):
        # if so we load it
        lst_bad = np.loadtxt(fname=bad_filename, dtype='bytes').astype(str).tolist()
        # it can happen that we have only one channel selected, this will be rendered as a string causing an error
        # if this is the case we make a list of this
        if isinstance(lst_bad, str):
            lst_bad = [lst_bad]
        return lst_bad
    else:
        return []


def markBadChannels(bids_path, butterflyFlag=False, lst_bad=[]):
    """
    1) check if the channels have already been marked. if so ask if we want to repeat the procedure
    2) load the data, interactive plot to mark the channels. then save a text file and mark them as bad also in bids
    """

    # 1)
    # compose the data filename
    bad_filename = f"{opt['badPath']}{bids_path.basename}_bads.txt"
    # check whether this already exists
    if not os.path.exists(bad_filename):
        flag = 'y'
    else:
        flag = input(f"Do you want to check {bids_path.basename} channels again? [y/n]")

    #2)
    if flag == 'y':
        # load data in bids
        # now bad channels are already marked
        raw = mne_bids.read_raw_bids(bids_path=bids_path, extra_params={'preload': True})

        # apply filter and downsample only for plotting
        raw.pick_types(meg=True, exclude=[]).filter(hp_plt, lp_plt).resample(sfreq_plt)

        # overwrite bad channels --> this should go after picking
        raw.info['bads'] = lst_bad

        # plot the data
        raw.plot(duration=10, n_channels=21, block=True, remove_dc=True, butterfly=butterflyFlag,
                 bad_color='r', group_by='position')

        # mark them as bad in bids
        mne_bids.mark_channels(bids_path=bids_path, ch_names=lst_bad, status='bad',
                               descriptions=['visual inspection' for _ in lst_bad])

        # and save the bad channels as text file
        # open it
        txt = open(bad_filename, 'w+')

        # create a header string and write to file
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        string = \
        '''# %s inspected by %s on %s
        # visualization settings: 
        # sampling rate: %i Hz
        # low pass freq: %i Hz
        # high pass freq: %f Hz
        # n. bad channels: %i \n''' % (bids_path.basename, getpass.getuser(), now, sfreq_plt,
                                       lp_plt, hp_plt, len(raw.info['bads']))
        txt.writelines(string)

        # write the bad channels
        txt.writelines(["%s \n" % item for item in raw.info['bads']])

        # check how many we selected
        if len(raw.info['bads']) > 12:
            warnings.warn(
                'you selected more than 12 bad channels; for maxfilter to work properly these should be =< 12')
            string = '''######################## warning #####################
            # you selected more than 12 bad channels; 
            # for maxfilter to work properly these should be =< 12
            ######################## warning #####################'''
            txt.writelines(string)

        # close the text file and finish
        txt.close()


if __name__ == "__main__":
    # ------------------------------------------------  define variables ------------------------------------------------- #
    lp_plt = 40
    hp_plt = .1
    sfreq_plt = 500

    # collect input
    sub_id = sys.argv[1]
    ses_id = sys.argv[2]

    butterflyFlag = bool(int(input(f"Which visualization? [1: butterfly / 0: all channels]")))

    # ------------------------------------------ subject-specific empty room --------------------------------------------- #
    # here we process the empty room first, to identify bad channels without the participant in the meg
    empty_path = mne_bids.BIDSPath(subject=sub_id, session=ses_id, task='noise', run=None, root=opt['bidsPath'])
    # check if the bad channel file exist and load it
    lst_bad = check_badfile(empty_path)
    # mark bad channels and create/update bad text file
    markBadChannels(empty_path, butterflyFlag, lst_bad)

    # --------------------------------------------------- task data ------------------------------------------------------ #
    # now we repeat for the task data
    task_fname_list = glob.glob(f"{opt['bidsPath']}sub-{sub_id}/ses-{ses_id}/meg/*run*_meg.fif")
    for run_id in range(1, len(task_fname_list)+1):
        # create the bids path
        if len(str(run_id)) == 1: run_id = f"0{run_id}"
        bids_path = mne_bids.BIDSPath(subject=sub_id, session=ses_id, task='task',
                                      run=run_id, root=opt['bidsPath'])
        # check if the bad channel text file exists
        lst_bad = check_badfile(bids_path)
        # if it doesnt we load the empty room one
        if not lst_bad:
            lst_bad = check_badfile(empty_path)
        # we now pass as input also the list of bad channels identified in the empty room
        markBadChannels(bids_path, butterflyFlag, lst_bad)

    # compute the reference run for maxmove realignment
    # this script was made by meg technicians and is hosted in the local server
