"""
Run the analysis pipeline in the server with parallel processing of the subjects

@author: giuliano giari, giuliano.giari@gmail.com
"""

import sys
import glob
from joblib import Parallel, delayed
from gft2_config import opt
from gft2_preprocessing import meg_preprocess, make_epochs, make_segments_epochs
from gft2_frequency import compute_coh_epochs, compute_fft_epochs
from gft2_stc import compute_fft_stc, compute_coh_stc
from gft2_src import make_src
from gft2_eye import eye_correlation, eye_preprocess, realign_to_dots, \
    compute_heatmaps, correlate_heatmaps, compare_angle_distributions

ses_id = sys.argv[1]

exp_id = 'exp_2' if ses_id == 'meg' else 'exp_1'
n_jobs = -1
for exp_id in [exp_id]:

    subj_list = [sub_id for sub_id in opt[exp_id]['subj_list'] if not sub_id in opt[exp_id]['to_exclude']]

    for ses_id in [ses_id]:
        
        if ses_id == 'anat':
            # create source space
            Parallel(n_jobs=n_jobs)(delayed(make_src)(sub_id, opt) for sub_id in subj_list)

        else:

            # preprocessing
            Parallel(n_jobs=n_jobs)(delayed(meg_preprocess)(sub_id, ses_id, opt.copy()) for sub_id in subj_list)
            
            # epoching
            Parallel(n_jobs=n_jobs)(delayed(make_epochs)(sub_id, ses_id, opt.copy()) for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(make_segments_epochs)(sub_id, ses_id, 'task', 15, opt.copy(),
                                                                  save=True) for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(make_segments_epochs)(sub_id, ses_id, 'task', 30, opt.copy(),
                                                                  save=True) for sub_id in subj_list)
            
            # eye tracker
            Parallel(n_jobs=n_jobs)(delayed(eye_preprocess)(sub_id, ses_id, opt.copy()) 
                for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(realign_to_dots)(sub_id, ses_id, opt.copy())
                                    for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(compute_heatmaps)(sub_id, ses_id, opt.copy())
                                    for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(correlate_heatmaps)(sub_id, ses_id, opt.copy())
                                    for sub_id in subj_list)
            if ses_id == 'dots':
                Parallel(n_jobs=n_jobs)(delayed(eye_correlation)(sub_id, ses_id, 'dist', opt.copy())
                                        for sub_id in subj_list)
            
            # frequency analysis
            Parallel(n_jobs=n_jobs)(delayed(compute_fft_epochs)(sub_id, ses_id, opt_local.copy()) for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(compute_coh_epochs)(sub_id, ses_id, opt_local.copy()) for sub_id in subj_list)
            
            # source analysis
            Parallel(n_jobs=n_jobs)(delayed(compute_fft_stc)(sub_id, ses_id, opt_local.copy()) for sub_id in subj_list)
            Parallel(n_jobs=n_jobs)(delayed(compute_coh_stc)(sub_id, ses_id, opt_local.copy()) for sub_id in subj_list)

