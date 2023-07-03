"""
configuration settings for gridft2 study
@author: giulianogiari
"""

# ------------------------------------------------- initialization --------------------------------------------------- #
import os, sys
import matplotlib as mpl
from matplotlib import cm
import numpy as np

opt = dict()

# ------------------------------------------------------ paths ------------------------------------------------------- #
if sys.platform == 'linux':
    # we have to check whether we are on the cluster or the meg server
    if os.path.exists('/mnt/storage/tier2/robbot/ProjectData/MEGRID/'):
        # we are on the meg server
        rootpath = '/mnt/storage/tier2/robbot/ProjectData/MEGRID/'
        opt['fifPath'] = f'{rootpath}fif/'
        opt['maxPath'] = f'{rootpath}derivatives/max/'
        opt['bidsPath'] = rootpath
    else: # we are on the cluster
        rootpath = '/mnt/storage/tier2/ROBBOT-extra/Projects/MEGRID/'
        # we keep the raw data on a different storage space
        opt['fifPath'] = '/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/fif/'
        opt['maxPath'] = '/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/derivatives/max/'
        opt['fsPath']  = '/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/freesurfer/'
        opt['mriPath'] = '/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/mri/'
        opt['bidsPath'] = '/mnt/storage/tier2/ROBBOT/ProjectData/MEGRID/'
elif sys.platform == 'darwin':
    # we are on my mac
    rootpath = '/Volumes/Projects/MEGRID/'
    opt['bidsPath'] = '/Volumes/ProjectData/MEGRID/'
    opt['fifPath'] = '/Users/giulianogiari/Desktop/fif/'
    opt['mriPath'] = '/Users/giulianogiari/Desktop/mri/'
    opt['maxPath'] = '/Users/giulianogiari/Desktop/derivatives/max/'
    opt['fsPath'] = '/Volumes/ProjectData/MEGRID/freesurfer/'
opt['derPath']   = f'{rootpath}derivatives/'
opt['prgPath']   = f'{rootpath}prg/analysis/'
opt['logPath']   = f'{rootpath}derivatives/log/'
opt['badPath']   = f'{rootpath}bad/'
opt['frqPath']   = f'{rootpath}derivatives/frq/'
opt['eyePath']   = f'{rootpath}derivatives/eye/'
opt['epoPath']   = f'{rootpath}derivatives/epo/'
opt['prePath']   = f'{rootpath}derivatives/pre/'
opt['infoPath']  = f'{rootpath}info/'
opt['figPath']   = f'{rootpath}derivatives/fig/'
opt['stcPath']   = f'{rootpath}derivatives/stc/'
opt['srcPath']   = f'{rootpath}derivatives/src/'
opt['behPath']   = f"{opt['bidsPath']}beh/"

# ----------------------------------------------- experiment 1 ------------------------------------------------------- #
opt['exp_1'] = {}
opt['exp_1']['subj_list'] = [f'sub-{sub_id}' for sub_id in ['19970805EEAT', '19970127MLAG', '19931222BLON',
                                                            '19820925PEGA', '19870521NRIA', '19940408MROS',
                                                            '19950121SRKN', '19940921BRFL', '19961224AAMN',
                                                            '19981201VRII', '19960711EEBN', '20020627GBVD',
                                                            '20010614DMCA', '19941010LCTY', '19940605SVTR',
                                                            '19930322ATBA', '19910513IACR', '20010802LLLO',
                                                            '20010926GABI', '19940721MRBU', '19970416RGBR',
                                                            '19851030GBGL', '19970323DNZN', '19990209CRSA']
                             ]
opt['exp_1']['to_exclude'] = [f'sub-{sub_id}' for sub_id in ['19931222BLON', '20020627GBVD']]
opt['exp_1']['no_mri']    = [f'sub-{sub_id}' for sub_id in ['19950121SRKN', '19961224AAMN']]
opt['exp_1']['ses_list'] = ['dots', 'lines', 'anat']

# ----------------------------------------------- experiment 2 ------------------------------------------------------- #
opt['exp_2'] = {}
opt['exp_2']['subj_list'] = [f'sub-{sub_id}' for sub_id in ['20000613BTBA', '19970522LINU', '20010722LLFL',
                                                            '19990729ANVN', '19960405EAFR', '20000217MCZN',
                                                            '19940521MRVN', '19961115GAPD', '19890618CAOM',
                                                            '19960904LUAT', '20011101DBBU', '19991017CIMN',
                                                            '20011224FARN', '19950621MRPV', '19950920MRRS',
                                                            '19961230RSTN', '20000519MRDM', '20011214CAAD',
                                                            '20000819NDAR', '19960330TTMD', '19901203ATBZ',
                                                            '19940424SFOH', '19930812ANVC', '20010710ANAB']
                             ]
opt['exp_2']['to_exclude'] = [f'sub-{sub_id}' for sub_id in ['20010722LLFL', '20000519MRDM']]
opt['exp_2']['no_mri']    = [] 
opt['exp_2']['ses_list'] = ['meg', 'anat']

# ------------------------------------------------- events ----------------------------------------------------------- #
opt['baserate_hz'] = 6
opt['baserate_ms'] = 166

opt['event_id'] = {'ang_res_15': 15, 'ang_res_30': 30, 'trial_end': 99, 'long_break_start': 150,
                   'long_break_end': 199, 'test_phase': 130, 'button_red': 253, 'button_blue': 254,
                   'empty': 1}

opt['ang_res'] = [15, 30]
opt['cycle_time'] = {'15': 4, '30': 2}
opt['n_cycles'] = {'15': 11, '30': 22}
opt['n_trj'] = 264
opt['n_trls'] = 48
opt['n_trj_in_cycle'] = {'15': 24, '30': 12}
opt['n_blocks'] = 6
opt['foi'] = {'15': {'4': 1, '6': 1.5, '8': 2},
              '30': {'4': 2, '6': 3}}
opt['cont_list'] = {'15': ['6>4', '6>8'],
                    '30': ['6>4']}

# ----------------------------------------------- preprocessing ------------------------------------------------------ #
# filters
opt['hp'] = .1
opt['lp'] = 40
# epochs
opt['tmin'] = -0.5 # s
opt['tmax'] = 44 # s
opt['empty_len'] = 'all' # all or trial, the duration of the emtpy room recording that will be considered
opt['diode_tolerance'] = 150  # samples
opt['reject_trls'] = True
opt['realign_trj'] = True
opt['starting_trj'] = 1
# number of segments into which to divide the time series
# it makes that this is a multiple of the number of cycles
# e.g. 15°: 4s (40s trial) --> 5 segments of 8 seconds (2 cycles);
#      30°: 2s (42s trial) --> 7 segments of 6 seconds (3 cycles).
opt['seg_info'] = {'15': {'n_seg': 5, 'seg_len': 8}, '30': {'n_seg': 7, 'seg_len': 6}}
opt['seg_overlap'] = 0 # in samples

# -------------------------------------------------- frequency ------------------------------------------------------- #
# fft
opt['frq_do'] = ['task']
opt['frq_foi_lims'] = [.1, 15]
opt['frq_out_fft'] = 'complex'
opt['frq_n_seg'] = 1 # this assumes that the segmentation has been done alrady
opt['frq_taper'] = 'boxcar' # boxcar to apply no taper,
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window for options

# coherence
opt['coh_do'] = ['task']
opt['coh_method'] = 'ding' 

# ------------------------------------------------ source space ------------------------------------------------------ #
opt['src_type'] = 'vol'
opt['src_bem_spacing'] = 4 
opt['src_vol_spacing'] = 5. # distance in mm between source points
opt['src_smooth'] = 'nearest' # https://mne.tools/stable/overview/implementation.html#about-smoothing

# ----------------------------------------------- source analysis ---------------------------------------------------- #
opt['stc_method'] = 'LCMV'
opt['stc_out']    = 'max-power'   
opt['stc_n_jobs'] = 5 # number of parallel jobs in source analysis     
opt['stc_lambda']   = 0.05            # lambda parameter for beamformer
opt['stc_ch_type'] = 'meg'
opt['stc_realign_trj'] = True
opt['stc_cov_data'] = 'noise'         
opt['stc_cov_method'] = 'empirical'
# reduce rank of the covariance matrix after tsss (Westner, 2022)
opt['stc_cov_rank']   = 'nfree'          # rank of the covariance matrix, nfree to calculate the actual df otherwise as mne
# empty room cov options
opt['stc_cov_empty_data'] = 'raw'        

# ------------------------------------------------- eye tracker ------------------------------------------------------ #
opt['eye_id'] = 'right'
opt['eye_blink'] = 'zero' 
# https://wiki.cimec.unitn.it/tiki-index.php?page=EYE+LINK+1000+PLUS
opt['eye_chan'] = {'RX': 'MISC001',  # (Right eye, X axis)',
                   'RY': 'MISC002',  # (Right eye, Y axis)',
                   'RP': 'MISC003',  # (Right eye, Pupil size)',
                   'LX': 'MISC004',  # (Left eye, X axis)',
                   'LY': 'MISC005',  # (Left eye, Y axis)',
                   'LP': 'MISC006'}  # (Left eye, Pupil size)'}

# coordinates of the fixation window, obtained in util_estimate_fixwin.m
# x is fixationWindow(1) and fixationWindow(3)
# while y is fixationWindow(2) and fixationWindow(4)
opt['fixWin'] = [608.3400, 429.0700, 831.6600, 650.9300]
opt['fixation_thresh'] = .8
opt['hmap_method'] = 'kde'
opt['hmap_downsampling'] = 1

# ------------------------------------------------- statistics ------------------------------------------------------- #
opt['cluster_method'] = 'cluster'
opt['ch_type_cluster'] = ['mag']
opt['tail'] = 0
opt['n_perms'] = 10000
opt['p_thresh'] = 0.05
opt['roi_list'] = ['entorhinal', 'hippocampus', 'parahippocampal',        # regions of interest
                   'lateraloccipital', 'precentral', 'pericalcarine']     # control regions

# -------------------------------------------------- plotting -------------------------------------------------------- #
# change matplotlib defaults
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
plot_style = 'paper'
if plot_style == 'paper':
    mpl.rc('axes', linewidth=1)
    mpl.rc('ytick', direction='in')
    mpl.rc('xtick', direction='in')
    mpl.rc('ytick.major', width=1)
    mpl.rc('xtick.major', width=1)
    mpl.rc('savefig', dpi= 500)
    mpl.rc('font', family='sans-serif', weight=500)
    mpl.rc({'font.sans-serif': 'Arial'})
    mpl.rc('axes.spines', right= False, top=False)
    mpl.rc('figure', frameon= False)
else:
    mpl.rc('axes', linewidth=5, titleweight='bold', labelweight='bold')
    mpl.rc('ytick', direction='in')
    mpl.rc('xtick', direction='in')
    mpl.rc('ytick.major', width=3)
    mpl.rc('xtick.major', width=3)
    mpl.rc('savefig', dpi= 500)
    mpl.rc('font', family='sans-serif', weight=500)
    mpl.rc({'font.sans-serif': 'Arial'})
    mpl.rc('axes.spines', right=False, top=False)
    mpl.rc('figure', frameon=False)
# colors
cmap_rois = cm.get_cmap('Blues')
opt['colors'] = dict()
opt['cmap_rois'] = {'rh': cmap_rois(.75), 'lh': cmap_rois(.95)}
opt['colors']['beh'] = dict()
opt['colors']['beh']['dots'] = [cm.get_cmap('Greens')(.9), cm.get_cmap('Greens')(.75), cm.get_cmap('Greens')(.6)]
opt['colors']['beh']['lines'] = [cm.get_cmap('Blues')(.9), cm.get_cmap('Blues')(.75), cm.get_cmap('Blues')(.6)]
opt['colors']['beh']['meg'] = [cm.get_cmap('Reds')(.9), cm.get_cmap('Reds')(.75), cm.get_cmap('Reds')(.6)]
opt['colors']['fold'] = dict()
opt['colors']['fold']['dots+lines'] = {4: cm.get_cmap('magma')(.25), 6: cm.get_cmap('magma')(.55), 8: cm.get_cmap('magma')(.8)}
opt['colors']['fold']['dots'] = {4: cm.get_cmap('magma')(.25), 6: cm.get_cmap('magma')(.55), 8: cm.get_cmap('magma')(.8)}
opt['colors']['fold']['lines'] = {4: cm.get_cmap('magma')(.25), 6: cm.get_cmap('magma')(.55), 8: cm.get_cmap('magma')(.8)}
opt['colors']['fold']['meg'] = {4: cm.get_cmap('magma')(.35), 6: cm.get_cmap('magma')(.6), 8: cm.get_cmap('magma')(.9)}
opt['colors']['topo'] = 'RdBu_r'
opt['colors']['topo_marker'] = np.array([255, 213, 0])/255
opt['colors']['fit'] = dict()
opt['colors']['fit']['QL'] = np.array([16, 79, 85])/255
opt['colors']['fit']['LQ'] = np.array([234, 221, 166])/255
opt['colors']['eye-grid'] = dict()
opt['colors']['eye-grid']['rh'] = np.array([160, 200, 166]) / 255
opt['colors']['eye-grid']['lh'] = np.array([16, 79, 100]) / 255
opt['colors']['pink'] = np.array([204, 158, 245]) / 255

# ------------------------------------------------ create folders ---------------------------------------------------- #
# loop through these folders and create them if they dont exist
for k, v in opt.items():
    if 'Path' in k:
        if not os.path.exists(v):
            os.mkdir(v)

# create subject specific log directories
for exp_id in ['1', '2']:

    for sub_id in opt[f'exp_{exp_id}']['subj_list']:

        if not os.path.exists(f"{opt['logPath']}{sub_id}/"):
            os.mkdir(f"{opt['logPath']}{sub_id}/")

        for ses_id in opt[f'exp_{exp_id}']['ses_list']:
            if not os.path.exists(f"{opt['logPath']}{sub_id}/ses-{ses_id}/"):
                os.mkdir(f"{opt['logPath']}{sub_id}/ses-{ses_id}/")

