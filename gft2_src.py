"""
functions to create source space
@author: giulianogiari
"""

import glob
import matplotlib
import mne
import numpy as np
import os
from mne.source_estimate import _volume_labels, _prepare_label_extraction

# to avoid figures to pop up
matplotlib.use('Agg')


def make_src(sub_id, opt_local):
    """
    Create source space and save diagnostic figures.
    If these files are already present the corresponding lines will be skipped
    """

    # define this subject filenames
    src_fname = f"{opt_local['srcPath']}{sub_id}_{opt_local['src_%s_spacing'%opt_local['src_type']]}_{opt_local['src_type']}-src.fif"
    bem_fname = f"{opt_local['srcPath']}{sub_id}_{opt_local['src_bem_spacing']}-bem.fif"

    # create bem head model
    if not os.path.exists(bem_fname):
        mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-anat/{sub_id}_ses-anat_log.log",
                         output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)
        model = mne.make_bem_model(subject=sub_id, ico=opt_local['src_bem_spacing'], conductivity=(0.3,),  # single shell
                                   subjects_dir=opt_local['fsPath'])
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_fname, bem)
        # plot bem
        fig = mne.viz.plot_bem(subject=sub_id, subjects_dir=opt_local['fsPath'], orientation='coronal',
                               brain_surfaces='white', show=False)
        fig.savefig(f"{opt_local['logPath']}{sub_id}/ses-anat/{sub_id}_bem.png", dpi=300)

    if not os.path.exists(src_fname):
        # create volume source space
        src = mne.setup_volume_source_space(subject=sub_id, pos=opt_local['src_vol_spacing'],
                                            subjects_dir=opt_local['fsPath'], mri='T1.mgz', bem=bem_fname)
        src.save(src_fname)

        # plot source space
        fig = mne.viz.plot_bem(subject=sub_id, subjects_dir=opt_local['fsPath'], orientation='coronal',
                               brain_surfaces='white', src=src, show=False)
        fig.savefig(f"{opt_local['logPath']}{sub_id}/ses-anat/{sub_id}_{opt_local['src_type']}.png", dpi=300)


def make_morpher(sub_id, ses_id, opt_local, info=None):
    """
    Compute and save source morph to fsaverage
    """
    morph_fname = f"{opt_local['srcPath']}{sub_id}_ses-{ses_id}_{opt_local['src_type']}_{opt_local['src_smooth']}-morph.h5"

    if not os.path.exists(morph_fname):
        mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-anat/{sub_id}_ses-anat_log.log",
                         output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)
        # load data
        if info is None:
            info = mne.io.read_info(glob.glob(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_epo.fif.gz")[0])
        fwd_ = make_forward(sub_id, ses_id, opt_local, info)
        src_fs = mne.read_source_spaces(f"{opt_local['srcPath']}sub-fsaverage_"
                                        f"{opt_local['src_%s_spacing' % opt_local['src_type']]}_{opt_local['src_type']}-src.fif")
        morpher_ = mne.compute_source_morph(fwd_['src'], subject_from=sub_id, subject_to='sub-fsaverage',
                                            subjects_dir=opt_local['fsPath'], smooth=opt_local['src_smooth'],
                                            src_to=src_fs)
        morpher_.save(morph_fname, overwrite=True)
    else:
        morpher_ = mne.read_source_morph(morph_fname)

    return morpher_


def make_forward(sub_id, ses_id, opt_local, info=None):
    """
    Compute and save forward solution
    """

    fwd_fname = f"{opt_local['srcPath']}{sub_id}_ses-{ses_id}_{opt_local['src_%s_spacing' % opt_local['src_type']]}_{opt_local['src_type']}-fwd.fif"

    if os.path.exists(fwd_fname):
        fwd_ = mne.read_forward_solution(fwd_fname) # is possible to select only few channels https://mne.tools/stable/generated/mne.read_forward_solution.html
    else:
        mne.set_log_file(f"{opt_local['logPath']}{sub_id}/ses-anat/{sub_id}_ses-anat_log.log",
                         output_format='%(asctime)s | %(levelname)s | %(message)s', overwrite=False)
        if info is None:
            info = mne.io.read_info(f"{opt_local['epoPath']}{sub_id}_ses-{ses_id}_task-task_epo.fif.gz")
        src_fname = f"{opt_local['srcPath']}{sub_id}_{opt_local['src_%s_spacing' % opt_local['src_type']]}_{opt_local['src_type']}-src.fif"
        bem_fname = f"{opt_local['srcPath']}{sub_id}_{opt_local['src_bem_spacing']}-bem.fif"
        trans_fname = f"{opt_local['srcPath']}{sub_id}_ses-{ses_id}_trans.fif"
        fwd_ = mne.make_forward_solution(info, trans_fname, src_fname, bem_fname, meg=True)
        mne.write_forward_solution(fwd_fname, fwd_)

    return fwd_


def copy_fsaverage(opt_local):
    """ Copies fsaverage folder to the project folder and creates the relevant surfaces """

    # copy its folder if its not there
    if not os.path.exists(f"{opt_local['fsPath']}sub-fsaverage"):
        mne.coreg.create_default_subject(fs_home=None, update=False, subjects_dir=opt_local['fsPath'],
                                         verbose=None)
    # run the watershed algorhythm
    os.system(f"mne watershed_bem -s sub-fsaverage -d {opt_local['fsPath']} --overwrite")

    # move files where mne expects them
    files_list = glob.glob(f"{opt_local['fsPath']}sub-fsaverage/bem/watershed/*")
    for fname in files_list:
        # extract the surface name
        surf_name = fname.split('fsaverage_')[-1].split('_surface')[0]
        # remove the symbolic link to file to avoid errors
        os.system(f"rm -f {opt_local['fsPath']}/sub-fsaverage/bem/{surf_name}.surf")
        # copy and rename the original file
        os.system(f"cp -rf {fname} {opt_local['fsPath']}/sub-fsaverage/bem/{surf_name}.surf")

    # create the source space
    os.mkdir(f"{opt_local['logPath']}sub-fsaverage/")
    os.mkdir(f"{opt_local['logPath']}sub-fsaverage/ses-anat/")
    make_src('fsaverage', opt_local)


def make_cortical_mask(opt_local):
    roi_list = ['Left-Cerebellum-Exterior', 'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
                'Right-Cerebellum-Exterior', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
                'Brain-Stem', '4th-Ventricle']
    # get the labels from the freesurfer parcellation
    mri_fname = f"{opt_local['fsPath']}sub-fsaverage/mri/aparc+aseg.mgz"
    # https://github.com/mne-tools/mne-python/blob/79ea57a4318d8d045f5966c26360b079f40a4865/mne/source_estimate.py#L3077
    src_fs = mne.read_source_spaces(f"{opt_local['srcPath']}sub-fsaverage_"
                                    f"{opt_local['src_%s_spacing' % opt_local['src_type']]}_{opt_local['src_type']}-src.fif")
    labels = _volume_labels(src_fs, mri_fname, mri_resolution=True)

    # get the corresponding indices in the source space
    label_vertidx, _ = _prepare_label_extraction(
        None, labels, src_fs, mode='mean', allow_empty=True, use_sparse=True)

    ind = np.zeros_like(src_fs[0]['vertno'], dtype=bool)
    for i, l in enumerate(labels):
        if l['name'] in roi_list:
            ind[label_vertidx[i].nonzero()[1]] = True
    return ind

