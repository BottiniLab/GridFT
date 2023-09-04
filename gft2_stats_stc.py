"""
functions for statistics at source level
@author: giuliano giari, giuliano.giari@gmail.com
"""

import h5py
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import warnings
from gft2_src import make_morpher, make_forward, make_cortical_mask
from nilearn import plotting
from scipy import stats, special
from mne.stats import permutation_cluster_1samp_test
from gft2_stc import read_stc
from gft2_utils import significance_bar
from gft2_frequency import pick_frequency
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img


def load_stc_data(ses_id, opt_local):
    """
    Load frequency data of the specified session
    :return:
    """
    group_fname = f"{opt_local['stcPath']}ses-{ses_id}_stc_desc-{opt_local['stc_method']}-{opt_local['stc_out']}-" \
                  f"{opt_local['src_type']}-coh.h5"

    exp_id = 'exp_2' if ses_id == 'meg' else 'exp_1'

    # load fs average source space
    src_fs = mne.read_source_spaces(f"{opt_local['srcPath']}sub-fsaverage_"
                                    f"{opt_local['src_%s_spacing' % opt_local['src_type']]}"
                                    f"_{opt_local['src_type']}-src.fif")
    connectivity = mne.spatial_src_adjacency(src_fs)

    data_stc = {ses_id: {ang_res: {fold_id: [] for fold_id in opt_local['foi'][str(ang_res)].keys()}
                         for ang_res in opt_local['ang_res']}}

    if not os.path.exists(group_fname):

        for sub_id in opt_local[exp_id]['subj_list']:
            if sub_id in opt_local[exp_id]['to_exclude']:
                print(sub_id, 'excluded')
                continue
            if ses_id != 'dots+lines':
                # load this here for convenience
                morpher = make_morpher(sub_id, ses_id, opt_local)
            for ang_res in opt_local['ang_res']:
                # load source data
                if ses_id == 'dots+lines':
                    # first get the data of the individual sessions and average
                    sub_stc = {fold_id: [] for fold_id, foi in opt_local['foi'][str(ang_res)].items()}
                    for ses_id_split in ['dots', 'lines']:
                        # load the data and the morpher
                        stc_ses = read_stc(f"{opt_local['stcPath']}{sub_id}_ses-{ses_id_split}_task-task_desc-"
                                           f"{opt_local['stc_out']}_{opt_local['stc_method']}_{ang_res}_coh"
                                           f"_{opt_local['src_type']}.h5", return_object=True)
                        morpher = make_morpher(sub_id, ses_id_split, opt_local)

                        for fold_id, foi in opt_local['foi'][str(ang_res)].items():
                            # print(f"requested {fold_id} fold - {foi} Hz")
                            # print('cropping time (freq) axis')
                            stc_crop = stc_ses.copy().crop(foi, foi)
                            # print(f"actual frequency {stc_crop.times[0]} Hz")
                            assert (float(foi) - stc_crop.times[0] <= 1e-10)
                            sub_stc[fold_id].append(morpher.apply(stc_crop).data.squeeze())
                    # then average the two session maps for each foi separately and append to group data
                    for fold_id, foi in opt_local['foi'][str(ang_res)].items():
                        data_stc[ses_id][ang_res][fold_id].append(np.mean(sub_stc[fold_id], 0))
                else:
                    stc_fname = f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_" \
                                f"{opt_local['stc_method']}_{ang_res}_coh_{opt_local['src_type']}.h5"
                    stc = read_stc(stc_fname, return_object=True)

                    for fold_id, foi in opt_local['foi'][str(ang_res)].items():
                        # print(f"requested {fold_id} fold - {foi} Hz")
                        # print('cropping time (freq) axis')
                        stc_crop = stc.copy().crop(foi, foi)
                        # print(f"actual frequency {stc_crop.times[0]} Hz")
                        assert (float(foi) - stc_crop.times[0] <= 1e-10)
                        data_stc[ses_id][ang_res][fold_id].append(morpher.apply(stc_crop).data.squeeze())
        # ------- save to h5 -------- #
        f = h5py.File(group_fname, 'w')
        for ang_res in opt_local['ang_res']:
            for fold_id in opt_local['foi'][str(ang_res)].keys():
                g = f.create_group(name=f'/{ses_id}/{ang_res}/{fold_id}fold')
                g.create_dataset(name='coh',
                                 data=np.array(data_stc[ses_id][ang_res][fold_id]))
        f.close()
    else:
        warnings.warn(f"loading {group_fname}...")
        # load the data to dictionary
        f = h5py.File(group_fname, 'r')
        for ang_res in opt_local['ang_res']:
            for fold_id in opt_local['foi'][str(ang_res)].keys():
                data_stc[ses_id][ang_res][fold_id] = np.array(f.get(f'/{ses_id}/{ang_res}/{fold_id}fold/coh'))
        f.close()
    return data_stc, connectivity, src_fs


def load_roi_data(ses_id, opt_local):
    """

    :param ses_id:
    :param opt_local:
    :return:
    """
    group_fname = f"{opt_local['stcPath']}ses-{ses_id}_ROI_desc-{opt_local['stc_method']}-{opt_local['stc_out']}-" \
                  f"{opt_local['src_type']}-coh"
    data_roi_h5 = {ses_id: {str(ang_res): {hemi: {roi: [] for roi in opt_local['roi_list']} for hemi in ['lh', 'rh']}
                         for ang_res in opt_local['ang_res']}}
    data_roi_csv = {k: [] for k in ['sub_id', 'ses_id', 'ang_res', 'hemi', 'roi', 'fold', 'coh', 'n_voxels']}
    freqs = {str(ang_res): None for ang_res in opt_local['ang_res']}

    if not os.path.exists(f"{group_fname}.h5"):

        exp_id = 'exp_2' if ses_id == 'meg' else 'exp_1'

        for i_sub, sub_id in enumerate(opt_local[exp_id]['subj_list']):

            if sub_id in opt_local[exp_id]['to_exclude']:
                print(sub_id, 'excluded')
                continue

            fwd = make_forward(sub_id, ses_id, opt_local.copy())

            for ang_res in opt_local['ang_res']:
                # load source data
                stc_fname = f"{opt_local['stcPath']}{sub_id}_ses-{ses_id}_task-task_desc-{opt_local['stc_out']}_" \
                                f"{opt_local['stc_method']}_{ang_res}_coh_{opt_local['src_type']}.h5"
                stc = read_stc(stc_fname, return_object=True)

                for hemi in ['lh', 'rh']:

                    mtl_data = []

                    for roi in opt_local['roi_list']:

                        if roi == 'hippocampus':
                            label_name = 'Left-Hippocampus' if hemi == 'lh' else 'Right-Hippocampus'
                        else:
                            label_name = f"ctx-{hemi}-{roi}"
                        data_array = stc.in_label(label=label_name, src=fwd['src'],
                                                  mri=f"{opt_local['fsPath']}{sub_id}/mri/aparc+aseg.mgz",
                                                  ).data
                        # average across voxels in the roi
                        roi_data = data_array.mean(0)

                        if roi == 'entorhinal' or roi == 'hippocampus' or roi == 'parahippocampal':
                            mtl_data.append(data_array)

                        # store data in dict to be saved as h5
                        data_roi_h5[ses_id][str(ang_res)][hemi][roi].append(roi_data)
                        # store data in dict to be saved as csv
                        for foi_id, foi in opt_local['foi'][str(ang_res)].items():
                            data_roi_csv['sub_id'].append(sub_id)
                            data_roi_csv['ses_id'].append(ses_id)
                            data_roi_csv['ang_res'].append(ang_res)
                            data_roi_csv['hemi'].append(hemi)
                            data_roi_csv['roi'].append(roi)
                            data_roi_csv['fold'].append(int(foi_id))
                            data_roi_csv['n_voxels'].append(data_array.shape[0])
                            data_roi_csv['coh'].append(roi_data[pick_frequency(stc.times, foi)])

                    # add the mtl roi
                    if i_sub == 0:
                        data_roi_h5[ses_id][str(ang_res)][hemi]['MTL'] = []
                    roi_data = np.concatenate(mtl_data, 0).mean(0)

                    data_roi_h5[ses_id][str(ang_res)][hemi]['MTL'].append(roi_data)
                    for foi_id, foi in opt_local['foi'][str(ang_res)].items():
                        data_roi_csv['sub_id'].append(sub_id)
                        data_roi_csv['ses_id'].append(ses_id)
                        data_roi_csv['ang_res'].append(ang_res)
                        data_roi_csv['hemi'].append(hemi)
                        data_roi_csv['roi'].append('MTL')
                        data_roi_csv['fold'].append(int(foi_id))
                        data_roi_csv['n_voxels'].append(data_array.shape[0])
                        data_roi_csv['coh'].append(np.concatenate(mtl_data, 0).mean(0)[pick_frequency(stc.times, foi)])

                freqs[str(ang_res)] = stc.times

        # ---------------------------- save to h5 ----------------------------- #
        f = h5py.File(f"{group_fname}.h5", 'w')
        # f.create_dataset(name='info', data=frq_event.info)
        for ang_res in opt_local['ang_res']:
            for hemi in ['lh', 'rh']:
                for roi in opt_local['roi_list']:
                    g = f.create_group(name=f'/{ses_id}/{ang_res}/{hemi}/{roi}')
                    g.create_dataset(name='coh',
                                     data=np.array(data_roi_h5[ses_id][str(ang_res)][hemi][roi]))
                # add mtl
                g = f.create_group(name=f'/{ses_id}/{ang_res}/{hemi}/MTL')
                g.create_dataset(name='coh',
                                 data=np.array(data_roi_h5[ses_id][str(ang_res)][hemi]['MTL']))
            f.create_dataset(name=f'/{ses_id}/{ang_res}/freqs', data=freqs[str(ang_res)])
        f.close()
        # save as csv
        df = pd.DataFrame.from_dict(data_roi_csv)
        df.to_csv(f"{group_fname}.csv")
    else:
        warnings.warn('loading data')
        f = h5py.File(f"{group_fname}.h5", 'r')
        for ang_res in opt_local['ang_res']:
            for hemi in ['lh', 'rh']:
                for roi in opt_local['roi_list']:
                    data_roi_h5[ses_id][str(ang_res)][hemi][roi] = np.array(f.get(f'/{ses_id}/{ang_res}/{hemi}/{roi}/coh/'))
                data_roi_h5[ses_id][str(ang_res)][hemi]['MTL'] = np.array(f.get(f'/{ses_id}/{ang_res}/{hemi}/MTL/coh/'))
            freqs[str(ang_res)] = np.array(f.get(f'/{ses_id}/{ang_res}/freqs'))
        f.close()

        df = pd.read_csv(f"{group_fname}.csv", index_col=0)

    return data_roi_h5, freqs, df


def _average_ses_roi(ses1, ses2, opt_local):
    """
    Average roi data of two sessions
    :param ses1:
    :param ses2:
    :param opt_local:
    :return:
    """
    group_fname = f"{opt_local['stcPath']}ses-{ses1}+{ses2}_ROI_desc-{opt_local['stc_method']}-{opt_local['stc_out']}-" \
                  f"{opt_local['src_type']}-coh"
    roi_dict = {f"{ses1}+{ses2}": {str(ang_res): {hemi: {} for hemi in ['lh', 'rh']}
                                   for ang_res in opt_local['ang_res']}}
    freqs = {str(ang_res): None for ang_res in opt_local['ang_res']}

    if not os.path.exists(f"{group_fname}.csv"):
        # average dots and lines
        # first load the individual datasets
        ses1_dict, freqs, ses1_df = load_roi_data(ses1, opt_local)
        ses2_dict, freqs, ses2_df = load_roi_data(ses2, opt_local)

        # merge dataframes
        df = pd.concat([ses1_df, ses2_df])
        roi_df = df.groupby(['sub_id', 'roi', 'hemi', 'fold', 'ang_res']).mean().reset_index()
        roi_df['ses_id'] = f"{ses1}+{ses2}"

        # merge dict
        for ang_res in opt_local['ang_res']:
            for hemi in ['lh', 'rh']:
                for roi in ses1_dict[ses1][str(ang_res)][hemi].keys():
                    roi_dict[f"{ses1}+{ses2}"][str(ang_res)][hemi][roi] = \
                        ( np.array(ses1_dict[ses1][str(ang_res)][hemi][roi]) + np.array(ses2_dict[ses2][str(ang_res)][hemi][roi]) ) / 2

        # ---------------------------- save to h5 ----------------------------- #
        f = h5py.File(f"{group_fname}.h5", 'w')
        for ang_res in opt_local['ang_res']:
            for hemi in ['lh', 'rh']:
                for roi in roi_dict[f"{ses1}+{ses2}"][str(ang_res)][hemi].keys():
                    g = f.create_group(name=f'/{ses1}+{ses2}/{ang_res}/{hemi}/{roi}')
                    g.create_dataset(name='coh',
                                     data=np.array(roi_dict[f"{ses1}+{ses2}"][str(ang_res)][hemi][roi]))
            f.create_dataset(name=f'/{ses1}+{ses2}/{ang_res}/freqs', data=freqs[str(ang_res)])
        f.close()
        # save as csv
        roi_df.to_csv(f"{group_fname}.csv")

    else:

        warnings.warn('loading data')
        f = h5py.File(f"{group_fname}.h5", 'r')
        for ang_res in f[f"{ses1}+{ses2}"].keys():
            for hemi in ['lh', 'rh']:
                for roi in f[f"{ses1}+{ses2}"][ang_res][hemi].keys():
                    roi_dict[f"{ses1}+{ses2}"][ang_res][hemi][roi] = np.array(
                        f.get(f"{ses1}+{ses2}/{ang_res}/{hemi}/{roi}/coh/"))
            freqs[str(ang_res)] = np.array(f.get(f"{ses1}+{ses2}/{ang_res}/freqs"))
        f.close()

        roi_df = pd.read_csv(f"{group_fname}.csv", index_col=0)

    return roi_dict, freqs, roi_df


def ttest_roi(ses_id, roi_list, opt_local):
    """

    :param ses_id:
    :param opt_local:
    :return:
    """

    # load the data
    if ses_id == 'dots+lines':
        _, _, roi_df = _average_ses_roi('dots', 'lines', opt_local)
    else:
        _, _, roi_df = load_roi_data(ses_id, opt_local)

    # initialize dict to store the results
    d = {k: [] for k in ['Angular\nResolution', 'Hemisphere', 'ROI', 'Comparison', 'df', 't', 'p', 'Sig.']}

    for ang_res in opt_local['ang_res']:

        # define figure size in cm
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
        cm = 1 / 2.54  # centimeters in inches
        fig, ax = plt.subplots(2, len(roi_list), figsize=(10, 4))
        # add a big axes for labels https://stackoverflow.com/questions/6963035/pyplot-common-axes-labels-for-subplots
        ax_label = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        for i_hemi, hemi in enumerate(['lh', 'rh']):
            for i_roi, roi in enumerate(roi_list):
                if all(['hippocampus' != roi, opt_local['src_type'] == 'srf']) or opt_local['src_type'] == 'vol':

                    this_df = roi_df.loc[(roi_df['hemi'] == hemi) & (roi_df['roi'] == roi) &
                                         (roi_df['ang_res'] == ang_res)]
                    # plot the frequency of interest
                    for i_fold, fold_id in enumerate(this_df['fold'].unique()):
                        fold_data = this_df.loc[this_df['fold'] == fold_id]['coh'].values
                        ax[i_hemi][i_roi].bar(x=i_fold,
                                              height=fold_data.mean(),
                                              # we add here another value to the color that indicates the alpha value
                                              color=np.concatenate([opt_local['colors']['fold'][ses_id][fold_id][:3], [.85]]),
                                              yerr=stats.sem(fold_data), width=.7, edgecolor='k',
                                              error_kw={'capsize': 5})
                    # plot participants
                    for sub_id, x in zip(this_df['sub_id'].unique(),
                                         np.random.uniform(-.2, .2, len(this_df['sub_id'].unique()))):
                        sub_data = this_df.loc[this_df['sub_id'] == sub_id].sort_values(by='fold')['coh'].values
                        ax[i_hemi][i_roi].plot(np.arange(len(this_df['fold'].unique())) + x,
                                               sub_data,
                                               '.-', color='gray', alpha=.5, ms=10, markeredgecolor='k',
                                               markeredgewidth=1)

                    # t-test of the pairs
                    for i_contrast, contrast_id in enumerate(opt_local['cont_list'][str(ang_res)]):
                        t, p = stats.ttest_rel(
                            this_df.loc[this_df['fold'] == int(contrast_id.split('>')[0])]['coh'].values,
                            this_df.loc[this_df['fold'] == int(contrast_id.split('>')[1])]['coh'].values,
                            alternative='two-sided')
                        print(ses_id, ang_res, hemi, roi, contrast_id, t, p)
                        significance_bar(t, p, [i_contrast + .1, i_contrast + .8],
                                         [this_df['coh'].max() * 1.1,
                                          this_df['coh'].max() * 1.1],
                                         ax[i_hemi][i_roi])
                        # add the results to a dict
                        d['Angular\nResolution'].append(f"{ang_res}°")
                        d['Hemisphere'].append('Left' if hemi == 'lh' else 'Right')
                        d['ROI'].append('Lat. occipital' if roi == 'lateraloccipital' else roi.title())
                        if ses_id == 'meg':
                            comp_str = ' vs '.join([f"Cont{str(f)}" \
                                            f"[{opt_local['foi'][str(ang_res)][str(f)]} Hz]"
                                        for f in contrast_id.split('>')])
                        else:
                            comp_str = ' vs '.join([f"{str(f)}" \
                                        f"[{opt_local['foi'][str(ang_res)][str(f)]} Hz]"
                                        for f in contrast_id.split('>')])
                        d['Comparison'].append(comp_str)
                        d['df'].append(len(this_df.loc[this_df['fold'] == int(contrast_id.split('>')[0])]['coh']) - 1)
                        d['t'].append(str(t)[:4] if str(t)[0] != '-' else str(t)[:5])
                        d['p'].append('<0.001' if p < 0.001 else str(p)[:5])
                        if p <= 0.001:
                            sig = '***'
                        elif p <= 0.01:
                            sig = '**'
                        elif p <= 0.05:
                            sig = '*'
                        else:
                            sig = 'n.s.'
                        d['Sig.'].append(sig)

                    # labels and things
                    ax[i_hemi][i_roi].set_xticks(np.arange(0, len(this_df['fold'].unique())))
                    xticklabels = [f"{str(f)} [{opt_local['foi'][str(ang_res)][str(f)]}]"
                                   for f in this_df['fold'].unique()]
                    ax[i_hemi][i_roi].set_xticklabels(xticklabels)
                    # ax[i_hemi][i_roi].set_ylim(0, ax[i_hemi][i_roi].get_ylim()
                    ylims = ax[i_hemi][i_roi].get_ylim()
                    ax[i_hemi][i_roi].set_yticks(np.linspace(ylims[0], ylims[1], 4))
                    ax[i_hemi][i_roi].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
                    ax[i_hemi][i_roi].set_xlim(-.5, len(this_df['fold'].unique()) - .5)
                    if i_hemi == 0:
                        if roi == 'MTL': roi = 'Medial temporal lobe'
                        elif roi == 'lateraloccipital': roi = 'Lateral occipital'
                        ax[i_hemi][i_roi].set_title(roi.title(), y = 1.1)
        ax[0][len(roi_list)-1].text(1.07, .45, 'Left',rotation=270, ha='center',
                                    color=opt_local['cmap_rois']['lh'], transform=ax[0][len(roi_list)-1].transAxes)
        ax[1][len(roi_list)-1].text(1.07, .45, 'Right', rotation=270, ha='center',
                                    color=opt_local['cmap_rois']['rh'], transform=ax[1][len(roi_list) - 1].transAxes)
        plt.xlabel("ContFold [Frequency in Hz]" if ses_id == 'meg' else "Fold [Frequency in Hz]")
        plt.text(-.06, .5, 'ITC', fontsize=12, rotation=90,
                 verticalalignment='center', horizontalalignment='right')
        # final things and save
        fig.suptitle(f"{ang_res}° resolution" if ses_id != 'meg' else f"Corresponding to {ang_res}° resolution", y=0.94,
                     horizontalalignment='center')
        fig.tight_layout()
        fig.savefig(f"{opt_local['figPath']}stc_{ses_id}_{opt_local['stc_method']}_{opt_local['stc_out']}_"
                    f"{opt_local['src_type']}_{ang_res}_coh-roi.tiff", dpi=500, pad_inches=0, bbox_inches='tight',
                    transparent=True)
    # clean up the dictionary by removing the duplicate rows in sequence
    for k in ['Angular\nResolution', 'Hemisphere', 'ROI']:
        d[k] = ['' if d[k][i] == d[k][i - 1] else d[k][i] for i in range(len(d[k]))]
    df = pd.DataFrame.from_dict(d)
    # plot results table
    #https: // www.sonofacorner.com / beautiful - tables /
    fig = plt.figure(figsize=(15, 10), dpi=500)
    ax = plt.subplot()
    fontsize = 12
    lw = 1
    nrows = df.shape[0]
    x_positions = [0, 5, 7.6, 12, 15.4, 17, 19, 21]
    ax.set_xlim(0, max(x_positions) + 1)
    y_positions = np.linspace(0, nrows * 2, nrows)
    ax.set_ylim(-3, nrows * 2 + 4)
    columns = df.keys()
    for j, column in enumerate(columns):
        # Add column names
        ax.annotate(
            xy=(x_positions[j], nrows * 2 + 3),
            text=column,
            ha='left' if j == 0 else 'center',
            va='bottom', weight='bold', fontsize=fontsize)
        # Add table's main text
        for y, i in zip(reversed(range(nrows)), range(nrows)):
            ax.annotate(
                xy=(x_positions[j], y_positions[y]),
                text=df[column].iloc[i],
                ha='left' if j == 0 else 'center',
                va='center', fontsize=fontsize)
    ax.set_axis_off()
    # Add dividing lines
    # angular resolutions
    for y in [nrows*2+2, -2, 12]:
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [y, y], lw=lw, color='black', marker='', zorder=4)
    # hemispheres
    for y in [5.5, 24.5]:
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [y, y], lw=lw, ls='--', color='gray', marker='', zorder=3)
    # roi
    for y_ind in [0, 1, 3, 4, 7, 9, 13, 15]: #[1, 3, 7, 9, 15, 19, 27, 31]:
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [y_positions[y_ind]+1.25, y_positions[y_ind]+1.25], lw=.8, ls=':',
                color='gray', marker='', zorder=2, alpha=.8)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [24.5, 24.5], lw=.8, ls=':', color='gray', marker='', zorder=3)
    fig.savefig(f"{opt_local['figPath']}stc_{ses_id}_pairwise_table.tiff", dpi=500, pad_inches=0,
                bbox_inches='tight', transparent=True)


def plot_spectrum_roi(ses_id, opt_local):
    """
    Plot $metric spectrum in the rois
    """

    # load the data
    if ses_id == 'dots+lines':
        roi_dict, freqs, _ = _average_ses_roi('dots', 'lines', opt_local)
    else:
        roi_dict, freqs, _ = load_roi_data(ses_id, opt_local)

    for roi in ['MTL']:
        fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
        vmin = np.min([v2[roi].mean(0).min() for v1 in roi_dict[ses_id].values() for v2 in v1.values()])
        vmax = np.max([v2[roi].mean(0).max() for v1 in roi_dict[ses_id].values() for v2 in v1.values()])
        for i_res, ang_res in enumerate(opt_local['ang_res']):
            for i_hemi, hemi in enumerate(['lh', 'rh']):
                m = roi_dict[ses_id][str(ang_res)][hemi][roi].mean(0)
                ax[i_res][i_hemi].plot(freqs[str(ang_res)], m, lw=3)
                ax[i_res][i_hemi].fill_between(freqs[str(ang_res)],
                                               m - np.std(roi_dict[ses_id][str(ang_res)][hemi][roi], 0),
                                               m + np.std(roi_dict[ses_id][str(ang_res)][hemi][roi], 0), alpha=.5)
                # plot lines corresponding to the folds
                for i_f, (fold_id, foi) in enumerate(opt_local['foi'][str(ang_res)].items()):
                    ax[i_res][i_hemi].vlines(foi, 0, vmax *1.5, ls='--',
                                             color=opt_local['colors']['fold'][ses_id][int(fold_id)])
                    ax[i_res][i_hemi].text(foi+0.1, vmax * .6, f"{str(fold_id)} Fold",
                                           rotation=90, fontsize=12,
                                           color=opt_local['colors']['fold'][ses_id][int(fold_id)])
                # base-rate and cycle rate
                for f, t, c in zip([1 if ang_res == 30 else .5, 6], ('Sequence rate', 'Presentation\nrate'), ('k', 'green')):
                    ax[i_res][i_hemi].vlines(f, 0, vmax*1.5, ls='--', color=c)
                    ax[i_res][i_hemi].text(f+0.1, vmax * .6, t, rotation=90, fontsize=12, color=c)

                if (i_res == 0 and i_hemi == 0) or (i_res == 0 and i_hemi == 1):
                    ax[i_res][i_hemi].set_title('Left Hemisphere' if hemi == 'lh' else 'Right Hemisphere')
                if (i_res == 0 and i_hemi == 0) or (i_res == 1 and i_hemi == 0):
                    ax[i_res][i_hemi].text(-.25, .5, f"{ang_res}°", transform=ax[i_res][i_hemi].transAxes,
                                           rotation='vertical')
        plt.suptitle(roi.title() if roi != 'MTL' else 'Medial temporal lobe')
        plt.setp(ax, xlabel='Frequency (Hz)', ylabel='ITC',
                 xlim=(min(freqs[str(ang_res)]), max(freqs[str(ang_res)])), ylim=(0, vmax))
        fig.tight_layout()
        fig.savefig(f"{opt_local['figPath']}stc_{ses_id}_{opt_local['stc_method']}_{opt_local['stc_out']}_"
                    f"{opt_local['src_type']}_{ang_res}_{metric}_{roi}-spectrum-roi.tiff", dpi=500)
        plt.close()


def conjunction(ses_id, p_thresh, alternative, opt_local):
    """ whole brain conjunction analysis (Nichols et al., 2005) """
    data_stc, connectivity, src_fs = load_stc_data(ses_id, opt_local)
    mask_ind = make_cortical_mask(opt_local)

    stat_res_stc = {ses_id: {str(ang_res): {} for ang_res in opt_local['ang_res']}}
    for i_ang, ang_res in enumerate(opt_local['ang_res']):
        for contrast_id in opt_local['cont_list'][str(ang_res)]:

            data_for_1samp = np.array(data_stc[ses_id][ang_res][contrast_id.split('>')[0]]) - \
                             np.array(data_stc[ses_id][ang_res][contrast_id.split('>')[1]])

            res = [stats.ttest_1samp(vox, 0, alternative=alternative) for vox in data_for_1samp.T]
            stat_res_stc[ses_id][str(ang_res)][contrast_id] = {}
            stat_res_stc[ses_id][str(ang_res)][contrast_id]['t'] = np.stack(res)[:, 0]
            stat_res_stc[ses_id][str(ang_res)][contrast_id]['p'] = np.stack(res)[:, 1]

    # combine the contrasts
    t_maps = np.stack([stat_res_stc[ses_id]['15']['6>8']['t'],
                       stat_res_stc[ses_id]['15']['6>4']['t'],
                       stat_res_stc[ses_id]['30']['6>4']['t']], 0)
    # combine the maps by taking the lowest t-score across contrasts at each voxel
    conjunction = np.min(t_maps, 0)
    # keep only the positive voxels. this is a test in which we look for the minimum t value, thus a negative t-value
    # will always be minimal but doesnt necessarily mean that all the voxels have a negative t-value
    conjunction[conjunction < 0] = 0
    # make the voxels corresponding to the mask to 0
    conjunction[mask_ind] = 0

    # define t-crit based on pval
    #p_thresh = 0.01
    if alternative == 'greater':
        thresh = stats.distributions.t.ppf(1-p_thresh, data_for_1samp.shape[0] - 1)
    elif alternative == 'less':
        thresh = stats.distributions.t.ppf(p_thresh, data_for_1samp.shape[0] - 1)
    elif alternative == 'two-sided':
        thresh = stats.distributions.t.ppf(1-p_thresh / 2, data_for_1samp.shape[0] - 1)
    # thresh = 2.518  # 0.01
    # thresh = 2.831  # 0.005
    vmax = np.max(conjunction)

    clu = mne.VolSourceEstimate(conjunction, vertices=[src_fs[0]['vertno']],
                                tmin=0, tstep=0, subject='sub-fsaverage')
    clu_vol = clu.as_volume(src_fs, dest='mri', format='nifti1')
    clu_vol.to_filename(f"{opt_local['figPath']}stc_{ses_id}_slices_conjunction_{p_thresh}_{alternative}.nii")

    for coord in ['doeller2010', 'nau2018']:
        im = plotting.plot_stat_map(clu_vol, display_mode='ortho', colorbar=True,
                                    output_file=f"{opt_local['figPath']}stc_{ses_id}_slices_conjunction_{p_thresh}_{alternative}_{coord}.tiff",
                                    cut_coords=opt_local['coordinates'][coord], cmap='Reds',
                                    draw_cross=False, dim=-.1, threshold=thresh, vmax=vmax)

    im = plotting.plot_stat_map(clu_vol, display_mode='ortho', colorbar=True,
                                output_file=f"{opt_local['figPath']}stc_{ses_id}_slices_conjunction_test.tiff",
                                cut_coords=(30, -18, -25), cmap='Reds',
                                draw_cross=False, dim=-.1, threshold=thresh, vmax=vmax)
    

def cluster_permutation_test():
    """
    Cluster permutation test at source level
    """
    # load data
    data_stc, connectivity, src_fs = load_stc_data(ses_id, metric, opt_local)
    mask_ind = make_cortical_mask(opt_local)

    # ----------------------------------------- cluster permutation test --------------------------------------------- #
    stats_fname = f"{opt_local['stcPath']}stats_{ses_id}_{opt_local['stc_method']}_{opt_local['cluster_method']}_" \
                  f"{opt_local['n_perms']}_{opt_local['p_thresh']}_{opt_local['src_type']}_{metric}_{opt_local['stc_ch_type']}.h5"

    if os.path.isfile(stats_fname):
        with open(stats_fname, 'rb') as f:
            stat_res_stc = pickle.load(f)
    else:
        threshold = -stats.distributions.t.ppf(opt_local['p_thresh'] / 2.,
                                               np.squeeze(data_stc[ses_id][15]['6']).shape[0] - 1)
    
        stat_res_stc = {ses_id: {str(ang_res): {} for ang_res in opt_local['ang_res']}}
        for i_ang, ang_res in enumerate(opt_local['ang_res']):

            f6 = np.array(data_stc[ses_id][ang_res]['6'])
            if ang_res == 30:
                fCont = np.array(data_stc[ses_id][ang_res]['4'])
            else:
                fCont = 0.5 * np.array(data_stc[ses_id][ang_res]['4']) + 0.5 * np.array(data_stc[ses_id][ang_res]['8'])

            stat_res_stc[ses_id][str(ang_res)] = permutation_cluster_1samp_test(
                f6-fCont, adjacency=connectivity, threshold=threshold, out_type='indices',
                n_permutations=opt_local['n_perms'], n_jobs=-1, seed=0, exclude=mask_ind)
        with open(stats_fname, 'wb') as f:
            pickle.dump(stat_res_stc, f)

    # get all t-scores to estimate the max values for the plotting scale
    vmax = np.max(np.concatenate([v[0] for k, v in stat_res_stc[ses_id].items()]))
    cortical_mask_ind = make_cortical_mask(opt_local)

    # ----------------------------------------------------- plot ----------------------------------------------------- #
    for i_ang, ang_res in enumerate(opt_local['ang_res']):
        # get the stat results
        t_obs, cluster_list, cluster_pval, h0 = stat_res_stc[ses_id][str(ang_res)]

        data_to_plot = t_obs.copy()
        data_to_plot[cortical_mask_ind] = 0
        #
        t = mne.VolSourceEstimate(data_to_plot.copy(), vertices=[src_fs[0]['vertno']],
                                  tmin=0, tstep=0, subject='sub-fsaverage').as_volume(src_fs, dest='mri', format='nifti1')

        cluster_mask = np.zeros_like(t_obs)
        # find the index of the most significant cluster
        good_cluster_ind = np.argmin(cluster_pval)
        # find the index of the voxels in the cluster
        vox_cluster_ind = cluster_list[good_cluster_ind][0]
        # keep only significant voxels
        cluster_mask[vox_cluster_ind] = 1
        # make a nifti object
        mask = mne.VolSourceEstimate(cluster_mask.copy(), vertices=[src_fs[0]['vertno']],
                                     tmin=0, tstep=0, subject='sub-fsaverage').as_volume(src_fs, dest='mri',
                                                                                         format='nifti1')
        # remove the time dimension
        mask = nib.Nifti1Image(mask.dataobj.squeeze(), mask.affine)
        # load the template image and resample the stat image and the mask to the template
        template = load_mni152_template(resolution=5)
        resampled_t = resample_to_img(t, template, interpolation='nearest')
        resampled_mask = resample_to_img(mask, template, interpolation='nearest')
        # plot
        display = plotting.plot_glass_brain(resampled_t, title=f"{ang_res}°", symmetric_cbar=True, colorbar=True, plot_abs=False,
                                            vmax=vmax, vmin=-vmax, cmap=opt_local['colors']['topo'])
        display.add_contours(resampled_mask, colors=['#35155D'], linewidths=2,
                             levels=np.array([1]))

        display.savefig(f"{opt_local['figPath']}cluster_{ses_id}_{opt_local['stc_method']}_"
                        f"{opt_local['cluster_method']}_{ang_res}_{metric}_{opt_local['stc_ch_type']}.png", dpi=500)


