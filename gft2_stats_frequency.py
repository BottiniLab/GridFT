"""
functions for statistics on the frequency analysis
@author: giuliano giari, giuliano.giari@gmail.com
"""

import glob
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns
import warnings
from gft2_frequency import _amplitude, _power, read_frq, pick_frequency
from gft2_utils import significance_bar
from mne.channels.layout import _merge_grad_data
from mne.stats import permutation_cluster_1samp_test
from mne.viz import plot_topomap
from scipy import stats, spatial


def load_frq_data(ses_id, opt_local):
    """
    Load frequency data of the specified session
    :return: dict
    """
    group_fname = f"{opt_local['frqPath']}ses-{ses_id}_frq_desc-coh.h5"
    exp_id = 'exp_2' if ses_id == 'meg' else 'exp_1'
    if not os.path.exists(group_fname):
        spctrm = {ses_id: {str(ang_res): [] for ang_res in opt_local['ang_res']}}
        freqs = {ses_id: {str(ang_res): [] for ang_res in opt_local['ang_res']}}
        for i_sub, sub_id in enumerate(opt_local[exp_id]['subj_list']):
            if sub_id in opt_local[exp_id]['to_exclude']:
                print(sub_id, 'excluded')
                continue
            frq_fname = f"{opt_local['frqPath']}{sub_id}_ses-{ses_id}_task-task_" \
                        f"desc-{opt_local['coh_method']}_coh_frq.h5"
            print(f"loading {sub_id} {ses_id}")
            frq_list = read_frq(frq_fname)

            # initialize the data dict here to use the proper event id
            for frq_event in frq_list:
                """
                # to check channel position
                # print(np.array([ch['loc'][:3] for ch in sub_frq[event_id].info['chs']])[293, :])
                """
                ang_res = str(list(frq_event.event_id.values())[0])
                spctrm[ses_id][ang_res].append(frq_event.pick_types(meg=True).data.squeeze())
                freqs[ses_id][ang_res] = frq_event.freqs
        info = frq_event.info
        # save to h5
        f = h5py.File(group_fname, 'w')
        for ang_res in opt_local['ang_res']:
            g = f.create_group(name=f'/{ses_id}/{ang_res}')
            g.create_dataset(name='freqs', data=freqs[ses_id][str(ang_res)])
            g.create_dataset(name='spctrm', data=spctrm[ses_id][str(ang_res)])
        f.close()
    else:
        warnings.warn(f'loading {group_fname}')
        # load the data to dictionary
        f = h5py.File(group_fname, 'r')
        spctrm = {ses_id: {str(ang_res): [] for ang_res in opt_local['ang_res']}}
        freqs = {ses_id: {str(ang_res): [] for ang_res in opt_local['ang_res']}}
        for ang_res in opt_local['ang_res']:
            spctrm[ses_id][str(ang_res)] = np.array(f.get(f'/{ses_id}/{str(ang_res)}/spctrm'))
            freqs[ses_id][str(ang_res)] = np.array(f.get(f'/{ses_id}/{str(ang_res)}/freqs'))
        f.close()
        # load info
        info = read_frq(sorted(glob.glob(f"{opt_local['frqPath']}/*_ses-{ses_id}_task-task_"
                                         f"desc-*_frq.h5"))[0])[0].info
    return spctrm, freqs, info



def cluster_permutation_frq(ses_id, opt_local):
    """
    Load data and do cluster permutation test
    :param ses_id:
    :param opt_local:
    :return:
    """
    # load data
    if ses_id == 'dots+lines':
        spctrm1, freqs, info = load_frq_data('dots', opt_local)
        spctrm2, freqs, info = load_frq_data('lines', opt_local)
        spctrm = {ses_id: {str(ang_res): (spctrm1['dots'][str(ang_res)]+spctrm2['lines'][str(ang_res)]) / 2
                            for ang_res in opt_local['ang_res']}}
        freqs = {ses_id: {str(ang_res): freqs['lines'][str(ang_res)] for ang_res in opt_local['ang_res']}}
    else:
        spctrm, freqs, info = load_frq_data(ses_id, opt_local)
    # and read the channel connectivity
    # we use the mags connectivity as we will anyway have 102 channels after averaging grads
    connectivity, _ = mne.channels.read_ch_adjacency(opt_local['prgPath'] + 'neighbours.mat')
    # and create info
    ch_type_ind = mne.io.pick.channel_indices_by_type(info, 'mag')['mag']
    # ----------------------------------------- cluster permutation test --------------------------------------------- #
    threshold = -stats.distributions.t.ppf(opt_local['p_thresh'] / 2., np.shape(spctrm[ses_id]['15'])[0] - 1)
    
    stat_res_sensor = {ses_id: {ang_res: {ch_type: {} for ch_type in ['mag']}
                                for ang_res in spctrm[ses_id].keys()}}
    for i_ang, ang_res in enumerate(opt_local['ang_res']):
        # select these channels
        ch_spctrm = np.array(spctrm[ses_id][str(ang_res)])[:, ch_type_ind, :]
        # and the frequencies of interest
        ch_foi_dict = {foi_id: ch_spctrm[..., pick_frequency(freqs[ses_id][str(ang_res)], foi)]
                       for foi_id, foi in opt_local['foi'][str(ang_res)].items()}
        # for each contrast
        for i_contrast, contrast_id in enumerate(opt_local['cont_list'][str(ang_res)]):
            # compute difference between conditions
            delta_foi = ch_foi_dict[contrast_id.split('>')[0]] - ch_foi_dict[contrast_id.split('>')[1]]
            # do the test
            stat_res_sensor[ses_id][str(ang_res)][ch_type][contrast_id] = permutation_cluster_1samp_test(
                delta_foi,
                adjacency=connectivity,
                tail=opt_local['tail'],
                threshold=threshold,
                out_type='indices',
                n_permutations=opt_local['n_perms'])
    # save
    with open(f"{opt_local['frqPath']}{ses_id}_tail-{opt_local['tail']}_perm-{opt_local['n_perms']}_cluster_test.p", 'wb') as f:
        pickle.dump(stat_res_sensor, f)


def plot_cluster_permutation_frq(ses_id, opt_local):
    """
    plot cluster permutation results
    :param ses_id:
    :param opt_local:
    :return:
    """
    # load statistics results and data
    with open(f"{opt_local['frqPath']}{ses_id}_tail-{opt_local['tail']}_perm-{opt_local['n_perms']}_cluster_test.p", 'rb') as f:
        stat_res_sensor = pickle.load(f)
    if ses_id == 'dots+lines':
        spctrm1, freqs, info = load_frq_data('dots', opt_local)
        spctrm2, freqs, info = load_frq_data('lines', opt_local)
        spctrm = {ses_id: {str(ang_res): (spctrm1['dots'][str(ang_res)] + spctrm2['lines'][str(ang_res)]) / 2
                               for ang_res in opt_local['ang_res']}}
        freqs = {ses_id: {str(ang_res): freqs['lines'][str(ang_res)] for ang_res in opt_local['ang_res']}}
    else:
        spctrm, freqs, info = load_frq_data(ses_id, opt_local)
    # and create info
    ch_type_ind = mne.io.pick.channel_indices_by_type(info, 'mag')['mag']
    info = mne.pick_info(info, ch_type_ind['mag'], copy=True)

    for i_ang, ang_res in enumerate(opt_local['ang_res']):

        # select these channels
        ch_spctrm = np.array(spctrm[ses_id][str(ang_res)])[:, ch_type_ind, :]
        # and the frequencies of interest
        ch_foi_dict = {foi_id: ch_spctrm[..., pick_frequency(freqs[ses_id][str(ang_res)], foi)]
                       for foi_id, foi in opt_local['foi'][str(ang_res)].items()}
        # for each contrast
        for i_contrast, contrast_id in enumerate(opt_local['cont_list'][str(ang_res)]):

            delta_foi = ch_foi_dict[contrast_id.split('>')[0]] - ch_foi_dict[contrast_id.split('>')[1]]
            # get clusters info
            t_map, clusters, p_values, h0 = stat_res_sensor[ses_id][str(ang_res)]['mag'][contrast_id]

            good_cluster_inds = np.arange(len(clusters))
            # ---------------------------------------- loop over clusters ---------------------------------------- #
            mask = np.zeros((t_map.shape[0], 1), dtype=bool)
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                # unpack cluster information, get unique indices
                thispval = p_values[clu_idx]
                print(ses_id, ang_res, contrast_id, 'cluster n.', clu_idx, ' p = ', thispval)
                if thispval > 0.05:
                    continue
                else:
                    ch_inds = np.squeeze(clusters[clu_idx])
                # create spatial mask highlighting the sensors that belong to the cluster
                mask[ch_inds, :] = True

            # ------------------------------------- delta SNR ------------------------------------------------ #
            fig = plt.figure(figsize=(3, 3))
            image, _ = plot_topomap(delta_foi.mean(0), info, cmap=opt_local['colors']['topo'], mask=mask,
                                    vmin=-np.max(delta_foi.mean(0)), vmax=np.max(delta_foi.mean(0)),
                                    axes=fig.add_axes([.1, .06, .8, 1.1]),
                                    mask_params=dict(marker='o', markerfacecolor=opt_local['colors']['topo_marker'],
                                                     markeredgecolor='k', linewidth=0, markersize=8))
            # add colorbar
            cbar = plt.colorbar(image, cax=fig.add_axes([.2, .13, .6, 0.04]), orientation='horizontal')
            # left: fig.add_axes([.85, 0.3, 0.03, 0.38])
            if ses_id == 'meg':
                cbar_label = f"$ITC_{{Cont{contrast_id.split('>')[0]}}}$ - " \
                             f"$ITC_{{Cont{contrast_id.split('>')[1]}}}$"
            else:
                cbar_label = f"$ITC_{{{contrast_id.split('>')[0]}}}$ - " \
                             f"$ITC_{{{contrast_id.split('>')[1]}}}$"
            cbar.set_label('ΔITC')
            cbar.ax.tick_params(width=1)
            fig.tight_layout()
            image.axes.set_title(f"{ang_res}°: {cbar_label}")
            fig.savefig(f"{opt_local['figPath']}sens_topo_{ses_id}_ang-{ang_res}_"
                        f"{contrast_id}_mag.tiff", dpi=500, bbox_inches="tight")
            

def topo_correlation(opt_local):
    """
    pairwise correlation between the topographies of the 6>4 contrast
    :param opt_local:
    :return:
    """
    # load the data and store in a dictionary
    spctrm_dots = load_frq_data('dots', opt_local)[0]
    spctrm_lines, freqs, info = load_frq_data('lines', opt_local)
    spctrm = {"dots+lines": {str(ang_res): (spctrm_dots['dots'][str(ang_res)] + spctrm_lines['lines'][str(ang_res)]) / 2
              for ang_res in opt_local['ang_res']}}
    spctrm = spctrm | load_frq_data('meg', opt_local)[0]
    freqs = {str(ang_res): freqs['lines'][str(ang_res)] for ang_res in opt_local['ang_res']}
    del spctrm_lines, spctrm_dots
    # keep only magnetometers
    mag_ind = mne.io.pick.channel_indices_by_type(info, 'mag')['mag']
    spctrm = {ses_id: {str(ang_res): spctrm[ses_id][str(ang_res)][:, mag_ind, :] for ang_res in opt_local['ang_res']}
              for ses_id in ["dots+lines", "meg"]}
    # compute contrasts 6>4 in each experiment/angular resolution
    contrast_dict = {ses_id: dict() for ses_id in ["dots+lines", "meg"]}
    for ses_id in ["dots+lines", "meg"]:
        for ang_res in opt_local["ang_res"]:
            ind6 = pick_frequency(freqs[str(ang_res)], opt_local['foi'][str(ang_res)]['6'])
            ind4 = pick_frequency(freqs[str(ang_res)], opt_local['foi'][str(ang_res)]['4'])
            contrast_dict[ses_id][ang_res] = spctrm[ses_id][str(ang_res)][..., ind6] - \
                                             spctrm[ses_id][str(ang_res)][..., ind4]
    # compute correlation within participant between 15 and 30
    corr_dict = {ses_id: [] for ses_id in ["dots+lines", "meg"]}
    for ses_id in ["dots+lines", "meg"]:
        for i_sub in range(contrast_dict[ses_id][15].shape[0]):
            corr_dict[ses_id].append(
                stats.spearmanr(contrast_dict[ses_id][15][i_sub, :], contrast_dict[ses_id][30][i_sub, :])[0])
    # plot and statistics
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for i_ses, ses_id in enumerate(["dots+lines", "meg"]):
        ax.bar(i_ses, np.mean(corr_dict[ses_id]), yerr=stats.sem(corr_dict[ses_id]),
               edgecolor='k', color=np.array([248, 250, 238])/255, error_kw={'capsize': 8})
        ax.plot(i_ses + np.random.uniform(-.2, .2, len(corr_dict[ses_id])), corr_dict[ses_id], '.', alpha=.5,
                color=np.array([91, 79, 143])/255, ms=15, markeredgewidth=2)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Grid-like response correlation between angular resolutions')
    ax.set_xticks([0, 1])
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[np.arange(0, len(yticks), 2)])
    ax.tick_params(length=0, axis='x')
    ax.set_xticklabels(['Spatial', 'Non-spatial'])
    # statistics
    ymax = np.max([np.max(corr_dict['dots+lines']), np.max(corr_dict['meg'])])
    ymin = np.min([np.min(corr_dict['dots+lines']), np.min(corr_dict['meg'])])
    t, p = stats.ttest_ind(np.arctanh(corr_dict['dots+lines']), np.arctanh(corr_dict['meg']))
    print('between', t, p)
    significance_bar(t, p, [0, 1], [ymax*1.2, ymax*1.2], ax)
    t, p = stats.ttest_1samp(np.arctanh(corr_dict['dots+lines']), 0)
    print('within-space', t, p)
    significance_bar(t, p, [-.2, .2], [ymax * 1.1, ymax * 1.1], ax)
    t, p = stats.ttest_1samp(np.arctanh(corr_dict['meg']), 0)
    print('within-non-space', t, p)
    significance_bar(t, p, [.8, 1.2], [ymax * 1.1, ymax * 1.1], ax)
    ax.set_ylim([ymin*1.3, ymax*1.3])
    fig.savefig(f"{opt_local['figPath']}sens_correlation_subject.tiff", dpi=500, pad_inches=0, bbox_inches='tight',
                transparent=True)
    
    for ang_res in [15, 30]:
        topo_space = contrast_dict['dots+lines'][ang_res].mean(0)
        topo_letter = contrast_dict['meg'][ang_res].mean(0)
        r, p = stats.pearsonr(topo_space, topo_letter)
        df = pd.DataFrame.from_dict({"$ITC_{6}$ - $ITC_{4}$": topo_space, "$ITC_{Cont6}$ - $ITC_{Cont4}$": topo_letter})
        im = sns.lmplot(data=df, x="$ITC_{6}$ - $ITC_{4}$", y="$ITC_{Cont6}$ - $ITC_{Cont4}$",
                        scatter_kws={'edgecolor': 'k', 'color': 'gray', 's': 20},
                        line_kws={'color': np.array([204, 158, 245])/255, 'label': f'r={r:1.3f}\np={p:1.3f}'})
        im.ax.legend()
        im.ax.set_title(f"{ang_res}° grid-like response correlation between experiments")
        xlims = [np.min(topo_space), np.max(topo_space)]
        #xlims = ax.ax.get_xlim()
        im.ax.set_xticks(np.linspace(xlims[0], xlims[1], 5))
        im.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        ylims = im.ax.get_ylim()
        im.ax.set_yticks(np.linspace(ylims[0], ylims[1], 5))
        im.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        im.savefig(f"{opt_local['figPath']}sens_between_correlation_{ang_res}.tiff", dpi=500, pad_inches=0,
                   bbox_inches='tight', transparent=True)

        