"""
functions for statistical analyses of eye tracker data
@author: giulianogiari
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from gft2_eye import eye_correlation
from gft2_stats_fit import linear_fit, quadratic_fit
from gft2_stats_stc import _average_ses_roi
from gft2_utils import significance_bar, raincloud
from scipy import stats


def load_eye_perc(ses_id, opt_local):
    """
    Load eye data
    """

    group_fname = f"{opt_local['eyePath']}ses-{ses_id}_eye_percentage.csv"

    if not os.path.exists(group_fname):
        exp_id = 'exp_2' if ses_id == 'meg' else 'exp_1'

        perc_dict = {k: [] for k in ['sub_id', 'ses_id', 'ang_res', 'fixation_perc']}
        for sub_id in opt_local[exp_id]['subj_list']:
            if sub_id in opt_local[exp_id]['to_exclude']:
                print(f"{sub_id} excluded from the analysis")
                continue
            csv_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_percentage.csv"
            csv = pd.read_csv(csv_fname, index_col=0)

            perc_dict['ang_res'].append('Total')
            for k in ['sub_id', 'ses_id', 'fixation_perc']:
                if k == 'fixation_perc':
                    perc_dict[k].append(csv[k].mean())
                else:
                    perc_dict[k].append(csv[k].values[0])

            for ang_res in opt_local['ang_res']:
                ang_df = csv.loc[csv['ang_res' ] == ang_res]
                for k in ['sub_id', 'ang_res', 'ses_id', 'fixation_perc']:
                    if k == 'fixation_perc':
                        perc_dict[k].append(ang_df[k].mean())
                    else:
                        perc_dict[k].append(ang_df[k].values[0])
        perc_df = pd.DataFrame.from_dict(perc_dict)
        perc_df.to_csv(group_fname)
        return perc_df
    else:
        return pd.read_csv(group_fname, index_col=0)


def _average_ses_eye(ses1, ses2, opt_local):
    """ average dots and lines session """
    group_fname = f"{opt_local['eyePath']}ses-{ses1}+{ses2}_eye_percentage.csv"

    if not os.path.exists(group_fname):
        ses1_df = load_eye_perc(ses1, opt_local)
        ses2_df = load_eye_perc(ses2, opt_local)

        # merge dataframes
        df = pd.concat([ses1_df, ses2_df])
        ses_df = df.groupby(['sub_id', 'ang_res']).mean().reset_index()
        ses_df['ses_id'] = f"{ses1}+{ses2}"
        ses_df.to_csv(group_fname)
        return ses_df
    else:
        return pd.read_csv(group_fname, index_col=0)


def plot_percentage(ses_id, opt_local):
    """
    Compute stats and plot percentage fixation
    """
    if ses_id == 'dots+lines':
        perc_df = _average_ses_eye('dots', 'lines', opt_local)
    else:
        perc_df = load_eye_perc(ses_id, opt_local)
    # plot average fixation percentage
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax = raincloud(perc_df, 'ang_res', 'fixation_perc', ax, opt_local['colors']['beh'][ses_id], ['Total', '15', '30'])
    # ttest
    t, p = stats.ttest_rel(perc_df.loc[perc_df['ang_res'] == 15, 'fixation_perc'].values,
                           perc_df.loc[perc_df['ang_res'] == 30, 'fixation_perc'].values)
    print(ses_id, 'fixation time: t(', len(perc_df.sub_id.unique()) - 1, ')=', t, 'p=', p)
    significance_bar(t, p, [1.1, 1.8],
                     [np.max(perc_df['fixation_perc'].values) * 1.01, np.max(perc_df['fixation_perc'].values) * 1.01], ax)
    plt.setp(ax, ylabel='Fixation time (%)', xlabel='Angular resolution', xticklabels=['Overall', '15°', '30°'],
             ylim=(.6, 1.02), yticks=(.6, .7, .8, .9, 1))
    ax.set_title(f'{ses_id.title()} session' if ses_id != 'meg' else 'Gaze behavior')
    fig.tight_layout()
    fig.savefig(f"{opt_local['figPath']}eye_{ses_id}_percentage.tiff", dpi=500, pad_inches=0, bbox_inches='tight',
                transparent=True)

    # plot individual trials percentage
    hist_val = []
    for sub_id in perc_df.sub_id.unique():
        csv_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_percentage.csv"
        csv = pd.read_csv(csv_fname, index_col=0)
        hist_val.extend(csv.fixation_perc.values)
        # plot individual subject
        fig, ax = plt.subplots(1,1, figsize=(4,3))
        ax.hist(csv.fixation_perc, color=opt_local['colors']['beh'][ses_id][-1])
        ax.vlines(.8, 0, ax.get_ylim()[1], ls='--', color='k')
        ax.set_xlabel('Fixation time (%)')
        ax.set_ylabel('Trial Count')
        ax.set_xlim(0, 1)
        ax.set_title(sub_id)
        fig.tight_layout()
        fig.savefig(f"{opt_local['figPath']}eye_{ses_id}_{sub_id}_percentage_hist.png", dpi=500, pad_inches=0,
                    bbox_inches='tight', transparent=True)
        plt.close()
    print(f'{sum(np.array(hist_val)<.8) / len(hist_val) *100} % of trials excluded because below threshold')
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.hist(hist_val, color=opt_local['colors']['beh'][ses_id][-1])
    ax.vlines(.8, 0, ax.get_ylim()[1], ls='--', color='k')
    ax.set_xlabel('Fixation time (%)')
    ax.set_ylabel('Trial Count')
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{opt_local['figPath']}eye_{ses_id}_percentage_hist.tiff", dpi=500, pad_inches=0, bbox_inches='tight',
                transparent=True)
    if ses_id == 'meg': # this participant is excluded
        perc_df = perc_df.drop(perc_df.loc[perc_df['sub_id']=='sub-20010722LLFL'].index)
    print(ses_id, 'fixation percentage M=', perc_df.loc[perc_df['ang_res'] == 'Total', 'fixation_perc'].mean() * 100,
          'SD=', np.std(perc_df.loc[perc_df['ang_res'] == 'Total', 'fixation_perc'] * 100))


def load_eye_corr(ses_id, opt_local):
    """
    load correlation data
    """
    group_fname = f"{opt_local['eyePath']}corr_{ses_id}.csv"
    out_dict = {k: [] for k in ['ang_res', 'sub_id', 'r']}

    if not os.path.exists(group_fname):
        for sub_id in opt_local['exp_1']['subj_list']:
            if sub_id in opt_local['exp_1']['to_exclude']: continue
            corr_dict = eye_correlation(sub_id, ses_id, opt_local)
            for ang_res in corr_dict.keys():
                out_dict['sub_id'].append(sub_id)
                out_dict['ang_res'].append(ang_res)
                out_dict['r'].append(np.mean(corr_dict[ang_res]))
        out_df = pd.DataFrame.from_dict(out_dict)
        out_df.to_csv(group_fname)
    else:  # load the output
        out_df = pd.read_csv(group_fname, index_col=0)
    return out_df


def plot_correlation(ses_id, opt_local):
    """
    plot the correlation results
    """
    assert ses_id == 'dots'
    # load correlation data
    eye_df = load_eye_corr(ses_id, opt_local)
    # plot average fixation percentage
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
    ses_ax = raincloud(eye_df, x='ang_res', y='r', ax=ax,
                       palette=[opt_local['colors']['beh'][ses_id][1], opt_local['colors']['beh'][ses_id][2]],
                       order=eye_df.ang_res.unique(), orientation='v', jitter=.2)
    ses_ax.set_title(f"{ses_id.title()} session" if ses_id != 'meg' else 'Letters session', y=1.1)
    plt.setp(ax, ylabel='Correlation (r)', xlabel='Angular resolution')
    # within ang res stats
    for i_ang, ang_res in enumerate(eye_df['ang_res'].unique()):
        t, p = stats.ttest_1samp(np.arctanh(eye_df.loc[eye_df['ang_res'] == ang_res, 'r'].values), 0)
        print(ses_id, ang_res, t, p)
        significance_bar(t, p, [i_ang-.2, i_ang+.2],
                         [np.max(eye_df['r'].values)*1.07, np.max(eye_df['r'].values)*1.07], ax)
    # between ang res stats
    t, p = stats.ttest_rel(np.arctanh(eye_df.loc[eye_df['ang_res'] == 15, 'r'].values),
                           np.arctanh(eye_df.loc[eye_df['ang_res'] == 30, 'r'].values))
    print(ses_id, 'between', t, p)
    # add significance bar
    significance_bar(t, p, [0, 1], [np.max(eye_df['r'].values)*1.3, np.max(eye_df['r'].values)*1.3], ax)
    # add horizontal dashed line at 0
    ax.axhline(0, ls='--', color='k', alpha=.5)
    fig.tight_layout()
    fig.savefig(f"{opt_local['figPath']}eye_{ses_id}_correlation.png", dpi=500, pad_inches=0, bbox_inches='tight',
                transparent=True)


def plot_red_dots(ses_id, opt_local):
    """

    :param ses_id:
    :param opt_local:
    :return:
    """
    # load the data
    df = pd.DataFrame()
    for sub_id in opt_local['exp_1']['subj_list']:
        if not sub_id in opt_local['exp_1']['to_exclude']:
            csv_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_desc-dotOnset.csv.gz"
            csv = pd.read_csv(csv_fname)
            #csv = _cleanup(csv, sub_id, ses_id, opt_local)
            # average across trials
            csv = csv.groupby(['times', 'dot_id', 'ang_res']).mean().reset_index()
            # add subejct id
            csv['sub_id'] = sub_id
            # concatente dfs
            df = pd.concat([df, csv])
    # plot the distance of the eyes from the dots
    m = df.groupby(['times', 'dot_id', 'ang_res']).mean().reset_index()
    sd = df.groupby(['times', 'dot_id', 'ang_res']).std().reset_index()
    # plot the time series for each condition
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for i_ang, ang_res in enumerate(m.ang_res.unique()):
        # get the data
        m_dot = m.loc[(m['ang_res'] == ang_res)].groupby(['times']).mean().reset_index()
        sd_dot = sd.loc[(sd['ang_res'] == ang_res)].groupby(['times']).mean().reset_index()
        # plot the data
        ax[i_ang].plot(m_dot['times'].values, m_dot['dist'].values, color=opt_local['colors']['beh'][ses_id][i_ang+1])
        # plot the shaded area
        ax[i_ang].fill_between(m_dot['times'].values, m_dot['dist'].values - sd_dot['dist'].values,
                               m_dot['dist'].values + sd_dot['dist'].values,
                               color=opt_local['colors']['beh'][ses_id][i_ang+1], alpha=.2)
        # plot vertical line at 0
        ax[i_ang].axvline(0, ls='--', color='k', alpha=.5)
        # add text at time 0 with label 'dot onset'
        ax[i_ang].text(0.05, 320, 'Target dot onset', ha='center', rotation=90, color='k', fontsize=9)
        ax[i_ang].set_xlabel('Time (s)')
        ax[i_ang].set_ylabel('Distance from target dot (pixel)')
        ax[i_ang].set_title(f"{ang_res}° resolution")
    plt.suptitle(f"{ses_id.title()} session")
    plt.setp(ax, xlim=(-1, 1), ylim=(250, 375))
    plt.savefig(f"{opt_local['figPath']}ses-{ses_id}_desc-redDots.png", pad_inches=0, bbox_inches='tight',
                    transparent=True, dpi=500)
    plt.close(fig)

    # define time windows
    time_before = [-0.050, 0]
    time_after = [0.05, 0.1]
    # take the median position in the time windows after the dot onset for each participant
    med_post = df.loc[(df['times'] >= time_after[0]) & (df['times'] <= time_after[1])].groupby(
        ['sub_id', 'ang_res']).mean().reset_index()
    # take the median position in the time windows before the dot onset for each participant
    med_pre = df.loc[(df['times'] >= time_before[0]) & (df['times'] <= time_before[1])].groupby(
        ['sub_id', 'ang_res']).mean().reset_index()
    for i_ang, ang_res in enumerate(m.ang_res.unique()):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        # get the data of this angular resolution
        post_ang = med_post.loc[(med_post['ang_res'] == ang_res)]
        post_ang['times'] = 'post'
        pre_ang = med_pre.loc[(med_pre['ang_res'] == ang_res)]
        pre_ang['times'] = 'pre'
        # plot the data
        raincloud(pd.concat([pre_ang, post_ang]), x='times', y='dist', order=['pre', 'post'],
                  ax=ax, palette=[opt_local['colors']['beh'][ses_id][i_ang+1], opt_local['colors']['beh'][ses_id][i_ang+1]])
        # t-test between the pre and post eye position
        t, p = stats.ttest_rel(pre_ang['dist'].values, post_ang['dist'].values)
        significance_bar(t, p, [0, 1], [np.max([pre_ang['dist'].values.mean(), post_ang['dist'].values.mean()]) * 1.2,
                                        np.max([pre_ang['dist'].values.mean(), post_ang['dist'].values.mean()]) * 1.2],
                         ax)
        print(ang_res, t, p)
        ax.set_title(f"{ang_res}° resolution")
        plt.setp(ax, xticks=(0, 1), xticklabels=(f'Before {time_before}', f'After {time_after}'),
                 ylabel='Distance from target dot (pixel)', xlabel='Time (s)')
        plt.savefig(f"{opt_local['figPath']}ses-{ses_id}_{ang_res}_desc-redDotsTtest_mean.png", pad_inches=0,
                    bbox_inches='tight', transparent=True, dpi=500)
        plt.close(fig)


def plot_heatmap(ses_id, opt_local):
    import matplotlib as mpl
    from gft2_eye import compute_heatmaps
    for sub_id in opt_local['exp_1']['subj_list']:
        if sub_id in opt_local['exp_1']['to_exclude']: continue

        hmap_dict = compute_heatmaps(sub_id, ses_id, opt_local)
        # select the central fixation window
        in_fixation = slice(int(opt_local['fixWin'][0]), int(opt_local['fixWin'][2])), \
                      slice(int(opt_local['fixWin'][1]), int(opt_local['fixWin'][3]))
        # initialize figure
        mpl.rc('axes.spines', right=True, top=True)
        mpl.rc('ytick.major', width=0)
        mpl.rc('xtick.major', width=0)
        for ang_res in hmap_dict.keys():
            # initialize figure
            fig, ax = plt.subplots(2, 3 if ang_res == '30' else 6, figsize=(4, 4), sharex=True, sharey=True)
            ax = ax.ravel()
            for i, angle in enumerate(hmap_dict[ang_res].keys()):
                # get the data in the central fixation window
                hmap = hmap_dict[ang_res][angle][in_fixation[0], in_fixation[1]]
                # plot heatmap
                ax[i].imshow(hmap, origin='lower', cmap='viridis', alpha=1)
                # set title
                ax[i].set_title(f"{angle}°")
            plt.setp(ax, xticks=[0, hmap.shape[0]//2, hmap.shape[0]], yticks=[0, hmap.shape[1]//2, hmap.shape[1]],
                     xticklabels=[-4.5, 0, 4.5], yticklabels=[-4.5, 0, 4.5])
            # add one label to the x axis
            fig.text(0.6, 0.14, 'Horizontal position (°)', ha='center')
            # add one label to the y axis
            fig.text(0, 0.5, 'Vertical position (°)', va='center', rotation='vertical')
            #fig.tight_layout()
            plt.subplots_adjust(hspace=-.1, wspace=.35, top=1, bottom=.1, left=.12, right=.99)
            fig.savefig(f"{opt_local['figPath']}{sub_id}_ses-{ses_id}_{ang_res}_hmap.png", pad_inches=0, bbox_inches='tight',
                        transparent=True, dpi=500)


def plot_heatmap_ang_correlation(opt_local):
    """

    :param ses_id:
    :param opt_local:
    :return:
    """
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for ses_id in ['dots', 'lines']:
        # load the data
        df = pd.DataFrame()
        for sub_id in opt_local['exp_1']['subj_list']:
            if not sub_id in opt_local['exp_1']['to_exclude']:
                csv_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_hmap_corr.csv"
                csv = pd.read_csv(csv_fname)
                df = pd.concat([df, csv])
        df = df.groupby(['sub_id', 'ang_res', 'ang_diff', 'ang_1', 'ang_2']).mean().reset_index()
        # plot
        for i, ang_res in enumerate(opt_local['ang_res']):
            df_ang = df.loc[df['ang_res'] == ang_res]
            print(ang_res, df_ang.groupby('sub_id').mean()['hmap_r'].mean(),
                  df_ang.groupby('sub_id').mean()['hmap_r'].std())
            im = sns.regplot(data=df_ang.groupby(['ang_diff', 'ang_1', 'ang_2']).mean().reset_index(),
                             x="ang_diff", y="hmap_r", ax=ax[i],
                             scatter_kws={'edgecolor': 'k', 'color': opt_local['colors']['beh'][ses_id][i+1],
                                          's': 30, 'alpha': .5},
                             line_kws={'color': opt_local['colors']['beh'][ses_id][i+1]})
            # set the yticks to 4 decimals
            im.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            im.set_title(f"{ang_res}° resolution")
            # compute correlation between heatmaps correlation and angular difference
            r = []
            for sub_id in df_ang.sub_id.unique():
                sub_df = df_ang.loc[df_ang['sub_id'] == sub_id]
                r.append(stats.pearsonr(sub_df['ang_diff'], sub_df['hmap_r'])[0])
            # test against zero
            t, p = stats.ttest_1samp(np.arctanh(r), 0)
            print(f"Correlation between heatmaps correlation and angular difference: {np.mean(r)} ± {np.std(r)}")
            print(f"t({len(r) - 1}) = {t}, p = {p}")
    plt.setp(ax, xlabel='Angular difference (°)', ylabel='Correlation (r)')
    fig.tight_layout()
    fig.savefig(f"{opt_local['figPath']}hmap_ang_corr.png", dpi=500, bbox_inches='tight')


def plot_heatmap_correlation(opt_local):
    """

    :param ses_id:
    :param opt_local:
    :return:
    """
    for ses_id in ['dots', 'lines']:
        # load the data
        df = pd.DataFrame()
        for sub_id in opt_local['exp_1']['subj_list']:
            if not sub_id in opt_local['exp_1']['to_exclude']:
                csv_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_hmap_corr.csv"
                csv = pd.read_csv(csv_fname)
                df = pd.concat([df, csv])
        df = df.groupby(['sub_id', 'ang_res']).mean().reset_index()

        # plot the correlation
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax = raincloud(df, x='ang_res', y='hmap_r', order=[15, 30], ax=ax, palette=opt_local['colors']['beh'][ses_id][1:])
        ax.set_xlabel('Angular resolution (°)')
        ax.set_ylabel('Correlation (r)')
        ax.set_title(f"{ses_id.title()} session")
        ax.set_ylim(.93, 1)
        fig.tight_layout()
        fig.savefig(f"{opt_local['figPath']}ses-{ses_id}_hmap_corr.png", pad_inches=0, bbox_inches='tight',
                transparent=True, dpi=500)


def eye_meg_correlation(opt_local):
    """
    Compute correlation between the slope of the grid-like effect and the slope of the eye effect
    :param opt_local:
    :return:
    """
    # read the meg data
    _, _, roi_df = _average_ses_roi('dots', 'lines', 'coh', opt_local)
    # select the data in the MTL roi
    roi_df = roi_df[roi_df['roi'] == 'MTL']
    # compute slopes in the MTL for each subject
    meg_df = {k: [] for k in ['sub_id', 'ang_res', 'hemi', 'slope']}
    for sub_id in roi_df.sub_id.unique():
        sub_df = roi_df[roi_df['sub_id'] == sub_id]
        for ang_res in sub_df.ang_res.unique():
            ang_df = sub_df[sub_df['ang_res'] == ang_res]
            for hemi in ['lh', 'rh']:
                hemi_df = ang_df[ang_df['hemi'] == hemi]
                if ang_res == 15:
                    mdl = quadratic_fit(hemi_df['fold'].values[:, None], hemi_df['coh'].values)[2]
                else:
                    mdl = linear_fit(hemi_df['fold'].values[:, None], hemi_df['coh'].values)[2]
                meg_df['sub_id'].append(sub_id)
                meg_df['ang_res'].append(ang_res)
                meg_df['hemi'].append(hemi)
                meg_df['slope'].append(mdl.params[1])
    meg_df = pd.DataFrame.from_dict(meg_df)
    # load the data from the eye tracking
    eye_df = {k: [] for k in ['sub_id', 'ang_res', 'ses_id', 'slope']}
    for ses_id in ['lines', 'dots']:
        for sub_id in opt_local['exp_1']['subj_list']:
            if not sub_id in opt_local['exp_1']['to_exclude']:
                csv_fname = f"{opt_local['eyePath']}{sub_id}_ses-{ses_id}_hmap_corr.csv"
                csv = pd.read_csv(csv_fname)
                for ang_res in csv.ang_res.unique():
                    # select data of this ang_res
                    csv_ang = csv[csv['ang_res'] == ang_res]
                    # compute correlation and slope
                    _, _, model = linear_fit(csv_ang['ang_diff'].values, csv_ang['hmap_r'].values)
                    # compute correlation and slope using scipy
                    eye_df['sub_id'].append(sub_id)
                    eye_df['ang_res'].append(ang_res)
                    eye_df['ses_id'].append(ses_id)
                    eye_df['slope'].append(model.params[1])
    eye_df = pd.DataFrame.from_dict(eye_df)
    # average across sessions
    eye_df = eye_df.groupby(['sub_id', 'ang_res']).mean().reset_index()
    # merge eye_df and meg_df
    df = pd.merge(meg_df, eye_df, on=['sub_id', 'ang_res'], suffixes=('_meg', '_eye'))
    # compute correlation between slopes in MEG and slopes in eye tracking, separately for the hemi and ang_res
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for i_ang, ang_res in enumerate(df.ang_res.unique()):
        for hemi in df.hemi.unique():
            df_ang_hemi = df[(df['ang_res'] == ang_res) & (df['hemi'] == hemi)]
            r, p = stats.pearsonr(df_ang_hemi['slope_meg'].values, df_ang_hemi['slope_eye'].values)
            # plot the data
            print(f"{ang_res} - {hemi}: r={r:.2f}, p={p:.3f}")
            sns.regplot(x='slope_meg', y='slope_eye', data=df_ang_hemi, ax=ax[i_ang], color=opt_local['colors']['eye-grid'][hemi],
                        label="Left Hemisphere" if hemi == 'lh' else "Right Hemisphere")
        ax[i_ang].set_title(f"{ang_res}° resolution")
        ax[i_ang].legend()
    plt.setp(ax, xlabel="Grid-like effect", ylabel="Spatial gaze modulation")
    fig.savefig(f"{opt_local['figPath']}eye-meg_correlation.png", dpi=500, bbox_inches='tight')

    