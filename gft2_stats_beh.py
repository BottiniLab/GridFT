"""
functions for analysis of behavioral data
@author: giulianogiari
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from gft2_utils import significance_bar, raincloud


def load_beh_data(ses_id, opt_local):
    """
    Load behavioral data and store in a pandas dataframe
    """
    group_fname = f"{opt_local['behPath']}ses-{ses_id}_behav_data.csv"

    if not os.path.exists(group_fname):

        exp_id = 'exp_2' if ses_id == 'meg' else 'exp_1'
        behav_data = {k: [] for k in ['RT', 'ang_res', 'sub_id', 'correct', 'trl_id', 'ses_id']}
        if exp_id == 'exp_2':
            behav_data['dot_id'] = []

        # loop through subjects
        for sub_id in opt_local[exp_id]['subj_list']:

            if sub_id == 'sub-20020627GBVD':
                print('participant', sub_id, 'excluded')
                continue

            for i_block in range(1, opt_local['n_blocks'] + 1):
                # find all files of this participant/sessions
                if exp_id == 'exp_2':
                    beh_fname = glob.glob(f"{opt_local['behPath']}{sub_id.split('-')[1]}/{sub_id.split('-')[1]}"
                                          f"_BLOCK0{i_block}.mat")[0]
                else:
                    beh_fname = glob.glob(f"{opt_local['behPath']}{sub_id.split('-')[1]}_{ses_id[0]}/" 
                                          f"{sub_id.split('-')[1]}_BLOCK0{i_block}_{ses_id[0]}.mat")[0]

                # we now load the matlab structure obtained during the experiment
                # this will be a object array of size n_blocks, with each block containing
                # fields: subID, ang_res, RT, correct, pressed
                behav_experiment = sio.loadmat(beh_fname, struct_as_record=False, squeeze_me=True)['behav_experiment']

                # extract the responses and reaction times for each dot presented independently
                for i_trl, trial in enumerate(behav_experiment):

                    if ses_id == 'meg':
                        for i_dot in range(2):
                            behav_data['sub_id'].append(f"sub-{str(trial.subID)}")
                            behav_data['ang_res'].append(str(trial.cond_id))
                            behav_data['RT'].append(trial.RT[i_dot])
                            behav_data['correct'].append(trial.correct[i_dot])
                            behav_data['trl_id'].append(f"b{i_block}_trl{i_trl}")
                            behav_data['dot_id'].append(i_dot+1)
                            behav_data['ses_id'].append(ses_id)

                    else:
                        # account for the fact that some subjects have different subID
                        if str(trial.subID) == '19990711EEBN': sub_id = 'sub-19960711EEBN'
                        elif str(trial.subID) == '1981030GBGL': sub_id = 'sub-19851030GBGL'
                        else: sub_id = f"sub-{str(trial.subID)}"
                        behav_data['sub_id'].append(sub_id)
                        behav_data['ang_res'].append(str(trial.ang_res))
                        behav_data['RT'].append(trial.RT)
                        behav_data['trl_id'].append(f"b{i_block}_trl{i_trl}")
                        behav_data['correct'].append(trial.correct)
                        behav_data['ses_id'].append(ses_id)

        # then transform to pandas dataframe and save
        df_behav = pd.DataFrame.from_dict(behav_data).astype({'sub_id': object, 'ang_res': object, 'RT': float,
                                                              'correct': int, 'trl_id': object, 'ses_id': object})
        df_behav.to_csv(group_fname)
        return df_behav
    else:
        return pd.read_csv(group_fname, index_col=0)


def compute_accuracy(ses_id, opt_local):
    """
    Load behavioral data and average the responses per participant
    """
    group_fname = f"{opt_local['behPath']}ses-{ses_id}_behav_accuracy.csv"
    if not os.path.exists(group_fname):
        # load data
        df_behav = load_beh_data(ses_id, opt_local)
        # average data of the same condition for each subject
        df_avg = df_behav.groupby(['sub_id']).mean().reset_index().rename(columns={'correct': 'accuracy'})
        df_avg['ang_res'] = 'Total'
        df_avg = pd.concat([df_avg,
                            df_behav.groupby(['sub_id', 'ang_res']).mean().reset_index().rename(columns={'correct': 'accuracy'})])
        if ses_id == 'meg':
            df_avg.drop(columns='dot_id', inplace=True)
        df_avg.to_csv(f"{opt_local['behPath']}ses-{ses_id}_beh_accuracy.csv")
        return df_avg
    else:
        return pd.read_csv(group_fname, index_col=0)


def plot_accuracy(ses_id, opt_local):
    """
    plot accuracy values for each session
    """
    df = compute_accuracy(ses_id, opt_local)
    m = df.loc[df['ang_res']=='Total', 'accuracy'].mean()
    sd = np.std(df.loc[df['ang_res']=='Total', 'accuracy'])
    print(ses_id, 'accuracy M=', m *100, 'SD=', np.std(df.loc[df['ang_res']=='Total', 'accuracy'] * 100 ))
    ind = df.loc[df['ang_res']=='Total', 'accuracy'].values < m-sd*2
    if any(ind):
        # subject id of the subject to exclude
        sub_list = df.loc[df['ang_res'] == 'Total', 'sub_id'][ind].values
        print(sub_list, '2SD below mean')
        # get the indices of data points to exclude
        ind = df['sub_id'].isin(sub_list).values

    fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharex=True)
    for i_measure, measure in enumerate(['accuracy']):
        ax = raincloud(df.copy(), 'ang_res', measure, axes, opt_local['colors']['beh'][ses_id], ['Total', 15, 30],
                       outliers=ind)
        # ttest
        t, p = ttest_rel(df.loc[df['ang_res'] == 15, measure].values,
                         df.loc[df['ang_res'] == 30, measure].values)
        print(ses_id, measure, 't(', len(df.sub_id.unique())-1,')=', t, 'p=', p)
        significance_bar(t, p, [1.1, 1.9],
                         [np.max(df[measure].values) * 1.01, np.max(df[measure].values) * 1.01], ax)
        if measure == 'accuracy':
            if ses_id != 'meg':
                ax.hlines(.5, -.5, 2.5, linestyle='--', color='gray', alpha=.9)
                ax.set_ylim(.48, 1)
            ax.set_ylabel('Accuracy (%)')
        elif measure == 'RT':
            ax.set_ylabel('Reaction times (s)')
            #ax.set_ylim(.2, 1)
    plt.setp(axes, xticklabels=['Overall', '15°', '30°'], xlabel='Angular Resolution')
    ax.set_title(f'{ses_id.title()} session' if ses_id != 'meg' else 'Task performance')
    fig.tight_layout()
    fig.savefig(f"{opt_local['figPath']}beh_{ses_id}_accuracy.tiff", dpi=500, pad_inches=0, bbox_inches='tight',
                transparent=True)
    
