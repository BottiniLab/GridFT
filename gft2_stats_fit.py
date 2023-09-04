"""
functions for model fitting analyses

@author: giuliano giari, giuliano.giari@gmail.com
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import statsmodels.api as sm
from gft2_stats_stc import load_roi_data, load_stc_data, _average_ses_roi
from nilearn import plotting
from scipy.stats import zscore


def bic_to_bf(bic1, bic0):
    """
    Compute bayes factor approximation from bic as in https://link.springer.com/article/10.3758/BF03194105 and
    https://link.springer.com/article/10.3758/s13428-010-0049-5
    :param bic1:
    :param bic0:
    :return: bayes factor for model 0 over model 1
    """
    return np.exp( (bic1 - bic0) / 2)


def quadratic_fit(X, y_true, model=None):
    """
    Fit linear regression with quadratic terms
    :param X:
    :param y:
    :return:
    """
    # add the quadratic term to the predictors
    X_poly = np.concatenate([X, X**2], 1)
    # z-score
    X_poly = zscore(X_poly, 0)
    # add constant
    X_poly = sm.add_constant(X_poly)
    # fit linear model with quadratic terms
    if model is None:
        model = sm.OLS(y_true, X_poly).fit()
    y_pred = model.predict(X_poly)
    return y_pred, model.bic, model


def linear_fit(X, y_true, model=None):
    """
    Fit linear regression
    :param X:
    :param y:
    :return:
    """
    # z-score
    X = zscore(X, 0)
    # add constant
    X = sm.add_constant(X)
    if model is None:
        model = sm.OLS(y_true, X).fit()
    y_pred = model.predict(X)
    return y_pred, model.bic, model


def roi_fit(ses_id, metric, roi_list, opt_local):
    """
    Fit linear and quadratic regression at the roi level
    :param ses_id:
    :param metric:
    :param opt_local:
    :return:
    """
    # load the data
    if ses_id == 'dots+lines':
        _, _, roi_df = _average_ses_roi('dots', 'lines', metric, opt_local)
    else:
        _, _, roi_df = load_roi_data(ses_id, metric, opt_local)
    # select 15° resolution
    roi_df = roi_df.loc[roi_df['ang_res']==15]
    for hemi in ['lh', 'rh']:
        fig, ax = plt.subplots(1, len(roi_list), figsize=(5*len(roi_list), 4))
        if len(roi_list)==1: ax = [ax]
        for i_roi, roi in enumerate(roi_list):
            # get the data
            roi_csv = roi_df.loc[(roi_df['hemi'] == hemi) & (roi_df['roi'] == roi)]
            X = roi_csv['fold'].astype(int).values.reshape(-1, 1)
            y = roi_csv['coh'].values.reshape(-1, 1)
            # linear regression
            lr_y_hat, lr_bic, _ = linear_fit(X.copy(), y)
            # quadratic fit
            qr_y_hat, qr_bic, _ = quadratic_fit(X.copy(), y)
            # plot dots + best fit line
            ax[i_roi].plot(X + np.random.uniform(-.2, .2, X.shape), y, '.', zorder=5, color='gray', alpha=.5, ms=15,
                           markeredgecolor='k', markeredgewidth=2)
            # calculate bayes factor
            for model_id, m0_bic, m1_bic, y_hat in zip(['LQ', 'QL'],
                                                       [lr_bic, qr_bic], [qr_bic, lr_bic],
                                                       [lr_y_hat, qr_y_hat]):
                # compute bayes factor from bic
                bf01 = bic_to_bf(m1_bic, m0_bic)
                print(ses_id, hemi, roi, model_id, 'BF=', bf01, 'posterior=', bf_to_posterior(bf01))
                ax[i_roi].plot(X.reshape([X.shape[0] // 3, 3])[0, :],
                               y_hat.reshape([X.shape[0] // 3, 3]).mean(0),
                               color=opt_local['colors']['fit'][model_id], ls='-',
                               label="Linear model" if model_id == 'LQ' else 'Quadratic model', lw=5)
            # ax[i_roi].set_title(roi.title() if roi != 'MTL' else 'Medial temporal lobe')
            ax[i_roi].legend()
            ax[i_roi].set_ylim(0, y.max() * 1.1)
            ax[i_roi].set_yticks(np.linspace(0, y.max() * 1.1, 4))
            ax[i_roi].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
            ax[i_roi].set_xticks(np.unique(X))
            ax[i_roi].set_xticklabels([f"{str(f)} [{opt_local['foi']['15'][str(f)]}]"
                                       for f in np.unique(X)])
        #fig.suptitle(f"{'Letters' if ses_id == 'meg' else ses_id} -  {'Right Hemisphere' if hemi == 'rh' else 'Left Hemisphere'}", fontweight='bold', fontsize=20)
        plt.setp(ax, xlabel="Fold [Frequency in Hz]", ylabel='ITC')
        fig.tight_layout()
        fig.savefig(f"{opt_local['figPath']}stc_{ses_id}_{opt_local['stc_method']}_{opt_local['stc_out']}_"
                    f"{hemi}_roi_fit.tiff", dpi=500)


def voxel_fit(ses_id, metric, opt_local):
    """

    :return:
    """

    data_stc, connectivity, src_fs = load_stc_data(ses_id, metric, opt_local)
    # mask_ind = make_cortical_mask(opt_local)

    out_dict = {model_id: {k: [] for k in ['bf', 'mse']} for model_id in ['LQ', 'QL']}

    ang_res = 15
    data_array = np.concatenate([np.squeeze(data_stc[ses_id][ang_res][fold_id])
                                 for fold_id in opt_local['foi'][str(ang_res)].keys()], 0)
    # this is a 2d array of shape (n_sub x n_fold) x n_voxels
    # to check how they are organized np.unique(data_array[:22,...]-data_stc[ses_id][ang_res]['4'])

    X = np.repeat([int(fold_id) for fold_id in opt_local['foi'][str(ang_res)].keys()],
                  np.squeeze(data_stc[ses_id][ang_res]['4']).shape[0]).reshape(-1, 1)

    for i_voxel in range(data_array.shape[1]):

        y = data_array[:, i_voxel]
        # linear regression
        lr_y_hat, lr_bic, _ = linear_fit(X.copy(), y)
        # quadratic fit
        qr_y_hat, qr_bic, _ = quadratic_fit(X.copy(), y)

        # calculate bayes factor
        for model_id, m0_bic, m1_bic, y_hat in zip(['LQ', 'QL'],
                                                      [lr_bic, qr_bic], [qr_bic, lr_bic],
                                                      [lr_y_hat, qr_y_hat]):
            # compute bayes factor from bic
            bf01 = bic_to_bf(m1_bic, m0_bic)
            # store the bayes factor
            out_dict[model_id]['bf'].append(bf01)

    for model_id in ['LQ', 'QL']:

        data_to_plot = np.array(out_dict[model_id]['bf'])
        #data_to_plot[mask_ind] = 0

        clu = mne.VolSourceEstimate(data_to_plot, vertices=[src_fs[0]['vertno']],
                                    tmin=0, tstep=0, subject='sub-fsaverage')
        clu_vol = clu.as_volume(src_fs, dest='mri', format='nifti1')

        thresh = np.max([3, np.percentile(data_to_plot, 99)])
        vmax = np.max(data_to_plot)

        # slices
        im = plotting.plot_stat_map(clu_vol, display_mode='ortho', colorbar=True, symmetric_cbar=False,
                                    output_file=f"{opt_local['figPath']}stc_{ses_id}_{model_id}.tiff",
                                    cut_coords=(30, -16, -25), cmap='Reds',
                                    draw_cross=False, dim=-.1, threshold=thresh, vmax=vmax)
        
