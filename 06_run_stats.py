#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:08:26 2021

@author: giulianogiari

========================================================================================================================
# 							                                                                                           #
#	                         run the statistics at the group level                                                     #
#							                                                                                           #
========================================================================================================================

"""

from gft2_config import opt
from gft2_stats_frequency import cluster_permutation_frq, plot_cluster_permutation_frq, topo_correlation
from gft2_stats_stc import ttest_roi, plot_spectrum_roi, conjunction
from gft2_stats_eye import plot_percentage, plot_heatmap_ang_correlation, plot_correlation, plot_red_dots, plot_heatmap_ang_correlation, plot_heatmap_correlation, eye_meg_correlation
from gft2_stats_beh import plot_accuracy
from gft2_stats_fit import voxel_fit, roi_fit

metric = 'coh'

# behavioral
for ses_id in ['dots', 'lines', 'meg']:
    plot_accuracy(ses_id, opt)
    plot_percentage(ses_id, opt)

# meg
for ses_id in ['dots', 'lines', 'dots+lines', 'meg']:
    # sensor level
    cluster_permutation_frq(ses_id, metric, opt)
    plot_cluster_permutation_frq(ses_id, metric, opt)
    # source
    # anova is run in R
    ttest_roi(ses_id, metric, roi_list, opt)
    if ses_id == 'dots+lines' or ses_id == 'meg':
        voxel_fit(ses_id, metric, opt)
        roi_fit(ses_id, metric, ['MTL'], opt)
        if ses_id == 'dots+lines':
            conjunction(ses_id, metric, 0.005, 'greater', opt)
            plot_spectrum_roi(ses_id, metric, opt)
            eye_meg_correlation(opt)
    elif ses_id == 'dots' or ses_id == 'lines':
        plot_red_dots(ses_id, opt)

plot_heatmap_ang_correlation(opt)
plot_heatmap_correlation(opt)
plot_correlation('dots', opt)
plot_heatmap_ang_correlation(opt)
topo_correlation(metric, opt)


