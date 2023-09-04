# @author: giuliano giari, giuliano.giari@gmail.com

# define packages -------------------

library(tidyverse)
library(ggpubr)
library(rstatix)
library(ggdist)
library(gghalves)

# define parameters -------------------
params.stc_method = 'LCMV'
params.stc_out = 'max-power'
params.stc_cov_data = 'noise'
params.stc_cov_method = 'empirical'
params.src_type = 'vol'
params.metric = 'coh'
params.ch_type = 'meg'

#params.stcPath <- '/Volumes/Projects/MEGRID/gft2/pilot_v2/derivatives/stc/'
params.stcPath <- paste('/Users/giulianogiari/Desktop/gridft2/pilot_v2/derivatives/stc/', sep='')

# load data and append in a big df -------------------
ses_csv_flist <- Sys.glob(paste(params.stcPath, 'ses-*_ROI_desc-', params.stc_method, '-', params.stc_out, '-',
                                params.src_type, '-', params.metric, '-', params.ch_type, '.csv', sep=''))

for (i_csv in 1 : length(ses_csv_flist)) {
  # load the csv file
  ses_csv_fname = ses_csv_flist[i_csv]
  ses_csv = read.csv(ses_csv_fname)
  # keep only the regions of interest
  ses_csv <- subset(ses_csv, roi == 'MTL' | roi == 'lateraloccipital' | roi == 'precentral')
  ses_csv[params.metric] <- as.numeric(ses_csv[params.metric][,1])
  # merge dataframes
  if ( i_csv == 1 ) { df <- subset(ses_csv)
  } else { df <- rbind(df, ses_csv) }
  
}

summary(df)

# define the variables as factors
df$X <- NULL
df$sub_id <- factor(df$sub_id)
df$ses_id <- factor(df$ses_id)
df$ang_res <- factor(df$ang_res)
df$hemi <- factor(df$hemi)
df$roi <- factor(df$roi)
df$fold <- factor(df$fold)
#df[paste('log', metric, sep='')] <- log(df[metric])
df = subset(df, select=c('sub_id', 'ses_id', 'hemi', 'ang_res', 'roi', 'fold',  params.metric))
#metric = paste('log', metric, sep='')
head(df)

# ANOVA 2-way: session (dots, lines), periodicity (15° resolution: 4-, 6-, 8-fold; 30° resolution: 4-, 6-fold ) in the MTL 

for ( this_hemi in c('lh', 'rh') ) {
  for ( this_res in c(15, 30) ) {
    # select these options
    anova_df <- subset(df, (ses_id == 'dots' | ses_id == 'lines') & hemi == this_hemi & ang_res == this_res & roi == 'MTL')
    
    # perform the test
    aov_table = anova_test(data = anova_df, dv = all_of(params.metric), wid = sub_id, within =  c('fold', 'ses_id') )
      
    # save output
    aov_fname = paste(params.stcPath, 'aov_within-dotsVSlines_hemi-', this_hemi, '_roi-MTL_res', this_res, sep='') 
    if ( is.null(aov_table$ANOVA) ) {
      write.csv(aov_table, paste(aov_fname, '_anova.csv', sep=''))
   	} else {
      write.csv(aov_table$ANOVA, paste(aov_fname, '_anova.csv', sep=''))
      write.csv(aov_table$`Mauchly's Test for Sphericity`, paste(aov_fname, '_mauchly.csv', sep=''))
      write.csv(aov_table$`Sphericity Corrections`, paste(aov_fname, '_sphericity.csv', sep=''))
    }
  }
}


# ANOVA 2-way: ROI (MTL, lateral occipital, precentral) x periodicity (15° resolution: 4-, 6-, 8-fold; 30° resolution: 4-, 6-fold) within session (dots+lines) 

within = c('fold', 'roi')
for ( this_ses in c('dots+lines', 'meg', 'dots', 'lines') ) {
  for ( this_hemi in c('lh', 'rh') ) {
    for ( this_res in c(15, 30) ) {
        # select the data
        anova_df <- subset(df, ses_id == this_ses & hemi == this_hemi & ang_res == this_res)
        # perform the test
        aov_table = anova_test(data=anova_df, dv='coh', wid=sub_id, within=all_of(within))
        # save output
        aov_fname = paste(params.stcPath, 'aov_within-', this_ses, '_hemi-', this_hemi, '_res-', this_res, sep='') 
        if ( is.null(aov_table$ANOVA) ) {
         write.csv(aov_table, paste(aov_fname, '_anova.csv', sep=''))
        } else {
          write.csv(aov_table$ANOVA, paste(aov_fname, '_anova.csv', sep=''))
          write.csv(aov_table$`Mauchly's Test for Sphericity`, paste(aov_fname, '_mauchly.csv', sep=''))
          write.csv(aov_table$`Sphericity Corrections`, paste(aov_fname, '_sphericity.csv', sep=''))
        }
    }
  }
}

# ANOVA 3-way: experiment (spatial, non-spatial) x roi (MTL, lateraloccipital, precentral) x periodicity
# separately for each hemisphere and angular resolution

df = subset(df, ( ses_id == 'dots+lines' | ses_id == 'meg'))

within = c('fold', 'roi')
for ( this_hemi in c('lh', 'rh') ) {
  for ( this_res in c(15, 30) ) {
    # select the data
    anova_df <- subset(df, hemi == this_hemi & ang_res == this_res)
    # perform the test
    aov_table = anova_test(data=anova_df, dv='coh', wid=sub_id, within=all_of(within), between='ses_id')
    # save output
    aov_fname = paste(params.stcPath, 'aov_between-dots+linesVSmeg_hemi-', this_hemi, '_res', this_res, sep='') 
   	if ( is.null(aov_table$ANOVA) ) {
      write.csv(aov_table, paste(aov_fname, '_anova.csv', sep=''))
    } else {
     write.csv(aov_table$ANOVA, paste(aov_fname, '_anova.csv', sep=''))
      write.csv(aov_table$`Mauchly's Test for Sphericity`, paste(aov_fname, '_mauchly.csv', sep=''))
      write.csv(aov_table$`Sphericity Corrections`, paste(aov_fname, '_sphericity.csv', sep=''))
    }
  }
}
