#!/bin/bash
# ======================================================================================================================
# do tsss max filter
# apply maxfilter to the raw data removing channels identified in previous step
# additionally it moves the data to the reference run
# and then transfers all the data to the maxfilter folder

# -- options --
# e.g. /mnt/storage/tier2/robbot/ProjectData/MEGRID/prg/analysis/04_doMaxFilter.sh

# ======================================================================================================================

function doMaxFilter () {
    # find the experiment in which this participant took part
    exp_id=$( awk -F"\t" -v pattern="$sub_id" '$1 == pattern { print $6 }' ${rootpath}participants.tsv )
    # then define the sessions variable accordingly
    if [ $exp_id == 1.0 ]; then ses_list=('dots' 'lines'); else ses_list=('meg'); fi

    for ses_id in ${ses_list[@]}; do
      # go the raw file directory
      cd $rootpath$1/ses-$ses_id/meg/
      # get all the .fif files
      raw_fnames=$(ls -- *.fif)
      # get the reference run id and check it exits. if it doesnt it means this participant channels have not been
      # checked yet
      ref_fname="${rootpath}bad/${1}_ses-${ses_id}_headPositionHistory.txt"
      if [ -f "$ref_fname" ]; then
        ref=$( tail -n 1 $ref_fname)
      else
        continue
      fi
      # for each run
      for fname in $raw_fnames; do
        # get the text file with the bad channels
        txt="${rootpath}bad/${fname%_meg.fif}_bads.txt"
        # if its not already been processed, ie if the file already exists in the max folder
        if ! compgen -G "$rootpath"derivatives/max/${fname%_meg.fif}*.fif; then
          # then check which maxfilter option to use
          # the following scripts are by the MEG technicians
          if [[ "$fname" == "$ref" || "$fname" == *"noise"* ]]; then
            # if its the reference run or the empty room dont move it
            doMaxFilterNoMoveTSSS $(echo "${fname}") $(echo "${txt}")
          else
            # otherwise move to reference run
            doMaxFilterAndMove $(echo "${fname}") $(echo "${txt}") $(echo "${ref}")
          fi
          # rename to match bids convetion and move to the output folder
          bids_fname=${fname%_meg*}_proc-tsss_meg
          mv ${fname%_meg*}*tsss*.fif ${rootpath}derivatives/max/${bids_fname}.fif
          mv ${fname%_meg*}*tsss*.log ${rootpath}derivatives/max/${bids_fname}.log
        fi
      done
    done
}

# ======================================================================================================================
# define the root path
rootpath=/mnt/storage/tier2/robbot/ProjectData/MEGRID/

# define the list of subjects that need to be processed
subj_list=$(cut -f1 ${rootpath}participants.tsv | tail -n +2)

# submit the jobs
for sub_id in ${subj_list[@]}; do
  # call the function
  doMaxFilter $sub_id
done
